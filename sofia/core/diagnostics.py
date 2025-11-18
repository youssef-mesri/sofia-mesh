"""Diagnostics helpers for mesh remeshing driver.

Functions operate on raw numpy arrays or a PatchBasedMeshEditor like
object passed in.
"""
from __future__ import annotations

import time
import logging
from typing import Dict, List, Tuple, Any

import numpy as np
from .constants import EPS_AREA
from .logging_utils import get_logger

logger = get_logger('sofia.diagnostics')

# Try importing numba for performance optimizations
try:
    import numba
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define dummy decorator if numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])


def compact_copy(editor):
    """Return a compacted (points, tris, mapping, active_idx) copy of the editor state.

    Tombstoned triangles (all -1) are removed and vertex indices remapped so the
    returned arrays are dense and suitable for conformity / inversion checks or plotting.

    mapping is returned as a numpy ndarray of shape (N_old,) with dtype int32:
      - mapping[old] = new compacted vertex index
      - mapping[old] = -1 for vertices not present in the compacted mesh
    """
    tris = np.ascontiguousarray(np.asarray(editor.triangles, dtype=np.int32))
    pts = np.ascontiguousarray(np.asarray(editor.points, dtype=np.float64))
    active_mask = ~np.all(tris == -1, axis=1)
    active_tris = tris[active_mask]
    active_idx = np.nonzero(active_mask)[0].tolist()
    if active_tris.size == 0:
        return pts, np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32), []
    used_verts = sorted(set(active_tris.flatten().tolist()))
    used_verts = [v for v in used_verts if v >= 0]
    # vectorized old->new array mapping
    n_old = int(pts.shape[0])
    old_to_new = np.full(n_old, -1, dtype=np.int32)
    for new_i, old_i in enumerate(used_verts):
        old_to_new[int(old_i)] = int(new_i)
    new_points = pts[used_verts]
    # vectorized remap of triangles using array indexing
    remapped = old_to_new[active_tris]
    # filter out any rows with -1 (shouldn't occur for active_tris, but keep safe)
    row_ok = ~np.any(remapped < 0, axis=1)
    new_tris = np.ascontiguousarray(remapped[row_ok]) if remapped.size else np.empty((0, 3), dtype=np.int32)
    return new_points, new_tris, old_to_new, active_idx


def _vectorized_boundary_edges(tris: np.ndarray) -> np.ndarray:
    """Return unique boundary edges (Nx2) using vectorized counting on compacted triangles."""
    if tris.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    a = tris[:, [0, 1]]; b = tris[:, [1, 2]]; c = tris[:, [2, 0]]
    edges = np.vstack((a, b, c)).astype(np.int32)
    edges.sort(axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return uniq[counts == 1]


@njit(cache=True)
def _extract_loops_numba(boundary_edges: np.ndarray, n_vertices: int):
    """Numba-optimized loop extraction from boundary edges.
    
    Returns arrays: loop_starts, loop_data
    - loop_starts[i] = start index in loop_data for loop i
    - loop_data = flattened vertex indices for all loops
    """
    if boundary_edges.shape[0] == 0:
        return np.array([0], dtype=np.int32), np.empty(0, dtype=np.int32)
    
    # Build adjacency list using arrays (Numba-friendly)
    # Count neighbors per vertex first
    neighbor_count = np.zeros(n_vertices, dtype=np.int32)
    for i in range(boundary_edges.shape[0]):
        a, b = boundary_edges[i, 0], boundary_edges[i, 1]
        neighbor_count[a] += 1
        neighbor_count[b] += 1
    
    # Allocate adjacency structure (max 2 neighbors per boundary vertex typically)
    max_neighbors = max(2, np.max(neighbor_count))
    adjacency = np.full((n_vertices, max_neighbors), -1, dtype=np.int32)
    adj_idx = np.zeros(n_vertices, dtype=np.int32)
    
    # Fill adjacency list
    for i in range(boundary_edges.shape[0]):
        a, b = boundary_edges[i, 0], boundary_edges[i, 1]
        adjacency[a, adj_idx[a]] = b
        adj_idx[a] += 1
        adjacency[b, adj_idx[b]] = a
        adj_idx[b] += 1
    
    # Find all vertices involved in boundary
    boundary_verts = np.unique(boundary_edges.flatten())
    
    # Track visited vertices
    visited = np.zeros(n_vertices, dtype=np.int8)
    
    # Pre-allocate output arrays (worst case: each vertex is its own loop)
    max_loops = len(boundary_verts)
    loop_starts_temp = np.zeros(max_loops + 1, dtype=np.int32)
    loop_data_temp = np.zeros(len(boundary_verts), dtype=np.int32)
    
    n_loops = 0
    data_idx = 0
    
    for start_idx in range(len(boundary_verts)):
        start_v = boundary_verts[start_idx]
        if visited[start_v]:
            continue
        
        # BFS to find component
        queue = np.zeros(n_vertices, dtype=np.int32)
        queue_start = 0
        queue_end = 1
        queue[0] = start_v
        visited[start_v] = 1
        
        component = np.zeros(n_vertices, dtype=np.int32)
        comp_size = 0
        
        while queue_start < queue_end:
            u = queue[queue_start]
            queue_start += 1
            component[comp_size] = u
            comp_size += 1
            
            # Add unvisited neighbors
            for j in range(neighbor_count[u]):
                w = adjacency[u, j]
                if w != -1 and not visited[w]:
                    visited[w] = 1
                    queue[queue_end] = w
                    queue_end += 1
        
        # Mark start of this loop
        loop_starts_temp[n_loops] = data_idx
        
        # Order the loop greedily
        if comp_size <= 2:
            # Trivial case - just copy sorted
            for i in range(comp_size):
                loop_data_temp[data_idx] = component[i]
                data_idx += 1
            # Sort the small array in-place
            if comp_size == 2 and loop_data_temp[data_idx-2] > loop_data_temp[data_idx-1]:
                tmp = loop_data_temp[data_idx-2]
                loop_data_temp[data_idx-2] = loop_data_temp[data_idx-1]
                loop_data_temp[data_idx-1] = tmp
        else:
            # Greedy ordering: follow edges
            cur = component[0]
            loop_data_temp[data_idx] = cur
            data_idx += 1
            
            prev = -1
            used = np.zeros(n_vertices, dtype=np.int8)
            used[cur] = 1
            
            for _ in range(comp_size - 1):
                # Find next neighbor
                found_next = False
                for j in range(neighbor_count[cur]):
                    nb = adjacency[cur, j]
                    if nb != -1 and nb != prev and not used[nb]:
                        loop_data_temp[data_idx] = nb
                        data_idx += 1
                        used[nb] = 1
                        prev = cur
                        cur = nb
                        found_next = True
                        break
                
                if not found_next:
                    break
        
        n_loops += 1
    
    # Mark end
    loop_starts_temp[n_loops] = data_idx
    
    # Return trimmed arrays
    return loop_starts_temp[:n_loops+1], loop_data_temp[:data_idx]


def extract_boundary_loops(points, tris) -> List[List[int]]:
    """Return explicit boundary loops as lists of vertex indices.

    This operates on compacted points/tris (no -1 tombstones). Each loop is a
    connected component of edges that appear exactly once across all triangles.
    Ordering around the loop is approximated by greedy neighbor walking; the
    loop may start at an arbitrary vertex, but contains each boundary vertex
    exactly once.
    """
    boundary_edges = _vectorized_boundary_edges(np.asarray(tris, dtype=np.int32))
    if boundary_edges.size == 0:
        return []
    
    # Use Numba version if available for large meshes
    if HAS_NUMBA and boundary_edges.shape[0] > 20:
        n_vertices = int(np.max(boundary_edges) + 1) if boundary_edges.size > 0 else 0
        loop_starts, loop_data = _extract_loops_numba(boundary_edges, n_vertices)
        
        # Convert flat representation to list of lists
        loops = []
        for i in range(len(loop_starts) - 1):
            start = loop_starts[i]
            end = loop_starts[i + 1]
            loops.append(loop_data[start:end].tolist())
        
        return loops
    
    # Fallback to Python implementation for small meshes or when Numba unavailable
    from collections import defaultdict, deque
    # Build adjacency per boundary edges
    adj = defaultdict(list)
    for a, b in boundary_edges:
        a = int(a); b = int(b)
        adj[a].append(b)
        adj[b].append(a)
    # Extract connected components and try to order vertices around each component
    visited = set()
    loops: List[List[int]] = []
    for v in list(adj.keys()):
        if v in visited:
            continue
        # BFS to get component set
        comp = []
        dq = deque([v])
        visited.add(v)
        comp_set = {v}
        while dq:
            u = dq.popleft()
            comp.append(u)
            for w in adj.get(u, []):
                if w not in comp_set:
                    comp_set.add(w)
                    visited.add(w)
                    dq.append(w)
        # Order loop greedily
        if len(comp_set) <= 2:
            loops.append(sorted(list(comp_set)))
            continue
        start = next(iter(comp_set))
        ordered = [start]
        prev = None
        cur = start
        used = set([start])
        # Follow adjacency choosing the next neighbor not equal to previous
        while True:
            nbrs = adj.get(cur, [])
            nxt = None
            for nb in nbrs:
                if nb == prev:
                    continue
                if nb not in used:
                    nxt = nb
                    break
            if nxt is None:
                break
            ordered.append(nxt)
            used.add(nxt)
            prev, cur = cur, nxt
            if cur == start:
                break
        loops.append(ordered)
    return loops


def count_boundary_loops(points, tris) -> int:
    """Return number of connected boundary loops in the mesh (compacted points/tris)."""
    return len(extract_boundary_loops(points, tris))


def compact_from_arrays(points_arr, tris_arr):
    """Compaction helper for raw arrays: return (new_points, new_tris, mapping, active_idx).

    mapping is an ndarray old->new (int32), -1 for removed.
    """
    pts = np.ascontiguousarray(np.asarray(points_arr, dtype=np.float64))
    tris = np.ascontiguousarray(np.asarray(tris_arr, dtype=np.int32))
    if tris.size == 0:
        return pts, np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32), []
    active_mask = ~np.all(tris == -1, axis=1)
    active_tris = tris[active_mask]
    active_idx = np.nonzero(active_mask)[0].tolist()
    used_verts = sorted(set(active_tris.flatten().tolist()))
    used_verts = [v for v in used_verts if v >= 0]
    n_old = int(pts.shape[0])
    old_to_new = np.full(n_old, -1, dtype=np.int32)
    for new_i, old_i in enumerate(used_verts):
        old_to_new[int(old_i)] = int(new_i)
    new_points = pts[used_verts]
    remapped = old_to_new[active_tris]
    row_ok = ~np.any(remapped < 0, axis=1)
    new_tris = np.ascontiguousarray(remapped[row_ok]) if remapped.size else np.empty((0, 3), dtype=np.int32)
    return new_points, new_tris, old_to_new, active_idx


def find_inverted_triangles(points, tris, eps=EPS_AREA):
    """Return list of (tri_idx, signed_area) for triangles inverted or near-degenerate."""
    try:
        from .geometry import triangles_signed_areas
        T = np.asarray(tris, dtype=int)
        if T.size == 0:
            return []
        areas = triangles_signed_areas(points, T)
        inv = []
        for i, a in enumerate(areas):
            if (a <= 0.0) or (abs(a) <= eps):
                inv.append((int(i), float(a)))
        return inv
    except Exception:
        inv = []
        for ti, t in enumerate(tris):
            try:
                p0 = points[int(t[0])]
                p1 = points[int(t[1])]
                p2 = points[int(t[2])]
                a = 0.5 * np.cross(p1 - p0, p2 - p0)
                if a <= 0.0 or abs(a) <= eps:
                    inv.append((ti, float(a)))
            except Exception:
                continue
        return inv


def detect_and_dump_pockets(editor, op_desc=None, dump: bool = None):
    """Detect quad pockets (boundary components of size 4 with no interior triangles).

    Returns (pockets, mapping) where pockets is a list of dicts for each pocket.

    Dumping behavior:
      - By default (dump=None), write a diagnostic NPZ only when logger level is DEBUG.
      - If dump is True, force writing regardless of level; if False, never write.
    """
    try:
        pts_c, tris_c, mapping_c, active_idx_c = compact_copy(editor)
        from matplotlib.path import Path
        from collections import defaultdict
        edge_count = defaultdict(int)
        for t in tris_c:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            for e in ((a, b), (b, c), (c, a)):
                edge_count[tuple(sorted(e))] += 1
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        adj = defaultdict(list)
        for a, b in boundary_edges:
            adj[a].append(b)
            adj[b].append(a)
        visited = set()
        pockets = []
        for v in adj:
            if v in visited:
                continue
            stack = [v]
            comp = set()
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                visited.add(u)
                for w in adj[u]:
                    if w not in comp:
                        stack.append(w)
            comp = sorted(comp)
            if len(comp) == 4:
                coords = pts_c[comp]
                poly = Path(coords)
                inside_tris = [tidx for tidx, t in enumerate(tris_c) if poly.contains_point(((pts_c[int(t[0])] + pts_c[int(t[1])] + pts_c[int(t[2])]) / 3.0))]
                inside_pts = [i for i, p in enumerate(pts_c) if poly.contains_point(p) and i not in comp]
                pockets.append({'verts': comp, 'coords': coords, 'inside_pts': inside_pts, 'inside_tris': inside_tris})
        if pockets:
            # Determine dumping policy: default only when DEBUG
            do_dump = logger.isEnabledFor(logging.DEBUG) if dump is None else bool(dump)
            if do_dump:
                fn = f"pocket_dump_{int(time.time())}.npz"
                try:
                    np.savez(fn, points=pts_c, tris=tris_c, pockets=pockets, op_desc=str(op_desc), mapping=mapping_c)
                    logger.debug('Wrote pocket diagnostic to %s op=%s', fn, op_desc)
                except Exception as e:
                    logger.debug('Could not write pocket diagnostic: %s', e)
        return pockets, mapping_c
    except Exception as e:
        logger.debug('detect_and_dump_pockets failed: %s', e)
    return [], (np.empty((0,), dtype=np.int32))


__all__ = [
    'compact_copy',
    'count_boundary_loops', 'extract_boundary_loops',
    'compact_from_arrays',
    'find_inverted_triangles',
    'detect_and_dump_pockets',
]

