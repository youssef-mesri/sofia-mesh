"""Shared mesh helper utilities extracted from legacy monolith to reduce cyclic imports."""
from __future__ import annotations
from typing import List, Sequence, Optional
import numpy as np
from .geometry import point_in_polygon

def boundary_cycle_from_incident_tris(triangles, incident_tris: Sequence[int], v_idx: int) -> Optional[List[int]]:
    """Reconstruct ordered neighbor cycle around v_idx from its incident triangle indices.
    Returns list of neighbor vertex indices in cyclic order or None if not a simple interior cycle.
    Conditions for success:
      - Each incident triangle contributes exactly two neighbors (so v_idx appears exactly once per tri)
      - Each neighbor has degree 2 in the local adjacency (simple cycle)
      - Walk traverses all neighbors without branching or early termination
    """
    adj = {}
    try:
        for t_idx in incident_tris:
            tri = triangles[t_idx]
            others = [int(x) for x in tri if int(x) != int(v_idx)]
            if len(others) != 2:
                return None
            a, b = others
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    except Exception:
        return None
    if not adj:
        return None
    for neis in adj.values():
        if len(neis) != 2:
            return None
    start = next(iter(adj))
    cycle = [start]
    prev = None
    curr = start
    while True:
        nexts = [n for n in adj[curr] if n != prev]
        if not nexts:
            break
        nxt = nexts[0]
        if nxt == start:
            break
        cycle.append(nxt)
        prev, curr = curr, nxt
        if len(cycle) > len(adj):
            break
    if len(cycle) != len(adj):
        return None
    return cycle

__all__ = [
    'boundary_cycle_from_incident_tris',
    'patch_nodes_for_triangles',
    'boundary_polygons_from_patch',
    'select_outer_polygon',
    'detect_empty_pockets',
    'extract_patch_nodes',
]


# -----------------------
# Patch and boundary helpers (moved from mesh_modifier2)
# -----------------------
def patch_nodes_for_triangles(triangles, tri_indices):
    nodes = set()
    for t in tri_indices:
        tri = triangles[t]
        if np.all(np.array(tri) == -1):
            continue
        nodes.update(tri)
    return sorted(nodes)


def boundary_polygons_from_patch(triangles, patch_tri_indices):
    """
    Return a list of boundary polygons (each polygon is a list of vertex indices in order).
    There may be multiple polygons if the patch has holes; we return all detected loops.
    Fallback: if no boundary found, return empty list.
    """
    patch_tri_indices = set(patch_tri_indices)
    # count edges inside patch
    edge_count = {}
    edge_tri = {}
    for t_idx in patch_tri_indices:
        tri = triangles[t_idx]
        # skip disabled triangles
        if np.all(np.array(tri) == -1):
            continue
        for i in range(3):
            a = int(tri[i]); b = int(tri[(i+1)%3])
            key = tuple(sorted((a,b)))
            edge_count[key] = edge_count.get(key, 0) + 1
            edge_tri.setdefault(key, []).append(t_idx)
    # boundary edges: those appearing exactly once inside the patch
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        return []
    # adjacency for walk
    adj = {}
    for a,b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    # collect loops by walking unvisited edges
    visited_edges = set()
    polygons = []
    for e in boundary_edges:
        if e in visited_edges:
            continue
        # start with oriented edge e
        a,b = e
        loop = [a, b]
        visited_edges.add(e)
        prev = a; current = b
        # walk until closed or stuck
        while True:
            neighbors = adj.get(current, [])
            # choose neighbor that isn't prev
            next_v = None
            for nb in neighbors:
                if nb != prev:
                    next_v = nb
                    break
            if next_v is None:
                # stuck (open chain) -> abort this loop
                break
            edge_key = tuple(sorted((current, next_v)))
            if edge_key in visited_edges:
                # if next_v equals loop[0] we closed; else it's already visited -> break
                if next_v == loop[0]:
                    loop.append(next_v)
                break
            loop.append(next_v)
            visited_edges.add(edge_key)
            prev, current = current, next_v
            if len(loop) > 10000:
                break
        # if loop closed (last equals first) remove duplicate end and accept
        if len(loop) >= 3 and loop[0] == loop[-1]:
            polygons.append(loop[:-1])
        else:
            # maybe walk found a chain but not closed; try to close by looking for connection back to start
            # fallback: ignore incomplete chain
            pass
    # If polygons empty, try convex hull as fallback
    if not polygons:
        return []
    # Return polygons
    return polygons


def select_outer_polygon(points, polygons):
    if not polygons:
        return None
    best = None
    best_area = -1.0
    for poly in polygons:
        coords = np.asarray([points[int(v)] for v in poly])
        # polygon area via shoelace (signed)
        x = coords[:,0]; y = coords[:,1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area > best_area:
            best_area = area
            best = poly
    return best


def detect_empty_pockets(points, triangles):
    """
    Detect boundary loops (pockets) in the compacted mesh and report whether
    each pocket is empty (i.e. contains no active triangles inside its polygon).

    Returns a list of dicts with keys:
      - vertices: list of vertex indices forming the boundary loop (ordered)
      - area: polygon area (positive)
      - tri_count: number of active triangle centroids strictly inside the polygon
      - vertex_count: number of mesh vertices (not on the loop) strictly inside
      - is_empty: True if tri_count == 0

    Note: triangles marked as [-1,-1,-1] are ignored (tombstoned).
    """
    pts = np.asarray(points)
    tris = np.asarray(triangles, dtype=int)
    # gather active triangle indices
    active_idx = [i for i, t in enumerate(tris) if not np.all(np.array(t) == -1)]
    if not active_idx:
        return []

    # use existing boundary extraction on the full active set
    polygons = boundary_polygons_from_patch(tris, active_idx)
    results = []
    # precompute centroids of active triangles
    tri_centroids = []
    for i in active_idx:
        t = tris[i]
        p0 = pts[int(t[0])]; p1 = pts[int(t[1])]; p2 = pts[int(t[2])]
        c = np.mean(np.vstack((p0, p1, p2)), axis=0)
        tri_centroids.append((i, c))

    # for each polygon compute area and test containment
    for poly in polygons:
        coords = [tuple(pts[int(v)]) for v in poly]
        # compute polygon area (shoelace)
        arr = np.asarray(coords)
        if arr.shape[0] < 3:
            continue
        x = arr[:,0]; y = arr[:,1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        # count triangles whose centroids lie strictly inside polygon
        tri_inside = 0
        for _, c in tri_centroids:
            if point_in_polygon(float(c[0]), float(c[1]), coords):
                tri_inside += 1

        # count vertices (excluding polygon vertices) inside polygon
        vert_inside = 0
        poly_set = set(int(v) for v in poly)
        for vidx, p in enumerate(pts):
            if int(vidx) in poly_set:
                continue
            if point_in_polygon(float(p[0]), float(p[1]), coords):
                vert_inside += 1

        results.append({
            'vertices': list(poly),
            'area': float(area),
            'tri_count': int(tri_inside),
            'vertex_count': int(vert_inside),
            'is_empty': (tri_inside == 0),
        })
    return results


def extract_patch_nodes(points, triangles, tri_idx, ring=1):
    tri = triangles[tri_idx]
    patch_tris = set([tri_idx])
    for _ in range(ring):
        new = set()
        for j in range(len(triangles)):
            if j in patch_tris: continue
            for t in list(patch_tris):
                if len(set(triangles[j]).intersection(triangles[t])) >= 2:
                    new.add(j); break
        if not new: break
        patch_tris.update(new)
    patch_nodes = set()
    for t in patch_tris:
        patch_nodes.update(triangles[t])
    return sorted(patch_nodes), sorted(patch_tris)

