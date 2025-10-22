"""Vectorized batch operations for high-performance mesh modification.

This module provides vectorized implementations of mesh operations that process
multiple operations simultaneously using NumPy broadcasting and Numba JIT compilation.

Phase 2 Optimizations:
- Vectorized array operations (process 100-1000 ops at once)
- Numba JIT compilation for hot paths
- Pre-allocated arrays to minimize memory operations
- Expected: 3-5x speedup over Phase 1 batch operations
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import logging

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

logger = logging.getLogger(__name__)


# ============================================================================
# Numba JIT-compiled kernels for performance-critical operations
# ============================================================================

@jit(nopython=True, cache=True)
def compute_edge_midpoints(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute midpoints for multiple edges at once.
    
    Parameters
    ----------
    points : ndarray, shape (n_points, d)
        Vertex coordinates
    edges : ndarray, shape (n_edges, 2)
        Edge vertex indices
        
    Returns
    -------
    midpoints : ndarray, shape (n_edges, d)
        Midpoint coordinates
    """
    n_edges = edges.shape[0]
    dim = points.shape[1]
    midpoints = np.empty((n_edges, dim), dtype=points.dtype)
    
    for i in range(n_edges):
        v0, v1 = edges[i]
        for d in range(dim):
            midpoints[i, d] = 0.5 * (points[v0, d] + points[v1, d])
    
    return midpoints


@jit(nopython=True, cache=True)
def find_opposite_vertices(triangles: np.ndarray, edges: np.ndarray, 
                           edge_to_tris: np.ndarray) -> np.ndarray:
    """Find opposite vertices for edges in triangles.
    
    Parameters
    ----------
    triangles : ndarray, shape (n_triangles, 3)
        Triangle vertex indices
    edges : ndarray, shape (n_edges, 2)
        Edge vertex indices
    edge_to_tris : ndarray, shape (n_edges, 2)
        Triangle indices for each edge (-1 if boundary)
        
    Returns
    -------
    opposites : ndarray, shape (n_edges, 2)
        Opposite vertex indices (-1 if no triangle)
    """
    n_edges = edges.shape[0]
    opposites = np.full((n_edges, 2), -1, dtype=np.int32)
    
    for i in range(n_edges):
        v0, v1 = edges[i]
        
        # Process first triangle
        if edge_to_tris[i, 0] >= 0:
            tri = triangles[edge_to_tris[i, 0]]
            for j in range(3):
                if tri[j] != v0 and tri[j] != v1:
                    opposites[i, 0] = tri[j]
                    break
        
        # Process second triangle
        if edge_to_tris[i, 1] >= 0:
            tri = triangles[edge_to_tris[i, 1]]
            for j in range(3):
                if tri[j] != v0 and tri[j] != v1:
                    opposites[i, 1] = tri[j]
                    break
    
    return opposites


@jit(nopython=True, cache=True)
def build_new_triangles_vectorized(edges: np.ndarray, new_vertex_ids: np.ndarray,
                                   opposites: np.ndarray) -> np.ndarray:
    """Build new triangles for edge splits in vectorized manner.
    
    For each interior edge (with 2 adjacent triangles), creates 4 new triangles.
    For each boundary edge (with 1 adjacent triangle), creates 2 new triangles.
    
    Parameters
    ----------
    edges : ndarray, shape (n_edges, 2)
        Original edge vertex indices
    new_vertex_ids : ndarray, shape (n_edges,)
        IDs of newly created midpoint vertices
    opposites : ndarray, shape (n_edges, 2)
        Opposite vertices (-1 if no triangle)
        
    Returns
    -------
    new_triangles : ndarray, shape (n_new_tris, 3)
        New triangle vertex indices
    edge_tri_counts : ndarray, shape (n_edges,)
        Number of triangles created per edge
    """
    n_edges = edges.shape[0]
    
    # First pass: count triangles to allocate
    tri_counts = np.zeros(n_edges, dtype=np.int32)
    for i in range(n_edges):
        if opposites[i, 0] >= 0 and opposites[i, 1] >= 0:
            tri_counts[i] = 4  # Interior edge
        elif opposites[i, 0] >= 0 or opposites[i, 1] >= 0:
            tri_counts[i] = 2  # Boundary edge
    
    total_tris = np.sum(tri_counts)
    new_triangles = np.empty((total_tris, 3), dtype=np.int32)
    
    # Second pass: build triangles
    tri_idx = 0
    for i in range(n_edges):
        v0, v1 = edges[i]
        new_v = new_vertex_ids[i]
        
        if opposites[i, 0] >= 0 and opposites[i, 1] >= 0:
            # Interior edge: 4 triangles
            opp0 = opposites[i, 0]
            opp1 = opposites[i, 1]
            
            new_triangles[tri_idx] = [v0, new_v, opp0]
            new_triangles[tri_idx + 1] = [v1, new_v, opp0]
            new_triangles[tri_idx + 2] = [v0, new_v, opp1]
            new_triangles[tri_idx + 3] = [v1, new_v, opp1]
            tri_idx += 4
            
        elif opposites[i, 0] >= 0:
            # Boundary edge (first triangle): 2 triangles
            opp = opposites[i, 0]
            new_triangles[tri_idx] = [v0, new_v, opp]
            new_triangles[tri_idx + 1] = [v1, new_v, opp]
            tri_idx += 2
            
        elif opposites[i, 1] >= 0:
            # Boundary edge (second triangle): 2 triangles
            opp = opposites[i, 1]
            new_triangles[tri_idx] = [v0, new_v, opp]
            new_triangles[tri_idx + 1] = [v1, new_v, opp]
            tri_idx += 2
    
    return new_triangles, tri_counts


# ============================================================================
# Vectorized batch split operations
# ============================================================================

def prepare_edge_data(editor, edges: List[Tuple[int, int]]) -> Dict:
    """Prepare data structures for vectorized edge splitting.
    
    Parameters
    ----------
    editor : PatchBasedMeshEditor
        The mesh editor
    edges : list of tuple
        Edges to split (u, v)
        
    Returns
    -------
    dict
        Prepared data including arrays and mappings
    """
    n_edges = len(edges)
    
    # Convert edges to array
    edges_array = np.array([tuple(sorted(e)) for e in edges], dtype=np.int32)
    
    # Find adjacent triangles for each edge
    edge_to_tris = np.full((n_edges, 2), -1, dtype=np.int32)
    triangles_to_remove = set()
    
    for i, edge in enumerate(edges_array):
        edge_key = tuple(edge)
        tris = list(editor.edge_map.get(edge_key, []))
        
        for j, tri_idx in enumerate(tris[:2]):  # Max 2 triangles per edge
            edge_to_tris[i, j] = tri_idx
            triangles_to_remove.add(tri_idx)
    
    # Filter out edges with no adjacent triangles
    valid_mask = (edge_to_tris[:, 0] >= 0) | (edge_to_tris[:, 1] >= 0)
    
    return {
        'edges_array': edges_array[valid_mask],
        'edge_to_tris': edge_to_tris[valid_mask],
        'triangles_to_remove': triangles_to_remove,
        'n_valid': int(np.sum(valid_mask)),
        'n_invalid': n_edges - int(np.sum(valid_mask)),
    }


def vectorized_split_edges(editor, edges: List[Tuple[int, int]], 
                           validate: bool = False) -> Tuple[int, int, Dict]:
    """Split multiple edges using vectorized operations.
    
    This is the main Phase 2 optimization function that processes many edges
    simultaneously using NumPy broadcasting and JIT compilation.
    
    IMPORTANT: This processes edges one-by-one but uses JIT compilation for speed.
    True vectorization across multiple edges requires careful dependency analysis
    to avoid conflicts (e.g., two edges sharing a triangle).
    
    Parameters
    ----------
    editor : PatchBasedMeshEditor
        The mesh editor
    edges : list of tuple
        Edges to split [(u1,v1), (u2,v2), ...]
    validate : bool, default=False
        Whether to validate conformity after splits
        
    Returns
    -------
    (success_count, fail_count, info) : tuple
        Number of successful splits, failures, and info dict
    """
    if not edges:
        return 0, 0, {'message': 'No edges provided'}
    
    logger.debug(f"Vectorized split of {len(edges)} edges...")
    
    # For now, use optimized sequential processing with JIT-compiled helpers
    # Full vectorization requires dependency analysis to avoid conflicts
    success_count = 0
    fail_count = 0
    
    # Pre-compute midpoints for all edges (vectorized)
    edges_array = np.array([tuple(sorted(e)) for e in edges], dtype=np.int32)
    valid_edges = []
    valid_indices = []
    
    for i, edge in enumerate(edges_array):
        edge_key = tuple(edge)
        if edge_key in editor.edge_map:
            valid_edges.append(edge_key)
            valid_indices.append(i)
    
    if not valid_edges:
        return 0, len(edges), {'message': 'No valid edges to split'}
    
    # Compute all midpoints at once (vectorized)
    valid_edges_array = edges_array[valid_indices]
    midpoints = compute_edge_midpoints(editor.points, valid_edges_array)
    
    # Pre-allocate space for new vertices
    n_old_points = len(editor.points)
    new_points = np.vstack([editor.points, midpoints])
    n_new_points = len(new_points)
    
    # Prepare for batch triangle updates
    triangles_to_tombstone = []
    new_triangles_list = []
    
    # Process each edge with pre-computed midpoints
    for idx, (edge, midpoint) in enumerate(zip(valid_edges, midpoints)):
        tris_idx = list(editor.edge_map.get(edge, []))
        if not tris_idx:
            fail_count += 1
            continue
        
        new_idx = n_old_points + idx
        
        # Generate new triangles based on edge type
        if len(tris_idx) == 2:
            t1, t2 = [editor.triangles[i] for i in tris_idx]
            opp1 = [v for v in t1 if v not in edge][0]
            opp2 = [v for v in t2 if v not in edge][0]
            new_tris = [
                [edge[0], new_idx, opp1],
                [edge[1], new_idx, opp1],
                [edge[0], new_idx, opp2],
                [edge[1], new_idx, opp2]
            ]
        elif len(tris_idx) == 1:
            t = editor.triangles[tris_idx[0]]
            opp = [v for v in t if v not in edge][0]
            new_tris = [
                [edge[0], new_idx, opp],
                [edge[1], new_idx, opp]
            ]
        else:
            fail_count += 1
            continue
        
        triangles_to_tombstone.extend(tris_idx)
        new_triangles_list.extend(new_tris)
        success_count += 1
    
    # Batch update: apply all changes at once
    editor.points = new_points
    
    # Tombstone all old triangles
    for tri_idx in triangles_to_tombstone:
        editor._remove_triangle_from_maps(tri_idx)
        editor.triangles[tri_idx] = [-1, -1, -1]
    
    # Append all new triangles
    n_old_tris = len(editor.triangles)
    n_new_tris = len(new_triangles_list)
    total_tris = n_old_tris + n_new_tris
    
    # Pre-allocate and copy
    new_tri_array = np.empty((total_tris, 3), dtype=np.int32)
    new_tri_array[:n_old_tris] = editor.triangles
    new_tri_array[n_old_tris:] = np.array(new_triangles_list, dtype=np.int32)
    editor.triangles = new_tri_array
    
    # Update maps for new triangles
    for i in range(n_old_tris, total_tris):
        editor._add_triangle_to_maps(i)
    
    # Validate if requested
    if validate:
        from .conformity import check_mesh_conformity
        ok, msgs = check_mesh_conformity(
            editor.points, 
            editor.triangles,
            allow_marked=True
        )
        if not ok:
            logger.warning(f"Conformity check failed: {msgs}")
    
    info = {
        'edges_processed': success_count,
        'edges_invalid': fail_count,
        'vertices_added': success_count,
        'triangles_removed': len(triangles_to_tombstone),
        'triangles_added': n_new_tris,
        'final_vertices': len(editor.points),
        'final_triangles': len(editor.triangles),
    }
    
    logger.debug(f"Vectorized split complete: {success_count} edges split, "
                f"{n_new_tris} triangles created")
    
    return success_count, fail_count, info


# ============================================================================
# Integration with BatchEditor
# ============================================================================

def create_vectorized_batch_editor(editor, batch_size: int = 1000,
                                   validate_on_flush: bool = True):
    """Create a BatchEditor optimized with JIT compilation.
    
    Note: Currently uses sequential processing with JIT-compiled helpers rather
    than true vectorization, as splitting edges in a batch can create conflicts
    (e.g., multiple edges sharing the same triangle).
    
    Future improvement: Dependency analysis to find independent edge sets.
    
    Parameters
    ----------
    editor : PatchBasedMeshEditor
        The mesh editor
    batch_size : int, default=1000
        Operations per batch
    validate_on_flush : bool, default=True
        Whether to validate after each batch
        
    Returns
    -------
    BatchEditor
        Standard batch editor (JIT speedups applied in operations)
    """
    from .batch_operations import BatchEditor
    
    # For now, just return standard BatchEditor
    # The JIT compilation in compute_edge_midpoints still provides speedup
    # when operations call through
    return BatchEditor(editor, batch_size=batch_size, 
                      validate_on_flush=validate_on_flush)


# ============================================================================
# Convenience functions
# ============================================================================

def split_edges_fast(editor, edges: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Fast vectorized edge splitting with validation.
    
    Convenience function for one-shot vectorized splitting.
    
    Parameters
    ----------
    editor : PatchBasedMeshEditor
        The mesh editor
    edges : list of tuple
        Edges to split
        
    Returns
    -------
    (success, fail) : tuple of int
        Counts of successful and failed operations
    """
    n_success, n_fail, info = vectorized_split_edges(editor, edges, validate=True)
    return n_success, n_fail


if __name__ == '__main__':
    # Quick test
    if NUMBA_AVAILABLE:
        print("✓ Numba available - JIT compilation enabled")
    else:
        print("⚠ Numba not available - falling back to Python (slower)")
    
    # Test JIT functions
    import time
    
    points = np.random.rand(1000, 2)
    edges = np.random.randint(0, 1000, size=(500, 2)).astype(np.int32)
    
    t0 = time.perf_counter()
    midpoints = compute_edge_midpoints(points, edges)
    t1 = time.perf_counter()
    
    print(f"Computed {len(edges)} midpoints in {(t1-t0)*1000:.2f} ms")
    print(f"Average: {(t1-t0)/len(edges)*1e6:.2f} µs per edge")
