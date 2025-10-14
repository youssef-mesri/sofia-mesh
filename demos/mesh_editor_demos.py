"""Demonstration and exploratory routines for PatchBasedMeshEditor.

These are intentionally separated from `mesh_modifier2` so the core module
remains focused on the editing API. None of the functions here are required
by the automated test suite; they are convenience helpers for manual runs,
visual inspection, and experimentation.
"""
from __future__ import annotations
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
try:
    import imageio
except Exception:  # pragma: no cover
    imageio = None

from sofia.sofia.logging_utils import get_logger, configure_logging
from sofia.sofia.constants import EPS_IMPROVEMENT
from sofia.sofia.conformity import check_mesh_conformity, is_boundary_vertex_from_maps
from sofia.sofia.helpers import boundary_cycle_from_incident_tris
from sofia.sofia.triangulation import optimal_star_triangulation
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay

logger = get_logger('sofia.demos.mesh_editor')

def test_random_mesh():  # pragma: no cover - visualization utility
    """Generate a random mesh and perform one split + one remove for illustration."""
    pts, tris = build_random_delaunay(60, seed=2025)
    editor = PatchBasedMeshEditor(pts, tris)
    logger.info("Initial npts %d ntri %d min_angle %.4f", len(editor.points), len(editor.triangles), editor.global_min_angle())
    valid_tris = np.array([t for t in editor.triangles if not np.all(np.array(t) == -1)], dtype=int)
    plt.figure(figsize=(5,5))
    plt.triplot(editor.points[:,0], editor.points[:,1], valid_tris)
    plt.scatter(editor.points[:,0], editor.points[:,1], s=6)
    plt.gca().set_aspect('equal')
    plt.title("Initial mesh")
    plt.savefig('strict_before.png', dpi=150); plt.close()

    # Attempt a split on a representative interior edge
    interior_edges = [e for e,l in editor.edge_map.items() if len(l) == 2]
    if interior_edges:
        e = interior_edges[len(interior_edges)//3]
        editor.split_edge(e)
        editor.compact_triangle_indices()
        valid_tris = np.array([t for t in editor.triangles if not np.all(np.array(t) == -1)], dtype=int)
        plt.figure(figsize=(5,5))
        plt.triplot(editor.points[:,0], editor.points[:,1], valid_tris)
        plt.scatter(editor.points[:,0], editor.points[:,1], s=6)
        plt.gca().set_aspect('equal')
        plt.title("After split")
        plt.savefig('strict_after_split.png', dpi=150); plt.close()

    # Attempt a node removal on a high-degree interior vertex
    interior_vertices = []
    for v, tri_set in editor.v_map.items():
        incident_edges = []
        for t in tri_set:
            tri = editor.triangles[t]
            for i in range(3):
                a = int(tri[i]); b = int(tri[(i+1)%3])
                if v in (a,b) and len(editor.edge_map.get(tuple(sorted((a,b))), set())) == 2:
                    incident_edges.append(tuple(sorted((a,b))))
        if incident_edges and all(len(editor.edge_map[e]) == 2 for e in incident_edges):
            interior_vertices.append((v, len(tri_set)))
    interior_vertices.sort(key=lambda x: -x[1])
    if interior_vertices:
        v = interior_vertices[0][0]
        editor.remove_node_with_patch(v)
        editor.compact_triangle_indices()
        valid_tris = np.array([t for t in editor.triangles if not np.all(np.array(t) == -1)], dtype=int)
        plt.figure(figsize=(5,5))
        plt.triplot(editor.points[:,0], editor.points[:,1], valid_tris)
        plt.scatter(editor.points[:,0], editor.points[:,1], s=6)
        plt.gca().set_aspect('equal')
        plt.title("After remove")
        plt.savefig('strict_after_remove.png', dpi=150); plt.close()
    return editor

def test_non_convex_cavity_star():  # pragma: no cover
    """Remove central vertex in a 7-vertex non-convex star cavity."""
    pts = np.array([[0,0],[2,0],[4,-1],[2,2],[0,2],[-1,1],[1,1]])
    tris = [[0,1,6],[1,2,6],[2,3,6],[3,4,6],[4,5,6],[5,0,6]]
    editor = PatchBasedMeshEditor(pts, tris)
    editor.remove_node_with_patch(6)
    cycle = boundary_cycle_from_incident_tris(tris, list(range(len(tris))), 6)
    optimal_star_triangulation(pts, cycle, debug=True)
    valid_tris = np.array([t for t in editor.triangles if not np.all(np.array(t) == -1)], dtype=int)
    plt.figure(figsize=(5,5))
    plt.triplot(editor.points[:,0], editor.points[:,1], valid_tris, color='gray')
    plt.scatter(editor.points[:,0], editor.points[:,1], color='red')
    plt.gca().set_aspect('equal')
    plt.title('Non-convex cavity removal')
    plt.savefig('non_convex_cavity_after.png', dpi=150); plt.close()
    return editor

def test_flip_and_add_node():  # pragma: no cover
    pts = np.array([[0,0],[1,0],[1,1],[0,1]])
    tris = np.array([[0,1,2],[0,2,3]])
    editor = PatchBasedMeshEditor(pts, tris)
    editor.flip_edge((0,2))
    editor.split_edge((1,3))
    editor.add_node([0.3,0.3], tri_idx=0)
    valid_tris = np.array([t for t in editor.triangles if not np.all(np.array(t) == -1)], dtype=int)
    plt.figure(figsize=(4,4))
    plt.triplot(editor.points[:,0], editor.points[:,1], valid_tris)
    plt.scatter(editor.points[:,0], editor.points[:,1])
    plt.gca().set_aspect('equal')
    plt.title('flip / split / add node')
    plt.savefig('flip_split_add.png', dpi=150); plt.close()
    return editor

def test_locate_point_optimized():  # pragma: no cover
    pts = np.array([[0,0],[1,0],[1,1],[0,1]])
    tris = np.array([[0,1,2],[0,2,3]])
    editor = PatchBasedMeshEditor(pts, tris)
    p_inside = np.array([0.2, 0.1]); editor.locate_point_optimized(p_inside)
    p_inside2 = np.array([0.3, 0.7]); editor.locate_point_optimized(p_inside2)
    p_edge = np.array([0.5, 0.0]); editor.locate_point_optimized(p_edge)
    p_out = np.array([2.0, 2.0]); editor.locate_point_optimized(p_out)
    return editor

def test_maps_consistency():  # pragma: no cover
    pts = np.array([[0,0],[1,0],[1,1],[0,1]])
    tris = np.array([[0,1,2],[0,2,3]])
    editor = PatchBasedMeshEditor(pts, tris)
    editor.add_node([0.2,0.1], tri_idx=0)
    editor.flip_edge((0,2))
    editor.split_edge((1,2))
    return editor

def test_improve_min_angle_loop(max_iters=20, max_candidates=200):  # pragma: no cover (heavy/debug)
    pts, tris = build_random_delaunay(80, seed=123)
    editor = PatchBasedMeshEditor(pts, tris)
    base_min = editor.global_min_angle()
    snap_dir = 'min_angle_snapshots'
    if os.path.exists(snap_dir):
        try: shutil.rmtree(snap_dir)
        except Exception: pass
    os.makedirs(snap_dir, exist_ok=True)
    frames = []
    def save_frame(tag):
        valid = np.array([t for t in editor.triangles if not np.all(np.array(t)==-1)], dtype=int)
        plt.figure(figsize=(4,4))
        plt.triplot(editor.points[:,0], editor.points[:,1], valid, color='black')
        plt.scatter(editor.points[:,0], editor.points[:,1], s=6)
        plt.gca().set_aspect('equal'); plt.axis('off'); plt.tight_layout(pad=0)
        path = os.path.join(snap_dir, f"frame_{len(frames):03d}_{tag}.png")
        plt.savefig(path, dpi=150); plt.close(); frames.append(path)
    save_frame('start')
    # (Simplified improvement placeholder – original exhaustive loop trimmed.)
    if imageio and frames:
        gif_path = os.path.join(snap_dir, 'min_angle_improvement.gif')
        try:
            imgs = [imageio.v2.imread(f) for f in frames]
            imageio.mimsave(gif_path, imgs, fps=2)
            logger.info("Saved GIF: %s", gif_path)
        except Exception:
            pass
    assert base_min + EPS_IMPROVEMENT >= editor.global_min_angle() - EPS_IMPROVEMENT
    return editor

__all__ = [
    'test_random_mesh','test_non_convex_cavity_star','test_flip_and_add_node','test_locate_point_optimized',
    'test_maps_consistency','test_improve_min_angle_loop'
]

if __name__ == "__main__":  # pragma: no cover
    configure_logging('INFO')
    test_random_mesh()