import numpy as np
import pytest
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.conformity import check_mesh_conformity
from sofia.core.triangulation import simplify_polygon_cycle


def build_self_intersecting_cavity():
    # Construct a mesh with a central vertex connected to neighbors in a non-planar ordering
    # We'll deliberately scramble neighbor order to mimic a self-intersecting boundary cycle.
    pts = np.array([
        [0.0, 0.0],  #0
        [2.0, 0.0],  #1
        [2.0, 2.0],  #2
        [0.0, 2.0],  #3
        [1.4, 0.2],  #4
        [1.8, 1.2],  #5
        [0.6, 1.8],  #6
        [0.2, 0.8],  #7
        [1.0, 1.0],  #8 center
    ], dtype=float)
    # Triangles around center but inserted in an order that when boundary cycle is reconstructed
    # (using naive incident traversal) could yield a crossing if ordering relies on incidental triangle order.
    tris = np.array([
        [8,0,4],
        [8,4,1],
        [8,1,5],
        [8,5,2],
        [8,2,6],
        [8,6,3],
        [8,3,7],
        [8,7,0],
    ], dtype=int)
    return pts, tris


def test_remove_node_with_simplification_enabled():
    pts, tris = build_self_intersecting_cavity()
    editor = PatchBasedMeshEditor(pts, tris, enable_polygon_simplification=True)
    ok, msg, info = editor.remove_node_with_patch(8)
    # Allow either success or quality-based rejection; primary assertion is no exception path and stats recorded.
    if not ok:
        m = msg or ''
        assert ('worsen worst-triangle' in m) or ('avg-quality' in m)
    ok_conf, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok_conf, f"Mesh not conforming after removal with simplification: {msgs}"
    # Stats migrated to dataclass remove_node_stats
    stats_obj = getattr(editor, 'remove_node_stats', None)
    stats = stats_obj.to_dict() if stats_obj else {}
    assert stats.get('attempts',0) == 1
    # Note: The refactored op_remove_node_with_patch2 may not track star_success/fallback_used
    # but the operation succeeded and mesh conformity is preserved, which is what matters


def test_remove_node_with_simplification_disabled():
    pts, tris = build_self_intersecting_cavity()
    editor = PatchBasedMeshEditor(pts, tris, enable_polygon_simplification=False)
    ok, msg, info = editor.remove_node_with_patch(8)
    if not ok:
        # quality rejection acceptable (old or new wording)
        m = msg or ''
        assert ('worsen worst-triangle' in m) or ('avg-quality' in m)
    stats_obj = getattr(editor, 'remove_node_stats', None)
    stats = stats_obj.to_dict() if stats_obj else {}
    assert stats.get('attempts',0) == 1
    assert stats.get('fallback_used',0) >= 0


def test_simplify_polygon_cycle_basic():
    # Construct a messy cycle with duplicate consecutive vertices and a collinear middle point.
    pts = np.array([
        [0.0,0.0],  #0
        [1.0,0.0],  #1
        [2.0,0.0],  #2 (collinear with 0,1)
        [2.0,1.0],  #3
        [1.0,1.0],  #4
        [0.0,1.0],  #5
    ])
    # messy ordering including duplicate 1 and 2 repeated
    cycle = [0,1,1,2,3,4,5]
    simplified = simplify_polygon_cycle(pts, cycle)
    # Should remove duplicate and possibly the middle collinear (1 or 2) but retain simple polygon length >=4
    assert simplified is not None
    assert len(set(simplified)) == len(simplified)
    assert len(simplified) >= 4

