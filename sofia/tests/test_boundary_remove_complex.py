import numpy as np

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.conformity import check_mesh_conformity
from sofia.core.config import BoundaryRemoveConfig
from sofia.core.helpers import boundary_polygons_from_patch, select_outer_polygon
from sofia.core.triangulation import polygon_signed_area


def square_with_center():
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1 (corner to remove)
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.5],  # 4 center
    ], dtype=float)
    tris = np.array([
        [4, 0, 1],
        [4, 1, 2],
        [4, 2, 3],
        [4, 3, 0],
    ], dtype=int)
    return pts, tris


def test_default_rejects_non_preserving_corner_after_splits():
    """By default, boundary removals must preserve cavity area. The corner-with-splits
    case changes the cavity area (cavity triangles vs. new triangulation), so the 
    operation must be rejected without passing an explicit BoundaryRemoveConfig that 
    allows area changes.
    """
    pts, tris = square_with_center()
    # Default editor (no explicit boundary_remove_config) should enforce area preservation
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True, enforce_split_quality=False)
    # Apply splits around corner 1: (0,1) and (1,2)
    ok_a, msg_a, _ = editor.split_edge((0, 1))
    assert ok_a, f"split_edge(0,1) failed: {msg_a}"
    ok_b, msg_b, _ = editor.split_edge((1, 2))
    assert ok_b, f"split_edge(1,2) failed: {msg_b}"
    # Attempt removal; expect rejection due to cavity-area preservation
    try:
        editor.enforce_remove_quality = False
    except Exception:
        pass
    ok_r, msg_r, _ = editor.remove_node_with_patch(1)
    assert not ok_r, "Removal should be rejected by default when cavity area is not preserved"
    msg_str = str(msg_r)
    assert ('cavity area changed' in msg_str or 'area changed' in msg_str or 'area preservation failed' in msg_str or 'area not preserved' in msg_str), \
        f"Unexpected rejection reason: {msg_r}"
