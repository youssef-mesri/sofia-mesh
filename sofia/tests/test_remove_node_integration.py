import numpy as np
from types import SimpleNamespace
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor
from sofia.sofia.conformity import check_mesh_conformity


def test_remove_node_prefers_area_preserving_candidate():
    # Build a simple quad with a center vertex (4) forming a star of 4 triangles
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.5],  # 4 center (to be removed)
    ], dtype=float)
    tris = np.array([
        [0,1,4],
        [1,2,4],
        [2,3,4],
        [3,0,4],
    ], dtype=int)

    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True)
    # Configure boundary remove to require area preservation
    br_cfg = SimpleNamespace()
    br_cfg.require_area_preservation = True
    br_cfg.area_tol_rel = 1e-6
    br_cfg.area_tol_abs_factor = 4.0
    br_cfg.prefer_area_preserving_star = True
    br_cfg.prefer_worst_angle_star = True
    editor.boundary_remove_config = br_cfg

    ok, msg, info = editor.remove_node_with_patch(4)
    assert ok, f"remove_node_with_patch failed: {msg}"

    # Collect active triangles (non-tombstoned)
    active_tris = [tuple(sorted(map(int, t))) for t in editor.triangles if not np.all(np.array(t) == -1)]
    active_set = set(active_tris)
    expected = {tuple(sorted([0,1,2])), tuple(sorted([0,2,3]))}
    assert active_set == expected, f"Expected area-preserving triangulation {expected}, got {active_set}"

    # Ensure mesh conformity (allow tombstoned triangles not expected here)
    ok_conf, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok_conf, f"Mesh not conforming after removal: {msgs}"
