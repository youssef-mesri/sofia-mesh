import numpy as np

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.conformity import check_mesh_conformity


def skinny_star():
    # Build a star where removing the center likely worsens worst min-angle
    # Outer hexagon slightly irregular to encourage skinny retriangulation
    R = 1.0
    pts_outer = np.array([
        [0.0, 0.0],
        [R, 0.0],
        [R, 0.8*R],
        [0.5*R, R],
        [0.0, R],
        [-0.2*R, 0.4*R],
    ], dtype=float)
    center = np.array([[0.3*R, 0.3*R]], dtype=float)
    pts = np.vstack([pts_outer, center])
    c = len(pts) - 1
    tris = np.array([[c, i, (i+1) % (len(pts)-1)] for i in range(len(pts)-1)], dtype=int)
    return pts, tris


def test_remove_node_allows_quality_degradation_when_flag_off():
    pts, tris = skinny_star()
    # Relax removal quality gating
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), enforce_remove_quality=False)
    # Attempt removal of center vertex
    v_center = len(pts) - 1
    ok, msg, _ = editor.remove_node_with_patch(v_center)
    assert ok, f"remove_node should succeed with relaxed quality: {msg}"

    # Conformity with tombstones allowed must hold
    ok1, msgs1 = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok1, f"Mesh not conforming after relaxed remove (pre-compact): {msgs1}"
    editor.compact_triangle_indices()
    ok2, msgs2 = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok2, f"Mesh not conforming after compaction: {msgs2}"
