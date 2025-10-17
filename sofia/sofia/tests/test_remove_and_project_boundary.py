import numpy as np
from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.conformity import check_mesh_conformity


def test_remove_degenerate_and_fill():
    pts, tris = build_random_delaunay(npts=30, seed=9)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # create a degenerate triangle by collapsing two vertices of tri 0
    if len(editor.triangles) > 0:
        t0 = editor.triangles[0]
        a, b, c = [int(x) for x in t0]
        editor.points[a] = editor.points[b].copy()
    res = editor.remove_degenerate_triangles()
    assert res['tombstoned'] >= 1
    # compacted result should be conforming (allowing no tombstones)
    ok, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok


def test_project_boundary_vertices():
    pts, tris = build_random_delaunay(npts=30, seed=5)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    loops = editor._extract_ordered_boundary_loops()
    if not loops:
        return
    loop = loops[0]
    v = loop[0]
    orig = editor.points[v].copy()
    editor.points[v] = orig + np.array([0.01, -0.02])
    moved = editor.project_boundary_vertices()
    assert moved > 0
    # after projection, the moved vertex should be closer to original boundary polyline
    assert np.linalg.norm(editor.points[v] - orig) < 0.05