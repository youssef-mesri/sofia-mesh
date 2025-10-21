import numpy as np
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.conformity import is_boundary_vertex_from_maps


def test_collapse_safe_basic():
    pts, tris = build_random_delaunay(npts=60, seed=0)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # pick an interior edge
    edge = None
    for e, ts in editor.edge_map.items():
        if len(ts) == 2:
            edge = e
            break
    assert edge is not None
    ok = editor.collapse_safe(edge)
    assert isinstance(ok, bool)


def test_collapse_safe_boundary_reject():
    pts, tris = build_random_delaunay(npts=40, seed=1)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # find an edge with at least one endpoint on boundary
    edge = None
    for e, ts in editor.edge_map.items():
        a, b = e
        if is_boundary_vertex_from_maps(a, editor.edge_map) or is_boundary_vertex_from_maps(b, editor.edge_map):
            edge = e
            break
    assert edge is not None
    assert editor.collapse_safe(edge) is False


def test_collapse_safe_degenerate_reject():
    pts, tris = build_random_delaunay(npts=30, seed=2)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # pick an interior edge and create a near-degenerate adjacent triangle
    edge = None
    for e, ts in editor.edge_map.items():
        if len(ts) == 2:
            edge = e
            break
    assert edge is not None
    u, v = edge
    # find a triangle incident to u but not containing v and collapse its area
    tris_u = [int(t) for t in editor.v_map[u]]
    target = None
    for ti in tris_u:
        tri = list(editor.triangles[ti])
        if v not in tri:
            target = tri
            idx = ti
            break
    if target is not None:
        # move one vertex very close to another to make small area
        a, b, c = [int(x) for x in target]
        editor.points[b] = editor.points[a].copy()
        assert editor.collapse_safe(edge, min_tri_area=1e-12) is False