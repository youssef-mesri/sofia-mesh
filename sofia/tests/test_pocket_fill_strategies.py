import numpy as np
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.triangulation import fill_pocket_quad, fill_pocket_steiner, fill_pocket_earclip
from sofia.core.conformity import check_mesh_conformity


def make_editor(points, triangles):
    return PatchBasedMeshEditor(np.asarray(points, dtype=float), np.asarray(triangles, dtype=int))


def test_fill_pocket_quad_success():
    # square plus an external triangle to seed mesh
    pts = [
        [0.0,0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0],
        [2.0,2.0]
    ]
    tris = [[4,0,1]]
    editor = make_editor(pts, tris)
    ok, details = fill_pocket_quad(editor, [0,1,2,3], min_tri_area=1e-12, reject_min_angle_deg=None)
    assert ok, f"quad strategy failed: {details}"
    assert details['method'] == 'quad'
    ok_conf, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok_conf, f"non conforming after quad fill: {msgs}"


def test_fill_pocket_steiner_success():
    # regular pentagon
    angles = np.linspace(0, 2*np.pi, 6)[:-1]
    ring = [[np.cos(a), np.sin(a)] for a in angles]
    pts = ring + [[3.0,3.0]]  # external triangle anchor
    tris = [[5,0,1]]
    editor = make_editor(pts, tris)
    ok, details = fill_pocket_steiner(editor, [0,1,2,3,4], min_tri_area=1e-14, reject_min_angle_deg=None)
    assert ok, f"steiner strategy failed: {details}"
    assert details['method'] == 'steiner'
    assert details['new_point_idx'] is not None
    ok_conf, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok_conf, f"non conforming after steiner fill: {msgs}"


def test_fill_pocket_earclip_degenerate_failure():
    # colinear points => earclip should fail producing no triangles
    pts = [[0,0],[1,0],[2,0],[3,0],[10,10]]
    tris = [[4,0,1]]
    editor = make_editor(pts, tris)
    ok, details = fill_pocket_earclip(editor, [0,1,2,3], min_tri_area=1e-12, reject_min_angle_deg=None)
    assert not ok
    assert 'earclip produced no triangles' in details['failure_reasons'] or 'earclip failed conformity' in '\n'.join(details['failure_reasons'])


def test_fill_pocket_earclip_success_pentagon():
    # pentagon should succeed if steiner is skipped (we call earclip directly)
    angles = np.linspace(0, 2*np.pi, 6)[:-1]
    ring = [[np.cos(a), np.sin(a)] for a in angles]
    pts = ring + [[4.0,4.0]]
    tris = [[5,0,1]]
    editor = make_editor(pts, tris)
    ok, details = fill_pocket_earclip(editor, [0,1,2,3,4], min_tri_area=1e-14, reject_min_angle_deg=None)
    assert ok, f"earclip strategy failed: {details}"
    assert details['method'] == 'earclip'
    ok_conf, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok_conf, f"non conforming after earclip fill: {msgs}"
