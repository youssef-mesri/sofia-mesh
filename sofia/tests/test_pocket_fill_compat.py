import numpy as np
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor
from sofia.sofia import pocket_fill as pf_compat
from sofia.sofia import triangulation as tri


def make_editor(points, triangles):
    return PatchBasedMeshEditor(np.asarray(points, dtype=float), np.asarray(triangles, dtype=int))


def _call_and_capture(fn, editor, verts, **kwargs):
    # create a deep copy of editor to ensure calls don't interfere
    import copy
    ed_copy = copy.deepcopy(editor)
    ok, details = fn(ed_copy, verts, **kwargs)
    return ok, details, ed_copy


def test_compat_fill_pocket_strategies_match():
    # prepare a pentagon scenario for steiner/earclip and a quad for quad
    angles = np.linspace(0, 2*np.pi, 6)[:-1]
    pentagon = [[np.cos(a), np.sin(a)] for a in angles]
    pts = pentagon + [[3.0, 3.0]]
    tris = [[5, 0, 1]]
    editor = make_editor(pts, tris)

    # steiner: compare results
    ok_a, det_a, ed_a = _call_and_capture(tri.fill_pocket_steiner, editor, [0,1,2,3,4], min_tri_area=1e-14, reject_min_angle_deg=None)
    ok_b, det_b, ed_b = _call_and_capture(pf_compat.fill_pocket_steiner, editor, [0,1,2,3,4], min_tri_area=1e-14, reject_min_angle_deg=None)
    assert ok_a == ok_b, f"steiner ok mismatch: {det_a} vs {det_b}"
    # If both succeeded, check method name and triangle counts and mesh conformity
    if ok_a:
        assert det_a.get('method') == det_b.get('method')
        assert len(det_a.get('triangles', [])) == len(det_b.get('triangles', []))

    # quad: square test
    pts_q = [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[2.0,2.0]]
    tris_q = [[4,0,1]]
    editor_q = make_editor(pts_q, tris_q)
    ok_a, det_a, ed_a = _call_and_capture(tri.fill_pocket_quad, editor_q, [0,1,2,3], min_tri_area=1e-12, reject_min_angle_deg=None)
    ok_b, det_b, ed_b = _call_and_capture(pf_compat.fill_pocket_quad, editor_q, [0,1,2,3], min_tri_area=1e-12, reject_min_angle_deg=None)
    assert ok_a == ok_b, f"quad ok mismatch: {det_a} vs {det_b}"
    if ok_a:
        assert det_a.get('method') == det_b.get('method')
        # verify mesh conformity for both
        from sofia.sofia.conformity import check_mesh_conformity
        ok_conf_a, _ = check_mesh_conformity(ed_a.points, ed_a.triangles, allow_marked=False)
        ok_conf_b, _ = check_mesh_conformity(ed_b.points, ed_b.triangles, allow_marked=False)
        assert ok_conf_a and ok_conf_b

    # earclip: pentagon should work
    editor_e = make_editor(pts, tris)
    ok_a, det_a, ed_a = _call_and_capture(tri.fill_pocket_earclip, editor_e, [0,1,2,3,4], min_tri_area=1e-14, reject_min_angle_deg=None)
    ok_b, det_b, ed_b = _call_and_capture(pf_compat.fill_pocket_earclip, editor_e, [0,1,2,3,4], min_tri_area=1e-14, reject_min_angle_deg=None)
    assert ok_a == ok_b, f"earclip ok mismatch: {det_a} vs {det_b}"
    if ok_a:
        assert det_a.get('method') == det_b.get('method')