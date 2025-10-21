import numpy as np
import types
import importlib
from sofia.core.operations import _evaluate_quality_change
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
import sofia.core.operations as ops_module


def make_simple_triangle_mesh():
    pts = np.array([[0.0,0.0],[1.0,0.0],[0.5,0.86602540378],[2.0,0.0]], dtype=float)
    # two triangles: an equilateral (0,1,2) and a skinny one (1,3,2)
    tris = [[0,1,2],[1,3,2]]
    return pts, tris


def test_evaluate_quality_change_accepts_when_quality_not_worse():
    pts, tris = make_simple_triangle_mesh()
    editor = PatchBasedMeshEditor(pts.copy(), np.array(tris, dtype=int))
    # old_tris = first triangle only (equilateral), new_tris = same
    old_tris = [tris[0]]
    new_tris = [tris[0]]
    # ensure operations uses the normalized quality metric by injecting a module-level helper
    def tri_q(points_arr, tris_arr):
        pts_a = np.asarray(points_arr, dtype=float)
        tris_a = np.asarray(tris_arr, dtype=int)
        if tris_a.size == 0:
            return np.empty((0,))
        p0 = pts_a[tris_a[:,0]]; p1 = pts_a[tris_a[:,1]]; p2 = pts_a[tris_a[:,2]]
        a = 0.5 * np.abs((p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0]))
        e0 = np.sum((p1-p0)**2, axis=1); e1 = np.sum((p2-p1)**2, axis=1); e2 = np.sum((p0-p2)**2, axis=1)
        denom = e0+e1+e2; safe = denom>0
        q = np.zeros(tris_a.shape[0], dtype=float); q[safe] = a[safe]/denom[safe]
        q = q * (12.0 / (np.sqrt(3.0)))
        return np.clip(q, 0.0, 1.0)
    import sofia.core.operations as ops_module
    import pytest
    pytest.MonkeyPatch().setattr(ops_module, '_triangle_qualities_norm', tri_q, raising=False)
    ok, msg = _evaluate_quality_change(editor, old_tris, new_tris, stats=None, op_label='test')
    assert ok and msg is None


def test_evaluate_quality_change_rejects_when_quality_degrades():
    pts, tris = make_simple_triangle_mesh()
    editor = PatchBasedMeshEditor(pts.copy(), np.array(tris, dtype=int))
    # old_tris = equilateral triangle, new_tris = skinny triangle
    old_tris = [tris[0]]
    new_tris = [tris[1]]
    # set small eps to make rejection likely
    editor.quality_metric_eps = 0.0
    # inject normalized-quality implementation to ensure metric is used
    def tri_q(points_arr, tris_arr):
        pts_a = np.asarray(points_arr, dtype=float)
        tris_a = np.asarray(tris_arr, dtype=int)
        if tris_a.size == 0:
            return np.empty((0,))
        p0 = pts_a[tris_a[:,0]]; p1 = pts_a[tris_a[:,1]]; p2 = pts_a[tris_a[:,2]]
        a = 0.5 * np.abs((p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0]))
        e0 = np.sum((p1-p0)**2, axis=1); e1 = np.sum((p2-p1)**2, axis=1); e2 = np.sum((p0-p2)**2, axis=1)
        denom = e0+e1+e2; safe = denom>0
        q = np.zeros(tris_a.shape[0], dtype=float); q[safe] = a[safe]/denom[safe]
        q = q * (12.0 / (np.sqrt(3.0)))
        return np.clip(q, 0.0, 1.0)
    import sofia.core.operations as ops_module
    import pytest
    pytest.MonkeyPatch().setattr(ops_module, '_triangle_qualities_norm', tri_q, raising=False)
    ok, msg = _evaluate_quality_change(editor, old_tris, new_tris, stats=None, op_label='test')
    assert not ok and msg is not None

def test_evaluate_quality_change_fallback_to_angle_on_exception(monkeypatch):
    pts, tris = make_simple_triangle_mesh()
    editor = PatchBasedMeshEditor(pts.copy(), np.array(tris, dtype=int))
    # monkeypatch _triangle_qualities_norm to raise
    def _raise(*a, **k):
        raise RuntimeError('boom')
    # monkeypatch the function on the operations module
    monkeypatch.setattr(ops_module, '_triangle_qualities_norm', _raise)
    # old/new that would otherwise be accept but will trigger fallback
    old_tris = [tris[0]]
    new_tris = [tris[0]]
    ok, msg = _evaluate_quality_change(editor, old_tris, new_tris, stats=None, op_label='test')
    # fallback compares angles and should accept since identical
    assert ok
