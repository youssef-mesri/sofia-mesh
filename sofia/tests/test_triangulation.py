import numpy as np
import pytest

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor


def make_editor(points, triangles):
    pts = np.asarray(points, dtype=float)
    tris = np.asarray(triangles, dtype=int)
    return PatchBasedMeshEditor(pts, tris)


def test_try_fill_quad_pocket_success():
    # square pts
    points = [
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [2.0, 2.0],  # 4 external
    ]
    # single external triangle to make mesh non-empty and valid
    triangles = [
        [4, 0, 1]
    ]
    editor = make_editor(points, triangles)
    before_ntri = len(editor.triangles)
    ok, details = editor.try_fill_pocket([0,1,2,3], min_tri_area=1e-12, reject_min_angle_deg=None)
    assert ok is True
    # details should include the triangles added (or method info)
    assert isinstance(details, dict)
    assert len(editor.triangles) >= before_ntri + 2


def test_try_fill_pentagon_earclip_success():
    # regular pentagon centered at origin radius 1
    angles = np.linspace(0, 2*np.pi, 6)[:-1]
    points = [[np.cos(a), np.sin(a)] for a in angles]
    # add an external triangle to have a non-empty active mesh
    points.append([3.0, 3.0])
    triangles = [[5, 0, 1]]
    editor = make_editor(points, triangles)
    verts = [0,1,2,3,4]
    before = len(editor.triangles)
    ok, details = editor.try_fill_pocket(verts, min_tri_area=1e-14, reject_min_angle_deg=None)
    assert ok is True
    assert isinstance(details, dict)
    # should add n-2 triangles for n-gon
    assert len(editor.triangles) >= before + (len(verts) - 2)


def test_try_fill_degenerate_failure():
    # colinear points (degenerate polygon)
    points = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [10.0, 10.0]
    ]
    triangles = [[5,0,1]]
    editor = make_editor(points, triangles)
    verts = [0,1,2,3,4]
    ok, details = editor.try_fill_pocket(verts, min_tri_area=1e-12, reject_min_angle_deg=None)
    assert ok is False
    # details should indicate failure reasons
    assert isinstance(details, dict)
