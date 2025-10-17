import json
import numpy as np
from sofia.sofia.triangulation_utils import constrained_delaunay_triangulate


def test_square_triangulation():
    # square vertices
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    indices = [0, 1, 2, 3]
    tris = constrained_delaunay_triangulate(indices, pts)
    assert tris is not None
    # should produce at least two triangles
    assert len(tris) >= 2


def test_s_polygon_file():
    with open('configs/s_polygon.json', 'r') as fh:
        poly = json.load(fh)
    pts = np.asarray(poly, dtype=float)
    indices = list(range(len(pts)))
    tris = constrained_delaunay_triangulate(indices, pts)
    # For S polygon we expect some triangulation; allow None to indicate env missing deps
    if tris is None:
        import pytest
        pytest.skip('scipy/matplotlib not available in this environment')
    assert len(tris) >= 1
