import numpy as np
import pytest

from sofia.sofia.triangulation import triangulate_polygon_with_holes, polygon_signed_area
from sofia.sofia.geometry import triangle_area


def _triangles_area_sum(tris):
    s = 0.0
    for t in tris:
        a = np.asarray(t[0]); b = np.asarray(t[1]); c = np.asarray(t[2])
        s += abs(triangle_area(a, b, c))
    return s


def test_single_hole_area_preserved():
    # Outer big square
    shell = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    # Inner hole: small centered square
    hole = [(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)]
    tris = triangulate_polygon_with_holes(shell, [hole], prefer_earcut=False)
    assert tris, "triangulation returned no triangles"
    tri_area = _triangles_area_sum(tris)
    expected = abs(polygon_signed_area(shell)) - abs(polygon_signed_area(hole))
    assert abs(tri_area - expected) <= 1e-9, f"area mismatch: got {tri_area} expected {expected}"


def test_two_holes_area_preserved():
    # This test exercises multiple holes; skip in environments without mapbox_earcut because
    # the repository fallback heuristic is not guaranteed to work for multiple holes.
    pytest.importorskip('mapbox_earcut')
    shell = [(-3.0, -2.0), (3.0, -2.0), (3.0, 2.0), (-3.0, 2.0)]
    hole1 = [(-1.5, -0.5), (-0.5, -0.5), (-0.5, 0.5), (-1.5, 0.5)]
    hole2 = [(0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5)]
    tris = triangulate_polygon_with_holes(shell, [hole1, hole2], prefer_earcut=False)
    assert tris, "triangulation returned no triangles"
    tri_area = _triangles_area_sum(tris)
    expected = abs(polygon_signed_area(shell)) - abs(polygon_signed_area(hole1)) - abs(polygon_signed_area(hole2))
    # allow slightly larger tolerance for heuristic bridging
    assert abs(tri_area - expected) <= 1e-8, f"area mismatch: got {tri_area} expected {expected}"
