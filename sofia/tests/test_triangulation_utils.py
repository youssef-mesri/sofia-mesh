import numpy as np

from sofia.core.triangulation import (
    optimal_star_triangulation,
    best_star_triangulation_by_min_angle,
    ear_clip_triangulation,
    point_in_polygon,
    polygon_signed_area,
    polygon_has_self_intersections,
    retriangulate_patch_strict
)
from sofia.core.geometry import triangle_area, triangle_angles


def _min_angle_for_star(points, tris):
    worst = 180.0
    for t in tris:
        a = triangle_angles(points[t[0]], points[t[1]], points[t[2]])
        worst = min(worst, min(a))
    return worst


def test_point_in_polygon_basic():
    square = [(0,0),(1,0),(1,1),(0,1)]
    assert point_in_polygon(0.5,0.5,square)
    assert not point_in_polygon(1.5,0.5,square)
    # edge tolerance: point on edge treated as outside by current even-odd rule if ray logic so allow either
    _ = point_in_polygon(1.0,0.5,square)


def test_polygon_signed_area_ccw_positive():
    square = [(0,0),(1,0),(1,1),(0,1)]
    assert polygon_signed_area(square) > 0
    assert abs(polygon_signed_area(square) - 1.0) < 1e-12


def test_ear_clip_triangulation_concave():
    # Simple concave "arrow" polygon
    pts = np.array([
        [0.0,0.0],  # 0
        [2.0,0.0],  # 1
        [2.0,1.0],  # 2
        [1.0,2.0],  # 3 (tip)
        [0.0,1.0],  # 4
    ])
    poly = [0,1,2,3,4]
    tris = ear_clip_triangulation(pts, poly)
    # Expect n-2 triangles for simple polygon if successful
    assert len(tris) == len(poly) - 2
    # Combined area equals polygon area (allow small tolerance)
    poly_area = polygon_signed_area([tuple(pts[i]) for i in poly])
    area_sum = sum(abs(triangle_area(pts[t[0]], pts[t[1]], pts[t[2]])) for t in tris)
    assert abs(area_sum - poly_area) < 1e-9


def test_optimal_star_triangulation_square():
    pts = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    boundary = [0,1,2,3]
    star = optimal_star_triangulation(pts, boundary)
    # Expect exactly 2 triangles covering square
    assert star is not None
    assert len(star) == 2
    used = sorted({v for tri in star for v in tri})
    assert used == boundary  # all vertices used


def test_best_star_triangulation_quality_matches_bruteforce():
    # Slightly irregular pentagon so quality differs among anchors
    pts = np.array([
        [0.0,0.0],   # 0
        [2.0,0.2],   # 1
        [2.1,1.5],   # 2
        [1.0,2.1],   # 3
        [-0.2,1.0],  # 4
    ])
    boundary = [0,1,2,3,4]
    best = best_star_triangulation_by_min_angle(pts, boundary)
    assert best is not None
    best_quality = _min_angle_for_star(pts, best)
    # brute force all anchors (similar enumeration to library function) to confirm optimality
    anchor_qualities = []
    n = len(boundary)
    for i in range(n):
        v0 = boundary[i]
        cycle = boundary[i:] + boundary[:i]
        tris = []
        degenerate = False
        for j in range(1,n-1):
            tri = [v0, cycle[j], cycle[j+1]]
            a = abs(triangle_area(pts[tri[0]], pts[tri[1]], pts[tri[2]]))
            if a < 1e-12:
                degenerate = True; break
            tris.append(tri)
        if degenerate:
            continue
        anchor_qualities.append(_min_angle_for_star(pts, tris))
    assert best_quality >= max(anchor_qualities) - 1e-12


def test_retriangulate_patch_strict_with_added_point():
    # Square split into two triangles; retriangulate with added center point should create a fan of 4 triangles
    pts = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    tris = np.array([[0,1,2],[0,2,3]], dtype=int)
    new_pt = np.array([[0.5,0.5]])
    new_pts, new_tris, ok, new_tri_list = retriangulate_patch_strict(pts, tris, [0,1], new_point_coords=new_pt, strict_mode='centroid')
    assert ok
    # new point should be appended
    assert len(new_pts) == len(pts) + 1
    center_idx = len(new_pts) - 1
    # Expect 4 triangles containing center
    center_tris = [t for t in new_tri_list if center_idx in t]
    assert len(center_tris) == 4
    # All areas positive
    for t in center_tris:
        a = triangle_area(new_pts[t[0]], new_pts[t[1]], new_pts[t[2]])
        assert a != 0.0


def test_ear_clip_triangulation_degenerate_line_polygon():
    # Polygon with collinear points forming effectively a line
    pts = np.array([[0.0,0.0],[1.0,0.0],[2.0,0.0]])
    poly = [0,1,2]
    # Area zero => no triangles
    from sofia.core.triangulation import ear_clip_triangulation
    tris = ear_clip_triangulation(pts, poly)
    assert tris == []


def test_ear_clip_triangulation_duplicate_vertices():
    pts = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0]])
    poly = [0,1,1,2]
    from sofia.core.triangulation import ear_clip_triangulation
    import pytest
    with pytest.raises(ValueError):
        ear_clip_triangulation(pts, poly)


def test_polygon_self_intersection_detection():
    # Simple bow-tie (figure-eight) self-intersecting polygon
    poly = [(0,0),(2,2),(0,2),(2,0)]
    assert polygon_has_self_intersections(poly)
    # Convex square no intersection
    square = [(0,0),(1,0),(1,1),(0,1)]
    assert not polygon_has_self_intersections(square)


def test_star_triangulation_self_intersection_error():
    import pytest
    pts = np.array([[0.,0.],[2.,2.],[0.,2.],[2.,0.]])  # bow-tie crossing if order used directly
    boundary = [0,1,2,3]
    with pytest.raises(ValueError):
        optimal_star_triangulation(pts, boundary)


def test_optimal_star_triangulation_duplicate_vertices():
    # Boundary contains duplicate vertex index -> algorithm should skip or return None
    pts = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    boundary = [0,1,1,2,3]  # duplicate 1
    import pytest
    with pytest.raises(ValueError):
        optimal_star_triangulation(pts, boundary)


def test_best_star_triangulation_by_min_angle_small_invalid():
    pts = np.array([[0.,0.],[1.,0.]])  # only 2 points
    boundary = [0,1]
    assert best_star_triangulation_by_min_angle(pts, boundary) is None


def test_retriangulate_patch_strict_invalid_patch():
    # Patch with fewer than 3 distinct nodes
    pts = np.array([[0.,0.],[1.,0.]])
    tris = np.array([[0,1,1]])  # degenerate triangle
    new_pts, new_tris, ok, lst = retriangulate_patch_strict(pts, tris, [0], new_point_coords=None)
    assert not ok
    assert len(lst) == 0
