import numpy as np
from sofia.sofia.operations import _triangle_qualities_norm


def test_triangle_qualities_norm_values():
    # Equilateral triangle side length 1
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.5, np.sqrt(3)/2.0])
    points = np.vstack([p0, p1, p2])
    tris = np.array([[0,1,2]], dtype=np.int32)
    q = _triangle_qualities_norm(points, tris)
    assert q.shape == (1,)
    assert q[0] > 0.999 and q[0] <= 1.0

    # Isosceles thin triangle -> lower quality
    p0 = np.array([0.0,0.0]); p1 = np.array([1.0,0.0]); p2 = np.array([0.5, 0.01])
    points2 = np.vstack([p0,p1,p2])
    tris2 = np.array([[0,1,2]], dtype=np.int32)
    q2 = _triangle_qualities_norm(points2, tris2)
    assert q2.shape == (1,)
    assert q2[0] < 0.1

    # Degenerate colinear -> zero
    p0 = np.array([0.0,0.0]); p1 = np.array([0.5,0.0]); p2 = np.array([1.0,0.0])
    points3 = np.vstack([p0,p1,p2])
    tris3 = np.array([[0,1,2]], dtype=np.int32)
    q3 = _triangle_qualities_norm(points3, tris3)
    assert q3.shape == (1,)
    assert q3[0] == 0.0
