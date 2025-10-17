import numpy as np
from scipy.spatial import Delaunay
from sofia.sofia.anisotropic_remesh import perform_local_flips_Delaunay


def test_flips_on_delaunay():
    pts = np.random.RandomState(0).rand(40, 2)
    tri = Delaunay(pts)
    T = tri.simplices.copy()
    flips = perform_local_flips_Delaunay(pts, T)
    assert flips == 0


def test_flips_on_quad():
    # build a convex quadrilateral with triangulation that needs flipping
    pts = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    # triangulation uses diagonal (0,2) but Delaunay should prefer (1,3)
    T = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    flips = perform_local_flips_Delaunay(pts, T)
    assert flips >= 1