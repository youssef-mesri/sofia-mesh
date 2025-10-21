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
    # Test that flipping terminates even on cocircular configurations
    # A perfect square has all 4 vertices cocircular, so both diagonals are equally valid
    # Delaunay triangulations. The algorithm should recognize this and not flip infinitely.
    pts = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    T = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    flips = perform_local_flips_Delaunay(pts, T)
    # Should perform 0 flips (cocircular) and terminate quickly without infinite loop
    assert flips == 0, f"Expected 0 flips on cocircular quad, got {flips}"