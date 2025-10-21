import numpy as np
from sofia.core.anisotropic_remesh import compute_Mhalf, compute_Mminushalf


def random_spd(seed=0):
    rng = np.random.RandomState(seed)
    A = rng.randn(2, 2)
    M = A @ A.T + np.eye(2) * 1e-3
    return M


def test_compute_Mhalf_reconstructs():
    M = random_spd(1)
    Mh = compute_Mhalf(M)
    recon = Mh @ Mh
    assert np.allclose(recon, M, atol=1e-10)


def test_compute_Mminushalf_inverts():
    M = random_spd(2)
    Mh = compute_Mhalf(M)
    Minvh = compute_Mminushalf(M)
    I_approx = Minvh @ Mh
    assert np.allclose(I_approx, np.eye(2), atol=1e-10)