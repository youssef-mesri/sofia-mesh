import numpy as np
import math
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.anisotropic_remesh import compute_Mhalf, compute_Mminushalf, optimize_vertex_metric_equilateral
from sofia.core.constants import EPS_TINY


def analytic_grad_isotropic(x, neighbor_coords):
    """Analytic gradient of E(x)=sum_j (||x_j-x|| - 1)^2 for isotropic metric.

    Returns gradient as numpy array shape (2,).
    """
    g = np.zeros(2, dtype=float)
    for xj in neighbor_coords:
        d = xj - x
        lj = np.linalg.norm(d)
        if lj <= 0:
            continue
        g += -2.0 * (lj - 1.0) / lj * d
    return g


def fd_grad(func, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for k in range(len(x)):
        e = np.zeros_like(x); e[k] = 1.0
        fp = func(x + h * e)
        fm = func(x - h * e)
        g[k] = (fp - fm) / (2.0 * h)
    return g


def test_analytic_vs_fd_gradient():
    # build a simple star around center at origin
    center = np.array([0.05, -0.02])
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    neigh = np.vstack([np.stack([np.cos(a), np.sin(a)]) for a in angles])
    # energy function
    def E(x):
        s = 0.0
        for xj in neigh:
            lj = np.linalg.norm(xj - x)
            s += (lj - 1.0)**2
        return float(s)

    g_fd = fd_grad(E, center, h=1e-6)
    g_an = analytic_grad_isotropic(center, neigh)
    assert np.allclose(g_fd, g_an, atol=1e-6, rtol=1e-4)


def test_near_singular_spd_and_optimizer():
    # Construct a near-singular SPD: eigenvalues [1e-18, 1e-2] rotated
    theta = 0.3
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    vals = np.array([1e-18, 1e-2])
    M = R @ np.diag(vals) @ R.T
    # compute sqrt and inverse sqrt should be finite (clipping inside functions)
    Mh = compute_Mhalf(M)
    Minvh = compute_Mminushalf(M)
    assert np.all(np.isfinite(Mh))
    assert np.all(np.isfinite(Minvh))
    I_approx = Minvh @ Mh
    assert np.allclose(I_approx, np.eye(2), atol=1e-8)

    # Now exercise optimizer with an anisotropic metric function that returns near-singular M
    pts, tris = build_random_delaunay(npts=30, seed=11)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # pick an interior vertex
    from sofia.core.conformity import is_boundary_vertex_from_maps
    v = next((i for i in range(len(editor.points)) if not is_boundary_vertex_from_maps(i, editor.edge_map)), None)
    assert v is not None

    def near_singular_metric(x):
        # small eigenvalue depends weakly on x to avoid exact singularity in tests
        eps = max(1e-18, 1e-16 + 1e-18 * (x[0] + x[1]))
        vals = np.array([eps, 1e-2])
        return R @ np.diag(vals) @ R.T

    newx, info = optimize_vertex_metric_equilateral(v, editor, near_singular_metric, max_iter=6)
    # optimizer should return finite coordinates and info dict
    assert np.all(np.isfinite(newx))
    assert isinstance(info, dict)
    assert np.isfinite(info.get('energy', 0.0))


def test_anisotropic_analytic_gradient_vs_fd():
    # Fixed rotation (eigenvectors) but eigenvalues vary with x -> anisotropic but analytically tractable
    theta = 0.37
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    # eigenvalue fields: lam1(x) = a1 + b1*x0, lam2(x) = a2 + b2*x1
    a1, b1 = 1.0, 0.2
    a2, b2 = 0.4, 0.15

    def lam1(x):
        return float(a1 + b1 * float(x[0]))

    def lam2(x):
        return float(a2 + b2 * float(x[1]))

    def metric_fn(x):
        L = np.diag([lam1(x), lam2(x)])
        return R @ L @ R.T

    from sofia.core.anisotropic_remesh import average_metric

    # build neighbors around center
    center = np.array([0.12, -0.08])
    angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    neigh = np.vstack([np.stack([0.9 * np.cos(a) + 0.05, 0.9 * np.sin(a) - 0.02]) for a in angles])

    def E(x):
        s = 0.0
        for xj in neigh:
            Mbar = average_metric(metric_fn(x), metric_fn(xj))
            d = xj - x
            l = np.sqrt(float(d @ (Mbar @ d)))
            s += (l - 1.0)**2
        return float(s)

    # finite-difference gradient
    def fd_grad_local(f, x, h=1e-7):
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        for k in range(len(x)):
            e = np.zeros_like(x); e[k] = 1.0
            fp = f(x + h * e)
            fm = f(x - h * e)
            g[k] = (fp - fm) / (2.0 * h)
        return g

    g_fd = fd_grad_local(E, center, h=1e-7)

    # analytic gradient under fixed-rotation assumption
    def analytic_grad(x):
        x = np.asarray(x, dtype=float)
        g_total = np.zeros(2, dtype=float)
        for xj in neigh:
            d = xj - x
            lam1_x = lam1(x); lam2_x = lam2(x)
            lam1_j = lam1(xj); lam2_j = lam2(xj)
            alpha1 = math.sqrt(lam1_x * lam1_j)
            alpha2 = math.sqrt(lam2_x * lam2_j)
            # Mbar = R diag(alpha1,alpha2) R^T
            Mbar = R @ np.diag([alpha1, alpha2]) @ R.T
            l = math.sqrt(float(d @ (Mbar @ d)))
            if l == 0.0:
                continue
            # derivatives of eigenvalues
            grad_lam1 = np.array([b1, 0.0], dtype=float)
            grad_lam2 = np.array([0.0, b2], dtype=float)
            # derivatives of alphas: dalpha_k/dx = 0.5 * sqrt(lam_k_j / lam_k_x) * grad_lam_k
            dalpha1 = 0.5 * math.sqrt(lam1_j / lam1_x) * grad_lam1
            dalpha2 = 0.5 * math.sqrt(lam2_j / lam2_x) * grad_lam2
            # Mbar d term
            Mbd = Mbar @ d
            # basis vectors (columns of R)
            s1 = R[:, 0]
            s2 = R[:, 1]
            # compute vector b = -2*Mbar*d + [ d^T (dMbar/dx_m) d ]_m
            b = -2.0 * Mbd
            # for each coordinate m, add sum_k (dalpha_k/dx_m) * (s_k^T d)^2
            proj1 = float(np.dot(s1, d))
            proj2 = float(np.dot(s2, d))
            b += np.array([dalpha1[0] * proj1 * proj1 + dalpha2[0] * proj2 * proj2,
                           dalpha1[1] * proj1 * proj1 + dalpha2[1] * proj2 * proj2], dtype=float)
            dl_dx = (0.5 / float(l)) * b
            g_total += 2.0 * (l - 1.0) * dl_dx
        return g_total

    g_an = analytic_grad(center)
    assert np.allclose(g_fd, g_an, atol=1e-6, rtol=1e-4)