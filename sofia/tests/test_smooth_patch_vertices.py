import numpy as np
from scipy.spatial import Delaunay
from sofia.sofia.anisotropic_remesh import smooth_patch_vertices


def test_smooth_changes_interior():
    pts = np.random.RandomState(1).rand(40, 2)
    tri = Delaunay(pts)
    T = tri.simplices.copy()
    # perturb interior vertices slightly
    pts_pert = pts.copy()
    pts_pert[5] += 0.05
    out, moved = smooth_patch_vertices(pts_pert, T, omega=0.6, iterations=2)
    assert moved > 0


def test_smooth_preserves_boundary():
    pts = np.random.RandomState(2).rand(30, 2)
    tri = Delaunay(pts)
    T = tri.simplices.copy()
    out, moved = smooth_patch_vertices(pts, T, omega=0.5, iterations=1)
    # compute boundary from edges
    from sofia.sofia.conformity import build_edge_to_tri_map
    edge_map = build_edge_to_tri_map(T)
    boundary_vs = set([u for e, ts in edge_map.items() if len(ts) == 1 for u in e])
    # check boundary vertices unchanged (within tolerance)
    for v in sorted(boundary_vs)[:5]:
        assert np.allclose(out[int(v)], pts[int(v)])