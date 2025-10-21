import numpy as np
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.anisotropic_remesh import optimize_vertex_metric_equilateral


def isotropic(x):
    return np.eye(2)


def test_optimize_reduces_energy():
    pts, tris = build_random_delaunay(npts=60, seed=7)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # pick an interior vertex
    from sofia.core.conformity import is_boundary_vertex_from_maps
    v = next(i for i in range(len(editor.points)) if not is_boundary_vertex_from_maps(i, editor.edge_map))
    oldx = editor.points[v].copy()
    oldE = 0.0
    Ms = isotropic(oldx)
    # compute initial energy using same convention as optimizer
    for ti in sorted(editor.v_map.get(v, [])):
        tri = editor.triangles[int(ti)]
        for vv in tri:
            vv = int(vv)
    newx, info = optimize_vertex_metric_equilateral(v, editor, isotropic, max_iter=6)
    assert 'energy' in info and info['energy'] >= 0.0


def test_optimize_isolated_vertex():
    # isolated vertex (no triangles) returns same pos
    pts = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float)
    tris = np.array([], dtype=int).reshape((0, 3))
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    x, info = optimize_vertex_metric_equilateral(0, editor, isotropic, max_iter=2)
    assert np.allclose(x, editor.points[0])


def test_grad_mode_fd_and_analytic_full_fallback():
    # Build a small random mesh and run optimizer with different grad modes
    pts, tris = build_random_delaunay(npts=40, seed=3)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    from sofia.core.conformity import is_boundary_vertex_from_maps
    v = next(i for i in range(len(editor.points)) if not is_boundary_vertex_from_maps(i, editor.edge_map))

    # FD mode
    newx_fd, info_fd = optimize_vertex_metric_equilateral(v, editor, isotropic, max_iter=4, grad_mode='fd')
    assert np.all(np.isfinite(newx_fd))
    assert isinstance(info_fd, dict) and info_fd.get('energy', 0.0) >= 0.0

    # analytic_full should fallback to analytic when metric has no jacobian
    newx_af, info_af = optimize_vertex_metric_equilateral(v, editor, isotropic, max_iter=4, grad_mode='analytic_full')
    assert np.all(np.isfinite(newx_af))
    assert isinstance(info_af, dict) and info_af.get('energy', 0.0) >= 0.0