"""Anisotropic local remeshing driver.

Provides `anisotropic_local_remesh(editor, metric_fn, alpha_split, beta_collapse, tol, max_iter, ...)`
which implements a practical variant of the provided pseudocode using existing
editor operations (split_edge, edge_collapse, flip_edge, move_vertices_to_barycenter).

The metric_fn is a callable M(x: (2,)) -> (2,2) symmetric positive-def matrix.
"""
from __future__ import annotations
import math
import logging
from typing import Callable, Iterable, List, Tuple, Set

import numpy as np
from .constants import EPS_TINY

from .mesh_modifier2 import PatchBasedMeshEditor

log = logging.getLogger('sofia.anisotropic')


def _average_metric(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return 0.5 * (A + B)


def _log_spd(M: np.ndarray) -> np.ndarray:
    """Matrix logarithm for 2x2 SPD matrix via eigendecomposition."""
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-16, None)
    logd = np.diag(np.log(vals))
    return vecs @ logd @ vecs.T


def _exp_spd(A: np.ndarray) -> np.ndarray:
    """Matrix exponential for symmetric matrix A via eigendecomposition."""
    vals, vecs = np.linalg.eigh(A)
    expd = np.diag(np.exp(vals))
    return vecs @ expd @ vecs.T


def _apply_dlog(M: np.ndarray, dM: np.ndarray) -> np.ndarray:
    """Apply the Fréchet derivative of log at M to perturbation dM.

    Uses eigen-decomposition: if M = V diag(lam) V^T and B = V^T dM V then
    Dlog(M)[dM] = V (C * B) V^T where C_ij = (log lam_i - log lam_j)/(lam_i - lam_j)
    and diagonal C_ii = 1/lam_i.
    """
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-16, None)
    Vt_dM_V = vecs.T @ dM @ vecs
    C = np.zeros((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            if i == j:
                C[i, j] = 1.0 / vals[i]
            else:
                C[i, j] = (math.log(vals[i]) - math.log(vals[j])) / (vals[i] - vals[j])
    Mout = vecs @ (C * Vt_dM_V) @ vecs.T
    return Mout


def _apply_dexp(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """Apply the Fréchet derivative of exp at A to perturbation dA.

    If A = V diag(alpha) V^T and B = V^T dA V then
    Dexp(A)[dA] = V (D * B) V^T where D_ij = (exp(alpha_i)-exp(alpha_j))/(alpha_i-alpha_j)
    and diagonal D_ii = exp(alpha_i).
    """
    vals, vecs = np.linalg.eigh(A)
    Vt_dA_V = vecs.T @ dA @ vecs
    D = np.zeros((2, 2), dtype=float)
    for i in range(2):
        for j in range(2):
            if i == j:
                D[i, j] = math.exp(vals[i])
            else:
                D[i, j] = (math.exp(vals[i]) - math.exp(vals[j])) / (vals[i] - vals[j])
    Mout = vecs @ (D * Vt_dA_V) @ vecs.T
    return Mout


def average_metric(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """Log-Euclidean average of two SPD matrices: exp(0.5*(log(M1)+log(M2)))."""
    A = _log_spd(M1)
    B = _log_spd(M2)
    return _exp_spd(0.5 * (A + B))


def _metric_half(M: np.ndarray) -> np.ndarray:
    """Return symmetric square-root of 2x2 SPD matrix M via eigendecomposition."""
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-16, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def compute_Mhalf(M: np.ndarray) -> np.ndarray:
    """Compute the symmetric matrix square-root M^{1/2} of a 2x2 SPD matrix M.

    Uses the eigendecomposition M = R diag(lambda) R^T and returns R diag(sqrt(lambda)) R^T.
    Numerical safety: eigenvalues are clipped to a small positive floor before sqrt.
    """
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-16, None)
    sqrtd = np.diag(np.sqrt(vals))
    return vecs @ sqrtd @ vecs.T


# Backwards-compatible alias used by the remesher implementation
_metric_half = compute_Mhalf


def compute_Mminushalf(M: np.ndarray) -> np.ndarray:
    """Compute the inverse symmetric matrix square-root M^{-1/2} of a 2x2 SPD matrix M.

    Uses eigendecomposition M = R diag(lambda) R^T and returns R diag(1/sqrt(lambda)) R^T.
    Eigenvalues are clipped to a small positive floor to avoid numerical overflow/division by zero.
    """
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 1e-16, None)
    inv_sqrtd = np.diag(1.0 / np.sqrt(vals))
    return vecs @ inv_sqrtd @ vecs.T



def _edge_length_in_metric(p: np.ndarray, q: np.ndarray, Mbar: np.ndarray) -> float:
    d = q - p
    return math.sqrt(float(d @ (Mbar @ d)))


def perform_local_flips_Delaunay(y_vertices, connectivity):
    """Perform Lawson-style local Delaunay flips in the transformed (metric) coordinates.

    Parameters
    ----------
    y_vertices : dict or ndarray
        If dict, maps vertex index -> 2-array (y coordinate). If ndarray, should be shape (N,2)
        and indexed by vertex id.
    connectivity : ndarray shape (M,3)
        Triangle list (mutable). Triangles are modified in-place when an edge flip occurs.

    Returns
    -------
    int
        Number of flips performed.

    Notes
    -----
    Uses the standard incircle predicate: point d is inside circumcircle of triangle (a,b,c)
    iff the lifted determinant has the appropriate sign; the sign is adjusted by the triangle
    orientation (signed area) to produce a robust test.
    """
    import numpy as _np
    from .geometry import triangle_area

    # accessor for y coordinates
    def _y(v):
        if isinstance(y_vertices, dict):
            return _np.asarray(y_vertices[int(v)], dtype=float)
        else:
            return _np.asarray(y_vertices[int(v)], dtype=float)

    def in_circumcircle(pa, pb, pc, pd):
        # Robust-ish incircle using 3x3 determinant form with orientation sign
        # Translate to reduce magnitude
        ax, ay = pa[0] - pd[0], pa[1] - pd[1]
        bx, by = pb[0] - pd[0], pb[1] - pd[1]
        cx, cy = pc[0] - pd[0], pc[1] - pd[1]
        a2 = ax*ax + ay*ay
        b2 = bx*bx + by*by
        c2 = cx*cx + cy*cy
        mat = _np.array([
            [ax, ay, a2],
            [bx, by, b2],
            [cx, cy, c2]
        ], dtype=float)
        det = float(_np.linalg.det(mat))
        orient = triangle_area(pa, pb, pc)
        if abs(orient) <= 1e-20:
            return False
        # For CCW (positive area), det > 0 means pd inside circle.
        # Use STRICT inequality to avoid infinite loops on cocircular configurations
        # (cocircular points should not trigger flips)
        tol = 1e-16
        if orient > 0.0:
            return det > tol
        else:
            return det < -tol

    T = _np.asarray(connectivity, dtype=_np.int32)
    flips = 0
    # build edge->tri adjacency
    def build_edge_map(T):
        emap = {}
        for ti, tri in enumerate(T):
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for e in ((a,b),(b,c),(c,a)):
                key = tuple(sorted(e))
                emap.setdefault(key, []).append(ti)
        return emap

    flipped = True
    while flipped:
        flipped = False
        edge_map = build_edge_map(T)
        # iterate over a snapshot of edges
        for (u,v), tris in list(edge_map.items()):
            if len(tris) != 2:
                continue
            t1, t2 = tris[0], tris[1]
            tri1 = [int(x) for x in T[int(t1)]]
            tri2 = [int(x) for x in T[int(t2)]]
            # find opposite vertices
            verts1 = set(tri1)
            verts2 = set(tri2)
            # ensure u and v present
            if not ({int(u), int(v)}.issubset(verts1) and {int(u), int(v)}.issubset(verts2)):
                continue
            c1 = int((verts1 - {int(u), int(v)}).pop())
            c2 = int((verts2 - {int(u), int(v)}).pop())
            pa, pb, pc1, pc2 = _y(u), _y(v), _y(c1), _y(c2)
            # If either opposite vertex is inside the other's circumcircle, flip
            if in_circumcircle(pa, pb, pc1, pc2) or in_circumcircle(pa, pb, pc2, pc1):
                # perform flip: replace triangles t1,t2 with (c1,c2,u) and (c2,c1,v)
                new1 = [c1, c2, int(u)]
                new2 = [c2, c1, int(v)]
                # ensure positive orientation using actual coordinates
                p1 = [_y(int(vv)) for vv in new1]
                if triangle_area(p1[0], p1[1], p1[2]) <= 0:
                    new1[1], new1[2] = new1[2], new1[1]
                p2 = [_y(int(vv)) for vv in new2]
                if triangle_area(p2[0], p2[1], p2[2]) <= 0:
                    new2[1], new2[2] = new2[2], new2[1]
                T[int(t1)] = _np.array(new1, dtype=_np.int32)
                T[int(t2)] = _np.array(new2, dtype=_np.int32)
                # fix orientation using geometric area test
                from .geometry import triangle_angles as _ta
                # correct orientation if triangle area negative
                for idx in (int(t1), int(t2)):
                    pts = [ _y(int(vv)) for vv in T[idx] ]
                    a_signed = triangle_area(pts[0], pts[1], pts[2])
                    if a_signed <= 0:
                        # swap last two vertices
                        T[idx][1], T[idx][2] = T[idx][2], T[idx][1]
                flips += 1
                flipped = True
                # break to rebuild edge map for correctness (Lawson commonly restarts)
                break
        # end for
    return flips


def smooth_patch_vertices(y_vertices, connectivity, boundary_vs=None, omega=0.5, iterations=1):
    """Laplacian-style smoothing in the transformed (y) coordinates.

    Parameters
    ----------
    y_vertices : dict or ndarray
        If dict, maps vertex index -> 2-array (y coordinate). If ndarray, should be shape (N,2)
        and indexed by vertex id. The function returns the same type with updated coordinates.
    connectivity : ndarray shape (M,3)
        Triangle list used to infer vertex neighbors.
    boundary_vs : iterable of ints or None
        Set of vertex indices to treat as boundary (they will not be moved). If None the function
        derives boundary vertices from connectivity (edges appearing only once).
    omega : float in [0,1]
        Relaxation factor: new_y = (1-omega)*y + omega*mean(neighbors).
    iterations : int
        Number of smoothing passes to perform.

    Returns
    -------
    (y_out, moved)
        y_out is same type as y_vertices with smoothed coordinates; moved is number of vertices moved
        in the last iteration.
    """
    import numpy as _np

    T = _np.asarray(connectivity, dtype=_np.int32)
    # Determine number of vertices
    if isinstance(y_vertices, dict):
        n_vs = max(int(k) for k in y_vertices.keys()) + 1 if y_vertices else 0
    else:
        y_arr = _np.asarray(y_vertices, dtype=float)
        n_vs = y_arr.shape[0]

    # Build neighbor lists
    nbrs = [[] for _ in range(n_vs)]
    for tri in T:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a < 0 or b < 0 or c < 0:
            continue
        nbrs[a].extend([b, c]); nbrs[b].extend([a, c]); nbrs[c].extend([a, b])
    # Deduplicate neighbor lists
    nbrs = [list(dict.fromkeys([int(x) for x in lst])) for lst in nbrs]

    # Compute boundary vertices if not provided
    if boundary_vs is None:
        edge_count = {}
        for tri in T:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for e in ((a,b),(b,c),(c,a)):
                key = tuple(sorted(e))
                edge_count[key] = edge_count.get(key, 0) + 1
        b_vs = set()
        for (u, v), cnt in edge_count.items():
            if cnt == 1:
                b_vs.add(int(u)); b_vs.add(int(v))
        boundary_vs = b_vs
    else:
        boundary_vs = set(int(x) for x in boundary_vs)

    # Convert y_vertices to array for computation, preserve dict semantics if needed
    is_dict = isinstance(y_vertices, dict)
    if is_dict:
        y_arr = _np.zeros((n_vs, 2), dtype=float)
        for i in range(n_vs):
            if int(i) in y_vertices:
                y_arr[i] = _np.asarray(y_vertices[int(i)], dtype=float)
            else:
                y_arr[i] = _np.array([0.0, 0.0])
    else:
        y_arr = _np.asarray(y_vertices, dtype=float).copy()

    moved = 0
    for it in range(int(max(1, iterations))):
        y_new = y_arr.copy()
        moved = 0
        for i in range(n_vs):
            if int(i) in boundary_vs:
                continue
            neigh = nbrs[int(i)]
            if not neigh:
                continue
            neigh_coords = y_arr[_np.array(neigh, dtype=int)]
            y_mean = _np.mean(neigh_coords, axis=0)
            y_i_new = (1.0 - float(omega)) * y_arr[int(i)] + float(omega) * y_mean
            if not _np.allclose(y_i_new, y_arr[int(i)], atol=1e-15, rtol=float(EPS_TINY)):
                moved += 1
            y_new[int(i)] = y_i_new
        y_arr = y_new

    # Return in the same type as input
    if is_dict:
        out = {i: y_arr[int(i)].copy() for i in range(n_vs) if i in y_vertices}
    else:
        out = y_arr
    return out, int(moved)


def optimize_vertex_metric_equilateral(v_idx, editor, metric_fn, max_iter=20, grad_eps=1e-6, ls_c=1e-4, ls_r=0.5, min_step=1e-8, grad_mode='analytic'):
    """Optimize position of vertex 'v_idx' to minimize sum_j (l_M(x,x_j) - 1)^2 where l_M uses
    the log-Euclidean averaged metric between x and neighbor positions x_j.

    This implementation uses a central finite-difference gradient and a simple backtracking
    Armijo line-search. It only updates the single vertex position and returns the proposed
    new coordinate (does not commit to the editor). Caller should decide to write to editor.points.

    Parameters
    ----------
    v_idx : int
        Vertex index to optimize.
    editor : PatchBasedMeshEditor
        Mesh editor providing points and v_map adjacency.
    metric_fn : callable
        Function x -> 2x2 SPD metric.
    max_iter : int
        Maximum gradient-descent iterations.
    grad_eps : float
        Finite-difference step for gradient approximation (baseline absolute eps).
    ls_c : float
        Armijo constant.
    ls_r : float
        Step reduction factor in backtracking line search.
    min_step : float
        Minimum allowed step size before giving up.

    Returns
    -------
    (x_new, info)
        x_new : ndarray shape (2,) proposed new coordinates
        info : dict with keys 'energy', 'grad_norm', 'steps_taken', 'converged'
    """
    import numpy as _np

    v = int(v_idx)
    pts = editor.points
    x0 = _np.asarray(pts[v], dtype=float).copy()
    neigh_tris = set(editor.v_map.get(v, []))
    neighbors = set()
    for ti in neigh_tris:
        tri = editor.triangles[int(ti)]
        for vv in tri:
            vv = int(vv)
            if vv != v:
                neighbors.add(vv)
    neighbors = sorted(int(w) for w in neighbors)
    if not neighbors:
        return x0, {'energy': 0.0, 'grad_norm': 0.0, 'steps_taken': 0, 'converged': True}

    def energy_at(x):
        E = 0.0
        Mx = metric_fn(_np.asarray(x, dtype=float))
        for j in neighbors:
            xj = _np.asarray(pts[int(j)], dtype=float)
            Mbar = average_metric(Mx, metric_fn(xj))
            d = xj - x
            l = _np.sqrt(float(d @ (Mbar @ d)))
            diff = l - 1.0
            E += diff * diff
        return float(E)

    def finite_grad(x):
        # central finite differences with step h scaled to |x| or grad_eps
        h = max(grad_eps, 1e-8 * max(1.0, _np.linalg.norm(x)))
        g = _np.zeros((2,), dtype=float)
        for k in range(2):
            e = _np.zeros((2,), dtype=float); e[k] = 1.0
            fp = energy_at(x + h * e)
            fm = energy_at(x - h * e)
            g[k] = (fp - fm) / (2.0 * h)
        return g

    def analytic_grad(x):
        """Compute analytic gradient of energy w.r.t x using chain rule.

        We compute dMx/dx numerically (central differences) but apply exact matrix
        Fréchet derivatives for log and exp to propagate to dMbar/dx, then form dl/dx.
        This reduces the number of expensive energy evaluations compared to naive FD.
        """
        x = _np.asarray(x, dtype=float)
        # small step for metric Jacobian (use grad_eps)
        h = max(float(grad_eps), 1e-10)
        # compute Mx at center
        Mx = metric_fn(x)
        # compute metric Jacobian dMx/dx_m via central differences
        dMx = []
        for m in range(2):
            e = _np.zeros(2, dtype=float); e[m] = 1.0
            Mp = metric_fn(x + h * e)
            Mm = metric_fn(x - h * e)
            dMx_m = (Mp - Mm) / (2.0 * h)
            dMx.append(dMx_m)

        g = _np.zeros((2,), dtype=float)
        for j in neighbors:
            xj = _np.asarray(pts[int(j)], dtype=float)
            Mj = metric_fn(xj)
            # build A = 0.5*(log(Mx)+log(Mj))
            A = 0.5 * (_log_spd(Mx) + _log_spd(Mj))
            Mbar = _exp_spd(A)
            d = xj - x
            l = _np.sqrt(float(d @ (Mbar @ d)))
            if l == 0.0:
                continue
            # compute contribution of dMbar/dx_m using chain rule: dMbar = Dexp(A)[0.5 * Dlog(Mx)[dMx]]
            b = -2.0 * (Mbar @ d)
            for m in range(2):
                # apply Dlog to dMx[m]
                dlog = _apply_dlog(Mx, dMx[m])
                dA = 0.5 * dlog
                dMbar = _apply_dexp(A, dA)
                scalar = float(d @ (dMbar @ d))
                b[m] += scalar
            dl_dx = (0.5 / float(l)) * b
            g += 2.0 * (l - 1.0) * dl_dx
        return g

    def analytic_full_grad(x):
        """Analytic gradient assuming metric_fn provides `jacobian(x)` returning [dM/dx0, dM/dx1].

        If metric_fn does not provide jacobian, fallback to `analytic_grad`.
        """
        if hasattr(metric_fn, 'jacobian') and callable(getattr(metric_fn, 'jacobian')):
            # get analytic dMx/dx_m from metric_fn
            dMx = metric_fn.jacobian(x)
            # ensure list-like of two 2x2 matrices
            if not (hasattr(dMx, '__len__') and len(dMx) == 2):
                return analytic_grad(x)
            x = _np.asarray(x, dtype=float)
            g = _np.zeros((2,), dtype=float)
            Mx = metric_fn(x)
            for j in neighbors:
                xj = _np.asarray(pts[int(j)], dtype=float)
                Mj = metric_fn(xj)
                A = 0.5 * (_log_spd(Mx) + _log_spd(Mj))
                Mbar = _exp_spd(A)
                d = xj - x
                l = _np.sqrt(float(d @ (Mbar @ d)))
                if l == 0.0:
                    continue
                b = -2.0 * (Mbar @ d)
                for m in range(2):
                    dlog = _apply_dlog(Mx, dMx[m])
                    dA = 0.5 * dlog
                    dMbar = _apply_dexp(A, dA)
                    scalar = float(d @ (dMbar @ d))
                    b[m] += scalar
                dl_dx = (0.5 / float(l)) * b
                g += 2.0 * (l - 1.0) * dl_dx
            return g
        else:
            return analytic_grad(x)

    x = x0.copy()
    E0 = energy_at(x)
    converged = False
    steps = 0
    for it in range(int(max_iter)):
        if grad_mode == 'fd':
            g = finite_grad(x)
        elif grad_mode == 'analytic':
            g = analytic_grad(x)
        elif grad_mode == 'analytic_full':
            g = analytic_full_grad(x)
        else:
            raise ValueError(f"unknown grad_mode '{grad_mode}'")
        gn = float(_np.linalg.norm(g))
        if gn <= float(EPS_TINY):
            converged = True
            break
        # descent direction
        d = g
        # line search: try step = 1.0, reduce by ls_r until Armijo satisfied
        step = 1.0
        f0 = E0
        found = False
        while step >= min_step:
            x_trial = x - step * d
            f_trial = energy_at(x_trial)
            # Armijo condition: f_trial <= f0 - c * step * ||d||^2
            if f_trial <= f0 - ls_c * step * (gn * gn):
                found = True
                break
            step *= ls_r
        if not found:
            # cannot find satisfactory step; quit
            break
        # accept
        x = x_trial
        E0 = f_trial
        steps += 1
        # small relative change test
        if abs(step * gn) < 1e-10:
            converged = True
            break

    info = {'energy': float(E0), 'grad_norm': float(_np.linalg.norm(g)), 'steps_taken': int(steps), 'converged': bool(converged)}
    return _np.asarray(x, dtype=float), info


def _collect_patch_triangles_around_vertices(editor: PatchBasedMeshEditor, verts: Iterable[int]) -> Set[int]:
    tris = set()
    for v in verts:
        for ti in editor.v_map.get(int(v), []):
            ti = int(ti)
            if not np.all(editor.triangles[ti] == -1):
                tris.add(ti)
    return tris


def _patch_vertices_from_tri_indices(editor: PatchBasedMeshEditor, tri_indices: Iterable[int]) -> List[int]:
    vs = set()
    for ti in tri_indices:
        t = [int(x) for x in editor.triangles[int(ti)]]
        for v in t:
            vs.add(v)
    return sorted(vs)


def debug_plot_patch_around_vertices(editor: PatchBasedMeshEditor,
                                     verts: Iterable[int],
                                     outname: str = 'patch_debug.png',
                                     highlight_boundary_loops: bool = True,
                                     annotate_vertices: bool = True):
    """Debug utility: collect the patch of triangles around `verts` and plot it.

    This uses `_collect_patch_triangles_around_vertices` to form the triangle set, then
    leverages `visualization.plot_mesh_by_tri_groups` to render those triangles filled.

    Parameters
    ----------
    editor : PatchBasedMeshEditor
        Mesh editor with current `points` and `triangles`.
    verts : iterable of int
        Vertex indices around which to collect the patch.
    outname : str
        Output image filename.
    highlight_boundary_loops : bool
        If True, overlay boundary loops in the rendered figure.
    """
    try:
        tri_patch = _collect_patch_triangles_around_vertices(editor, verts)
        from .visualization import plot_mesh_by_tri_groups
        tri_groups = {'patch': sorted(int(t) for t in tri_patch)}
        plot_mesh_by_tri_groups(
            editor,
            tri_groups,
            outname=outname,
            highlight_boundary_loops=bool(highlight_boundary_loops),
            annotate_vertices=(list(verts) if annotate_vertices else None),
            annotate_color=(0.85, 0.2, 0.2),
            annotate_size=28.0,
            annotate_labels=True,
        )
        log.info("[debug] saved patch plot around verts=%s to %s (ntri=%d)",
                 list(verts), outname, len(tri_patch))
    except Exception as e:
        log.warning("[debug] failed to plot patch: %s", e)


def anisotropic_local_remesh(editor: PatchBasedMeshEditor,
                             metric_fn: Callable[[np.ndarray], np.ndarray],
                             alpha_split: float = 1.5,
                             beta_collapse: float = 0.5,
                             tol: float = 0.05,
                             max_iter: int = 6,
                             verbose: bool = False,
                             do_global_smoothing: bool = True,
                             do_cleanup: bool = True):
    """Perform anisotropic local remeshing on `editor` guided by `metric_fn`.

    Parameters
    - editor: PatchBasedMeshEditor
    - metric_fn: callable taking (x: np.ndarray(shape=(2,))) and returning 2x2 SPD matrix
    - alpha_split: target upper bound for metric lengths (split if > alpha_split)
    - beta_collapse: target lower bound for metric lengths (collapse if < beta_collapse)
    - tol: convergence tolerance on metric edge length error (|lM - 1|)
    - max_iter: maximum outer iterations
    - verbose: log additional info
    - do_global_smoothing: if True, perform step 6 global metric-space smoothing
    - do_cleanup: if True, perform step 7 cleanup & validation (degenerate removal, boundary projection)

    Notes
    - This is a pragmatic, local implementation: it uses the editor's split/collapse/flip
      operations and performs simple metric-space smoothing using the local Mhalf transform.
    """
    def logv(*a, **k):
        if verbose:
            log.info(*a, **k)

    iter_no = 0
    converged = False

    for iter_no in range(1, max_iter + 1):
        logv('anisotropic remesh iter %d', iter_no)

        # 1) evaluate metric-lengths on all internal edges
        lM = {}
        internal_edges = []
        for e, ts in editor.edge_map.items():
            # only consider active adjacency (filter tombstoned tris)
            active_ts = [int(t) for t in ts if not np.all(editor.triangles[int(t)] == -1)]
            if len(active_ts) == 0:
                continue
            u, v = int(e[0]), int(e[1])
            p = np.asarray(editor.points[u], dtype=float)
            q = np.asarray(editor.points[v], dtype=float)
            Mbar = average_metric(metric_fn(p), metric_fn(q))
            lm = _edge_length_in_metric(p, q, Mbar)
            lM[(u, v)] = lm
            internal_edges.append((u, v))

        if not internal_edges:
            logv('no internal edges; exiting')
            break

        # Normalize target: we aim lM ~= 1.0
        # 2) mark edges to split / collapse
        edges_to_split = [e for e in internal_edges if lM.get(e, 0.0) > float(alpha_split)]
        edges_to_collapse = [e for e in internal_edges if lM.get(e, 0.0) < float(beta_collapse)]

        # Sort splits desc, collapses asc
        edges_to_split.sort(key=lambda e: lM[e], reverse=True)
        edges_to_collapse.sort(key=lambda e: lM[e])

        modified_vertices = set()

        # 3) refinement: split edges
        splits = 0
        for e in edges_to_split:
            try:
                ok, msg, info = editor.split_edge((int(e[0]), int(e[1])))
            except Exception as ex:
                ok = False; msg = str(ex); info = None
            logv('split edge %s -> %s (%s)', e, ok, msg)
            if ok:
                splits += 1
                # Determine the new vertex index appended by split_edge.
                # Our editor's split_edge returns info with 'npts' (post-op count),
                # and appends exactly one new vertex at the end. Prefer direct length.
                try:
                    new_idx = int(np.asarray(editor.points).shape[0] - 1)
                except Exception:
                    new_idx = None
                if (new_idx is None or new_idx < 0) and info and isinstance(info, dict) and ('npts' in info):
                    try:
                        new_idx = int(info['npts']) - 1
                    except Exception:
                        new_idx = None
                if new_idx is not None and new_idx >= 0:
                    modified_vertices.add(new_idx)
                # endpoints are also part of modified region
                modified_vertices.add(int(e[0])); modified_vertices.add(int(e[1]))

        # 4) simplification: collapse edges
        collapses = 0
        for e in edges_to_collapse:
            try:
                # attempt collapse, rely on editor to refuse unsafe collapses
                ok, msg, info = editor.edge_collapse((int(e[0]), int(e[1])))
            except Exception as ex:
                ok = False; msg = str(ex); info = None
            logv('collapse edge %s -> %s (%s)', e, ok, msg)
            if ok:
                collapses += 1
                modified_vertices.add(int(e[0])); modified_vertices.add(int(e[1]))

        logv('iter=%d splits=%d collapses=%d modified_verts=%d', iter_no, splits, collapses, len(modified_vertices))

        # 5) local remesh in metric space for patches around modified regions
        if modified_vertices:
            tri_patch = _collect_patch_triangles_around_vertices(editor, modified_vertices)
            patch_vs = _patch_vertices_from_tri_indices(editor, tri_patch)
            # Optional: visualize the current patch vertices for debugging
            if verbose and patch_vs:
                try:
                    debug_plot_patch_around_vertices(
                        editor,
                        patch_vs,
                        outname=f'patch_vs_iter{iter_no}.png',
                        highlight_boundary_loops=True,
                        annotate_vertices=True,
                    )
                except Exception:
                    # plotting is best-effort; ignore any failures in non-interactive contexts
                    pass
            if patch_vs:
                # average metric over patch
                Ms = [metric_fn(np.asarray(editor.points[int(v)], dtype=float)) for v in patch_vs]
                Mpatch = sum(Ms) / float(len(Ms))
                T = _metric_half(Mpatch)
                Tinv = np.linalg.inv(T)
                # transform patch vertex positions
                y_coords = {v: T @ np.asarray(editor.points[int(v)], dtype=float) for v in patch_vs}
                # simple local flip strategy: try flipping interior edges in patch if improves min-angle in metric space
                # collect interior edges of the patch
                patch_edges = set()
                for ti in tri_patch:
                    t = [int(x) for x in editor.triangles[int(ti)]]
                    for a,b in ((t[0],t[1]), (t[1],t[2]), (t[2],t[0])):
                        key = tuple(sorted((int(a), int(b))))
                        patch_edges.add(key)
                # attempt flips
                for e in list(patch_edges):
                    # only flip if edge is interior (2 adjacent tris)
                    ets = editor.edge_map.get(tuple(sorted((int(e[0]), int(e[1])))), [])
                    active_adj = [int(t) for t in ets if not np.all(editor.triangles[int(t)] == -1)]
                    if len(active_adj) != 2:
                        continue
                    # compute min-angle before
                    def _min_angle_metric(tri_idx):
                        tri = [int(x) for x in editor.triangles[int(tri_idx)]]
                        ys = [y_coords.get(v, T @ np.asarray(editor.points[int(v)], dtype=float)) for v in tri]
                        angs = []
                        try:
                            from .geometry import triangle_angles
                            angs = triangle_angles(ys[0], ys[1], ys[2])
                        except Exception:
                            return 0.0
                        return min(angs)
                    pre_min = min(_min_angle_metric(active_adj[0]), _min_angle_metric(active_adj[1]))
                    ok_flip, msg_flip, _ = editor.flip_edge((int(e[0]), int(e[1])))
                    if not ok_flip:
                        continue
                    post_min = min(_min_angle_metric(active_adj[0]), _min_angle_metric(active_adj[1]))
                    if post_min + float(EPS_TINY) < pre_min:
                        # revert flip by flipping back
                        try:
                            editor.flip_edge((int(e[0]), int(e[1])))
                        except Exception:
                            pass
                    else:
                        # accept flip
                        modified_vertices.update(e)

                # smooth patch vertices in metric-space (y coordinates barycenter)
                # compute boundary vertex set once
                try:
                    from .refinement import list_boundary_edges_only
                    b_edges = list_boundary_edges_only(editor)
                    boundary_vs = set([int(u) for e in b_edges for u in e])
                except Exception:
                    boundary_vs = set()
                for v in patch_vs:
                    if int(v) in boundary_vs:
                        # avoid moving boundary vertices here
                        continue
                    nbrs = set()
                    for ti in editor.v_map.get(int(v), []):
                        t = editor.triangles[int(ti)]
                        for vv in t:
                            vv = int(vv)
                            if vv != int(v):
                                nbrs.add(vv)
                    if not nbrs:
                        continue
                    ys = [y_coords.get(int(w), T @ np.asarray(editor.points[int(w)], dtype=float)) for w in nbrs]
                    y_new = sum(ys) / float(len(ys))
                    x_new = Tinv @ y_new
                    #editor.points[int(v)] = np.asarray(x_new)
                    modified_vertices.add(int(v))

        # 6) global smoothing: optimize vertices toward metric-equilateral configuration
        # Here we perform one pass of metric-space barycentric smoothing for interior vertices
        if do_global_smoothing:
            moved = 0
            for v in list(editor.v_map.keys()):
                v = int(v)
                # skip boundary
                try:
                    from .refinement import list_boundary_edges_only
                    b_edges = list_boundary_edges_only(editor)
                    boundary_vs_glob = set([int(u) for e in b_edges for u in e])
                except Exception:
                    boundary_vs_glob = set()
                if v in boundary_vs_glob:
                    continue
                p = np.asarray(editor.points[v], dtype=float)
                Mv = metric_fn(p)
                T = _metric_half(Mv)
                Tinv = np.linalg.inv(T)
                yv = T @ p
                nbrs = set()
                for ti in editor.v_map.get(v, []):
                    t = editor.triangles[int(ti)]
                    for vv in t:
                        vv = int(vv)
                        if vv != v:
                            nbrs.add(vv)
                if not nbrs:
                    continue
                ys = [T @ np.asarray(editor.points[int(w)], dtype=float) for w in nbrs]
                y_new = sum(ys) / float(len(ys))
                x_new = Tinv @ y_new
                editor.points[v] = np.asarray(x_new)
                moved += 1
            logv('metric-smoothing moved=%d vertices', moved)
        else:
            logv('skipping global metric smoothing (do_global_smoothing=False)')

        # 7) cleanup & validation
        #try:
            # remove degenerate triangles and project boundary vertices if helper exists
        #    if hasattr(editor, 'remove_degenerate_triangles'):
        #        try:
        #            editor.remove_degenerate_triangles()
        #        except Exception:
        #            pass
        #    if hasattr(editor, 'project_boundary_vertices'):
        #        try:
        #            editor.project_boundary_vertices()
        #        except Exception:
        #            pass
        #except Exception:
        #    pass
        if do_cleanup:
            try:
                # remove degenerate triangles and project boundary vertices if helper exists
                if hasattr(editor, 'remove_degenerate_triangles'):
                    try:
                        editor.remove_degenerate_triangles()
                    except Exception:
                        pass
                if hasattr(editor, 'project_boundary_vertices'):
                    try:
                        editor.project_boundary_vertices()
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            logv('skipping cleanup & validation (do_cleanup=False)')

        # recompute lM and check convergence
        lM_new = []
        for e, ts in editor.edge_map.items():
            active_ts = [int(t) for t in ts if not np.all(editor.triangles[int(t)] == -1)]
            if not active_ts:
                continue
            u, v = int(e[0]), int(e[1])
            p = np.asarray(editor.points[u], dtype=float)
            q = np.asarray(editor.points[v], dtype=float)
            Mbar = average_metric(metric_fn(p), metric_fn(q))
            lm = _edge_length_in_metric(p, q, Mbar)
            lM_new.append(abs(lm - 1.0))
        err = max(lM_new) if lM_new else 0.0
        logv('iter=%d metric-edge err=%.6g splits=%d collapses=%d', iter_no, err, splits, collapses)
        if err < float(tol) and not edges_to_split and not edges_to_collapse:
            converged = True
            break

    return editor, {'iter': iter_no, 'converged': converged}
