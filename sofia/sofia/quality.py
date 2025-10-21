"""Mesh quality metrics and helpers.

This module provides:
- mesh_min_angle: global minimum internal angle across active triangles
- worst_min_angle: compute worst (global min) of per-triangle minimum angles
- non_worsening_quality: gate based on worst-min-angle not decreasing
- compute_h: compute a global mesh size h using various metrics
"""

from __future__ import annotations

import numpy as np
from .geometry import (
    triangles_min_angles,
    triangle_angles,
    EPS_MIN_ANGLE_DEG,
    triangle_area,
)
from typing import Iterable, Optional, Sequence


def mesh_min_angle(points, triangles) -> float:
    """Return the global minimum internal angle across active triangles.

    Parameters
    ----------
    points : (N, 2) array-like
    triangles : (M, 3) array-like (may contain tombstoned rows of [-1,-1,-1])

    Returns
    -------
    float
        Minimum internal angle in degrees across all active triangles, or NaN if none.
    """
    tris = np.asarray(triangles, dtype=int)
    if tris.size == 0:
        return float('nan')
    mask = ~np.all(tris == -1, axis=1)
    active = tris[mask]
    if active.size == 0:
        return float('nan')
    mins = triangles_min_angles(points, active)
    # guard against NaNs from degenerate geometry
    mins = mins[np.isfinite(mins)]
    return float(mins.min()) if mins.size else float('nan')


__all__ = [
    'mesh_min_angle',
]

def worst_min_angle(points, triangles_iter):
    """Return the minimum (over triangles) of each triangle's minimum angle in degrees.

    points : (N,2) array
    triangles_iter : iterable of triangle index triplets (list/array/tuples)
    """
    w = 180.0
    for tri in triangles_iter:
        try:
            a,b,c = int(tri[0]), int(tri[1]), int(tri[2])
            angs = triangle_angles(points[a], points[b], points[c])
            tri_min = min(angs)
            if tri_min < w:
                w = tri_min
        except Exception:
            continue
    return w

def non_worsening_quality(pre_min, post_min, eps=EPS_MIN_ANGLE_DEG):
    """Return True if post worst-min-angle is not worse than pre by more than eps."""
    return post_min >= pre_min - eps


def _active_tri_indices(editor) -> list:
    tris = editor.triangles
    return [i for i, t in enumerate(tris) if not np.all(np.array(t) == -1)]


def _list_internal_edges(editor):
    internals = []
    for e, ts in editor.edge_map.items():
        ts_list = [int(t) for t in ts]
        # Only edges shared by exactly two active triangles
        if len(ts_list) == 2 and not any(np.all(editor.triangles[int(t)] == -1) for t in ts_list):
            internals.append(tuple(sorted((int(e[0]), int(e[1])))))
    # Deduplicate
    internals = list({tuple(x) for x in internals})
    return internals


def compute_h(
    editor,
    metric: str,
    tri_indices: Optional[Sequence[int]] = None,
    spd_metric: Optional[np.ndarray] = None,
) -> float:
    """Compute mesh size h for a chosen metric, optionally on a patch and/or in a metric space.

    Parameters
    ----------
    editor : PatchBasedMeshEditor
        Mesh editor providing `points`, `triangles`, and `edge_map`.
    metric : str
        Metric name, one of:
          - 'avg_internal_edge_length' (default)
          - 'median_internal_edge_length'
          - 'avg_longest_edge' (per-triangle longest edge, averaged)
          - 'median_longest_edge'
          - 'avg_equilateral_h' (sqrt(4*A/sqrt(3)) averaged over triangles)
    tri_indices : optional sequence of int
        If provided, restrict the computation to this set of triangle indices
        (tombstoned triangles are ignored). For internal-edge metrics, edges
        are considered internal to the patch (shared by exactly two triangles in
        the provided set).
    spd_metric : optional (2,2) SPD matrix
        If provided, compute distances and areas in the metric space induced by
        this SPD matrix M. The metric length of a vector v is sqrt(v^T M v), and
        triangle area is scaled by sqrt(det(M)) relative to Euclidean. Defaults
        to the identity matrix if None.

    Returns
    -------
    float
        The computed h value according to the requested metric.
    """
    metric = (metric or 'avg_internal_edge_length').lower()
    pts = editor.points

    # Prepare SPD metric (use identity if None); we only need M and sqrt(det(M))
    M = None
    detL = 1.0  # area scaling factor (|det(L)| where L^T L = M)
    if spd_metric is not None:
        try:
            M = np.asarray(spd_metric, dtype=float)
            if M.shape != (2, 2):
                M = None
            else:
                # symmetricize mildly to reduce numerical asymmetry
                M = 0.5 * (M + M.T)
                detM = float(np.linalg.det(M))
                if detM <= 0.0 or not np.isfinite(detM):
                    M = None
                else:
                    detL = float(np.sqrt(detM))
        except Exception:
            M = None

    def _edge_len(u: int, v: int) -> float:
        pu = pts[int(u)]; pv = pts[int(v)]
        d = pu - pv
        if M is None:
            return float(np.hypot(d[0], d[1]))
        try:
            return float(np.sqrt(d @ (M @ d)))
        except Exception:
            # conservative fallback to Euclidean if M application fails
            return float(np.hypot(d[0], d[1]))

    # Helper: active triangle indices considering optional patch restriction
    if tri_indices is None:
        active_tris = _active_tri_indices(editor)
    else:
        active_tris = []
        ts = editor.triangles
        for ti in tri_indices:
            try:
                ti = int(ti)
                if ti < 0 or ti >= len(ts):
                    continue
                if not np.all(np.asarray(ts[int(ti)]) == -1):
                    active_tris.append(int(ti))
            except Exception:
                continue

    # Internal edge based metrics
    if metric in ('avg_internal_edge_length', 'median_internal_edge_length'):
        if tri_indices is None:
            edges = _list_internal_edges(editor)
        else:
            # Build edge counts from the patch and keep those shared by exactly two triangles
            counts = {}
            ts = editor.triangles
            for ti in active_tris:
                a, b, c = [int(x) for x in ts[int(ti)]]
                for e in ((a, b), (b, c), (c, a)):
                    key = tuple(sorted((int(e[0]), int(e[1]))))
                    counts[key] = counts.get(key, 0) + 1
            edges = [e for e, k in counts.items() if int(k) == 2]
        if not edges:
            return 0.0
        lens = [ _edge_len(u, v) for (u, v) in edges ]
        if not lens:
            return 0.0
        return float(np.mean(lens)) if metric.startswith('avg_') else float(np.median(lens))

    # Per-triangle metrics over the (possibly restricted) set
    tri_vals = []
    ts = editor.triangles
    for ti in active_tris:
        a, b, c = [int(x) for x in ts[int(ti)]]
        pa, pb, pc = pts[a], pts[b], pts[c]
        if metric in ('avg_longest_edge', 'median_longest_edge'):
            Ls = [ _edge_len(a, b), _edge_len(b, c), _edge_len(c, a) ]
            tri_vals.append(max(Ls))
        elif metric == 'avg_equilateral_h':
            A = abs(triangle_area(pa, pb, pc))
            # Area under SPD metric scales by |det(L)| where L^T L = M
            if M is not None:
                A = float(detL * A)
            # Equilateral side s such that A = (sqrt(3)/4) s^2 -> s = sqrt(4A/sqrt(3))
            tri_vals.append(float(np.sqrt(4.0 * A / np.sqrt(3.0))))
        else:
            # Fallback to default metric
            return compute_h(editor, 'avg_internal_edge_length', tri_indices=tri_indices, spd_metric=spd_metric)
    if not tri_vals:
        return 0.0
    if metric.startswith('avg_'):
        return float(np.mean(tri_vals))
    elif metric.startswith('median_'):
        return float(np.median(tri_vals))
    return float(np.mean(tri_vals))


# -----------------------
# Area preservation validation
# -----------------------

class AreaPreservationChecker:
    """Validates area-preserving constraints for mesh operations.
    
    Used primarily for boundary vertex removal operations where the
    removed cavity area must match the candidate triangulation area
    within specified tolerances.
    """
    
    def __init__(self, config=None):
        """Initialize with optional configuration.
        
        Args:
            config: BoundaryRemoveConfig or None (defaults to strict preservation)
        """
        from .constants import EPS_TINY, EPS_AREA
        
        if config is None:
            # Default: strict area preservation required
            self.require_preservation = True
            self.rel_tolerance = EPS_TINY
            self.abs_tolerance = 4.0 * EPS_AREA
        else:
            # Use config settings
            self.require_preservation = bool(
                getattr(config, 'require_area_preservation', True)
            )
            self.rel_tolerance = float(
                getattr(config, 'area_tol_rel', EPS_TINY)
            )
            abs_factor = float(
                getattr(config, 'area_tol_abs_factor', 4.0)
            )
            self.abs_tolerance = abs_factor * EPS_AREA
    
    def compute_cavity_area(self, points, triangles, cavity_indices):
        """Compute total area of cavity triangles.
        
        Args:
            points: (N, 2) array of vertex coordinates
            triangles: (M, 3) array of triangle indices
            cavity_indices: Indices of triangles in the cavity
            
        Returns:
            float: Total absolute area of cavity triangles
        """
        total = 0.0
        for ti in cavity_indices:
            tri = triangles[int(ti)]
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            p0, p1, p2 = points[a], points[b], points[c]
            total += abs(triangle_area(p0, p1, p2))
        return total
    
    def check(self, removed_area, candidate_area):
        """Check if candidate preserves area within tolerance.
        
        Args:
            removed_area: Area of removed cavity triangles
            candidate_area: Area of candidate triangulation
            
        Returns:
            tuple: (ok: bool, error_message: str or None)
        """
        # If preservation not required, always accept
        if not self.require_preservation:
            return True, None
        
        # If either area is None/invalid, skip check
        if removed_area is None or candidate_area is None:
            return True, None
        
        # Compute tolerance: max of absolute and relative tolerances
        max_diff = max(
            self.abs_tolerance,
            self.rel_tolerance * max(1.0, removed_area)
        )
        
        # Check if difference is within tolerance
        area_diff = abs(candidate_area - removed_area)
        if area_diff > max_diff:
            return False, (
                f"area not preserved: removed={removed_area:.6e} "
                f"candidate={candidate_area:.6e} diff={area_diff:.6e} "
                f"tolerance={max_diff:.6e}"
            )
        
        return True, None


# extend the public export list with additional quality helpers
__all__.extend(["worst_min_angle", "non_worsening_quality", "compute_h", "AreaPreservationChecker"])


def _triangle_qualities_norm(points_arr, tris_arr):
    """Module-level normalized triangle quality helper.

    Returns an array of per-triangle quality values in [0,1].
    Kept here so other modules can import and tests can monkeypatch it.
    """
    try:
        pts = np.asarray(points_arr, dtype=np.float64)
        tris = np.asarray(tris_arr, dtype=np.int32)
        if tris.size == 0:
            return np.empty((0,), dtype=np.float64)
        p0 = pts[tris[:, 0]]
        p1 = pts[tris[:, 1]]
        p2 = pts[tris[:, 2]]
        a = 0.5 * np.abs((p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0]))
        e0 = np.sum((p1 - p0)**2, axis=1)
        e1 = np.sum((p2 - p1)**2, axis=1)
        e2 = np.sum((p0 - p2)**2, axis=1)
        denom = e0 + e1 + e2
        safe = denom > 0
        q = np.zeros(tris.shape[0], dtype=np.float64)
        q[safe] = a[safe] / denom[safe]
        norm_factor = 12.0 / (np.sqrt(3.0))
        q = q * norm_factor
        q = np.clip(q, 0.0, 1.0)
        return q
    except Exception:
        out = []
        for tri in tris_arr:
            try:
                p0 = points_arr[int(tri[0])]; p1 = points_arr[int(tri[1])]; p2 = points_arr[int(tri[2])]
                a = abs(triangle_area(p0, p1, p2))
                e0 = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
                e1 = (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
                e2 = (p0[0]-p2[0])**2 + (p0[1]-p2[1])**2
                denom = e0 + e1 + e2
                qv = 0.0
                if denom > 0:
                    qv = (a / denom) * (12.0 / (3.0**0.5))
                    if qv > 1.0: qv = 1.0
                out.append(qv)
            except Exception:
                out.append(0.0)
        return np.array(out, dtype=np.float64)
