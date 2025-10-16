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


def compute_h(editor, metric: str) -> float:
    """Compute global mesh size h for a chosen metric.

    Supported metrics:
      - 'avg_internal_edge_length' (default)
      - 'median_internal_edge_length'
      - 'avg_longest_edge' (per-triangle longest edge, averaged)
      - 'median_longest_edge'
      - 'avg_equilateral_h' (sqrt(4*A/sqrt(3)) averaged over triangles)
    """
    metric = (metric or 'avg_internal_edge_length').lower()
    # Internal edge based
    if metric in ('avg_internal_edge_length', 'median_internal_edge_length'):
        edges = _list_internal_edges(editor)
        if not edges:
            return 0.0
        lens = []
        for u, v in edges:
            d = editor.points[int(u)] - editor.points[int(v)]
            lens.append(float(np.hypot(d[0], d[1])))
        if not lens:
            return 0.0
        return float(np.mean(lens)) if metric.startswith('avg_') else float(np.median(lens))
    # Per-triangle metrics
    tri_vals = []
    for ti in _active_tri_indices(editor):
        a, b, c = [int(x) for x in editor.triangles[int(ti)]]
        pa, pb, pc = editor.points[a], editor.points[b], editor.points[c]
        if metric in ('avg_longest_edge', 'median_longest_edge'):
            Ls = [float(np.hypot(*(pa - pb))), float(np.hypot(*(pb - pc))), float(np.hypot(*(pc - pa)))]
            tri_vals.append(max(Ls))
        elif metric == 'avg_equilateral_h':
            A = abs(triangle_area(pa, pb, pc))
            # Equilateral side s such that A = (sqrt(3)/4) s^2 -> s = sqrt(4A/sqrt(3))
            tri_vals.append(float(np.sqrt(4.0 * A / np.sqrt(3.0))))
        else:
            # Fallback
            return compute_h(editor, 'avg_internal_edge_length')
    if not tri_vals:
        return 0.0
    if metric.startswith('avg_'):
        return float(np.mean(tri_vals))
    elif metric.startswith('median_'):
        return float(np.median(tri_vals))
    return float(np.mean(tri_vals))


# extend the public export list with additional quality helpers
__all__.extend(["worst_min_angle", "non_worsening_quality", "compute_h"])


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
