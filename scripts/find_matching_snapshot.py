#!/usr/bin/env python3
"""Find earliest per-op compacted snapshot in gtrace that matches the final failing mesh from gfail_seed_0.npz.
Matching is attempted by exact (indices) equality first, then by geometric triangle set equality (coordinates rounded).
"""
import numpy as np
import os
from sofia.sofia.logging_utils import get_logger

logger = get_logger('sofia.scripts.find_matching_snapshot')

GFAIL = 'diagnostics/gfail_seed_0.npz'
GTRACE = 'diagnostics/gtrace_seed_0.npz'

def tri_geom_set(pts, tris, ndigits=9):
    # pts: (N,2), tris: (M,3)
    s = set()
    for t in tris:
        coords = [tuple(np.round(pts[int(i)], ndigits)) for i in t]
        coords_sorted = tuple(sorted(coords))
        s.add(coords_sorted)
    return s


def main():
    if not os.path.exists(GFAIL):
        logger.error('Missing %s', GFAIL); return 2
    if not os.path.exists(GTRACE):
        logger.error('Missing %s', GTRACE); return 2
    g = np.load(GFAIL, allow_pickle=True)
    pts_after = g['pts_after']
    tris_after = g['tris_after']

    trace = np.load(GTRACE, allow_pickle=True)
    steps = trace['steps']
    pts_arr = trace['pts_comp']
    tris_arr = trace['tris_comp']
    actions = trace.get('actions', None)

    logger.info('gfail pts_after %s tris_after %s', pts_after.shape, tris_after.shape)
    target_geom = tri_geom_set(pts_after, tris_after)

    for i, step in enumerate(steps):
        pts = pts_arr[i]
        tris = tris_arr[i]
        if pts is None or tris is None:
            continue
        # exact index-level compare (fast)
        try:
            if pts.shape == pts_after.shape and tris.shape == tris_after.shape and np.array_equal(pts, pts_after) and np.array_equal(tris, tris_after):
                logger.info('Exact equal match at step %d', int(step))
                if actions is not None:
                    logger.info('action: %s', actions[i])
                return 0
        except Exception:
            pass
        # geometry-based compare
        try:
            geom = tri_geom_set(pts, tris)
            if geom == target_geom:
                logger.info('Geometric match at step %d index %d', int(step), i)
                if actions is not None:
                    logger.info('action: %s', actions[i])
                return 0
        except Exception as e:
            logger.exception('compare error at step %d: %s', int(step), e)
            continue
    logger.info('No matching per-op snapshot found')
    return 1

if __name__ == '__main__':
    raise SystemExit(main())
