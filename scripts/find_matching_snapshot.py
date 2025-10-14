#!/usr/bin/env python3
"""Find earliest per-op compacted snapshot in gtrace that matches the final failing mesh from gfail_seed_0.npz.
Matching is attempted by exact (indices) equality first, then by geometric triangle set equality (coordinates rounded).
"""
import numpy as np
import os

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
        print('Missing', GFAIL); return 2
    if not os.path.exists(GTRACE):
        print('Missing', GTRACE); return 2
    g = np.load(GFAIL, allow_pickle=True)
    pts_after = g['pts_after']
    tris_after = g['tris_after']

    trace = np.load(GTRACE, allow_pickle=True)
    steps = trace['steps']
    pts_arr = trace['pts_comp']
    tris_arr = trace['tris_comp']
    actions = trace.get('actions', None)

    print('gfail pts_after', pts_after.shape, 'tris_after', tris_after.shape)
    target_geom = tri_geom_set(pts_after, tris_after)

    for i, step in enumerate(steps):
        pts = pts_arr[i]
        tris = tris_arr[i]
        if pts is None or tris is None:
            continue
        # exact index-level compare (fast)
        try:
            if pts.shape == pts_after.shape and tris.shape == tris_after.shape and np.array_equal(pts, pts_after) and np.array_equal(tris, tris_after):
                print('Exact equal match at step', int(step))
                if actions is not None:
                    print('action:', actions[i])
                return 0
        except Exception:
            pass
        # geometry-based compare
        try:
            geom = tri_geom_set(pts, tris)
            if geom == target_geom:
                print('Geometric match at step', int(step), 'index', i)
                if actions is not None:
                    print('action:', actions[i])
                return 0
        except Exception as e:
            print('compare error at step', int(step), e)
            continue
    print('No matching per-op snapshot found')
    return 1

if __name__ == '__main__':
    raise SystemExit(main())
