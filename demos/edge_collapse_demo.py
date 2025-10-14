#!/usr/bin/env python3
"""
Demo: Collapse a set of the shortest interior edges.

Build a random Delaunay mesh, select interior edges (shared by exactly two
triangles), sort by length ascending, and attempt to collapse them in order
until a target count is reached. Uses the editor method `edge_collapse`.

Outputs before/after PNGs.
"""
from __future__ import annotations

import argparse
import logging
from typing import List, Tuple

import numpy as np

from sofia.sofia.logging_utils import configure_logging, get_logger
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.sofia.visualization import plot_mesh

log = get_logger('sofia.demo.edge_collapse')


def list_interior_edges(editor: PatchBasedMeshEditor) -> List[Tuple[int,int]]:
    return [tuple(sorted((int(a), int(b)))) for (a,b), ts in editor.edge_map.items() if len(ts) == 2]


def edge_length(editor: PatchBasedMeshEditor, e: Tuple[int,int]) -> float:
    u, v = int(e[0]), int(e[1])
    d = editor.points[u] - editor.points[v]
    return float(np.hypot(d[0], d[1]))


def run_edge_collapse_demo(npts: int, seed: int, k: int, out_before: str, out_after: str, log_failures: bool = False):
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

    # Plot before
    plot_mesh(editor, outname=out_before)
    log.info('Wrote %s', out_before)

    successes = 0
    attempts = 0
    # Safety cap to avoid long loops in pathological cases
    max_attempts = max(5*k, k + 20)

    while successes < k and attempts < max_attempts:
        edges = list_interior_edges(editor)
        if not edges:
            log.info('No more interior edges.')
            break
        # Sort by length (shortest first)
        edges.sort(key=lambda e: edge_length(editor, e))

        progressed = False
        for e in edges:
            attempts += 1
            ok, msg, _ = editor.edge_collapse(e)
            if ok:
                successes += 1
                progressed = True
                log.info('Collapsed edge %s (success=%d/%d)', e, successes, k)
                break  # recompute edge list after each success
            else:
                if log_failures:
                    log.debug('Failed to collapse %s: %s', e, msg)
            if attempts >= max_attempts or successes >= k:
                break
        if not progressed:
            # No edge succeeded in this round; likely quality gate prevents further collapses.
            log.info('No further collapses possible under current quality guard (attempts=%d).', attempts)
            break

    log.info('Edge collapse summary: successes=%d, attempts=%d', successes, attempts)

    # Compact and plot after
    editor.compact_triangle_indices()
    plot_mesh(editor, outname=out_after)
    log.info('Wrote %s', out_after)


def main():
    ap = argparse.ArgumentParser(description='Collapse the shortest interior edges of a random mesh')
    ap.add_argument('--npts', type=int, default=60, help='number of initial random points')
    ap.add_argument('--seed', type=int, default=7, help='random seed')
    ap.add_argument('--k', type=int, default=20, help='target number of edge collapses')
    ap.add_argument('--out-before', type=str, default='collapse_before.png')
    ap.add_argument('--out-after', type=str, default='collapse_after.png')
    ap.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')
    ap.add_argument('--log-failures', action='store_true', help='log failed collapse attempts at DEBUG level')
    args = ap.parse_args()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    if args.log_failures and args.log_level.upper() != 'DEBUG':
        # Encourage DEBUG when inspecting failures
        log.warning('--log-failures is most useful with --log-level DEBUG')

    run_edge_collapse_demo(
        npts=args.npts,
        seed=args.seed,
        k=args.k,
        out_before=args.out_before,
        out_after=args.out_after,
        log_failures=bool(args.log_failures),
    )


if __name__ == '__main__':
    main()
