#!/usr/bin/env python3
"""
Demo: Run patch driver in parallel safely using per-run context isolation.

This script launches several independent remeshing runs concurrently using
ThreadPoolExecutor. Each run has its own RNG seeds, logger, and per-run hook
stored in contextvars (see sofia.sofia.run_context).

Usage (optional args):
  python demos/parallel_patch.py --runs 3 --workers 3 --npts 80 --iterations 5 --log-level INFO

Notes:
- Plotting is disabled for speed and headless safety.
- Logging is configured once; each run logs on a distinct child logger.
- A light compact_check_hook is registered per run via run_context.set_context.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import logging
import random
from typing import Dict, Any, Tuple

import numpy as np

from sofia.sofia.constants import EPS_AREA
from sofia.sofia.config import PatchDriverConfig, RemeshConfig
from sofia.sofia.logging_utils import configure_logging, get_logger
from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.patch_driver import run_patch_batch_driver
from sofia.sofia.run_context import set_context


def _worker(run_id: int, seed: int, npts: int, max_iterations: int, angle_unit: str, threshold: float, log_prefix: str) -> Tuple[int, Dict[str, Any]]:
    # Per-thread/run hook to demonstrate isolation: periodically log a compact check marker.
    def compact_check_hook(editor, pass_type: str, pass_idx: int):
        if pass_idx % 25 == 0:  # keep sparse to avoid log spam
            lg = get_logger(f'{log_prefix}.run{run_id}')
            lg.debug('hook: pass=%s idx=%d pts=%d tris=%d', pass_type, pass_idx, len(editor.points), len(editor.triangles))
    set_context({'compact_check_hook': compact_check_hook})

    # Independent mesh/editor per run
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

    # Run-specific logger
    logger = get_logger(f'{log_prefix}.run{run_id}')
    logger.info('Start run=%d seed=%d npts=%d ntri=%d min_angle=%.3f deg', run_id, seed, len(editor.points), len(editor.triangles), editor.global_min_angle())

    # Patch driver configuration: lightweight and fast for demo
    cfg = PatchDriverConfig(
        threshold=threshold,
        max_iterations=max_iterations,
        patch_radius=1,
        top_k=50,
        disjoint_on='tri',
        allow_overlap=False,
        batch_attempts=2,
        min_triangle_area=EPS_AREA,
        min_triangle_area_fraction=None,
        reject_min_angle_deg=None,
        auto_fill_pockets=False,
        autofill_min_triangle_area=None,
        autofill_reject_min_angle_deg=None,
        angle_unit=angle_unit,
        log_dir=None,
        out_prefix=f'run{run_id}',
        plot_every=10**9,  # effectively disable periodic plots
        use_greedy_remesh=False,
        greedy_vertex_passes=1,
        greedy_edge_passes=1,
        gif_capture=False,
        gif_dir=f'patch_frames_run{run_id}',
        gif_out=f'patch_run_{run_id}.gif',
        gif_fps=4,
    )
    remesh_cfg = RemeshConfig.from_patch_config(cfg)

    result = run_patch_batch_driver(
        editor,
        cfg,
        rng=random.Random(seed),
        np_rng=np.random.RandomState(seed),
        logger=logger,
        greedy_remesh=None,
        plot_mesh=None,  # no plotting in demo
        remesh_config=remesh_cfg,
    )
    logger.info('Done run=%d: %s', run_id, result)
    return run_id, result


def main():
    ap = argparse.ArgumentParser(description='Demo parallel patch driver runs')
    ap.add_argument('--runs', type=int, default=3, help='Number of parallel runs to launch')
    ap.add_argument('--workers', type=int, default=None, help='Max workers (default: equal to runs)')
    ap.add_argument('--npts', type=int, default=80, help='Points per synthetic mesh')
    ap.add_argument('--iterations', type=int, default=5, help='Max iterations per run')
    ap.add_argument('--threshold', type=float, default=20.0, help='Target min-angle threshold (deg)')
    ap.add_argument('--angle-unit', type=str, choices=['deg','rad'], default='deg')
    ap.add_argument('--seed', type=int, default=42, help='Base RNG seed (each run offsets this)')
    ap.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')
    args = ap.parse_args()

    # Configure logging once for all child loggers
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(level)

    runs = int(args.runs)
    workers = int(args.workers or runs)
    log_prefix = 'sofia.demos.parallel'

    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i in range(runs):
            seed_i = args.seed + i
            fut = ex.submit(_worker, i, seed_i, args.npts, args.iterations, args.angle_unit, args.threshold, log_prefix)
            futs.append(fut)
        results = {}
        for fut in futures.as_completed(futs):
            run_id, res = fut.result()
            results[run_id] = res

    # Compact summary
    root_logger = get_logger(log_prefix)
    for rid in sorted(results):
        r = results[rid]
        root_logger.info('summary run=%d: iterations=%s final_min_angle=%.3f deg npts=%d ntri=%d',
                         rid, r.get('iterations'), float(r.get('final_min_angle', float('nan'))), r.get('npts'), r.get('ntri'))


if __name__ == '__main__':
    main()
