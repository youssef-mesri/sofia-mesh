#!/usr/bin/env python3
import time
import pstats
import cProfile
import argparse
import numpy as np

from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.sofia.remesh_driver import greedy_remesh
from sofia.sofia.config import GreedyConfig, RemeshConfig


def run_profile(
    amortized: bool,
    npts: int = 600,
    seed: int = 0,
    vertex_passes: int = 2,
    edge_passes: int = 2,
    strict: bool = False,
    reject_crossings: bool = False,
    reject_new_loops: bool = False,
    strict_check_cooldown: int = 8,
    compact_end_of_pass: bool = False,
):
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    # Exercise simulated preflight in ops so cooldown can matter
    editor.simulate_compaction_on_commit = True

    # Minimal greedy configuration; keep defaults simple
    g_cfg = GreedyConfig(
        max_vertex_passes=vertex_passes,
        max_edge_passes=edge_passes,
        strict=strict,
        reject_crossings=reject_crossings,
        reject_new_loops=reject_new_loops,
        force_pocket_fill=False,
        verbose=False,
        allow_flips=True,
    )
    # Driver-level cooldown: amortized mode uses configured cooldown; non-amortized forces 0
    g_cfg.strict_check_cooldown = int(strict_check_cooldown if amortized else 0)
    # Optional end-of-pass compaction toggle
    try:
        g_cfg.compact_end_of_pass = bool(compact_end_of_pass)
    except Exception:
        # Backward compatibility if field is absent in older builds
        pass
    r_cfg = RemeshConfig(greedy=g_cfg)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    greedy_remesh(editor, config=r_cfg.greedy)
    # End-of-run compaction (include in profile) — only if tombstones present
    do_compact = False
    try:
        do_compact = bool(getattr(editor, 'has_tombstones', None) and editor.has_tombstones())
    except Exception:
        do_compact = False
    if do_compact:
        try:
            editor._maybe_compact(force=True)
        except Exception:
            # Fallback to direct compaction
            editor.compact_triangle_indices()
    pr.disable()
    t1 = time.perf_counter()

    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats('cumtime')

    # Extract counts/times for key routines
    def stat_of(fn_name_substr):
        total_time = 0.0
        ncalls = 0
        for (filename, lineno, funcname), stat in stats.stats.items():
            if fn_name_substr in funcname or fn_name_substr in filename:
                cc, nc, tt, ct, callers = stat
                ncalls += nc
                total_time += ct
        return ncalls, total_time

    sim_calls, sim_time = stat_of('simulate_compaction_and_check')
    compact_calls, compact_time = stat_of('compact_triangle_indices')

    avg_sim = (sim_time / sim_calls) if sim_calls else 0.0
    return {
        'mode': 'amortized' if amortized else 'non_amortized',
        'wall_time_s': t1 - t0,
        'simulate_compaction_and_check': {'calls': sim_calls, 'cumtime_s': sim_time},
        'compact_triangle_indices': {'calls': compact_calls, 'cumtime_s': compact_time},
        'sim_avg_time_ms': avg_sim * 1000.0,
        'ntri_final': int(len(editor.triangles)),
        'npts_final': int(len(editor.points)),
    }


def main():
    parser = argparse.ArgumentParser(description='Profile amortized vs non-amortized remeshing.')
    parser.add_argument('--npts', type=int, default=600, help='Number of random points for the mesh')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--vertex-passes', type=int, default=2, help='Greedy max vertex passes')
    parser.add_argument('--edge-passes', type=int, default=2, help='Greedy max edge passes')
    parser.add_argument('--strict', action='store_true', help='Enable greedy strict mode')
    parser.add_argument('--reject-crossings', action='store_true', help='Reject crossing edges via simulation')
    parser.add_argument('--reject-new-loops', action='store_true', help='Reject increases in boundary loops')
    parser.add_argument('--strict-check-cooldown', type=int, default=8, help='Cooldown for strict crossing simulation in amortized mode (0 = every time)')
    parser.add_argument('--compact-end-of-pass', action='store_true', help='Compact triangles/vertices at the end of each greedy pass')
    args = parser.parse_args()

    print("Profiling with amortization policy (strict-check cooldown enabled)...")
    res_am = run_profile(
        amortized=True,
        npts=args.npts,
        seed=args.seed,
        vertex_passes=args.vertex_passes,
        edge_passes=args.edge_passes,
        strict=args.strict,
        reject_crossings=args.reject_crossings,
        reject_new_loops=args.reject_new_loops,
        strict_check_cooldown=args.strict_check_cooldown,
        compact_end_of_pass=args.compact_end_of_pass,
    )
    print(res_am)

    print("\nProfiling without amortization policy (strict-check cooldown disabled)...")
    res_no = run_profile(
        amortized=False,
        npts=args.npts,
        seed=args.seed,
        vertex_passes=args.vertex_passes,
        edge_passes=args.edge_passes,
        strict=args.strict,
        reject_crossings=args.reject_crossings,
        reject_new_loops=args.reject_new_loops,
        strict_check_cooldown=args.strict_check_cooldown,
        compact_end_of_pass=args.compact_end_of_pass,
    )
    print(res_no)

    # Compact summary
    ratio = res_no['wall_time_s'] / max(1e-9, res_am['wall_time_s'])
    print(f"\nSpeedup (amortized vs non_amortized): {ratio:.2f}x faster")


if __name__ == '__main__':
    main()
