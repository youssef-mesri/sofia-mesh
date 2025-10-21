#!/usr/bin/env python3
"""
Quick profiling harness for typical meshes.

- Runs a greedy remesh on a random Delaunay mesh
- Prints per-operation aggregated stats from PatchBasedMeshEditor
- Optionally records a cProfile session and prints hotspots
- Can dump raw .pstats for further analysis with snakeviz or gprof2dot
"""
import argparse
import json
import io
import time
import numpy as np

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.core.remesh_driver import greedy_remesh
from sofia.core.config import GreedyConfig
from sofia.core.stats import print_stats as print_op_stats


def run_once(npts: int, seed: int, vertex_passes: int, edge_passes: int, strict: bool,
             reject_crossings: bool, reject_new_loops: bool,
             profile: bool, profile_out: str):
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

    g_cfg = GreedyConfig(
        max_vertex_passes=vertex_passes,
        max_edge_passes=edge_passes,
        strict=strict,
        reject_crossings=reject_crossings,
        reject_new_loops=reject_new_loops,
        verbose=False,
        allow_flips=True,
    )

    t0 = time.perf_counter()
    if profile:
        import cProfile, pstats
        pr = cProfile.Profile()
        pr.enable()
        greedy_remesh(editor, config=g_cfg)
        pr.disable()
        t1 = time.perf_counter()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats('cumtime').print_stats(25)
        print("\n[hotspots] top 25 by cumulative time:\n" + s.getvalue())
        if profile_out:
            try:
                pr.dump_stats(profile_out)
                print(f"[hotspots] raw pstats written to {profile_out}")
            except Exception as e:
                print(f"[hotspots] failed to write pstats to {profile_out}: {e}")
    else:
        greedy_remesh(editor, config=g_cfg)
        t1 = time.perf_counter()

    print("\n[per-op stats]")
    print_op_stats(editor.stats_summary(), pretty=True)
    print(f"\n[summary] wall_time_s={t1 - t0:.3f}  ntri={len(editor.triangles)}  npts={len(editor.points)}")


def main():
    ap = argparse.ArgumentParser(description='Profile typical greedy run and print per-op timings and hotspots.')
    ap.add_argument('--npts', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--vertex-passes', type=int, default=2)
    ap.add_argument('--edge-passes', type=int, default=2)
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--reject-crossings', action='store_true')
    ap.add_argument('--reject-new-loops', action='store_true')
    ap.add_argument('--profile', action='store_true', help='Enable cProfile and print top hotspots')
    ap.add_argument('--profile-out', type=str, default=None, help='Write raw cProfile stats to this .pstats file when --profile is set')
    args = ap.parse_args()

    run_once(
        npts=args.npts,
        seed=args.seed,
        vertex_passes=args.vertex_passes,
        edge_passes=args.edge_passes,
        strict=args.strict,
        reject_crossings=args.reject_crossings,
        reject_new_loops=args.reject_new_loops,
        profile=args.profile,
        profile_out=args.profile_out,
    )


if __name__ == '__main__':
    main()
