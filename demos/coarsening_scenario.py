#!/usr/bin/env python3
"""
Demo: Apply a coarsening scenario (edge collapses, remove_node, smoothing) to an initial mesh.

Example JSON (see configs/coarsening_scenario.json):
{
  "mesh": { "type": "random_delaunay", "npts": 80, "seed": 7 },
  "auto": {
    "pre_smooth": true,
    "barycenter_passes": 1,
    "collapse_shortest_edges": true,
    "collapse_threshold_factor": 0.7,
    "max_collapse_per_iter": 50,
    "remove_low_degree_vertices": true,
    "remove_degree_max": 4,
    "remove_max_per_iter": 20,
    "allow_boundary_collapse": false,
    "allow_boundary_remove": false,
    "relax_remove_quality": false,
    "iterations": 3
  },
  "plot": { "out_before": "coarsen_before.png", "out_after": "coarsen_after.png" }
}
"""
from __future__ import annotations

import argparse
import cProfile as _cprof
import io as _io
import json
import logging
import pstats as _pstats
from typing import Any, Dict, Tuple

import numpy as np

from sofia.core.logging_utils import configure_logging, get_logger
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.core.visualization import plot_mesh
from sofia.core.quality import compute_h

log = get_logger('sofia.demo.coarsen')


def load_mesh_from_cfg(mesh_cfg: Dict[str, Any]) -> PatchBasedMeshEditor:
    mtype = (mesh_cfg or {}).get('type', 'random_delaunay')
    if mtype == 'random_delaunay':
        npts = int(mesh_cfg.get('npts', 80))
        seed = int(mesh_cfg.get('seed', 7))
        pts, tris = build_random_delaunay(npts=npts, seed=seed)
        return PatchBasedMeshEditor(pts.copy(), tris.copy())
    raise ValueError(f"Unsupported mesh type: {mtype}")


def internal_edges(editor: PatchBasedMeshEditor):
    edges = []
    for e, ts in editor.edge_map.items():
        active_adj = [int(t) for t in ts if not np.all(editor.triangles[int(t)] == -1)]
        if len(active_adj) == 2:
            edges.append(tuple(sorted((int(e[0]), int(e[1])))))
    # dedupe
    return list({tuple(x) for x in edges})


def avg_internal_edge_length(editor: PatchBasedMeshEditor) -> float:
    es = internal_edges(editor)
    if not es:
        return 0.0
    Ls = []
    for u, v in es:
        d = editor.points[int(u)] - editor.points[int(v)]
        Ls.append(float(np.hypot(d[0], d[1])))
    return float(np.mean(Ls)) if Ls else 0.0


def is_boundary_vertex(editor: PatchBasedMeshEditor, v: int) -> bool:
    for ti in editor.v_map.get(int(v), []):
        t = editor.triangles[int(ti)]
        for i in range(3):
            u, w = int(t[i]), int(t[(i+1)%3])
            key = tuple(sorted((u, w)))
            if v in (u, w):
                # Optimized: count non-tombstoned triangles without list comprehension
                incident_tris = editor.edge_map.get(key, [])
                count = sum(1 for tt in incident_tris if editor.triangles[int(tt)][0] != -1)
                if count == 1:
                    return True
    return False


def degree(editor: PatchBasedMeshEditor, v: int) -> int:
    # Degree defined as number of unique neighbors in star
    neighbors = set()
    for ti in editor.v_map.get(int(v), []):
        t = editor.triangles[int(ti)]
        for x in t:
            if int(x) != int(v):
                neighbors.add(int(x))
    return len(neighbors)


def auto_coarsen(editor: PatchBasedMeshEditor, auto_cfg: Dict[str, Any]):
    """Coarsen the mesh using edge collapses, optional removals, and smoothing.

    Supports two modes:
    - threshold mode (default): collapse edges shorter than collapse_threshold_factor * avg_internal_edge_length
    - target-h mode: if 'target_h_factor' provided (>1), drive coarsening until
      compute_h(editor, h_metric) >= initial_h * factor, collapsing edges shorter than
      a dynamic threshold thr = (2/factor) * curr_h per-iteration (symmetry with refinement).
    """
    # Smoothing (pre)
    pre = bool(auto_cfg.get('pre_smooth', True))
    passes = max(1, int(auto_cfg.get('barycenter_passes', 1)))
    if pre:
        moves = 0
        for _ in range(passes):
            moves += editor.move_vertices_to_barycenter()
        log.info('coarsen: pre-smooth moves=%d (passes=%d)', moves, passes)

    # Optionally relax remove quality gating for this demo run
    relax_remove_quality = bool(auto_cfg.get('relax_remove_quality', False))
    if relax_remove_quality:
        try:
            editor.enforce_remove_quality = False
            log.info('coarsen: enforce_remove_quality disabled for this run')
        except Exception:
            pass

    iters = max(1, int(auto_cfg.get('iterations', 3)))
    allow_boundary_collapse = bool(auto_cfg.get('allow_boundary_collapse', False))
    allow_boundary_remove = bool(auto_cfg.get('allow_boundary_remove', False))
    collapse_thr_factor = float(auto_cfg.get('collapse_threshold_factor', 0.7))
    max_collapse = int(auto_cfg.get('max_collapse_per_iter', 50))
    do_collapse = bool(auto_cfg.get('collapse_shortest_edges', True))
    do_remove = bool(auto_cfg.get('remove_low_degree_vertices', True))
    remove_degree_max = int(auto_cfg.get('remove_degree_max', 4))
    remove_max = int(auto_cfg.get('remove_max_per_iter', 20))
    use_remove_patch2 = bool(auto_cfg.get('use_remove_patch2', False))

    # Target-h driven coarsening (2h, 4h, ...)
    factor = auto_cfg.get('target_h_factor', None)
    if factor is not None and do_collapse:
        try:
            factor = float(factor)
        except Exception:
            factor = None
        if factor is not None and factor > 1.0:
            h_metric = str(auto_cfg.get('h_metric', 'avg_internal_edge_length'))
            initial_h = compute_h(editor, h_metric)
            tol = float(auto_cfg.get('h_tolerance', 1e-6))
            max_h_iters = int(auto_cfg.get('max_h_iters', iters))
            if initial_h <= 0.0:
                log.info('coarsen(h): no internal edges; skipping target-h coarsening')
            else:
                target_h = initial_h * float(factor)
                log.info('coarsen(h): metric=%s initial_h=%.6g target_h=%.6g (factor=%.3g)', h_metric, initial_h, target_h, factor)
                # Dynamic threshold per iteration:
                # Use thr = (factor/2) * curr_h so larger factors collapse more edges.
                # This mirrors refinement where factor=0.5 -> thr==curr_h for splits;
                # here factor=2 -> thr==curr_h for collapses, factor=4 -> thr==2*curr_h (more aggressive).
                def compute_threshold(curr_h: float) -> float:
                    return (float(factor) / 2.0) * float(curr_h)
                for it in range(max_h_iters):
                    curr_h_iter = compute_h(editor, h_metric)
                    if curr_h_iter >= target_h - tol:
                        break
                    thr = compute_threshold(curr_h_iter)
                    # Collapse shortest internal edges first if below threshold
                    edges = internal_edges(editor)
                    def elen(e):
                        u, v = int(e[0]), int(e[1])
                        d = editor.points[u] - editor.points[v]
                        return float(np.hypot(d[0], d[1]))
                    edges.sort(key=elen)  # shortest first
                    collapsed = 0
                    for e in edges:
                        if elen(e) >= thr:
                            break
                        if not allow_boundary_collapse:
                            u, v = int(e[0]), int(e[1])
                            if is_boundary_vertex(editor, u) or is_boundary_vertex(editor, v):
                                continue
                        ok, msg, _ = editor.edge_collapse(e)
                        log.debug('coarsen(h): collapse %s -> %s (%s)', e, ok, msg)
                        if ok:
                            collapsed += 1
                            if collapsed >= max_collapse:
                                break
                    # Optional remove-node pass per iteration
                    removed = 0
                    if do_remove:
                        verts = list(editor.v_map.keys())
                        scored = []
                        for v in verts:
                            v = int(v)
                            if not allow_boundary_remove and is_boundary_vertex(editor, v):
                                continue
                            deg = degree(editor, v)
                            if deg <= remove_degree_max:
                                scored.append((v, deg))
                        scored.sort(key=lambda x: (x[1], x[0]))
                        for v, degv in scored:
                            if use_remove_patch2:
                                from sofia.core.operations import op_remove_node_with_patch2
                                ok, msg, _ = op_remove_node_with_patch2(editor, int(v))
                            else:
                                ok, msg, _ = editor.remove_node_with_patch(int(v))
                            log.debug('coarsen(h): remove_node v=%d deg=%d -> %s (%s)', v, degv, ok, msg)
                            if ok:
                                removed += 1
                                if removed >= remove_max:
                                    break
                    # Post-smoothing per-iter (use configured passes)
                    moves = 0
                    for _ in range(passes):
                        moves += editor.move_vertices_to_barycenter()
                    curr_h_after = compute_h(editor, h_metric)
                    log.info('coarsen(h): iter=%d collapsed=%d removed=%d curr_h=%.6g target_h=%.6g threshold=%.6g post_smooth_moves=%d',
                             it, collapsed, removed, curr_h_after, target_h, thr, moves)
                    if collapsed == 0 and removed == 0:
                        break
            # Done with target-h path; return to avoid running threshold mode below
            return
        elif factor is not None:
            log.warning('coarsen(h): target_h_factor=%.3g <= 1; expected >1 for coarsening. Falling back to threshold mode.', factor)

    for it in range(iters):
        # Edge collapse pass (short edges first)
        if do_collapse:
            avgL = avg_internal_edge_length(editor)
            if avgL > 0:
                thr = float(collapse_thr_factor) * avgL
            else:
                thr = 0.0
            edges = internal_edges(editor)
            def elen(e):
                u, v = int(e[0]), int(e[1])
                d = editor.points[u] - editor.points[v]
                return float(np.hypot(d[0], d[1]))
            edges.sort(key=elen)  # shortest first
            collapsed = 0
            for e in edges:
                if elen(e) > thr:
                    break
                if not allow_boundary_collapse:
                    # Skip if either endpoint is boundary
                    u, v = int(e[0]), int(e[1])
                    if is_boundary_vertex(editor, u) or is_boundary_vertex(editor, v):
                        continue
                ok, msg, _ = editor.edge_collapse(e)
                log.debug('coarsen: collapse %s -> %s (%s)', e, ok, msg)
                if ok:
                    collapsed += 1
                    if collapsed >= max_collapse:
                        break
            log.info('coarsen: iter=%d collapsed=%d thr=%.6g avgL=%.6g', it, collapsed, thr, avgL)

        # Remove-node pass (low-degree vertices)
        if do_remove:
            verts = list(editor.v_map.keys())
            # degree ascending, prefer interior unless allowed
            scored = []
            for v in verts:
                v = int(v)
                if not allow_boundary_remove and is_boundary_vertex(editor, v):
                    continue
                deg = degree(editor, v)
                if deg <= remove_degree_max:
                    scored.append((v, deg))
            scored.sort(key=lambda x: (x[1], x[0]))
            removed = 0
            for v, deg in scored:
                if use_remove_patch2:
                    from sofia.core.operations import op_remove_node_with_patch2
                    ok, msg, _ = op_remove_node_with_patch2(editor, int(v))
                else:
                    ok, msg, _ = editor.remove_node_with_patch(int(v))
                log.debug('coarsen: remove_node v=%d deg=%d -> %s (%s)', v, deg, ok, msg)
                if ok:
                    removed += 1
                    if removed >= remove_max:
                        break
            log.info('coarsen: iter=%d removed=%d (deg<=%d)', it, removed, remove_degree_max)

        # Smoothing (post per-iter)
        moves = 0
        for _ in range(passes):
            moves += editor.move_vertices_to_barycenter()
        log.info('coarsen: post-smooth moves=%d (passes=%d)', moves, passes)


def _run(args):
    """Main logic extracted for profiling."""
    with open(args.scenario, 'r') as f:
        scenario = json.load(f)

    mesh_cfg = scenario.get('mesh', {})
    plot_cfg = scenario.get('plot', {})
    out_before = plot_cfg.get('out_before', 'coarsen_before.png')
    out_after = plot_cfg.get('out_after', 'coarsen_after.png')

    editor = load_mesh_from_cfg(mesh_cfg)
    plot_mesh(editor, outname=out_before)
    log.info('Wrote %s', out_before)

    auto_cfg = scenario.get('auto', None)
    if auto_cfg:
        # Route boundary handling through existing virtual_boundary_mode if specified (default False here)
        if 'virtual_boundary_mode' in auto_cfg:
            try:
                editor.virtual_boundary_mode = bool(auto_cfg.get('virtual_boundary_mode', False))
            except Exception:
                pass
        auto_coarsen(editor, auto_cfg)
    else:
        log.info('No auto block provided; nothing to do')

    editor.compact_triangle_indices()
    plot_mesh(editor, outname=out_after)
    log.info('Wrote %s', out_after)


def main():
    ap = argparse.ArgumentParser(description='Apply coarsening scenario (collapse_edge, remove_node, smoothing) from JSON')
    ap.add_argument('--scenario', type=str, required=True, help='Path to scenario JSON file')
    ap.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')
    ap.add_argument('--profile', action='store_true', help='Enable cProfile and print top hotspots')
    ap.add_argument('--profile-out', type=str, default=None, help='Write raw cProfile stats to this .pstats file when --profile is set')
    ap.add_argument('--profile-top', type=int, default=25, help='How many entries to show in hotspots (default: 25)')
    args = ap.parse_args()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.profile:
        pr = _cprof.Profile()
        pr.enable()
        _run(args)
        pr.disable()
        
        if args.profile_out:
            pr.dump_stats(args.profile_out)
            log.info('Wrote profiling stats to %s', args.profile_out)
        
        s = _io.StringIO()
        ps = _pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(args.profile_top)
        log.info('Profile (top %d by cumulative time):\n%s', args.profile_top, s.getvalue())
    else:
        _run(args)


if __name__ == '__main__': 
    main()
