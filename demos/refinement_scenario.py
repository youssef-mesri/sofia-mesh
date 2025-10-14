#!/usr/bin/env python3
"""
Demo: Apply a refinement scenario from JSON to an initial mesh.

The scenario describes a list of operations (add_node, split_edge) to apply
to a mesh constructed from a simple config (random Delaunay by default).

Example JSON (see configs/refinement_scenario.json):
{
  "mesh": { "type": "random_delaunay", "npts": 60, "seed": 7 },
  "ops": [
    { "op": "add_node", "mode": "tri_centroid", "tri_idx": 0 },
    { "op": "split_edge", "mode": "longest_in_tri", "tri_idx": 0 }
  ],
  "plot": { "out_before": "refine_before.png", "out_after": "refine_after.png" }
}
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, Tuple

import numpy as np

from sofia.sofia.logging_utils import configure_logging, get_logger
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.sofia.visualization import plot_mesh
from sofia.sofia.geometry import triangle_area, EPS_AREA
from sofia.sofia.quality import compute_h

log = get_logger('sofia.demo.refine')


def load_mesh_from_cfg(mesh_cfg: Dict[str, Any]) -> PatchBasedMeshEditor:
    mtype = (mesh_cfg or {}).get('type', 'random_delaunay')
    if mtype == 'random_delaunay':
        npts = int(mesh_cfg.get('npts', 60))
        seed = int(mesh_cfg.get('seed', 7))
        pts, tris = build_random_delaunay(npts=npts, seed=seed)
        return PatchBasedMeshEditor(pts.copy(), tris.copy())
    raise ValueError(f"Unsupported mesh type: {mtype}")


def tri_centroid(editor: PatchBasedMeshEditor, tri_idx: int) -> np.ndarray:
    t = editor.triangles[int(tri_idx)]
    coords = editor.points[[int(t[0]), int(t[1]), int(t[2])]]
    return np.mean(coords, axis=0)


def longest_edge_in_tri(editor: PatchBasedMeshEditor, tri_idx: int) -> Tuple[int, int]:
    t = [int(x) for x in editor.triangles[int(tri_idx)]]
    a, b, c = t
    p = editor.points
    pairs = [(a, b), (b, c), (c, a)]
    def d2(e):
        u, v = e
        dv = p[int(u)] - p[int(v)]
        return float(dv[0] * dv[0] + dv[1] * dv[1])
    pairs.sort(key=d2, reverse=True)
    return tuple(pairs[0])  # type: ignore


def apply_op(editor: PatchBasedMeshEditor, op: Dict[str, Any]) -> None:
    kind = op.get('op')
    if kind == 'add_node':
        mode = op.get('mode', 'tri_centroid')
        if mode == 'tri_centroid':
            tri_idx = int(op.get('tri_idx', 0))
            pt = tri_centroid(editor, tri_idx)
            ok, msg, _ = editor.add_node(pt, tri_idx=tri_idx)
        elif mode == 'point':
            pt = np.array(op['point'], dtype=float)
            tri_idx = op.get('tri_idx', None)
            ok, msg, _ = editor.add_node(pt, tri_idx=int(tri_idx) if tri_idx is not None else None)
        else:
            raise ValueError(f"Unsupported add_node mode: {mode}")
        log.info('add_node(%s) -> %s (%s)', mode, ok, msg)
    elif kind == 'split_edge':
        mode = op.get('mode', 'longest_in_tri')
        if mode == 'edge_by_vertices':
            u, v = [int(x) for x in op['edge']]
            ok, msg, _ = editor.split_edge((u, v))
        elif mode == 'longest_in_tri':
            tri_idx = int(op.get('tri_idx', 0))
            e = longest_edge_in_tri(editor, tri_idx)
            ok, msg, _ = editor.split_edge(e)
        else:
            raise ValueError(f"Unsupported split_edge mode: {mode}")
        log.info('split_edge(%s) -> %s (%s)', mode, ok, msg)
    else:
        raise ValueError(f"Unsupported op: {kind}")


def active_tri_indices(editor: PatchBasedMeshEditor):
    tris = editor.triangles
    return [i for i, t in enumerate(tris) if not np.all(np.array(t) == -1)]


def compute_avg_triangle_area(editor: PatchBasedMeshEditor) -> float:
    idxs = active_tri_indices(editor)
    if not idxs:
        return 0.0
    areas = []
    for i in idxs:
        a, b, c = [int(x) for x in editor.triangles[int(i)]]
        areas.append(abs(triangle_area(editor.points[a], editor.points[b], editor.points[c])))
    return float(np.mean(areas)) if areas else 0.0


def list_internal_edges(editor: PatchBasedMeshEditor):
    internals = []
    for e, ts in editor.edge_map.items():
        ts_list = [int(t) for t in ts]
        # Only edges shared by exactly two active triangles
        if len(ts_list) == 2 and not any(np.all(editor.triangles[int(t)] == -1) for t in ts_list):
            internals.append(tuple(sorted((int(e[0]), int(e[1])))))
    # Deduplicate
    internals = list({tuple(x) for x in internals})
    return internals


def compute_avg_internal_edge_length(editor: PatchBasedMeshEditor) -> float:
    edges = list_internal_edges(editor)
    if not edges:
        return 0.0
    lengths = []
    for u, v in edges:
        d = editor.points[int(u)] - editor.points[int(v)]
        lengths.append(float(np.sqrt(d[0]*d[0] + d[1]*d[1])))
    return float(np.mean(lengths)) if lengths else 0.0


## compute_h moved to sofia.sofia.quality


def refine_to_target_h(editor: PatchBasedMeshEditor, auto_cfg: Dict[str, Any]):
    """Simple refinement to reach target h by scanning and splitting long internal edges.

    - h defined via `h_metric` (default: average internal edge length)
    - target_h = initial_h * target_h_factor (e.g., 0.5 for h/2)
    - per-iter dynamic threshold: thr = 2 * factor * curr_h (so factor=0.5 -> thr==curr_h)
    - loop: sort internal edges by length; split while length > thr
    - stop when curr_h <= target_h + tol or when an iteration makes no splits
    """
    factor = auto_cfg.get('target_h_factor', None)
    if factor is None:
        return
    try:
        factor = float(factor)
    except Exception:
        log.error('auto: invalid target_h_factor=%r', factor)
        return
    h_metric = str(auto_cfg.get('h_metric', 'avg_internal_edge_length'))
    initial_h = compute_h(editor, h_metric)
    if initial_h <= 0.0:
        log.info('auto: no internal edges; skipping refine_to_target_h')
        return
    target_h = initial_h * float(factor)
    max_iters = int(auto_cfg.get('max_h_iters', 10))
    max_splits_per_iter = auto_cfg.get('max_h_splits_per_iter', None)
    max_splits_per_iter = int(max_splits_per_iter) if max_splits_per_iter is not None else None
    tol = float(auto_cfg.get('h_tolerance', 1e-6))
    log.info('auto(h): metric=%s initial_h=%.6g target_h=%.6g (factor=%.3g)', h_metric, initial_h, target_h, factor)
    # Minimal simple scan implementation
    def compute_threshold(curr_h: float) -> float:
        # dynamic threshold; equals curr_h when factor=0.5
        return 2.0 * float(factor) * float(curr_h)

    total_splits = 0
    for it in range(max_iters):
        curr_h_iter = compute_h(editor, h_metric)
        if curr_h_iter <= target_h + tol:
            break
        thr = compute_threshold(curr_h_iter)
        edges = list_internal_edges(editor)
        # Sort longer edges first for faster reduction
        def edge_len(e):
            u, v = int(e[0]), int(e[1])
            d = editor.points[u] - editor.points[v]
            return float(np.hypot(d[0], d[1]))
        edges.sort(key=edge_len, reverse=True)
        splits = 0
        for e in edges:
            if edge_len(e) <= thr:
                break
            ok, _, _ = editor.split_edge((int(e[0]), int(e[1])))
            if ok:
                splits += 1
                total_splits += 1
                if max_splits_per_iter is not None and splits >= max_splits_per_iter:
                    break
        curr_h_after = compute_h(editor, h_metric)
        log.info('auto(h.simple): iter=%d splits=%d curr_h=%.6g target_h=%.6g threshold=%.6g', it, splits, curr_h_after, target_h, thr)
        if splits == 0:
            break
    final_h = compute_h(editor, h_metric)
    log.info('auto(h): final_h=%.6g (initial_h=%.6g target_h=%.6g metric=%s strategy=%s total_splits=%d)', final_h, initial_h, target_h, h_metric, 'simple', total_splits)


def auto_refine(editor: PatchBasedMeshEditor, auto_cfg: Dict[str, Any]):
    order = auto_cfg.get('order', 'tri_first')  # or 'edge_first'
    do_tris = bool(auto_cfg.get('refine_large_tris', True))
    do_edges = bool(auto_cfg.get('split_long_edges', True))
    do_collapse = bool(auto_cfg.get('collapse_shortest_edges', False))
    has_target_h = auto_cfg.get('target_h_factor', None) is not None
    collapse_order = str(auto_cfg.get('collapse_order', 'after'))  # 'before' | 'after'
    # Optional limits to prevent huge refinements
    max_tri_ops = auto_cfg.get('max_tri_ops', None)
    max_edge_ops = auto_cfg.get('max_edge_ops', None)

    def refine_tris():
        avg_area = compute_avg_triangle_area(editor)
        log.info('auto: avg triangle area=%.6g', avg_area)
        tri_ids = active_tri_indices(editor)
        ops = 0
        for ti in tri_ids:
            a, b, c = [int(x) for x in editor.triangles[int(ti)]]
            area = abs(triangle_area(editor.points[a], editor.points[b], editor.points[c]))
            if area > avg_area:
                pt = tri_centroid(editor, int(ti))
                ok, msg, _ = editor.add_node(pt, tri_idx=int(ti))
                log.debug('auto add_node tri=%d area=%.3g -> %s (%s)', ti, area, ok, msg)
                ops += 1
                if max_tri_ops is not None and ops >= int(max_tri_ops):
                    break
        log.info('auto: add_node ops applied=%d', ops)

    def split_edges():
        avg_len = compute_avg_internal_edge_length(editor)
        log.info('auto: avg internal edge length=%.6g', avg_len)
        edges = list_internal_edges(editor)
        ops = 0
        for e in edges:
            u, v = int(e[0]), int(e[1])
            d = editor.points[u] - editor.points[v]
            L = float(np.sqrt(d[0]*d[0] + d[1]*d[1]))
            if L > avg_len:
                ok, msg, _ = editor.split_edge((u, v))
                log.debug('auto split_edge %s L=%.3g -> %s (%s)', e, L, ok, msg)
                ops += 1
                if max_edge_ops is not None and ops >= int(max_edge_ops):
                    break
        log.info('auto: split_edge ops applied=%d', ops)

    def collapse_shortest_edges():
        if not do_collapse:
            return
        target = int(auto_cfg.get('collapse_k', 0) or 0)
        if target <= 0:
            return
        log_failures = bool(auto_cfg.get('collapse_log_failures', False))
        # collect interior edges (exactly 2 adjacent tris) and sort by length
        edges = [tuple(sorted((int(a), int(b)))) for (a,b), ts in editor.edge_map.items() if len(ts) == 2]
        edges.sort(key=lambda e: float(np.linalg.norm(editor.points[int(e[0])] - editor.points[int(e[1])])))
        successes = 0
        attempts = 0
        max_attempts = max(5*target, target + 20)
        i = 0
        while successes < target and attempts < max_attempts and i < len(edges):
            e = edges[i]
            attempts += 1
            ok, msg, _ = editor.edge_collapse(e)
            if ok:
                successes += 1
                log.debug('auto collapse_edge %s -> %s', e, ok)
                # refresh edge list after each success as topology changed
                edges = [tuple(sorted((int(a), int(b)))) for (a,b), ts in editor.edge_map.items() if len(ts) == 2]
                edges.sort(key=lambda e: float(np.linalg.norm(editor.points[int(e[0])] - editor.points[int(e[1])])))
                i = 0
                continue
            else:
                if log_failures:
                    log.debug('auto collapse_edge %s -> %s (%s)', e, ok, msg)
            i += 1
        log.info('auto: collapse_edge successes=%d attempts=%d target=%d', successes, attempts, target)

    # Optional collapse pass before refine/split
    if do_collapse and collapse_order == 'before':
        collapse_shortest_edges()

    if has_target_h:
        # Drive mesh to target h; skip standard refine/split phases
        refine_to_target_h(editor, auto_cfg)
    else:
        if order == 'edge_first':
            if do_edges:
                split_edges()
            if do_tris:
                refine_tris()
        else:
            if do_tris:
                refine_tris()
            if do_edges:
                split_edges()
    # Optional collapse pass after refine/split
    if do_collapse and collapse_order == 'after':
        collapse_shortest_edges()
    # Optional smoothing: move nodes to barycenter after all ops
    if bool(auto_cfg.get('move_to_barycenter', True)):
        passes = max(1, int(auto_cfg.get('barycenter_passes', 1)))
        total_moves = 0
        for _ in range(passes):
            total_moves += editor.move_vertices_to_barycenter()
        log.info('auto: barycenter moves applied=%d (passes=%d)', total_moves, passes)


## moved to package ops as op_move_vertices_to_barycenter


def main():
    ap = argparse.ArgumentParser(description='Apply refinement scenario (add_node, split_edge) from JSON')
    ap.add_argument('--scenario', type=str, required=True, help='Path to scenario JSON file')
    ap.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')
    args = ap.parse_args()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    with open(args.scenario, 'r') as f:
        scenario = json.load(f)

    mesh_cfg = scenario.get('mesh', {})
    plot_cfg = scenario.get('plot', {})
    out_before = plot_cfg.get('out_before', 'refine_before.png')
    out_after = plot_cfg.get('out_after', 'refine_after.png')

    editor = load_mesh_from_cfg(mesh_cfg)
    plot_mesh(editor, outname=out_before)
    log.info('Wrote %s', out_before)

    # Either apply explicit ops list, or an auto refinement block
    auto_cfg = scenario.get('auto', None)
    if auto_cfg:
        # Optional toggle: enforce split quality (improvement) vs relax (refinement)
        if 'enforce_split_quality' in auto_cfg:
            val = bool(auto_cfg.get('enforce_split_quality', True))
            try:
                editor.enforce_split_quality = val
            except Exception:
                pass
        auto_refine(editor, auto_cfg)
    else:
        for i, op in enumerate(scenario.get('ops', [])):
            try:
                apply_op(editor, op)
            except Exception as e:
                log.error('Failed to apply op %d: %s', i, e)
                raise

    editor.compact_triangle_indices()
    plot_mesh(editor, outname=out_after)
    log.info('Wrote %s', out_after)


if __name__ == '__main__':
    main()
