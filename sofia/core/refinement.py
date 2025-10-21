"""Refinement helpers exported for demos and package usage.

Provides a stable implementation of `refine_to_target_h(editor, auto_cfg)`
that drives the mesh toward a target average internal edge length by
iteratively splitting long edges. This module intentionally keeps
dependencies lightweight and does not perform debug plotting by default.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.quality import compute_h

log = logging.getLogger('sofia.refinement')


def list_refinable_edges(editor: PatchBasedMeshEditor, include_boundary: bool = False) -> List[Tuple[int, int]]:
    """Return candidate edges for refinement.

    - interior edges have exactly 2 adjacent active triangles
    - boundary edges (deg==1) included only when include_boundary=True
    """
    edges = []
    for e, ts in editor.edge_map.items():
        ts_list = [int(t) for t in ts]
        # filter out tombstoned triangles
        ts_list = [t for t in ts_list if not np.all(editor.triangles[int(t)] == -1)]
        deg = len(ts_list)
        if deg == 2 or (include_boundary and deg == 1):
            edges.append(tuple(sorted((int(e[0]), int(e[1])))))
    # Deduplicate
    edges = list({tuple(x) for x in edges})
    return edges


def list_boundary_edges_only(editor: PatchBasedMeshEditor) -> List[Tuple[int, int]]:
    edges = []
    for e, ts in editor.edge_map.items():
        ts_list = [int(t) for t in ts if not np.all(editor.triangles[int(t)] == -1)]
        if len(ts_list) == 1:
            edges.append(tuple(sorted((int(e[0]), int(e[1])))))
    return list({tuple(x) for x in edges})


def refine_to_target_h(editor: PatchBasedMeshEditor, auto_cfg: Dict[str, Any]):
    """Refine mesh edges until the average edge length reaches target_h.

    The behaviour mirrors the demo implementation but omits debug plotting
    and heavy verification. Expects `auto_cfg` to provide:
      - target_h_factor (float): relative target factor (e.g. 0.5)
      - h_metric (str): metric name passed to compute_h (default 'avg_internal_edge_length')
      - max_h_iters (int)
      - max_h_splits_per_iter (int|None)
      - h_tolerance (float)
      - include_boundary_edges (bool)

    This function mutates the provided editor in-place.
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

    def compute_threshold(curr_h: float) -> float:
        # dynamic threshold; equals curr_h when factor=0.5
        return 2.0 * float(factor) * float(curr_h)

    include_boundary = bool(auto_cfg.get('include_boundary_edges', False))
    total_splits = 0

    for it in range(max_iters):
        curr_h_iter = compute_h(editor, h_metric)
        if curr_h_iter <= target_h + tol:
            break
        thr = compute_threshold(curr_h_iter)

        def edge_len(e: Tuple[int, int]) -> float:
            u, v = int(e[0]), int(e[1])
            d = editor.points[u] - editor.points[v]
            return float(np.hypot(d[0], d[1]))

        splits = 0
        # Optionally handle boundary edges first
        if include_boundary:
            b_edges = list_boundary_edges_only(editor)
            b_edges.sort(key=edge_len, reverse=True)
            for e in b_edges:
                if edge_len(e) <= thr:
                    break
                prev_nv = len(editor.points)
                ok, msg, _ = editor.split_edge((int(e[0]), int(e[1])))
                if ok:
                    splits += 1
                    total_splits += 1
                    if len(editor.points) > prev_nv:
                        # new vertex added; nothing else required here
                        pass
                    if max_splits_per_iter is not None and splits >= max_splits_per_iter:
                        break

        # Interior edges
        if max_splits_per_iter is None or splits < max_splits_per_iter:
            i_edges = [e for e in list_refinable_edges(editor, include_boundary=False)]
            i_edges.sort(key=edge_len, reverse=True)
            for e in i_edges:
                if edge_len(e) <= thr:
                    break
                prev_nv = len(editor.points)
                ok, _, _ = editor.split_edge((int(e[0]), int(e[1])))
                if ok:
                    splits += 1
                    total_splits += 1
                    if len(editor.points) > prev_nv:
                        pass
                    if max_splits_per_iter is not None and splits >= max_splits_per_iter:
                        break

        curr_h_after = compute_h(editor, h_metric)
        log.info('auto(h.simple): iter=%d splits=%d curr_h=%.6g target_h=%.6g threshold=%.6g', it, splits, curr_h_after, target_h, thr)
        if splits == 0:
            break

    final_h = compute_h(editor, h_metric)
    log.info('auto(h): final_h=%.6g (initial_h=%.6g target_h=%.6g metric=%s strategy=%s total_splits=%d)',
             final_h, initial_h, target_h, h_metric, 'simple', total_splits)


__all__ = ['refine_to_target_h', 'list_refinable_edges', 'list_boundary_edges_only']
