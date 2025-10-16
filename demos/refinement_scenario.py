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
import io as _io
import cProfile as _cprof
import pstats as _pstats
import json
import logging
from typing import Any, Dict, Tuple

import numpy as np

from sofia.sofia.logging_utils import configure_logging, get_logger
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.sofia.visualization import plot_mesh
from sofia.sofia.geometry import triangle_area, triangle_angles, EPS_AREA
from sofia.sofia.quality import compute_h
import os as _os
import matplotlib as _mpl
if not _os.environ.get('MPLBACKEND'):
    try:
        _mpl.use('Agg')
    except Exception:
        pass
import matplotlib.pyplot as plt

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


def list_refinable_edges(editor: PatchBasedMeshEditor, include_boundary: bool = False):
    """Return edges considered for splitting.

    - When include_boundary=False: only edges with exactly 2 adjacent active triangles (interior)
    - When include_boundary=True: also include edges with exactly 1 adjacent active triangle (boundary)
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

def list_boundary_edges_only(editor: PatchBasedMeshEditor):
    edges = []
    for e, ts in editor.edge_map.items():
        ts_list = [int(t) for t in ts if not np.all(editor.triangles[int(t)] == -1)]
        if len(ts_list) == 1:
            edges.append(tuple(sorted((int(e[0]), int(e[1])))))
    return list({tuple(x) for x in edges})


def _remove_bad_quality_vertices(editor: PatchBasedMeshEditor,
                                 min_deg: float = 20.0,
                                 max_rem: int = 2,
                                 allow_boundary: bool = True,
                                 prefix: str = 'sanity') -> int:
    """Remove up to max_rem vertices whose local star min-angle is below min_deg.
    Returns number of successful removals.
    """
    def local_star_min_angle(v: int) -> float:
        tris_idx = [int(t) for t in editor.v_map.get(int(v), []) if not np.all(editor.triangles[int(t)] == -1)]
        if not tris_idx:
            return 180.0
        mins = []
        for ti in tris_idx:
            a, b, c = [int(x) for x in editor.triangles[int(ti)]]
            A, B, C = triangle_angles(editor.points[a], editor.points[b], editor.points[c])
            mins.append(float(min(A, B, C)))
        return float(min(mins)) if mins else 180.0

    def is_boundary_vertex(v: int) -> bool:
        # deg==1 on any incident edge implies boundary
        for ti in editor.v_map.get(int(v), []):
            t = editor.triangles[int(ti)]
            for i in range(3):
                u, w = int(t[i]), int(t[(i+1)%3])
                key = tuple(sorted((u, w)))
                if v in (u, w):
                    inc = [tt for tt in editor.edge_map.get(key, []) if not np.all(editor.triangles[int(tt)] == -1)]
                    if len(inc) == 1:
                        return True
        return False

    all_vertices = list(editor.v_map.keys())
    scored = [(int(v), local_star_min_angle(int(v))) for v in all_vertices]
    scored = [(v, ang) for v, ang in scored if ang < float(min_deg)]
    if not allow_boundary:
        scored = [(v, ang) for v, ang in scored if not is_boundary_vertex(int(v))]
    scored.sort(key=lambda x: x[1])
    removed = 0
    for v, ang in scored:
        ok_rm, msg_rm, _ = editor.remove_node_with_patch(int(v))
        log.debug('%s remove_node v=%d (min_angle=%.3g) -> %s (%s)', prefix, v, ang, ok_rm, msg_rm)
        if ok_rm:
            removed += 1
            if removed >= int(max_rem):
                break
    if removed:
        log.info('%s: removed=%d (min_angle<%.3g)', prefix, removed, min_deg)
    return removed


def _compact_for_plot(editor: PatchBasedMeshEditor):
    """Return (new_points, new_tris, mapping, active_idx).
    mapping: orig_vertex_index -> compacted vertex index used in new_points/new_tris
    active_idx: list of original triangle indices kept (non-tombstoned), in order of new_tris rows.
    """
    tris = np.array(editor.triangles)
    pts = np.array(editor.points)
    active_mask = ~np.all(tris == -1, axis=1)
    active_tris = tris[active_mask]
    active_idx = np.nonzero(active_mask)[0].tolist()
    used_verts = sorted({int(v) for t in active_tris for v in t if int(v) >= 0})
    mapping = {old: new for new, old in enumerate(used_verts)}
    new_points = pts[used_verts]
    new_tris = []
    for t in active_tris:
        if np.any(np.array(t) < 0):
            continue
        try:
            new_tris.append([mapping[int(t[0])], mapping[int(t[1])], mapping[int(t[2])]])
        except KeyError:
            continue
    new_tris = np.array(new_tris, dtype=int)
    return new_points, new_tris, mapping, active_idx


def _boundary_edges_from_compacted(new_tris: np.ndarray):
    """Compute boundary edges (deg==1) from compacted triangles array."""
    from collections import defaultdict
    edge_count = defaultdict(int)
    for t in new_tris:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        for e in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(e))
            edge_count[key] += 1
    return [e for e, c in edge_count.items() if c == 1]


def plot_mesh_with_annotations(editor: PatchBasedMeshEditor, outname: str,
                               new_vertices: list[int] | None = None,
                               show_boundary_edges: bool = True,
                               boundary_color=(0.85, 0.2, 0.2)):
    """Plot the mesh like plot_mesh, and overlay boundary edges and newly added vertices.

    new_vertices: list of original vertex indices to highlight (e.g., those added in the last step).
    """
    new_vertices = new_vertices or []
    pts, tris, mapping, _ = _compact_for_plot(editor)
    if tris.size == 0:
        plt.figure(figsize=(6, 6))
        plt.title('empty mesh')
        plt.savefig(outname, dpi=150)
        plt.close()
        return
    plt.figure(figsize=(6, 6))
    plt.triplot(pts[:, 0], pts[:, 1], tris, lw=0.6)
    plt.scatter(pts[:, 0], pts[:, 1], s=6)
    # Boundary edges overlay
    if show_boundary_edges:
        b_edges = _boundary_edges_from_compacted(tris)
        for u, v in b_edges:
            p, q = pts[int(u)], pts[int(v)]
            plt.plot([p[0], q[0]], [p[1], q[1]], color=boundary_color, linewidth=1.4)
    # Newly added vertices overlay (map original indices to compacted)
    for ov in new_vertices:
        cv = mapping.get(int(ov))
        if cv is None:
            continue
        p = pts[int(cv)]
        plt.plot([p[0]], [p[1]], marker='o', markersize=5.0, color=(0.1, 0.7, 0.2))
    plt.gca().set_aspect('equal')
    plt.title(outname)
    plt.savefig(outname, dpi=150)
    plt.close()


def _verify_boundary_split(editor: PatchBasedMeshEditor, edge, new_idx: int):
    """Sanity-check connectivity for a boundary split.
    Returns (ok, messages). Checks:
    - old edge removed
    - new edges (u,new) and (new,v) exist
    - both new edges are boundary (deg==1)
    - new vertex referenced by exactly 2 active triangles
    """
    msgs = []
    ok = True
    u, v = int(edge[0]), int(edge[1])
    old = tuple(sorted((u, v)))
    if old in editor.edge_map:
        ok = False; msgs.append(f"old edge {old} still present")
    e1 = (min(u, new_idx), max(u, new_idx))
    e2 = (min(v, new_idx), max(v, new_idx))
    if e1 not in editor.edge_map:
        ok = False; msgs.append(f"edge {e1} missing")
    if e2 not in editor.edge_map:
        ok = False; msgs.append(f"edge {e2} missing")
    # boundary degree check
    if e1 in editor.edge_map:
        deg1 = len([t for t in editor.edge_map[e1] if not np.all(editor.triangles[int(t)] == -1)])
        if deg1 != 1:
            ok = False; msgs.append(f"edge {e1} deg={deg1} (expected 1)")
    if e2 in editor.edge_map:
        deg2 = len([t for t in editor.edge_map[e2] if not np.all(editor.triangles[int(t)] == -1)])
        if deg2 != 1:
            ok = False; msgs.append(f"edge {e2} deg={deg2} (expected 1)")
    # new vertex star size
    if new_idx not in editor.v_map:
        ok = False; msgs.append(f"new vertex {new_idx} not in v_map")
    else:
        tri_refs = [ti for ti in editor.v_map[new_idx] if not np.all(editor.triangles[int(ti)] == -1)]
        if len(tri_refs) != 2:
            ok = False; msgs.append(f"new vertex {new_idx} referenced by {len(tri_refs)} active tris (expected 2)")
    return ok, msgs, (e1, e2)


def compute_avg_edge_length(editor: PatchBasedMeshEditor, include_boundary: bool = False) -> float:
    edges = list_refinable_edges(editor, include_boundary=include_boundary)
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

    include_boundary = bool(auto_cfg.get('include_boundary_edges', False))
    total_splits = 0
    # Debug plot options
    plot_cfg = auto_cfg if isinstance(auto_cfg, dict) else {}
    annotate_debug = bool(plot_cfg.get('annotate_debug', False))
    annotate_dir = str(plot_cfg.get('annotate_dir', 'refine_debug'))
    annotate_stride = max(1, int(plot_cfg.get('annotate_stride', 1)))
    if annotate_debug:
        try:
            _os.makedirs(annotate_dir, exist_ok=True)
        except Exception:
            annotate_debug = False
    for it in range(max_iters):
        curr_h_iter = compute_h(editor, h_metric)
        if curr_h_iter <= target_h + tol:
            break
        thr = compute_threshold(curr_h_iter)
        # Two-phase: boundary edges first (deg==1), then interior (deg==2)
        def edge_len(e):
            u, v = int(e[0]), int(e[1])
            d = editor.points[u] - editor.points[v]
            return float(np.hypot(d[0], d[1]))
        splits = 0
        iter_new_vertices_boundary = []
        iter_new_vertices_interior = []
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
                    # record just-added vertex index
                    if len(editor.points) > prev_nv:
                        new_idx = len(editor.points) - 1
                        iter_new_vertices_boundary.append(new_idx)
                        # connectivity verification (only after first iteration, per user report)
                        if it >= 1 and annotate_debug:
                            okc, msgs_c, new_edges = _verify_boundary_split(editor, (int(e[0]), int(e[1])), new_idx)
                            if not okc:
                                out_err = _os.path.join(annotate_dir, f"iter{it:03d}_boundary_anomaly.png")
                                plot_mesh_with_annotations(editor, outname=out_err, new_vertices=[new_idx], show_boundary_edges=True)
                                log.warning('Boundary split anomaly at iter=%d edge=%s: %s; snapshot=%s', it, e, msgs_c, out_err)
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
                        iter_new_vertices_interior.append(len(editor.points) - 1)
                    if max_splits_per_iter is not None and splits >= max_splits_per_iter:
                        break
        # Annotate per-iteration boundary-first behavior
        if annotate_debug and (it % annotate_stride == 0):
            if iter_new_vertices_boundary:
                out_b = _os.path.join(annotate_dir, f"iter{it:03d}_boundary.png")
                plot_mesh_with_annotations(editor, outname=out_b, new_vertices=iter_new_vertices_boundary, show_boundary_edges=True)
            if iter_new_vertices_interior:
                out_i = _os.path.join(annotate_dir, f"iter{it:03d}_interior.png")
                plot_mesh_with_annotations(editor, outname=out_i, new_vertices=iter_new_vertices_interior, show_boundary_edges=True)
        curr_h_after = compute_h(editor, h_metric)
        log.info('auto(h.simple): iter=%d splits=%d curr_h=%.6g target_h=%.6g threshold=%.6g', it, splits, curr_h_after, target_h, thr)
        if splits == 0:
            break
        # Optional sanity removal pass to clean up local degradations
        sanity_on = bool(auto_cfg.get('sanity_remove_bad_quality', False))
        if sanity_on and ((it + 1) % int(auto_cfg.get('sanity_every', 1)) == 0):
            min_deg = float(auto_cfg.get('sanity_min_angle_deg', 20.0))
            max_rem = int(auto_cfg.get('sanity_max_removals_per_iter', 2))
            allow_boundary = bool(auto_cfg.get('sanity_allow_boundary', True))
            # collect candidate vertices with poor local min-angle
            def local_star_min_angle(v: int) -> float:
                tris_idx = [int(t) for t in editor.v_map.get(int(v), []) if not np.all(editor.triangles[int(t)] == -1)]
                if not tris_idx:
                    return 180.0
                mins = []
                for ti in tris_idx:
                    a, b, c = [int(x) for x in editor.triangles[int(ti)]]
                    A, B, C = triangle_angles(editor.points[a], editor.points[b], editor.points[c])
                    mins.append(float(min(A, B, C)))
                return float(min(mins)) if mins else 180.0
            # Boundary predicate
            def is_boundary_vertex(v: int) -> bool:
                # deg==1 on any incident edge implies boundary
                for ti in editor.v_map.get(int(v), []):
                    t = editor.triangles[int(ti)]
                    for i in range(3):
                        u, w = int(t[i]), int(t[(i+1)%3])
                        key = tuple(sorted((u, w)))
                        if v in (u, w):
                            inc = [tt for tt in editor.edge_map.get(key, []) if not np.all(editor.triangles[int(tt)] == -1)]
                            if len(inc) == 1:
                                return True
                return False
            # Build and sort candidates
            all_vertices = list(editor.v_map.keys())
            scored = [(v, local_star_min_angle(int(v))) for v in all_vertices]
            scored = [(int(v), ang) for v, ang in scored if ang < min_deg]
            # Prefer non-boundary unless allowed
            if not allow_boundary:
                scored = [(v, ang) for v, ang in scored if not is_boundary_vertex(int(v))]
            # Sort by ascending min angle (worst first)
            scored.sort(key=lambda x: x[1])
            removed = 0
            for v, ang in scored:
                ok_rm, msg_rm, _ = editor.remove_node_with_patch(int(v))
                log.debug('sanity remove_node v=%d (min_angle=%.3g) -> %s (%s)', v, ang, ok_rm, msg_rm)
                if ok_rm:
                    removed += 1
                    if removed >= max_rem:
                        break
            if removed:
                log.info('auto(h.simple): iter=%d sanity_remove applied=%d (min_angle<%.3g)', it, removed, min_deg)
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
        include_boundary = bool(auto_cfg.get('include_boundary_edges', False))
        avg_len = compute_avg_edge_length(editor, include_boundary=include_boundary)
        label = 'internal+boundary' if include_boundary else 'internal'
        log.info('auto: avg %s edge length=%.6g', label, avg_len)
        ops = 0
        # Debug plot options (reuse auto_cfg keys under 'plot' if available)
        plot_cfg = auto_cfg if isinstance(auto_cfg, dict) else {}
        annotate_debug = bool(plot_cfg.get('annotate_debug', False))
        annotate_dir = str(plot_cfg.get('annotate_dir', 'refine_debug'))
        if annotate_debug:
            try:
                _os.makedirs(annotate_dir, exist_ok=True)
            except Exception:
                annotate_debug = False
        # Pass 1: boundary edges only
        if include_boundary:
            b_edges = list_boundary_edges_only(editor)
            new_boundary_vertices = []
            for e in b_edges:
                u, v = int(e[0]), int(e[1])
                d = editor.points[u] - editor.points[v]
                L = float(np.sqrt(d[0]*d[0] + d[1]*d[1]))
                if L > avg_len:
                    prev_nv = len(editor.points)
                    ok, msg, _ = editor.split_edge((u, v))
                    log.debug('auto split_edge(boundary-first) %s L=%.3g -> %s (%s)', e, L, ok, msg)
                    if ok:
                        ops += 1
                        if len(editor.points) > prev_nv:
                            new_boundary_vertices.append(len(editor.points) - 1)
                        if max_edge_ops is not None and ops >= int(max_edge_ops):
                            log.info('auto: split_edge ops applied=%d (boundary-first limit)', ops)
                            if annotate_debug and new_boundary_vertices:
                                out_b = _os.path.join(annotate_dir, f"split_boundary.png")
                                plot_mesh_with_annotations(editor, outname=out_b, new_vertices=new_boundary_vertices, show_boundary_edges=True)
                            return
            if annotate_debug and new_boundary_vertices:
                out_b = _os.path.join(annotate_dir, f"split_boundary.png")
                plot_mesh_with_annotations(editor, outname=out_b, new_vertices=new_boundary_vertices, show_boundary_edges=True)
        # Pass 2: interior edges
        edges = list_refinable_edges(editor, include_boundary=False)
        new_interior_vertices = []
        for e in edges:
            u, v = int(e[0]), int(e[1])
            d = editor.points[u] - editor.points[v]
            L = float(np.sqrt(d[0]*d[0] + d[1]*d[1]))
            if L > avg_len:
                prev_nv = len(editor.points)
                ok, msg, _ = editor.split_edge((u, v))
                log.debug('auto split_edge %s L=%.3g -> %s (%s)', e, L, ok, msg)
                if ok:
                    ops += 1
                    if len(editor.points) > prev_nv:
                        new_interior_vertices.append(len(editor.points) - 1)
                    if max_edge_ops is not None and ops >= int(max_edge_ops):
                        break
        log.info('auto: split_edge ops applied=%d', ops)
        # Optional sanity removal pass after edge splits
        sanity_on = bool(auto_cfg.get('sanity_remove_bad_quality', False))
        if sanity_on:
            min_deg = float(auto_cfg.get('sanity_min_angle_deg', 20.0))
            max_rem = int(auto_cfg.get('sanity_max_removals_per_iter', 2))
            allow_boundary = bool(auto_cfg.get('sanity_allow_boundary', True))
            def local_star_min_angle(v: int) -> float:
                tris_idx = [int(t) for t in editor.v_map.get(int(v), []) if not np.all(editor.triangles[int(t)] == -1)]
                if not tris_idx:
                    return 180.0
                mins = []
                for ti in tris_idx:
                    a, b, c = [int(x) for x in editor.triangles[int(ti)]]
                    A, B, C = triangle_angles(editor.points[a], editor.points[b], editor.points[c])
                    mins.append(float(min(A, B, C)))
                return float(min(mins)) if mins else 180.0
            def is_boundary_vertex(v: int) -> bool:
                for ti in editor.v_map.get(int(v), []):
                    t = editor.triangles[int(ti)]
                    for i in range(3):
                        u, w = int(t[i]), int(t[(i+1)%3])
                        key = tuple(sorted((u, w)))
                        if v in (u, w):
                            inc = [tt for tt in editor.edge_map.get(key, []) if not np.all(editor.triangles[int(tt)] == -1)]
                            if len(inc) == 1:
                                return True
                return False
            all_vertices = list(editor.v_map.keys())
            scored = [(int(v), local_star_min_angle(int(v))) for v in all_vertices]
            scored = [(v, ang) for v, ang in scored if ang < min_deg]
            if not allow_boundary:
                scored = [(v, ang) for v, ang in scored if not is_boundary_vertex(int(v))]
            scored.sort(key=lambda x: x[1])
            removed = 0
            for v, ang in scored:
                ok_rm, msg_rm, _ = editor.remove_node_with_patch(int(v))
                log.debug('sanity remove_node v=%d (min_angle=%.3g) -> %s (%s)', v, ang, ok_rm, msg_rm)
                if ok_rm:
                    removed += 1
                    if removed >= max_rem:
                        break
            if removed:
                log.info('auto(split): sanity_remove applied=%d (min_angle<%.3g)', removed, min_deg)
        if annotate_debug and new_interior_vertices:
            out_i = _os.path.join(annotate_dir, f"split_interior.png")
            plot_mesh_with_annotations(editor, outname=out_i, new_vertices=new_interior_vertices, show_boundary_edges=True)

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

    # Smoothing passes configuration
    # Legacy behavior: scenario had 'move_to_barycenter' and 'barycenter_passes'.
    # We interpret legacy 'move_to_barycenter' as a pre-pass by default to satisfy
    # the request to apply barycenter moves first.
    pre_smooth = bool(auto_cfg.get('move_to_barycenter_first', auto_cfg.get('move_to_barycenter', False)))
    post_smooth = bool(auto_cfg.get('move_to_barycenter_post', False))
    bary_passes = max(1, int(auto_cfg.get('barycenter_passes', 1)))

    # Pre-pass: move interior vertices to barycenter before any operations
    if pre_smooth:
        total_moves_pre = 0
        for _ in range(bary_passes):
            total_moves_pre += editor.move_vertices_to_barycenter()
        log.info('auto: pre barycenter moves applied=%d (passes=%d)', total_moves_pre, bary_passes)
        # Immediately apply a removal pass to clean up worst local quality after smoothing
        pre_rm_min = float(auto_cfg.get('pre_remove_min_angle_deg', auto_cfg.get('sanity_min_angle_deg', 20.0)))
        pre_rm_max = int(auto_cfg.get('pre_remove_max', auto_cfg.get('sanity_max_removals_per_iter', 2)))
        pre_rm_allow_boundary = bool(auto_cfg.get('pre_remove_allow_boundary', auto_cfg.get('sanity_allow_boundary', True)))
        _remove_bad_quality_vertices(editor, min_deg=pre_rm_min, max_rem=pre_rm_max, allow_boundary=pre_rm_allow_boundary, prefix='pre-remove')

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
    # Always apply a post-pass smoothing to improve quality
    total_moves_post = 0
    for _ in range(bary_passes):
        total_moves_post += editor.move_vertices_to_barycenter()
    log.info('auto: post barycenter moves applied=%d (passes=%d)', total_moves_post, bary_passes)


## moved to package ops as op_move_vertices_to_barycenter


def main():
    ap = argparse.ArgumentParser(description='Apply refinement scenario (add_node, split_edge) from JSON')
    ap.add_argument('--scenario', type=str, required=True, help='Path to scenario JSON file')
    ap.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO')
    ap.add_argument('--profile', action='store_true', help='Enable cProfile and print top hotspots')
    ap.add_argument('--profile-out', type=str, default=None, help='Write raw cProfile stats to this .pstats file when --profile is set')
    ap.add_argument('--profile-top', type=int, default=25, help='How many entries to show in hotspots (default: 25)')
    ap.add_argument('--op-stats', action='store_true', help='Print per-operation stats summary at the end')
    ap.add_argument('--no-plot', action='store_true', help='Skip before/after plotting (useful for pure performance profiling)')
    args = ap.parse_args()

    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    def _run():
        with open(args.scenario, 'r') as f:
            scenario = json.load(f)

        mesh_cfg = scenario.get('mesh', {})
        plot_cfg = scenario.get('plot', {})
        out_before = plot_cfg.get('out_before', 'refine_before.png')
        out_after = plot_cfg.get('out_after', 'refine_after.png')

        editor = load_mesh_from_cfg(mesh_cfg)
        if not args.no_plot:
            plot_mesh(editor, outname=out_before)
            log.info('Wrote %s', out_before)

        # Either apply explicit ops list, or an auto refinement block
        auto_cfg = scenario.get('auto', None)
        if auto_cfg:
            # Enable virtual boundary topological mode when requested or when refining boundary edges
            try:
                if 'virtual_boundary_mode' in auto_cfg:
                    editor.virtual_boundary_mode = bool(auto_cfg.get('virtual_boundary_mode', False))
                    log.info('virtual_boundary_mode from scenario: %s', editor.virtual_boundary_mode)
                elif bool(auto_cfg.get('include_boundary_edges', False)):
                    editor.virtual_boundary_mode = True
                    log.info('virtual_boundary_mode enabled by include_boundary_edges')
            except Exception:
                pass
            # Optional toggle: enforce split quality (improvement) vs relax (refinement)
            if 'enforce_split_quality' in auto_cfg:
                val = bool(auto_cfg.get('enforce_split_quality', True))
                try:
                    editor.enforce_split_quality = val
                except Exception:
                    pass
            # Optional toggle: enable quad fast-path in remove_node
            if 'enable_remove_quad_fastpath' in auto_cfg:
                try:
                    editor.enable_remove_quad_fastpath = bool(auto_cfg.get('enable_remove_quad_fastpath', False))
                    log.info('enable_remove_quad_fastpath from scenario: %s', editor.enable_remove_quad_fastpath)
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
        if not args.no_plot:
            plot_mesh(editor, outname=out_after)
            log.info('Wrote %s', out_after)
        if args.op_stats:
            try:
                editor.print_stats(pretty=True)
            except Exception:
                # best-effort: stats are optional
                pass

    if args.profile:
        pr = _cprof.Profile()
        pr.enable()
        _run()
        pr.disable()
        s = _io.StringIO()
        _pstats.Stats(pr, stream=s).sort_stats('cumtime').print_stats(max(1, int(args.profile_top)))
        log.info('Profile (top %d by cumulative time):\n%s', int(args.profile_top), s.getvalue())
        if args.profile_out:
            try:
                pr.dump_stats(args.profile_out)
                log.info('Raw pstats written to %s', args.profile_out)
            except Exception as e:
                log.warning('Failed to write pstats to %s: %s', args.profile_out, e)
    else:
        _run()


if __name__ == '__main__':
    main()
