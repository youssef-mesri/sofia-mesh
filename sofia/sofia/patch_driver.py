"""Advanced patch/batch remeshing driver logic.

This module ports the former heavy logic from the legacy `debug_check` file
into a cleaner, parameterized implementation. It exposes a small surface:

  - PatchDriverConfig: configuration dataclass
  - pick_operation_for_patch(editor, patch, rng)
  - apply_patch_operation(editor, patch, rng, config)
  - run_patch_batch_driver(editor, config, rng, np_rng, *, logger)

It depends only on:
  * mesh_modifier2 (editor + conformity + metrics)
  * diagnostics (compact_copy, find_inverted_triangles, detect_and_dump_pockets, count_boundary_loops, compact_from_arrays)
  * visualization.plot_mesh for periodic snapshots (optional)

The acceptance policy mirrors the earlier implementation:
  * Reject if conformity fails, triangles inverted, new boundary loops created
  * Reject if local worst triangle does not improve by > EPS_IMPROVEMENT
  * Optional reject-min-angle policy (config.reject_min_angle_deg)
  * Optional pocket detection & (if enabled) auto-fill via editor.try_fill_pocket
"""
from __future__ import annotations

import csv
import math
import os
import time
from typing import Optional, Tuple, Any, Dict, List
import random

import numpy as np

from .constants import EPS_AREA, EPS_IMPROVEMENT
from .config import PatchDriverConfig, RemeshConfig
from .diagnostics import (
    compact_copy,
    find_inverted_triangles,
    detect_and_dump_pockets,
    count_boundary_loops,
    compact_from_arrays,
)
from .mesh_modifier2 import (
    check_mesh_conformity,
    triangle_angles,
    is_boundary_vertex_from_maps,
)
from .geometry import triangles_min_angles
from .patch_batching import (
    build_patches_from_metrics_strict,
    triangle_metric,
    partition_batches,
)
from .gif_capture import GifCapture
from .logging_utils import get_logger

def _local_min_angle_vec(points, triangles, tri_indices):
    if not tri_indices:
        return float('nan')
    T = np.asarray([triangles[int(t)] for t in tri_indices], dtype=int)
    if T.size == 0:
        return float('nan')
    mask = ~np.all(T == -1, axis=1)
    T = T[mask]
    if T.size == 0:
        return float('nan')
    mins = triangles_min_angles(points, T)
    mins = mins[np.isfinite(mins)]
    return float(mins.min()) if mins.size else float('nan')


def pick_operation_for_patch(editor, patch, rng: random.Random):
    """Pick an operation (add | remove | flip | split) and parameter.

    Policy (biased to refinement): try add (60%), remove (50%), edge ops else fallback add.
    """
    tri_set = set(patch.get('tris', []))
    vert_set = set(patch.get('verts', []))

    interior_edges = [e for e, tset in editor.edge_map.items() if set(tset).issubset(tri_set)]
    interior_edges = [e for e in interior_edges if len(editor.edge_map.get(e, [])) >= 1]
    interior_vs = [v for v in vert_set if not is_boundary_vertex_from_maps(int(v), editor.edge_map)]
    interior_vs = [v for v in interior_vs if len(editor.v_map.get(v, [])) >= 3]

    if tri_set and rng.random() < 0.6:
        tri_idx = rng.choice(list(tri_set))
        tri = editor.triangles[int(tri_idx)]
        if not np.all(np.array(tri) == -1):
            pts = editor.points[[int(tri[0]), int(tri[1]), int(tri[2])]]
            centroid = pts.mean(axis=0)
            return 'add', (tri_idx, centroid)

    if interior_vs and rng.random() < 0.5:
        return 'remove', int(rng.choice(interior_vs))

    if interior_edges:
        e = rng.choice(interior_edges)
        r = rng.random()
        if r < 0.5:
            return 'flip', e
        else:
            return 'split', e

    if tri_set:
        tri_idx = next(iter(tri_set))
        tri = editor.triangles[int(tri_idx)]
        if not np.all(np.array(tri) == -1):
            pts = editor.points[[int(tri[0]), int(tri[1]), int(tri[2])]]
            centroid = pts.mean(axis=0)
            return 'add', (tri_idx, centroid)

    return None, None


def _local_min_angle(editor, tris_idx):
    return _local_min_angle_vec(editor.points, editor.triangles, [int(t) for t in tris_idx])


def apply_patch_operation(editor, patch, rng: random.Random, config: PatchDriverConfig, logger):
    """Attempt a single patch operation respecting acceptance policy.

    Returns tuple (ok, info, op, op_param, local_before, local_after, tri_count_before, tri_count_after, rejected_tris)
    """
    tri_indices = [int(t) for t in patch.get('tris', [])]
    tri_count_before = sum(1 for t in tri_indices if not np.all(np.array(editor.triangles[int(t)]) == -1))
    local_before = _local_min_angle(editor, tri_indices) if tri_indices else float('nan')

    import copy
    pts_snap = editor.points.copy(); tris_snap = editor.triangles.copy()
    edge_map_snap = copy.deepcopy(editor.edge_map)
    v_map_snap = copy.deepcopy(editor.v_map)

    def rollback():
        editor.points = pts_snap
        editor.triangles = tris_snap
        editor.edge_map = copy.deepcopy(edge_map_snap)
        editor.v_map = copy.deepcopy(v_map_snap)
        editor._update_maps()

    op, param = pick_operation_for_patch(editor, patch, rng)
    if op is None:
        return False, 'no-op', None, None, local_before, None, tri_count_before, None, []

    if op == 'flip':
        ok, msg, _ = editor.flip_edge(param)
    elif op == 'split':
        ok, msg, _ = editor.split_edge(param)
    elif op == 'remove':
        try:
            ok, msg, _ = editor.remove_node_with_patch(param, force_strict=True)
        except TypeError:
            ok, msg, _ = editor.remove_node_with_patch(param)
    elif op == 'add':
        tri_idx, centroid = param
        ok, msg, _ = editor.add_node(centroid, tri_idx=tri_idx)
    else:
        ok = False; msg = 'unknown-op'

    if not ok:
        rollback()
        return False, msg, op, param, local_before, None, tri_count_before, None, []

    # Validate global conformity & inversion (allow_marked True first, then compact strict)
    ok_conf, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    pts_c, tris_c, mapping_c, active_idx_c = compact_copy(editor)
    inverted = find_inverted_triangles(pts_c, tris_c, eps=config.min_triangle_area)
    if not ok_conf or inverted:
        rollback()
        return False, 'conf-or-inverted', op, param, local_before, None, tri_count_before, None, [ai for ai,_ in inverted]

    tri_count_after = sum(1 for t in tri_indices if not np.all(np.array(editor.triangles[int(t)]) == -1))
    # recalc local min across patch + newly added tris (approx by scanning patch indices + tail)
    local_after = _local_min_angle(editor, tri_indices)

    # strict compact validation
    pts_comp, tris_comp, mapping_comp, active_idx_comp = compact_copy(editor)
    ok_comp, msgs_comp = check_mesh_conformity(pts_comp, tris_comp, allow_marked=False)
    inv_comp = find_inverted_triangles(pts_comp, tris_comp, eps=config.min_triangle_area)
    pre_pts, pre_tris, _, _ = compact_from_arrays(pts_snap, tris_snap)
    pre_loops = count_boundary_loops(pre_pts, pre_tris) if pre_tris is not None else None
    post_loops = count_boundary_loops(pts_comp, tris_comp)
    if (not ok_comp) or inv_comp or (config.reject_new_loops and (pre_loops is not None and post_loops > pre_loops)):
        rollback()
        return False, 'compact-invalid', op, param, local_before, None, tri_count_before, None, [active_idx_comp[i] for i,_ in inv_comp if i < len(active_idx_comp)]

    # optional crossing-edges rejection using simulation (mirrors greedy strict check)
    if getattr(config, 'reject_crossings', False):
        try:
            from .conformity import simulate_compaction_and_check
            ok_sim, msgs_sim, _ = simulate_compaction_and_check(editor.points, editor.triangles, reject_crossing_edges=True)
            if not ok_sim:
                rollback()
                return False, 'crossing-edges', op, param, local_before, local_after, tri_count_before, None, []
        except Exception:
            # If simulation fails unexpectedly, be conservative and reject
            rollback()
            return False, 'crossing-sim-error', op, param, local_before, local_after, tri_count_before, None, []

    # reject-min-angle policy
    if config.reject_min_angle_deg is not None and (not math.isnan(local_after)) and local_after < config.reject_min_angle_deg:
        rollback()
        return False, 'reject-min-angle', op, param, local_before, local_after, tri_count_before, None, []

    # improvement policy
    if not (math.isnan(local_before) or math.isnan(local_after)) and local_after <= local_before + EPS_IMPROVEMENT:
        rollback()
        return False, 'no-improve', op, param, local_before, local_after, tri_count_before, None, []

    # pocket detection (fail on empty pocket unless autofill allowed & succeeds)
    pockets, mapping = detect_and_dump_pockets(editor, op_desc=f"{op} param={param}")
    if pockets:
        if config.auto_fill_pockets:
            # attempt fill empty pockets using editor.try_fill_pocket
            filled_all = True
            for pk in pockets:
                if pk.get('inside_pts'):  # not empty
                    continue
                try:
                    # mapping: global_old->local; invert
                    local_to_global = {v_new: v_old for v_old, v_new in mapping.items()}
                    verts_global = [local_to_global[int(v)] for v in pk['verts']]
                    # Prefer config-provided overrides; otherwise fall back to current run's thresholds
                    use_min_area = (
                        config.autofill_min_triangle_area
                        if config.autofill_min_triangle_area is not None
                        else config.min_triangle_area if config.min_triangle_area is not None else EPS_AREA
                    )
                    use_reject_min = (
                        config.autofill_reject_min_angle_deg
                        if config.autofill_reject_min_angle_deg is not None
                        else config.reject_min_angle_deg
                    )
                    res = editor.try_fill_pocket(verts_global, min_tri_area=use_min_area, reject_min_angle_deg=use_reject_min)
                    if isinstance(res, tuple):
                        ok_fill = bool(res[0])
                    else:
                        ok_fill = bool(res)
                    if not ok_fill:
                        filled_all = False
                        break
                except Exception:
                    filled_all = False
                    break
            if not filled_all:
                rollback()
                return False, 'empty-pocket', op, param, local_before, local_after, tri_count_before, None, []
        else:
            # just fail if any empty pocket present
            for pk in pockets:
                if not pk.get('inside_pts'):
                    rollback()
                    return False, 'empty-pocket', op, param, local_before, local_after, tri_count_before, None, []

    return True, 'ok', op, param, local_before, local_after, tri_count_before, tri_count_after, []


def run_patch_batch_driver(editor, config: PatchDriverConfig, rng: random.Random, np_rng, *, logger, greedy_remesh=None, plot_mesh=None, remesh_config: RemeshConfig=None, rejected_log_path: Optional[str]=None):
    """Execute patch/batch iterative improvement until threshold or stagnation.

    Returns dict with summary stats.
    """
    # Determine angle display conversion
    base_logger = get_logger('sofia.driver')  # compact summaries on root driver logger, like greedy
    angle_factor = 1.0 if config.angle_unit == 'deg' else math.pi / 180.0
    def disp_angle(deg_val):
        try:
            return float(deg_val) * angle_factor
        except Exception:
            return deg_val

    # Optional fraction-based min triangle area
    if config.min_triangle_area_fraction is not None:
        pts_c0, tris_c0, _, _ = compact_copy(editor)
        areas0 = []
        for t in tris_c0:
            if np.any(np.array(t) < 0):
                continue
            p0 = pts_c0[int(t[0])]; p1 = pts_c0[int(t[1])]; p2 = pts_c0[int(t[2])]
            areas0.append(abs(0.5 * np.cross(p1 - p0, p2 - p0)))
        if areas0:
            config.min_triangle_area = float(config.min_triangle_area_fraction) * float(np.mean(areas0))
            logger.info('Setting min_triangle_area via fraction %.3f -> %.6g', config.min_triangle_area_fraction, config.min_triangle_area)

    # main loop
    iter_no = 0
    stagnation = 0
    max_stagnation = 50
    csv_writer = None; csv_file = None
    rejected_ops: List[Dict[str, Any]] = []
    if config.log_dir:
        os.makedirs(config.log_dir, exist_ok=True)
        csv_path = os.path.join(config.log_dir, f'patch_log_{int(time.time())}.csv')
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['iter','batch','patch_id','seed_tri','op','param','attempt','result','g_before','g_after','time_s','local_before','local_after'])

    logger.info('Patch driver start: npts=%d ntri=%d min_angle=%.4f %s', len(editor.points), len(editor.triangles), disp_angle(editor.global_min_angle()), config.angle_unit)

    gif_cap = GifCapture(config.gif_capture, config.gif_dir, config.gif_out, fps=config.gif_fps, max_frames=1500, logger=logger)
    def _save_frame(tag):
        if not config.gif_capture:
            return
        def _plot(ed, path):
            if plot_mesh is not None:
                plot_mesh(ed, outname=path)
            else:
                from visualization import plot_mesh as _pm
                _pm(ed, outname=path)
        gif_cap.save(editor, tag, _plot)
    if config.gif_capture:
        _save_frame('start')
    while iter_no < config.max_iterations:
        iter_start = time.time()
        g_before_iter = editor.global_min_angle()
        iter_applied = 0
        iter_attempts = 0
        patches_count = 0
        batches_count = 0
        if config.use_greedy_remesh and greedy_remesh is not None:
            logger.info('Greedy pass at iter %d', iter_no)
            if remesh_config is not None:
                # adapt unified config's greedy section, overriding passes
                g_cfg = remesh_config.greedy
                g_cfg.max_vertex_passes = config.greedy_vertex_passes
                g_cfg.max_edge_passes = config.greedy_edge_passes
                greedy_remesh(editor, config=g_cfg)
            else:
                greedy_remesh(editor, max_vertex_passes=config.greedy_vertex_passes, max_edge_passes=config.greedy_edge_passes, verbose=False)

        # Build list of triangles below threshold (vectorized)
        tris_arr = np.asarray(editor.triangles, dtype=int)
        if tris_arr.size == 0:
            bad_tris = []
        else:
            mask_act = ~np.all(tris_arr == -1, axis=1)
            idx_all = np.arange(len(tris_arr), dtype=int)
            idx_active = idx_all[mask_act]
            active_tris = tris_arr[mask_act]
            if active_tris.size == 0:
                bad_tris = []
            else:
                mins = triangles_min_angles(editor.points, active_tris)
                bad_mask = mins < float(config.threshold)
                bad_tris = idx_active[bad_mask].tolist()
        if not bad_tris:
            logger.info('All triangles above threshold after %d iterations', iter_no)
            break
        scored = [(triangle_metric(editor, t), t) for t in bad_tris]
        scored.sort(reverse=True)
        # Build patches (node-centered strict)
        patches = build_patches_from_metrics_strict(editor, node_top_k=config.top_k, edge_top_k=0, radius=config.patch_radius, disjoint_on=config.disjoint_on, allow_overlap=config.allow_overlap)
        patches_count = len(patches)
        if not patches:
            logger.info('No patches constructed; stopping')
            # Per-iteration compact summary before stopping
            g_after_iter = editor.global_min_angle()
            base_logger.info('iter %d: patches=%d batches=%d applied=%d attempts=%d stagnation=%d min_angle=%.3f->%.3f %s time=%.3fs',
                             iter_no, 0, 0, iter_applied, iter_attempts, stagnation, disp_angle(g_before_iter), disp_angle(g_after_iter), config.angle_unit, time.time() - iter_start)
            break
        batches = partition_batches(patches)
        pid_to_patch = {p['id']: p for p in patches}
        batches_count = len(batches)
        logger.info('Iter %d: %d patches in %d batches (bad tris %d)', iter_no, patches_count, batches_count, len(bad_tris))

        any_success = False
        for b_idx, batch in enumerate(batches):
            batch_start = time.time(); batch_success = 0
            for pid in batch:
                patch = pid_to_patch.get(pid)
                if not patch:
                    continue
                for attempt in range(config.batch_attempts):
                    iter_attempts += 1
                    g_before = editor.global_min_angle(); t0 = time.time()
                    ok, info, op, op_param, local_before, local_after, tri_count_before, tri_count_after, rejected = apply_patch_operation(editor, patch, rng, config, logger)
                    t1 = time.time(); g_after = editor.global_min_angle()
                    if ok:
                        any_success = True; batch_success += 1; iter_applied += 1
                        logger.debug('Patch %s %s op=%s %s g: %.3f->%.3f', pid, info, op, op_param, g_before, g_after)
                        _save_frame(f'iter{iter_no}_pid{pid}_{op}')
                        break  # stop attempts for this patch
                    else:
                        logger.debug('Patch %s attempt %d failed (%s)', pid, attempt, info)
                        # Collect a compact rejection record for diagnostics
                        try:
                            rejected_ops.append({
                                'iter': int(iter_no),
                                'batch': int(b_idx),
                                'patch_id': int(pid),
                                'attempt': int(attempt),
                                'op': str(op),
                                'reason': str(info),
                                'g_before': float(g_before),
                                'g_after': float(g_after),
                                'local_before': None if local_before is None else float(local_before),
                                'local_after': None if local_after is None else float(local_after),
                            })
                        except Exception:
                            pass
                    if csv_writer:
                        seed_tri = next(iter(patch['tris'])) if patch.get('tris') else ''
                        csv_writer.writerow([iter_no, b_idx, pid, seed_tri, op, op_param, attempt, info, f'{disp_angle(g_before):.6f}', f'{disp_angle(g_after):.6f}', f'{t1-t0:.4f}', local_before if local_before is not None else '', local_after if local_after is not None else ''])
            batch_time = time.time() - batch_start
            logger.info('Batch %d applied=%d time=%.3fs min_angle=%.3f', b_idx+1, batch_success, batch_time, editor.global_min_angle())
            if editor.global_min_angle() >= config.threshold:
                logger.info('Threshold reached mid-iteration')
                break
        stagnation = stagnation + 1 if not any_success else 0
        # Per-iteration compact summary
        g_after_iter = editor.global_min_angle()
        base_logger.info('iter %d: patches=%d batches=%d applied=%d attempts=%d stagnation=%d min_angle=%.3f->%.3f %s time=%.3fs',
                         iter_no, patches_count, batches_count, iter_applied, iter_attempts, stagnation, disp_angle(g_before_iter), disp_angle(g_after_iter), config.angle_unit, time.time() - iter_start)
        if (iter_no % config.plot_every == 0) and plot_mesh is not None:
            outname = f"{config.out_prefix}_iter{iter_no}.png"
            plot_mesh(editor, outname=outname)
            _save_frame(f'iter{iter_no}_snapshot')
        if stagnation >= 50:
            logger.info('Stagnation (%d) reached; stopping', stagnation)
            break
        iter_no += 1

    # Compact only if there are tombstones to clean up
    try:
        if getattr(editor, 'has_tombstones', None) and editor.has_tombstones():
            editor.compact_triangle_indices()
    except Exception:
        # Be resilient; skip compaction if helper unavailable
        pass
    # Optionally write a compact JSON log of rejected operations
    if rejected_log_path and rejected_ops:
        try:
            import json, time as _time
            # Summarize counts per reason to keep file small
            counts: Dict[str, int] = {}
            for r in rejected_ops:
                counts[r['reason']] = counts.get(r['reason'], 0) + 1
            # Cap detailed records to first 200 entries for size
            detailed = rejected_ops[:200]
            payload = {
                'timestamp': _time.time(),
                'total_rejections': len(rejected_ops),
                'counts_by_reason': counts,
                'samples': detailed,
            }
            with open(rejected_log_path, 'w') as f:
                json.dump(payload, f, indent=2)
            logger.info('Wrote rejected ops log to %s (total=%d)', rejected_log_path, len(rejected_ops))
        except Exception as e:
            logger.warning('Could not write rejected ops log %s: %s', rejected_log_path, e)
    if plot_mesh is not None:
        plot_mesh(editor, outname=f"{config.out_prefix}_final.png")
    _save_frame('final')
    gif_cap.finalize()
    if csv_writer:
        csv_file.close()
    return {
        'iterations': iter_no,
        'final_min_angle': editor.global_min_angle(),
        'npts': len(editor.points),
        'ntri': len(editor.triangles),
    }


__all__ = [
    'PatchDriverConfig',
    'pick_operation_for_patch',
    'apply_patch_operation',
    'run_patch_batch_driver',
]
