#!/usr/bin/env python3
"""Remeshing driver / CLI tool.

Iteratively improves a random (or provided) triangulated mesh by applying
local patch-based operations (flip, split, remove, add) guided by a quality
metric (triangle min-angle) while enforcing a series of geometric / topological
invariants (conformity, positive area, pocket avoidance, no new holes, optional
min-angle rejection, etc.).

This file was previously named `debug_check.py` but has been renamed to
`remesh_driver.py` to better reflect its role as an executable driver rather
than an internal debug helper. The legacy module name still exists as a
deprecated shim for backward compatibility.

Key Responsibilities (high-level):
  * CLI argument parsing and run orchestration (`main`).
  * Patch construction and batching (delegates to `patch_batching`).
  * Operation selection and guarded application (`apply_patch_operation`).
  * Greedy vertex/edge passes (`greedy_remesh`).
  * Mesh compaction utilities for diagnostics (`compact_copy`).
  * Visual / CSV diagnostics (plotting, per-iteration logging).

Suggested Future Modularization (TODO):
  1. Extract plotting routines into `visualization.py` (plot_mesh, plot_patches_map).
  2. Move low-level geometric helpers (count_boundary_loops, find_inverted_triangles, pocket detection) into a `diagnostics.py`.
  3. Isolate operation acceptance policy (improvement, rejection thresholds, pocket autofill) into a strategy object so alternative policies are easy to test.
  4. Replace extensive in-function rollback logic with a context manager that snapshots editor state and auto-rolls back on failure.
  5. Provide a slimmer public API surface (export only driver entrypoints, keep implementation details private with leading underscore).

The code below is intentionally copied verbatim from the former module except
for docstring / logger naming tweaks and an added `__all__` to define the
export surface.
"""

# NOTE: The original implementation lives below without substantive changes
# except for renaming the logger to 'remesh_driver'.

import argparse
import logging
import csv
import os
import time
import random
import warnings
import json
import numpy as np
import math
from .constants import EPS_AREA, EPS_IMPROVEMENT
from .gif_capture import GifCapture
from .config import RemeshConfig, GreedyConfig, PatchDriverConfig

# Local modularized helpers
from .diagnostics import (
    compact_copy,
    count_boundary_loops,
    compact_from_arrays,
    find_inverted_triangles,
    detect_and_dump_pockets,
)
from .visualization import plot_mesh

from .mesh_modifier2 import (
    build_random_delaunay,
    PatchBasedMeshEditor,
    check_mesh_conformity,
    triangle_angles,
    is_boundary_vertex_from_maps,
    build_edge_to_tri_map,
)
from .patch_batching import (
    build_patches_from_triangles,
    build_patches_from_metrics_strict,
    partition_batches,
    patch_boundary_loops,
    triangle_metric,
)
from .patch_driver import (
    PatchDriverConfig,
    run_patch_batch_driver,
)
from .logging_utils import get_logger, configure_logging

logger = get_logger('sofia.driver')

# (All remaining contents copied from former debug_check.py)

# Legacy globals (deprecated). Kept for API skin but not used by drivers.
MIN_TRI_AREA = EPS_AREA
REJECT_MIN_ANGLE_DEG = None
AUTO_FILL_POCKETS = False
ALLOW_FLIPS = True
# Back-compat symbol; hook is now read from per-run context, setting this variable has no effect.
COMPACT_CHECK_HOOK = None
from .run_context import get as _ctx_get, set_context as _ctx_set

def tri_min_angle(points, tri):
    p0, p1, p2 = points[int(tri[0])], points[int(tri[1])], points[int(tri[2])]
    angs = triangle_angles(p0, p1, p2)
    return min(angs)

def plot_mesh(editor, outname="mesh.png", annotate_bad=None, annotate_rejected=None, **kwargs):  # backward compat re-export
    from .visualization import plot_mesh as _pm
    return _pm(
        editor,
        outname=outname,
        annotate_bad=annotate_bad,
        annotate_rejected=annotate_rejected,
        **kwargs,
    )

def build_patch_adjacency(patches, editor):
    # ...existing code...
    tri_to_pids = {}
    for p in patches:
        pid = p['id']
        for t in p.get('tris', []):
            tri_to_pids.setdefault(int(t), set()).add(pid)
    pid_neighbors = {p['id']: set() for p in patches}
    edge_map = build_edge_to_tri_map(editor.triangles)
    for edge, tri_set in edge_map.items():
        owners = set()
        for t in tri_set:
            owners.update(tri_to_pids.get(int(t), set()))
        owners = list(owners)
        for i in range(len(owners)):
            for j in range(i+1, len(owners)):
                pid_neighbors[owners[i]].add(owners[j])
                pid_neighbors[owners[j]].add(owners[i])
    return pid_neighbors


def greedy_remesh(editor, max_vertex_passes=1, max_edge_passes=1, verbose=False,
                  strict=False, reject_crossings=False, reject_new_loops=False,
                  annotate_failures=False, force_pocket_fill=False,
                  rejected_log_path=None,
                  gif_capture=False, gif_dir='greedy_frames', gif_out='greedy_run.gif', gif_fps=4,
                  *, config: GreedyConfig = None):
    """Greedy remeshing policy (in-place).

    Loop 1: iterate interior vertices; attempt barycenter move, high-degree removal (>6), or centroid add
    Loop 2: iterate interior edges; attempt split on very obtuse opposite angles then flip if improvement.
    """
    def local_min_angle_for_tris(pts, tris, tri_indices):
        # Vectorized: gather active triangles and compute min-angles in batch
        idx = [int(t) for t in tri_indices]
        if not idx:
            return float('nan')
        T = np.asarray([tris[i] for i in idx], dtype=int)
        if T.size == 0:
            return float('nan')
        mask = ~np.all(T == -1, axis=1)
        T = T[mask]
        if T.size == 0:
            return float('nan')
        from .geometry import triangles_min_angles
        mins = triangles_min_angles(pts, T)
        mins = mins[np.isfinite(mins)]
        return float(mins.min()) if mins.size else float('nan')

    # Track diagnostics if requested
    rejected_ops = []  # list of dicts with keys phase, op, reason

    # GIF capture via helper
    # Prefer a passed GreedyConfig if provided (unified configuration pathway)
    if config is not None:
        max_vertex_passes = config.max_vertex_passes
        max_edge_passes = config.max_edge_passes
        strict = config.strict
        reject_crossings = config.reject_crossings
        reject_new_loops = config.reject_new_loops
        force_pocket_fill = config.force_pocket_fill
        gif_capture = config.gif_capture
        gif_dir = config.gif_dir
        gif_out = config.gif_out
        gif_fps = config.gif_fps
        verbose = config.verbose or verbose
    else:
        # Legacy positional invocation path (encourage unified config usage)
        warnings.warn(
            "greedy_remesh positional/keyword arguments without `config=` are deprecated and will be removed in a future release; "
            "construct a GreedyConfig and pass greedy_remesh(..., config=cfg) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    gif_cap = GifCapture(gif_capture, gif_dir, gif_out, fps=gif_fps, max_frames=800, logger=logger)
    def _save_frame(tag):
        gif_cap.save(editor, tag, lambda ed, path: plot_mesh(ed, outname=path))
    if gif_capture:
        _save_frame('start')

    def _snapshot():
        import copy
        return (editor.points.copy(), editor.triangles.copy(), copy.deepcopy(editor.edge_map), copy.deepcopy(editor.v_map))

    def _restore(snap):
        # Restore arrays and cached maps without rebuilding globally
        pts, tris, e_map, v_map = snap
        editor.points = pts
        editor.triangles = tris
        editor.edge_map = e_map
        editor.v_map = v_map

    # Driver-level cooldown state for crossing simulation
    _strict_cd = 0
    def _strict_check(pre_loops=None):
        compact_pts, compact_tris, _, _ = compact_copy(editor)
        ok_conf, msgs = check_mesh_conformity(compact_pts, compact_tris, allow_marked=False)
        if not ok_conf:
            return False, 'conf', msgs
        # pocket / boundary loop detection
        loops_now = count_boundary_loops(compact_pts, compact_tris)
        if reject_new_loops and pre_loops is not None and loops_now > pre_loops:
            return False, 'loops', loops_now
        # crossing edges (simulate)
        if reject_crossings:
            from .conformity import simulate_compaction_and_check
            # Amortize strict crossing simulation if configured
            do_check = True
            try:
                cd = int(getattr(config, 'strict_check_cooldown', 0)) if config else 0
            except Exception:
                cd = 0
            nonlocal _strict_cd
            if cd > 0:
                if _strict_cd > 0:
                    do_check = False
                    _strict_cd -= 1
                else:
                    _strict_cd = cd
            if not do_check:
                return True, 'ok', None
            ok_sim, msgs_sim = simulate_compaction_and_check(editor.points, editor.triangles, reject_crossing_edges=True)
            if not ok_sim:
                for m in msgs_sim:
                    if 'crossing edges' in m:
                        return False, 'cross', m
        return True, 'ok', None

    # Pre-loop loop count (if needed)
    pre_compact_pts, pre_compact_tris, _, _ = compact_copy(editor)
    base_loops = count_boundary_loops(pre_compact_pts, pre_compact_tris) if reject_new_loops else None

    # Vertex-centric passes
    for vp in range(max_vertex_passes):
        changed = False
        v_moved = v_removed = v_added = 0
        pass_rejected = 0
        # Precompute boundary flags and degrees once per vertex pass
        npts = len(editor.points)
        is_boundary = np.zeros(npts, dtype=bool)
        for (a, b), ts in editor.edge_map.items():
            if len(ts) == 1:
                if a < npts: is_boundary[a] = True
                if b < npts: is_boundary[b] = True
        degree = np.zeros(npts, dtype=np.int32)
        for v, inc in editor.v_map.items():
            if v < npts:
                degree[v] = len(inc)
        interior_vertices = np.where(~is_boundary & (degree >= 3))[0].tolist()
        # mean area / edge length for refinement heuristics (vectorized)
        try:
            compact_pts, compact_tris, _, _ = compact_copy(editor)
            if compact_tris.size:
                from .geometry import triangles_signed_areas
                signed = triangles_signed_areas(compact_pts, compact_tris)
                areas = np.abs(signed)
                mean_area = float(np.mean(areas)) if areas.size else 0.0
                p0 = compact_pts[compact_tris[:,0]]; p1 = compact_pts[compact_tris[:,1]]; p2 = compact_pts[compact_tris[:,2]]
                e01 = np.linalg.norm(p0 - p1, axis=1)
                e12 = np.linalg.norm(p1 - p2, axis=1)
                e20 = np.linalg.norm(p2 - p0, axis=1)
                edge_lengths = np.concatenate([e01, e12, e20])
                mean_edge = float(np.mean(edge_lengths)) if edge_lengths.size else 0.0
            else:
                mean_area = 0.0; mean_edge = 0.0
        except Exception:
            mean_area = 0.0; mean_edge = 0.0
        for v in interior_vertices:
            tri_indices = list(editor.v_map.get(int(v), ()))
            if not tri_indices:
                continue
            pre_min = local_min_angle_for_tris(editor.points, editor.triangles, tri_indices)
            # Move to barycenter
            coords = np.vstack([editor.points[int(x)] for t in tri_indices for x in editor.triangles[int(t)] if x != -1])
            if coords.size:
                bary = np.mean(coords.reshape(-1, 2), axis=0)
                snap = _snapshot()
                pts_candidate = editor.points.copy(); pts_candidate[int(v)] = bary
                active_tris = [tuple(t) for t in editor.triangles if not np.all(np.array(t) == -1)]
                try:
                    ok_conf, _ = check_mesh_conformity(pts_candidate, np.array(active_tris, dtype=int), allow_marked=False)
                except Exception:
                    ok_conf = False
                if ok_conf:
                    post_min = local_min_angle_for_tris(pts_candidate, editor.triangles, tri_indices)
                    if post_min > pre_min + EPS_IMPROVEMENT:
                        editor.points = pts_candidate
                        if strict:
                            ok_s, reason, extra = _strict_check(base_loops)
                            if not ok_s:
                                if verbose:
                                    logger.info('greedy_remesh(str): revert move v=%d reason=%s', v, reason)
                                rejected_ops.append({'phase':'vertex','op':'move','reason':reason})
                                _restore(snap)
                                continue
                        logger.debug('greedy_remesh: moved vertex %d %.4f->%.4f', v, pre_min, post_min)
                        _save_frame(f'move_v{v}')
                        v_moved += 1
                        changed = True
                        continue
            # Remove high-degree vertex
            degree = len(editor.v_map.get(int(v), []))
            if degree > 6:
                snap = _snapshot()
                ok, msg, info = editor.remove_node_with_patch(int(v), force_strict=False)
                if ok:
                    if strict:
                        ok_s, reason, extra = _strict_check(base_loops)
                        if not ok_s:
                            if verbose:
                                logger.info('greedy_remesh(str): revert remove v=%d reason=%s', v, reason)
                            rejected_ops.append({'phase':'vertex','op':'remove','reason':reason})
                            pass_rejected += 1
                            _restore(snap)
                            continue
                    logger.debug('greedy_remesh: removed vertex %d degree=%d', v, degree)
                    _save_frame(f'remove_v{v}')
                    v_removed += 1
                    changed = True
                    continue
                else:
                    # candidate removal failed (quality, conformity, or compaction preflight)
                    pass_rejected += 1
            # Add node in worst triangle (centroid) based on area/edge heuristics
            # Batch-evaluate candidate triangle min-angles and select the worst (smallest)
            active_pairs = [(int(ti), editor.triangles[int(ti)]) for ti in tri_indices if int(ti) < len(editor.triangles)]
            active_pairs = [(i, tri) for (i, tri) in active_pairs if not np.all(np.asarray(tri) == -1)]
            worst_tri = None; worst_ang = 180.0
            tri_min_map = {}
            if active_pairs:
                try:
                    from .geometry import triangles_min_angles
                    T = np.asarray([tri for _, tri in active_pairs], dtype=int)
                    mins = triangles_min_angles(editor.points, T)
                    for (i, _), m in zip(active_pairs, mins):
                        if np.isfinite(m):
                            tri_min_map[i] = float(m)
                    if tri_min_map:
                        worst_tri, worst_ang = min(tri_min_map.items(), key=lambda kv: kv[1])
                except Exception:
                    pass
            if worst_tri is not None:
                pts_tri = editor.points[editor.triangles[worst_tri]]
                centroid = np.mean(pts_tri, axis=0)
                try:
                    a0,a1,a2 = pts_tri[0], pts_tri[1], pts_tri[2]
                    tri_area = abs(np.cross(a1 - a0, a2 - a0)) * 0.5
                    e0 = np.linalg.norm(a0 - a1); e1 = np.linalg.norm(a1 - a2); e2 = np.linalg.norm(a2 - a0)
                    longest_edge = max(e0, e1, e2)
                except Exception:
                    tri_area = 0.0; longest_edge = 0.0
                tri_min_ang = tri_min_map.get(int(worst_tri), float('nan'))
                refine_by_area = (mean_area > 0 and tri_area > 1.5 * mean_area and not np.isnan(tri_min_ang) and tri_min_ang > 15.0)
                refine_by_edge = (mean_edge > 0 and longest_edge > 1.5 * mean_edge and not np.isnan(tri_min_ang) and tri_min_ang > 10.0)
                if refine_by_area or refine_by_edge:
                    snap = _snapshot()
                    ok, msg, info = editor.add_node(centroid, tri_idx=worst_tri)
                    if ok:
                        if strict:
                            ok_s, reason, extra = _strict_check(base_loops)
                            if not ok_s:
                                if verbose:
                                    logger.info('greedy_remesh(str): revert add tri=%d reason=%s', worst_tri, reason)
                                rejected_ops.append({'phase':'vertex','op':'add','reason':reason})
                                pass_rejected += 1
                                _restore(snap)
                                continue
                        logger.debug('greedy_remesh: added node tri=%d area=%.3g edge=%.3g', worst_tri, tri_area, longest_edge)
                        _save_frame(f'add_tri{worst_tri}')
                        v_added += 1
                        changed = True
                        continue
                    else:
                        # add attempt failed
                        pass_rejected += 1
        # Per-pass summary (vertex)
        try:
            cur_min = editor.global_min_angle()
        except Exception:
            cur_min = float('nan')
        logger.info('vertex pass %d: moved=%d removed=%d added=%d rejected=%d changed=%s min_angle=%.3f deg',
                    vp, v_moved, v_removed, v_added, pass_rejected, changed, cur_min)
        if not changed:
            break
        hook = _ctx_get('compact_check_hook')
        if hook is not None:
            try:
                hook(editor, pass_type='vertex', pass_idx=vp)
            except Exception:
                pass
        # Optional end-of-pass compaction (vertex pass) - only if there were changes and tombstones exist
        if changed and (config is not None and getattr(config, 'compact_end_of_pass', False)):
            if getattr(editor, 'has_tombstones', None) and editor.has_tombstones():
                try:
                    editor._maybe_compact(force=True)
                except Exception:
                    editor.compact_triangle_indices()

    # Edge-centric passes
    for ep in range(max_edge_passes):
        changed = False
        e_flipped = e_split = 0
        pass_rejected = 0
        interior_edges = [e for e, s in editor.edge_map.items() if len(s) == 2]
        for e in interior_edges:
            tris_idx = sorted(list(editor.edge_map.get(e, [])))
            if len(tris_idx) != 2:
                continue
            pre_min = local_min_angle_for_tris(editor.points, editor.triangles, tris_idx)
            # Opposite angles
            try:
                tri0 = editor.triangles[int(tris_idx[0])]; tri1 = editor.triangles[int(tris_idx[1])]
                opp0 = [v for v in tri0 if v not in e][0]; opp1 = [v for v in tri1 if v not in e][0]
                angs0 = triangle_angles(editor.points[int(tri0[0])], editor.points[int(tri0[1])], editor.points[int(tri0[2])])
                angs1 = triangle_angles(editor.points[int(tri1[0])], editor.points[int(tri1[1])], editor.points[int(tri1[2])])
                idx0 = int(list(tri0).index(opp0)); idx1 = int(list(tri1).index(opp1))
                opp_angle0 = angs0[idx0]; opp_angle1 = angs1[idx1]
            except Exception:
                opp_angle0 = opp_angle1 = 0.0
            # Split heuristics
            if opp_angle0 > 120.0 or opp_angle1 > 120.0:
                snap = _snapshot()
                ok_split, msg_split, info_split = editor.split_edge(e)
                if ok_split:
                    active_tris = [tuple(t) for t in editor.triangles if not np.all(np.array(t) == -1)]
                    try:
                        ok_conf, _ = check_mesh_conformity(editor.points, np.array(active_tris, dtype=int), allow_marked=False)
                    except Exception:
                        ok_conf = False
                    if ok_conf:
                        post_min = local_min_angle_for_tris(editor.points, editor.triangles, tris_idx)
                        if post_min > pre_min + EPS_IMPROVEMENT:
                            if strict:
                                ok_s, reason, extra = _strict_check(base_loops)
                                if not ok_s:
                                    if verbose:
                                        logger.info('greedy_remesh(str): revert split e=%s reason=%s', e, reason)
                                    rejected_ops.append({'phase':'edge','op':'split','reason':reason})
                                    pass_rejected += 1
                                    _restore(snap)
                                    continue
                            logger.debug('greedy_remesh: split edge %s %.3f->%.3f', e, pre_min, post_min)
                            _save_frame(f'split_{e[0]}_{e[1]}')
                            e_split += 1
                            changed = True
                            continue
                        else:
                            # no improvement; revert and count rejection
                            pass_rejected += 1
                _restore(snap)
                if not ok_split or not ok_conf:
                    # split attempt failed or violated conformity
                    pass_rejected += 1
            allow_flips = config.allow_flips if config is not None else True
            if allow_flips:
                snap = _snapshot()
                ok_flip, msg, _ = editor.flip_edge(e)
                if not ok_flip:
                    pass_rejected += 1
                    continue
            else:
                continue
            active_tris = [tuple(t) for t in editor.triangles if not np.all(np.array(t) == -1)]
            try:
                ok_conf, _ = check_mesh_conformity(editor.points, np.array(active_tris, dtype=int), allow_marked=False)
            except Exception:
                ok_conf = False
            if not ok_conf:
                editor.flip_edge(e)
                pass_rejected += 1
                _restore(snap)
                continue
            post_min = local_min_angle_for_tris(editor.points, editor.triangles, tris_idx)
            if post_min > pre_min + EPS_IMPROVEMENT:
                if strict:
                    ok_s, reason, extra = _strict_check(base_loops)
                    if not ok_s:
                        if verbose:
                            logger.info('greedy_remesh(str): revert flip e=%s reason=%s', e, reason)
                        rejected_ops.append({'phase':'edge','op':'flip','reason':reason})
                        pass_rejected += 1
                        _restore(snap)
                        continue
                logger.debug('greedy_remesh: flipped edge %s %.3f->%.3f', e, pre_min, post_min)
                _save_frame(f'flip_{e[0]}_{e[1]}')
                e_flipped += 1
                changed = True
            else:
                editor.flip_edge(e)
                pass_rejected += 1
                _restore(snap)
        # Per-pass summary (edge)
        try:
            cur_min = editor.global_min_angle()
        except Exception:
            cur_min = float('nan')
        logger.info('edge pass %d: flipped=%d split=%d rejected=%d changed=%s min_angle=%.3f deg',
                    ep, e_flipped, e_split, pass_rejected, changed, cur_min)
        if not changed:
            break
        # Optional end-of-pass compaction (edge pass) - only if there were changes and tombstones exist
        if changed and (config is not None and getattr(config, 'compact_end_of_pass', False)):
            if getattr(editor, 'has_tombstones', None) and editor.has_tombstones():
                try:
                    editor._maybe_compact(force=True)
                except Exception:
                    editor.compact_triangle_indices()
        hook = _ctx_get('compact_check_hook')
        if hook is not None:
            try:
                hook(editor, pass_type='edge', pass_idx=ep)
            except Exception:
                pass
    # Post-pass pocket fill (always if force_pocket_fill; else only when not strict)
    if (not strict) or force_pocket_fill:
        try:
            # If mesh currently has no active triangles and forced pocket fill is requested, attempt a proper
            # pocket triangulation using the convex hull of existing points (ordered) rather than ad-hoc fan seeding.
            if force_pocket_fill:
                active_tris = [t for t in editor.triangles if not np.all(np.array(t) == -1)]
                if len(active_tris) == 0 and len(editor.points) >= 3:
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(editor.points)
                        hull_verts = list(hull.vertices)
                        # Ensure CCW ordering for consistency (ConvexHull gives CCW, but enforce)
                        # Attempt pocket fill using existing API (quad / steiner / earclip)
                        ok, details = editor.try_fill_pocket(hull_verts)
                        if ok:
                            if verbose:
                                logger.info('greedy_remesh: triangulated empty mesh via pocket API (%d hull verts -> %d tris)', len(hull_verts), len(editor.triangles))
                        else:
                            # Record failure reasons
                            fr = details.get('failure_reasons') if isinstance(details, dict) else None
                            reason = 'hull-fill-fail' + (':' + ','.join(fr) if fr else '')
                            rejected_ops.append({'phase':'pocket','op':'seed','reason':reason})
                            if verbose:
                                logger.info('greedy_remesh: hull pocket fill failed (%s)', reason)
                    except Exception as e:
                        # Fallback: if ConvexHull fails (collinear), try sequential order
                        rejected_ops.append({'phase':'pocket','op':'seed','reason':f'hull-exception:{e}'})
                        if verbose:
                            logger.info('greedy_remesh: ConvexHull exception during seed (%s)', e)
                        if len(editor.points) >= 3:
                            seq = list(range(len(editor.points)))
                            try:
                                ok, details = editor.try_fill_pocket(seq)
                                if ok:
                                    if verbose:
                                        logger.info('greedy_remesh: triangulated empty mesh via sequential pocket API (%d pts)', len(seq))
                                else:
                                    fr = details.get('failure_reasons') if isinstance(details, dict) else None
                                    reason = 'seq-fill-fail' + (':' + ','.join(fr) if fr else '')
                                    rejected_ops.append({'phase':'pocket','op':'seed','reason':reason})
                                    if verbose:
                                        logger.info('greedy_remesh: sequential pocket fill failed (%s)', reason)
                            except Exception:
                                rejected_ops.append({'phase':'pocket','op':'seed','reason':'seq-exception'})
                                if verbose:
                                    logger.info('greedy_remesh: failed sequential pocket fill fallback (seq-exception)')
            # After potential empty-mesh triangulation, attempt to fill any 4-vertex empty pockets detected.
            from diagnostics import detect_and_dump_pockets
            pockets, mapping = detect_and_dump_pockets(editor, op_desc='post-greedy')
            filled = 0
            for pk in pockets:
                if pk.get('inside_pts'):
                    continue  # skip non-empty pockets
                verts_local = pk.get('verts', [])
                if len(verts_local) < 3:
                    continue
                # For now we only aggressively auto-fill quads (len==4) unless force_pocket_fill is set; in force mode allow any size
                if (len(verts_local) != 4) and not force_pocket_fill:
                    continue
                # mapping may be a dict (old->new) or a numpy array old->new (-1 for removed)
                try:
                    import numpy as _np
                except Exception:
                    _np = None
                if _np is not None and isinstance(mapping, _np.ndarray):
                    old_to_new = mapping
                    if old_to_new.size == 0 or not _np.any(old_to_new >= 0):
                        verts_global = []
                    else:
                        n_new = int(_np.max(old_to_new)) + 1
                        new_to_old = _np.full((n_new,), -1, dtype=_np.int32)
                        mask = old_to_new >= 0
                        old_idx = _np.nonzero(mask)[0]
                        new_idx = old_to_new[mask]
                        new_to_old[new_idx] = old_idx
                        verts_global = [int(new_to_old[int(v)]) if (0 <= int(v) < n_new) else None for v in verts_local]
                else:
                    inv_map = {v_new: v_old for v_old, v_new in mapping.items()} if isinstance(mapping, dict) else {}
                    verts_global = [inv_map.get(int(v)) for v in verts_local]
                if any(vg is None for vg in verts_global):
                    continue
                try:
                    res = editor.try_fill_pocket(verts_global)
                    ok_fill = bool(res[0]) if isinstance(res, tuple) else bool(res)
                    if ok_fill:
                        filled += 1
                except Exception:
                    continue
            if verbose and filled:
                logger.info('greedy_remesh: filled %d pockets post-pass', filled)
            if filled:
                _save_frame(f'pocket_fill_{filled}')
        except Exception:
            pass

    if annotate_failures and rejected_ops:
        logger.info('greedy_remesh strict rejected %d ops: %s', len(rejected_ops), rejected_ops[:10])
    if rejected_log_path and rejected_ops:
        try:
            import json, time
            payload = {
                'timestamp': time.time(),
                'n_rejected': len(rejected_ops),
                'rejections': rejected_ops,
            }
            with open(rejected_log_path, 'w') as f:
                json.dump(payload, f, indent=2)
            if verbose:
                logger.info('greedy_remesh: wrote rejected ops log to %s', rejected_log_path)
        except Exception as e:
            logger.warning('Could not write rejected ops log: %s', e)
    # Final GIF assembly
    if gif_capture:
        gif_cap.finalize()
    return True

def main(argv=None):  # pragma: no cover
    """Unified CLI for remeshing.

    Modes:
      greedy (default): run simple greedy vertex/edge improvement passes.
      patch:   run advanced patch/batch driver (ported from legacy logic).
    """
    parser = argparse.ArgumentParser(prog='remesh_driver', description='Mesh remeshing driver (greedy, patch, or config dump).')
    parser.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default='INFO', help='Logging verbosity (default: INFO)')
    sub = parser.add_subparsers(dest='mode', help='mode of operation')

    # Greedy mode arguments
    p_greedy = sub.add_parser('greedy', help='Simple greedy vertex/edge improvement loop (default)')
    p_greedy.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default=None, help='Logging verbosity override for this command')
    p_greedy.add_argument('--npts', type=int, default=80)
    p_greedy.add_argument('--seed', type=int, default=42)
    p_greedy.add_argument('--vertex-passes', type=int, default=2)
    p_greedy.add_argument('--edge-passes', type=int, default=2)
    p_greedy.add_argument('--out', type=str, default='greedy_run.png')
    p_greedy.add_argument('--no-plot', action='store_true')
    p_greedy.add_argument('--strict', action='store_true', help='Enable strict rollback on any conformity / crossing / new pocket loop issue')
    p_greedy.add_argument('--reject-crossings', action='store_true', help='Reject operations introducing edge crossings')
    p_greedy.add_argument('--reject-new-loops', action='store_true', help='Reject operations that increase boundary loop count')
    p_greedy.add_argument('--annotate-failures', action='store_true', help='Log first few rejected ops in strict mode')
    p_greedy.add_argument('--force-pocket-fill', action='store_true', help='Force pocket fill pass even in strict mode')
    p_greedy.add_argument('--rejected-log', type=str, default=None, help='Write JSON file with rejected op stats')
    p_greedy.add_argument('--gif-out', type=str, default=None, help='Write an animated GIF of accepted greedy operations (path or filename)')
    p_greedy.add_argument('--gif-fps', type=int, default=4, help='Frames per second for GIF (default=4)')
    p_greedy.add_argument('--profile', action='store_true', help='Profile the run with cProfile and print top hotspots')
    p_greedy.add_argument('--profile-out', type=str, default=None, help='Write raw cProfile stats to this .pstats file when --profile is set')
    p_greedy.add_argument('--config-json', type=str, default=None, help='Path to JSON file containing greedy configuration (as produced by config-dump)')
    # Visualization options
    p_greedy.add_argument('--loop-color-mode', type=str, choices=['per-loop','uniform'], default='per-loop', help='Boundary loop color mode for plots (default: per-loop)')
    p_greedy.add_argument('--loop-vertex-labels', action='store_true', help='Label boundary loop vertices with their order indices')

    # Patch driver mode arguments
    p_patch = sub.add_parser('patch', help='Advanced patch/batch quality improvement driver')
    p_patch.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default=None, help='Logging verbosity override for this command')
    p_patch.add_argument('--npts', type=int, default=120)
    p_patch.add_argument('--seed', type=int, default=42)
    p_patch.add_argument('--threshold', type=float, default=20.0, help='Target min-angle (deg) to stop')
    p_patch.add_argument('--max-iterations', type=int, default=500)
    p_patch.add_argument('--strict', action='store_true', help='Enable strict rollback on loop increases and edge crossings (toggles both guards)')
    p_patch.add_argument('--patch-radius', type=int, default=1)
    p_patch.add_argument('--top-k', type=int, default=80)
    p_patch.add_argument('--disjoint-on', type=str, choices=['tri','vertex'], default='tri')
    p_patch.add_argument('--allow-overlap', action='store_true')
    p_patch.add_argument('--batch-attempts', type=int, default=2)
    p_patch.add_argument('--min-triangle-area', type=float, default=EPS_AREA)
    p_patch.add_argument('--min-triangle-area-fraction', type=float, default=None)
    p_patch.add_argument('--reject-min-angle', type=float, default=None, help='Reject ops whose local min-angle falls below this (deg)')
    p_patch.add_argument('--reject-new-loops', action='store_true', help='Reject operations that increase boundary loop count (strict topology)')
    p_patch.add_argument('--reject-crossings', action='store_true', help='Reject operations introducing edge crossings (via simulation)')
    p_patch.add_argument('--auto-fill-pockets', action='store_true')
    p_patch.add_argument('--autofill-min-triangle-area', type=float, default=None)
    p_patch.add_argument('--autofill-reject-min-angle', type=float, default=None)
    p_patch.add_argument('--angle-unit', type=str, choices=['deg','rad'], default='deg')
    p_patch.add_argument('--log-dir', type=str, default=None)
    p_patch.add_argument('--out-prefix', type=str, default='patch_run')
    p_patch.add_argument('--plot-every', type=int, default=50)
    p_patch.add_argument('--use-greedy-remesh', action='store_true', help='Run a greedy pass each iteration')
    p_patch.add_argument('--greedy-vertex-passes', type=int, default=1)
    p_patch.add_argument('--greedy-edge-passes', type=int, default=1)
    p_patch.add_argument('--no-plot', action='store_true')
    p_patch.add_argument('--no-boundary-highlight', action='store_true', help='Disable boundary loop highlighting in plots')
    p_patch.add_argument('--no-crossing-highlight', action='store_true', help='Disable crossing edge overlay in plots')
    p_patch.add_argument('--loop-color-mode', type=str, choices=['per-loop','uniform'], default='per-loop', help='Boundary loop color mode for plots (default: per-loop)')
    p_patch.add_argument('--loop-vertex-labels', action='store_true', help='Label boundary loop vertices with their order indices')
    p_patch.add_argument('--gif-out', type=str, default=None, help='Write an animated GIF for patch driver iterations (path or filename)')
    p_patch.add_argument('--gif-fps', type=int, default=4, help='Frames per second for patch GIF')
    p_patch.add_argument('--profile', action='store_true', help='Profile the run with cProfile and print top hotspots')
    p_patch.add_argument('--profile-out', type=str, default=None, help='Write raw cProfile stats to this .pstats file when --profile is set (patch mode)')
    p_patch.add_argument('--rejected-log', type=str, default=None, help='Write JSON file with rejected op stats (patch mode)')
    p_patch.add_argument('--config-json', type=str, default=None, help='Path to JSON file containing patch configuration (as produced by config-dump)')

    # Config dump mode (outputs effective JSON for reproducibility)
    p_dump = sub.add_parser('config-dump', help='Emit effective configuration as JSON (no remeshing)')
    p_dump.add_argument('--log-level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default=None, help='Logging verbosity override for this command')
    p_dump.add_argument('--mode', dest='dump_mode', choices=['greedy','patch'], required=True, help='Target configuration to dump')
    # Greedy options (subset)
    p_dump.add_argument('--vertex-passes', type=int, default=2)
    p_dump.add_argument('--edge-passes', type=int, default=2)
    p_dump.add_argument('--strict', action='store_true')
    p_dump.add_argument('--reject-crossings', action='store_true')
    p_dump.add_argument('--reject-new-loops', action='store_true')
    p_dump.add_argument('--force-pocket-fill', action='store_true')
    p_dump.add_argument('--gif-out', type=str, default=None)
    p_dump.add_argument('--gif-fps', type=int, default=4)
    # Patch options
    p_dump.add_argument('--threshold', type=float, default=20.0)
    p_dump.add_argument('--max-iterations', type=int, default=500)
    p_dump.add_argument('--patch-radius', type=int, default=1)
    p_dump.add_argument('--top-k', type=int, default=80)
    p_dump.add_argument('--disjoint-on', type=str, choices=['tri','vertex'], default='tri')
    p_dump.add_argument('--allow-overlap', action='store_true')
    p_dump.add_argument('--batch-attempts', type=int, default=2)
    p_dump.add_argument('--min-triangle-area', type=float, default=EPS_AREA)
    p_dump.add_argument('--min-triangle-area-fraction', type=float, default=None)
    p_dump.add_argument('--reject-min-angle', type=float, default=None)
    p_dump.add_argument('--auto-fill-pockets', action='store_true')
    p_dump.add_argument('--autofill-min-triangle-area', type=float, default=None)
    p_dump.add_argument('--autofill-reject-min-angle', type=float, default=None)
    p_dump.add_argument('--angle-unit', type=str, choices=['deg','rad'], default='deg')
    p_dump.add_argument('--log-dir', type=str, default=None)
    p_dump.add_argument('--out-prefix', type=str, default='patch_run')
    p_dump.add_argument('--plot-every', type=int, default=50)
    p_dump.add_argument('--use-greedy-remesh', action='store_true')
    p_dump.add_argument('--greedy-vertex-passes', type=int, default=1)
    p_dump.add_argument('--greedy-edge-passes', type=int, default=1)
    p_dump.add_argument('--gif-out-patch', type=str, default=None)
    p_dump.add_argument('--gif-fps-patch', type=int, default=4)

    # Default to greedy if no subcommand
    args = parser.parse_args(argv)
    mode = args.mode or 'greedy'
    if mode == 'config-dump':
        if args.dump_mode == 'greedy':
            g_cfg = GreedyConfig(
                max_vertex_passes=args.vertex_passes,
                max_edge_passes=args.edge_passes,
                strict=args.strict,
                reject_crossings=args.reject_crossings,
                reject_new_loops=args.reject_new_loops,
                force_pocket_fill=args.force_pocket_fill,
                gif_capture=bool(getattr(args, 'gif_out', None)),
                gif_dir='greedy_frames',
                gif_out=os.path.basename(args.gif_out) if getattr(args, 'gif_out', None) else 'greedy_run.gif',
                gif_fps=getattr(args, 'gif_fps', 4),
                allow_flips=True,
            )
            payload = {
                'type': 'greedy',
                'config': g_cfg.__dict__,
            }
        else:  # patch
            p_cfg = PatchDriverConfig(
                threshold=args.threshold,
                max_iterations=args.max_iterations,
                reject_new_loops=True if args.strict else False,
                reject_crossings=True if args.strict else False,
                patch_radius=args.patch_radius,
                top_k=args.top_k,
                disjoint_on=args.disjoint_on,
                allow_overlap=args.allow_overlap,
                batch_attempts=args.batch_attempts,
                min_triangle_area=args.min_triangle_area,
                min_triangle_area_fraction=args.min_triangle_area_fraction,
                reject_min_angle_deg=args.reject_min_angle,
                auto_fill_pockets=args.auto_fill_pockets,
                autofill_min_triangle_area=args.autofill_min_triangle_area,
                autofill_reject_min_angle_deg=args.autofill_reject_min_angle,
                angle_unit=args.angle_unit,
                log_dir=args.log_dir,
                out_prefix=args.out_prefix,
                plot_every=args.plot_every,
                use_greedy_remesh=args.use_greedy_remesh,
                greedy_vertex_passes=args.greedy_vertex_passes,
                greedy_edge_passes=args.greedy_edge_passes,
                gif_capture=bool(getattr(args, 'gif_out_patch', None)),
                gif_dir='patch_frames',
                gif_out=os.path.basename(args.gif_out_patch) if getattr(args, 'gif_out_patch', None) else 'patch_run.gif',
                gif_fps=getattr(args, 'gif_fps_patch', 4),
            )
            payload = {
                'type': 'patch',
                'config': p_cfg.__dict__,
            }
        print(json.dumps(payload, indent=2))
        return

    # precedence: subcommand --log-level (if provided) > global --log-level > INFO
    chosen = getattr(args, 'log_level', None)
    if not chosen:
        chosen = parser.parse_args([]).log_level  # fallback to global default 'INFO'
    level = getattr(logging, str(chosen or 'INFO').upper(), logging.INFO)
    # Configure sofia logger family without touching root logger
    configure_logging(level)
    # Initialize per-run context (no-op defaults). External callers may set this via run_context.set_context.
    _ctx_set({'compact_check_hook': COMPACT_CHECK_HOOK})
    random.seed(getattr(args, 'seed', 42))
    np.random.seed(getattr(args, 'seed', 42))

    def _run_greedy():
        pts, tris = build_random_delaunay(npts=args.npts, seed=args.seed)
        editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
        # Use a more specific child logger for greedy mode
        g_logger = get_logger('sofia.driver.greedy')
        g_logger.info('Greedy start: npts=%d ntri=%d min_angle=%.3f deg', len(editor.points), len(editor.triangles), editor.global_min_angle())
        if args.config_json:
            try:
                with open(args.config_json, 'r') as f:
                    cfg_payload = json.load(f)
                # Accept either direct dict or {'type':'greedy','config':{...}}
                cfg_dict = cfg_payload.get('config', cfg_payload)
                # Filter to GreedyConfig fields
                allowed = set(GreedyConfig().__dict__.keys())
                filtered = {k: v for k, v in cfg_dict.items() if k in allowed}
                # Override passes if user specified via flags (explicit override precedence)
                filtered.setdefault('max_vertex_passes', args.vertex_passes)
                filtered.setdefault('max_edge_passes', args.edge_passes)
                g_cfg = GreedyConfig(**filtered)
                g_logger.info('Loaded greedy config from %s', args.config_json)
            except Exception as e:
                g_logger.error('Failed to load greedy config JSON (%s); falling back to CLI flags', e)
                g_cfg = GreedyConfig(
                    max_vertex_passes=args.vertex_passes,
                    max_edge_passes=args.edge_passes,
                    strict=args.strict,
                    reject_crossings=args.reject_crossings,
                    reject_new_loops=args.reject_new_loops,
                    force_pocket_fill=args.force_pocket_fill,
                    verbose=True,
                    gif_capture=bool(getattr(args, 'gif_out', None)),
                    gif_dir='greedy_frames',
                    gif_out=os.path.basename(args.gif_out) if getattr(args, 'gif_out', None) else 'greedy_run.gif',
                    gif_fps=getattr(args, 'gif_fps', 4),
                    allow_flips=True,
                )
        else:
            g_cfg = GreedyConfig(
                max_vertex_passes=args.vertex_passes,
                max_edge_passes=args.edge_passes,
                strict=args.strict,
                reject_crossings=args.reject_crossings,
                reject_new_loops=args.reject_new_loops,
                force_pocket_fill=args.force_pocket_fill,
                verbose=True,
                gif_capture=bool(getattr(args, 'gif_out', None)),
                gif_dir='greedy_frames',
                gif_out=os.path.basename(args.gif_out) if getattr(args, 'gif_out', None) else 'greedy_run.gif',
                gif_fps=getattr(args, 'gif_fps', 4),
                allow_flips=True,
            )
        ok = greedy_remesh(editor, config=g_cfg)
        g_logger.info('Greedy finished ok=%s min_angle=%.3f deg', ok, editor.global_min_angle())
        if not args.no_plot:
            try:
                plot_mesh(editor, outname=args.out, highlight_boundary_loops=True, loop_color_mode=args.loop_color_mode, loop_vertex_labels=args.loop_vertex_labels)
            except Exception as e:
                logger.warning('Plotting failed: %s', e)

    if mode == 'greedy':
        def _greedy_body():
            pts, tris = build_random_delaunay(npts=args.npts, seed=args.seed)
            editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
            g_logger = get_logger('sofia.driver.greedy')
            g_logger.info('Greedy start: npts=%d ntri=%d min_angle=%.3f deg', len(editor.points), len(editor.triangles), editor.global_min_angle())
            if args.config_json:
                try:
                    with open(args.config_json, 'r') as f:
                        cfg_payload = json.load(f)
                    cfg_dict = cfg_payload.get('config', cfg_payload)
                    allowed = set(GreedyConfig().__dict__.keys())
                    filtered = {k: v for k, v in cfg_dict.items() if k in allowed}
                    filtered.setdefault('max_vertex_passes', args.vertex_passes)
                    filtered.setdefault('max_edge_passes', args.edge_passes)
                    g_cfg = GreedyConfig(**filtered)
                    g_logger.info('Loaded greedy config from %s', args.config_json)
                except Exception as e:
                    g_logger.error('Failed to load greedy config JSON (%s); falling back to CLI flags', e)
                    g_cfg = GreedyConfig(
                        max_vertex_passes=args.vertex_passes,
                        max_edge_passes=args.edge_passes,
                        strict=args.strict,
                        reject_crossings=args.reject_crossings,
                        reject_new_loops=args.reject_new_loops,
                        force_pocket_fill=args.force_pocket_fill,
                        verbose=True,
                        gif_capture=bool(getattr(args, 'gif_out', None)),
                        gif_dir='greedy_frames',
                        gif_out=os.path.basename(args.gif_out) if getattr(args, 'gif_out', None) else 'greedy_run.gif',
                        gif_fps=getattr(args, 'gif_fps', 4),
                        allow_flips=True,
                    )
            else:
                g_cfg = GreedyConfig(
                    max_vertex_passes=args.vertex_passes,
                    max_edge_passes=args.edge_passes,
                    strict=args.strict,
                    reject_crossings=args.reject_crossings,
                    reject_new_loops=args.reject_new_loops,
                    force_pocket_fill=args.force_pocket_fill,
                    verbose=True,
                    gif_capture=bool(getattr(args, 'gif_out', None)),
                    gif_dir='greedy_frames',
                    gif_out=os.path.basename(args.gif_out) if getattr(args, 'gif_out', None) else 'greedy_run.gif',
                    gif_fps=getattr(args, 'gif_fps', 4),
                    allow_flips=True,
                )
            remesh_cfg = RemeshConfig(greedy=g_cfg)
            greedy_remesh(
                editor,
                annotate_failures=args.annotate_failures,
                rejected_log_path=args.rejected_log,
                config=remesh_cfg.greedy,
            )
            editor.compact_triangle_indices()
            g_logger.info('Greedy done: min_angle=%.3f deg ntri=%d', editor.global_min_angle(), len(editor.triangles))
            if not args.no_plot:
                plot_mesh(
                    editor,
                    outname=args.out,
                    loop_color_mode=getattr(args, 'loop_color_mode', 'per-loop'),
                    loop_vertex_labels=bool(getattr(args, 'loop_vertex_labels', False)),
                )
                g_logger.info('Wrote %s', args.out)
            if getattr(args, 'gif_out', None):
                g_logger.info('GIF frames directory: %s (output filename: %s)', 'greedy_frames', os.path.basename(args.gif_out))

        if getattr(args, 'profile', False):
            import cProfile, pstats, io
            pr = cProfile.Profile()
            pr.enable()
            _greedy_body()
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            ps.print_stats(20)
            logger.info('Profile (top 20 by cumulative time):\n%s', s.getvalue())
            out = getattr(args, 'profile_out', None)
            if out:
                try:
                    pr.dump_stats(out)
                    logger.info('Raw pstats written to %s', out)
                except Exception as e:
                    logger.warning('Failed to write pstats to %s: %s', out, e)
        else:
            _greedy_body()
        return

    if mode == 'patch':
        p_logger = get_logger('sofia.driver.patch')
        def _patch_body():
            pts, tris = build_random_delaunay(npts=args.npts, seed=args.seed)
            editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
        if args.config_json:
            try:
                with open(args.config_json, 'r') as f:
                    patch_payload = json.load(f)
                cfg_dict = patch_payload.get('config', patch_payload)
                allowed = set(PatchDriverConfig().__dict__.keys())
                filtered = {k: v for k, v in cfg_dict.items() if k in allowed}
                # Provide defaults for fields not present in JSON using CLI values (so user can override just a subset)
                defaults_from_cli = dict(
                    threshold=args.threshold,
                    max_iterations=args.max_iterations,
                    patch_radius=args.patch_radius,
                    top_k=args.top_k,
                    disjoint_on=args.disjoint_on,
                    allow_overlap=args.allow_overlap,
                    batch_attempts=args.batch_attempts,
                    min_triangle_area=args.min_triangle_area,
                    min_triangle_area_fraction=args.min_triangle_area_fraction,
                    reject_min_angle_deg=args.reject_min_angle,
                    reject_new_loops=args.reject_new_loops or args.strict,
                    reject_crossings=args.reject_crossings or args.strict,
                    autofill_min_triangle_area=args.autofill_min_triangle_area,
                    autofill_reject_min_angle_deg=args.autofill_reject_min_angle,
                    angle_unit=args.angle_unit,
                    log_dir=args.log_dir,
                    out_prefix=args.out_prefix,
                    plot_every=args.plot_every,
                    use_greedy_remesh=args.use_greedy_remesh,
                    greedy_vertex_passes=args.greedy_vertex_passes,
                    greedy_edge_passes=args.greedy_edge_passes,
                    gif_capture=bool(getattr(args, 'gif_out', None)),
                    gif_dir='patch_frames',
                    gif_out=os.path.basename(args.gif_out) if getattr(args, 'gif_out', None) else 'patch_run.gif',
                    gif_fps=getattr(args, 'gif_fps', 4),
                )
                for k, v in defaults_from_cli.items():
                    filtered.setdefault(k, v)
                # --strict overrides JSON for guards
                if args.strict:
                    filtered['reject_new_loops'] = True
                    filtered['reject_crossings'] = True
                cfg = PatchDriverConfig(**filtered)
                p_logger.info('Loaded patch config from %s', args.config_json)
            except Exception as e:
                p_logger.error('Failed to load patch config JSON (%s); falling back to CLI args', e)
                cfg = PatchDriverConfig(
                    threshold=args.threshold,
                    max_iterations=args.max_iterations,
                    patch_radius=args.patch_radius,
                    top_k=args.top_k,
                    disjoint_on=args.disjoint_on,
                    allow_overlap=args.allow_overlap,
                    batch_attempts=args.batch_attempts,
                    min_triangle_area=args.min_triangle_area,
                    min_triangle_area_fraction=args.min_triangle_area_fraction,
                    reject_min_angle_deg=args.reject_min_angle,
                    reject_new_loops=args.reject_new_loops or args.strict,
                    reject_crossings=args.reject_crossings or args.strict,
                    autofill_min_triangle_area=args.autofill_min_triangle_area,
                    autofill_reject_min_angle_deg=args.autofill_reject_min_angle,
                    angle_unit=args.angle_unit,
                    log_dir=args.log_dir,
                    out_prefix=args.out_prefix,
                    plot_every=args.plot_every,
                    use_greedy_remesh=args.use_greedy_remesh,
                    greedy_vertex_passes=args.greedy_vertex_passes,
                    greedy_edge_passes=args.greedy_edge_passes,
                    gif_capture=bool(getattr(args, 'gif_out', None)),
                    gif_dir='patch_frames',
                    gif_out=os.path.basename(args.gif_out) if getattr(args, 'gif_out', None) else 'patch_run.gif',
                    gif_fps=getattr(args, 'gif_fps', 4),
        )
        else:
            cfg = PatchDriverConfig(
                threshold=args.threshold,
                max_iterations=args.max_iterations,
                patch_radius=args.patch_radius,
                top_k=args.top_k,
                disjoint_on=args.disjoint_on,
                allow_overlap=args.allow_overlap,
                batch_attempts=args.batch_attempts,
                min_triangle_area=args.min_triangle_area,
                min_triangle_area_fraction=args.min_triangle_area_fraction,
                reject_min_angle_deg=args.reject_min_angle,
                reject_new_loops=args.reject_new_loops or args.strict,
                reject_crossings=args.reject_crossings or args.strict,
                autofill_min_triangle_area=args.autofill_min_triangle_area,
                autofill_reject_min_angle_deg=args.autofill_reject_min_angle,
                angle_unit=args.angle_unit,
                log_dir=args.log_dir,
                out_prefix=args.out_prefix,
                plot_every=args.plot_every,
                use_greedy_remesh=args.use_greedy_remesh,
                greedy_vertex_passes=args.greedy_vertex_passes,
                greedy_edge_passes=args.greedy_edge_passes,
                gif_capture=bool(getattr(args, 'gif_out', None)),
                gif_dir='patch_frames',
                gif_out=os.path.basename(args.gif_out) if getattr(args, 'gif_out', None) else 'patch_run.gif',
                gif_fps=getattr(args, 'gif_fps', 4),
            )
        p_logger.info('Patch driver config: %s', cfg)
        remesh_cfg = RemeshConfig.from_patch_config(cfg)
        # Wrap plot_mesh to inject visualization flags
        def _plot(editor_inst, outname):
            from .visualization import plot_mesh as _pm
            return _pm(
                editor_inst,
                outname=outname,
                highlight_boundary_loops=not args.no_boundary_highlight,
                highlight_crossings=not args.no_crossing_highlight,
                loop_color_mode=getattr(args, 'loop_color_mode', 'per-loop'),
                loop_vertex_labels=bool(getattr(args, 'loop_vertex_labels', False)),
            )
            result = run_patch_batch_driver(
                editor,
                cfg,
                rng=random.Random(args.seed),
                np_rng=np.random.RandomState(args.seed),
                logger=p_logger,
                greedy_remesh=greedy_remesh if cfg.use_greedy_remesh else None,
                plot_mesh=None if args.no_plot else _plot,
                remesh_config=remesh_cfg,
                rejected_log_path=args.rejected_log,
            )
            p_logger.info('Patch driver finished: %s', result)

        if getattr(args, 'profile', False):
            import cProfile, pstats, io
            pr = cProfile.Profile()
            pr.enable()
            _patch_body()
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            ps.print_stats(20)
            logger.info('Profile (top 20 by cumulative time):\n%s', s.getvalue())
            out = getattr(args, 'profile_out', None)
            if out:
                try:
                    pr.dump_stats(out)
                    logger.info('Raw pstats written to %s', out)
                except Exception as e:
                    logger.warning('Failed to write pstats to %s: %s', out, e)
        else:
            _patch_body()
        return

    parser.error(f"Unknown mode {mode}")


__all__ = [
    'greedy_remesh',
    'tri_min_angle',
    'plot_mesh',
    'PatchDriverConfig',
    'run_patch_batch_driver',
    'main',
    'MIN_TRI_AREA', 'REJECT_MIN_ANGLE_DEG', 'AUTO_FILL_POCKETS', 'ALLOW_FLIPS', 'COMPACT_CHECK_HOOK'
]

if __name__ == '__main__':  # pragma: no cover
    main()
