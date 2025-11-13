#!/usr/bin/env python3
"""Replay the debug_check batch/attempt loop for seed=0 and stop at the first conformity issue.

Saves diagnostics to diagnostics/trace_seed0_replay_{mode}.npz and prints a readable trace.
"""
import os
import argparse
import time
import numpy as np
from collections import defaultdict, Counter

from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor, check_mesh_conformity
import sofia.core.remesh_driver as debug_check
from sofia.core.remesh_driver import compact_copy, apply_patch_operation, build_patches_from_metrics_strict, partition_batches
from sofia.core.logging_utils import get_logger

logger = get_logger('sofia.scripts.trace_seed0_replay')

OUT = 'diagnostics'
os.makedirs(OUT, exist_ok=True)


def detect_issues(editor):
    pts_c, tris_c, mapping, active_idx = compact_copy(editor)
    ok, msgs = check_mesh_conformity(pts_c, tris_c, allow_marked=False)
    # duplicate triangles
    tri_sets = [tuple(sorted(map(int, t))) for t in tris_c]
    ctr = Counter(tri_sets)
    dup = {t: c for t, c in ctr.items() if c > 1}
    # edge multiplicity
    edge_count = defaultdict(int)
    for t in tris_c:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        for e in [(a,b),(b,c),(c,a)]:
            edge = tuple(sorted(e))
            edge_count[edge] += 1
    nm_edges = {e:c for e,c in edge_count.items() if c > 2}
    issues = {
        'conform_ok': bool(ok),
        'conform_msgs': msgs,
        'dup_count': sum(c-1 for c in ctr.values() if c>1),
        'dup_examples': list(dup.items())[:10],
        'nm_edge_count': len(nm_edges),
        'nm_edge_examples': list(nm_edges.items())[:10],
        'pts_c': pts_c,
        'tris_c': tris_c,
    }
    return issues


def snapshot_state(editor):
    pts_c, tris_c, mapping, active_idx = compact_copy(editor)
    return {'pts_c': pts_c, 'tris_c': tris_c, 'active_idx': active_idx}


def run_replay(seed=0, allow_flips=True, max_iters=50, batch_attempts=2, patch_radius=1, top_k=80):
    logger.info('Running replay seed=%d allow_flips=%s', seed, allow_flips)
    pts, tris = build_random_delaunay(npts=60, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    debug_check.ALLOW_FLIPS = allow_flips

    trace = []
    iter_no = 0
    while iter_no < max_iters:
        # build patches
        patches = build_patches_from_metrics_strict(editor, node_top_k=top_k, edge_top_k=0, radius=patch_radius, disjoint_on='tri', allow_overlap=False)
        if not patches:
            logger.info('No patches built; stopping')
            break
        pid_to_patch = {p['id']: p for p in patches}
        batches = partition_batches(patches)
        logger.info('Iter %d: built %d patches in %d batches', iter_no, len(patches), len(batches))

        for b_idx, batch in enumerate(batches):
            blocked_pids = set()
            for pid in batch:
                if pid in blocked_pids:
                    # skip because neighbor applied
                    trace.append({'action': ('skip_blocked', pid), 'before': snapshot_state(editor), 'after_issues': detect_issues(editor)})
                    continue
                patch = pid_to_patch.get(pid)
                if patch is None:
                    continue
                for attempt in range(batch_attempts):
                    before = snapshot_state(editor)
                    ok, info, op, op_param, local_min_before, local_min_after, tri_count_before, tri_count_after, rejected = apply_patch_operation(editor, patch, debug_check.rng if hasattr(debug_check, 'rng') else __import__('random').Random(seed))
                    # record action
                    rec = {
                        'iter': iter_no,
                        'batch': b_idx,
                        'patch_id': pid,
                        'attempt': attempt,
                        'result_ok': ok,
                        'info': info,
                        'op': op,
                        'op_param': op_param,
                        'before': before,
                    }
                    # inspect issues after the op
                    issues = detect_issues(editor)
                    rec['after_issues'] = issues
                    trace.append(rec)
                    if ok:
                        # block neighbors conservatively
                        nbrs = debug_check.build_patch_adjacency(patches, editor).get(pid, set())
                        blocked_pids.update(nbrs)
                        # stop on first violating issues
                        if issues['dup_count'] > 0 or issues['nm_edge_count'] > 0 or not issues['conform_ok']:
                            logger.warning('Issue after op %s on patch %s iter %d batch %d attempt %d', rec['op'], pid, iter_no, b_idx, attempt)
                            logger.warning('issues: %d dup examples=%s nm_edges=%d %s', issues['dup_count'], issues['dup_examples'], issues['nm_edge_count'], issues['nm_edge_examples'])
                            fname = os.path.join(OUT, f'trace_seed{seed}_replay_flips_{"on" if allow_flips else "off"}.npz')
                            np.savez(fname, trace=trace)
                            logger.info('Wrote trace to %s', fname)
                            return trace
                        # if accepted but no issues, break to next patch
                        break
                    else:
                        # if op failed, continue trying other attempts
                        continue
            # end batch
        iter_no += 1
    # no issues found in max_iters
    fname = os.path.join(OUT, f'trace_seed{seed}_replay_flips_{"on" if allow_flips else "off"}_noissue.npz')
    np.savez(fname, trace=trace)
    logger.info('No issue detected in replay; wrote trace to %s', fname)
    return trace


if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--allow-flips', action='store_true', help='Enable flips in the replay')
    p.add_argument('--max-iters', type=int, default=50)
    p.add_argument('--batch-attempts', type=int, default=2)
    p.add_argument('--patch-radius', type=int, default=1)
    p.add_argument('--top-k', type=int, default=80)
    args = p.parse_args()
    # run with flips according to flag
    run_replay(seed=args.seed, allow_flips=args.allow_flips, max_iters=args.max_iters, batch_attempts=args.batch_attempts, patch_radius=args.patch_radius, top_k=args.top_k)
