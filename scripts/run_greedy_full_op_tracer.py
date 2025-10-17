#!/usr/bin/env python3
"""Run greedy_remesh for a given seed and record every apply_patch_operation call's outcome
and compacted mesh snapshot. Saves diagnostics to diagnostics/gtrace_seed_<seed>.npz.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import random
import sofia.sofia.remesh_driver as debug_check
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor, check_mesh_conformity
from sofia.sofia.logging_utils import get_logger

logger = get_logger('sofia.scripts.run_greedy_full_op_tracer')
from sofia.sofia.remesh_driver import compact_copy

OUTDIR = os.path.join(os.getcwd(), 'diagnostics')

def run_seed(seed=0, max_vertex_passes=2, max_edge_passes=2, outname=None):
    fname = os.path.join(OUTDIR, f'gfail_seed_{seed}.npz')
    if os.path.exists(fname):
        data = np.load(fname, allow_pickle=True)
        pts_before = data['pts_before']
        tris_before = data['tris_before']
    else:
        logger.info('No gfail file for seed %d', seed)
        return 2
    editor = PatchBasedMeshEditor(pts_before.copy(), tris_before.copy())
    random.seed(seed)
    np.random.seed(seed)
    debug_check.rng = random.Random(seed)
    debug_check.np_rng = np.random.RandomState(seed)

    orig_apply = debug_check.apply_patch_operation
    records = []
    step = {'i':0}

    def wrapped_apply(editor_arg, patch, rng):
        step['i'] += 1
        res = orig_apply(editor_arg, patch, rng)
        ok = res[0]
        op = res[2] if len(res) > 2 else None
        param = res[3] if len(res) > 3 else None
        # compacted snapshot after operation
        try:
            pts_c, tris_c, mapping, active_idx = compact_copy(editor_arg)
        except Exception as e:
            pts_c = None; tris_c = None
        # check conformity
        msgs_allow = []
        msgs_strict = []
        try:
            msgs_allow = check_mesh_conformity(editor_arg.points, editor_arg.triangles, allow_marked=True)[1]
        except Exception:
            pass
        try:
            if pts_c is not None:
                msgs_strict = check_mesh_conformity(pts_c, tris_c, allow_marked=False)[1]
        except Exception:
            pass
        rec = {'step': step['i'], 'action': res[2] if len(res)>2 else None, 'param': param, 'ok': ok,
               'msgs_allow': msgs_allow, 'msgs_strict': msgs_strict, 'pts_comp': pts_c, 'tris_comp': tris_c}
        records.append(rec)
        return res

    debug_check.apply_patch_operation = wrapped_apply
    logger.info('Running greedy_remesh for seed %d', seed)
    try:
        ok = debug_check.greedy_remesh(editor, max_vertex_passes=max_vertex_passes, max_edge_passes=max_edge_passes, verbose=False)
        logger.info('greedy_remesh finished, ok=%s', ok)
    finally:
        debug_check.apply_patch_operation = orig_apply
    # save records
    out = outname or os.path.join(OUTDIR, f'gtrace_seed_{seed}.npz')
    # convert records to arrays suitable for np.savez
    steps = np.array([r['step'] for r in records], dtype=int)
    actions = np.array([r['action'] for r in records], dtype=object)
    params = np.array([r['param'] for r in records], dtype=object)
    oks = np.array([r['ok'] for r in records], dtype=bool)
    msgs_allow = np.array([r['msgs_allow'] for r in records], dtype=object)
    msgs_strict = np.array([r['msgs_strict'] for r in records], dtype=object)
    pts_comp = np.array([r['pts_comp'] for r in records], dtype=object)
    tris_comp = np.array([r['tris_comp'] for r in records], dtype=object)
    logger.info('Saving recorded ops to %s', out)
    np.savez_compressed(out, steps=steps, actions=actions, params=params, oks=oks, msgs_allow=msgs_allow, msgs_strict=msgs_strict, pts_comp=pts_comp, tris_comp=tris_comp)
    return 0

if __name__ == '__main__':
    seed = 0
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    sys.exit(run_seed(seed))
