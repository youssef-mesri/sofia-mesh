#!/usr/bin/env python3
"""Re-run greedy_remesh starting from the failing snapshot saved in diagnostics/gfail_seed_0.npz
and step through every operation (using the driver's apply_patch_operation) to detect exactly which
committed op causes the compacted mesh to become non-conforming. Saves a diagnostic .npz on first failure.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import sofia.sofia.remesh_driver as debug_check
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor, check_mesh_conformity
from sofia.sofia.logging_utils import get_logger

logger = get_logger('sofia.scripts.reproduce_gfail_seed0_from_fuzz')
from sofia.sofia.remesh_driver import compact_copy

OUTDIR = os.path.join(os.getcwd(), 'diagnostics')

def main():
    fname = os.path.join(OUTDIR, 'gfail_seed_0.npz')
    data = np.load(fname, allow_pickle=True)
    seed = int(data['seed'].tolist()) if 'seed' in data else 0
    pts_before = data['pts_before']
    tris_before = data['tris_before']
    logger.info('Loaded gfail_seed_0: seed=%d pts_before.shape=%s tris_before.shape=%s', seed, pts_before.shape, tris_before.shape)
    editor = PatchBasedMeshEditor(pts_before.copy(), tris_before.copy())

    # Prepare RNGs used by debug_check
    debug_check.rng = random.Random(seed)
    debug_check.np_rng = np.random.RandomState(seed)
    # also seed global random and numpy for modules that use module-level RNG
    random.seed(seed)
    np.random.seed(seed)

    # stash original function
    orig_apply = debug_check.apply_patch_operation
    step = {'count': 0}

    def wrapped_apply(editor_arg, patch, rng):
        step['count'] += 1
        res = orig_apply(editor_arg, patch, rng)
        ok = res[0]
        op = res[2] if len(res) > 2 else None
        param = res[3] if len(res) > 3 else None
        if ok:
            # after a successful operation, compact and check strict conformity
            try:
                # compacted copy
                pts_c, tris_c, mapping, active_idx = compact_copy(editor_arg)
            except Exception as e:
                logger.exception('compact_copy failed at step %d error: %s', step['count'], e)
                pts_c = None; tris_c = None
            if pts_c is not None:
                ok_allow, msgs_allow = check_mesh_conformity(editor_arg.points, editor_arg.triangles, allow_marked=True)
                ok_strict, msgs_strict = check_mesh_conformity(pts_c, tris_c, allow_marked=False)
                if (not ok_allow) or (not ok_strict):
                    out = os.path.join(OUTDIR, f'repro_first_offending_seed0_step{step["count"]}.npz')
                    logger.warning('Detected compacted failure at step %d op=%s param=%s', step['count'], op, param)
                    np.savez_compressed(out,
                                         step=step['count'],
                                         op=op,
                                         param=param,
                                         msgs_allow=np.array(msgs_allow, dtype=object),
                                         msgs_strict=np.array(msgs_strict, dtype=object),
                                         pts_comp=pts_c,
                                         tris_comp=tris_c,
                                         pts_full=editor_arg.points,
                                         tris_full=editor_arg.triangles)
                    logger.info('Wrote %s', out)
                    # exit process with non-zero to signal detection
                    sys.exit(0)
        return res

    # monkeypatch
    debug_check.apply_patch_operation = wrapped_apply
    logger.info('Starting greedy_remesh with seed %d', seed)
    try:
        ok = debug_check.greedy_remesh(editor, max_vertex_passes=2, max_edge_passes=2, verbose=True)
        logger.info('greedy_remesh finished, ok=%s', ok)
    except SystemExit:
        logger.info('Stopped due to detected offending op (saved NPZ).')
        return 0
    finally:
        # restore
        debug_check.apply_patch_operation = orig_apply
    logger.info('No offending commit detected during greedy_remesh run')
    return 0

if __name__ == '__main__':
    sys.exit(main())
