#!/usr/bin/env python3
"""Run greedy_remesh from gfail_seed_0.before and perform compact+strict check after each pass via COMPACT_CHECK_HOOK.
If a failure is detected, save the compacted offending snapshot and exit.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sofia.core.remesh_driver as debug_check
from sofia.core.run_context import set_context
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, check_mesh_conformity
from sofia.core.logging_utils import get_logger

logger = get_logger('sofia.scripts.compact_check_runner_seed0')

OUT = 'diagnostics'
NPZ = os.path.join(OUT, 'gfail_seed_0.npz')
if not os.path.exists(NPZ):
    logger.error('Missing %s', NPZ); sys.exit(2)
D = np.load(NPZ, allow_pickle=True)
seed = int(D['seed'].tolist()) if 'seed' in D else 0
pts_before = D['pts_before']; tris_before = D['tris_before']

editor = PatchBasedMeshEditor(pts_before.copy(), tris_before.copy())

# prepare RNGs
import random
debug_check.rng = random.Random(seed)
debug_check.np_rng = np.random.RandomState(seed)
random.seed(seed); np.random.seed(seed)

# define hook
def compact_hook(ed, pass_type, pass_idx):
    pts_c, tris_c, _, _ = debug_check.compact_copy(ed)
    ok, msgs = check_mesh_conformity(pts_c, tris_c, allow_marked=False)
    logger.info('compact_hook: pass_type=%s pass_idx=%s ok=%s msgs=%s', pass_type, pass_idx, ok, msgs)
    if not ok:
        fn = os.path.join(OUT, f'compact_hook_failure_{pass_type}_{pass_idx}.npz')
        np.savez_compressed(fn, pts=pts_c, tris=tris_c, msgs=np.array(msgs, dtype=object), pass_type=pass_type, pass_idx=pass_idx)
        logger.info('Wrote %s', fn)
        sys.exit(0)

# install hook in per-run context (back-compat global may be ignored by driver)
debug_check.COMPACT_CHECK_HOOK = compact_hook
set_context({'compact_check_hook': compact_hook})
logger.info('Running greedy_remesh with compact hook...')
ok = debug_check.greedy_remesh(editor, max_vertex_passes=5, max_edge_passes=5, verbose=True)
logger.info('greedy_remesh finished, ok=%s', ok)
logger.info('No compact-hook failure detected')
