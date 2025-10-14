#!/usr/bin/env python3
"""Fuzz greedy_remesh across random seeds and save diagnostics for failing cases.

Writes .npz files into ./diagnostics/ named gfail_seed_<seed>.npz containing before/after meshes
and the conformity messages. Limits number of saved failures to avoid excessive output.
"""
import os
import numpy as np
from mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor, check_mesh_conformity
from sofia.sofia.remesh_driver import greedy_remesh, compact_copy


OUTDIR = os.path.join(os.getcwd(), 'diagnostics')
os.makedirs(OUTDIR, exist_ok=True)


def run_fuzz(max_seeds=1000, max_failures=20, npts=60, vertex_passes=2, edge_passes=2):
    found = 0
    for seed in range(max_seeds):
        pts, tris = build_random_delaunay(npts=npts, seed=seed)
        editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

        # compact copy before operations for stable before-state
        pts_before, tris_before, mapping, active_idx = compact_copy(editor)
        # sanity: ensure before-state conforms (strict)
        ok_before, msgs_before = check_mesh_conformity(pts_before, tris_before, allow_marked=False)

        # run greedy_remesh
        ok = greedy_remesh(editor, max_vertex_passes=vertex_passes, max_edge_passes=edge_passes, verbose=False)

        # check conformity allowing tombstones
        ok_allow, msgs_allow = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
        # compact and check strict conformity
        editor.compact_triangle_indices()
        pts_after, tris_after, mapping2, active_idx2 = compact_copy(editor)
        ok_strict, msgs_strict = check_mesh_conformity(pts_after, tris_after, allow_marked=False)

        # decide if failure: either non-conforming with tombstones allowed OR strict fail after compaction
        fail_condition = (not ok_allow) or (not ok_strict)
        if fail_condition:
            fname = os.path.join(OUTDIR, f'gfail_seed_{seed}.npz')
            print(f'[FOUND] seed={seed} ok_before={ok_before} ok_allow={ok_allow} ok_strict={ok_strict} -> dumping {fname}')
            np.savez(fname,
                     seed=seed,
                     pts_before=pts_before,
                     tris_before=tris_before,
                     msgs_before=np.array(msgs_before, dtype=object),
                     pts_after=pts_after,
                     tris_after=tris_after,
                     msgs_allow=np.array(msgs_allow, dtype=object),
                     msgs_strict=np.array(msgs_strict, dtype=object))
            found += 1
            if found >= max_failures:
                print(f'Reached max_failures={max_failures}; stopping fuzz.')
                break
        # small progress indicator
        if seed % 100 == 0 and seed > 0:
            print(f'Checked {seed} seeds, found {found} failures so far...')

    print(f'Fuzz finished: checked up to seed {seed} (max_seeds={max_seeds}), failures={found}')


if __name__ == '__main__':
    run_fuzz(max_seeds=500, max_failures=20, npts=60, vertex_passes=2, edge_passes=2)
