"""Enhanced per-commit checker: replay greedy_remesh from diagnostics/gfail_seed_0.npz and on any successful commit
save both pre-commit compacted mesh and post-commit compacted mesh plus raw editor arrays and offending op details.
"""
import sys
import numpy as np
import os
import time

from sofia.sofia.remesh_driver import compact_copy, check_mesh_conformity, find_inverted_triangles, MIN_TRI_AREA, greedy_remesh
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor

NPZ_IN = 'diagnostics/gfail_seed_0.npz'
OUT_PREFIX = 'diagnostics/per_commit_prepost_failure'


def save_prepost(editor, op_name, op_param, pre_comp, pre_raw_pts, pre_raw_tris, post_comp, post_raw_pts, post_raw_tris, msgs_comp, inv_comp):
    ts = int(time.time())
    fn = f"{OUT_PREFIX}_{op_name}_{ts}.npz"
    try:
        np.savez(fn,
                 pre_pts=pre_comp[0], pre_tris=pre_comp[1],
                 post_pts=post_comp[0], post_tris=post_comp[1],
                 raw_pre_pts=pre_raw_pts, raw_pre_tris=pre_raw_tris,
                 raw_post_pts=post_raw_pts, raw_post_tris=post_raw_tris,
                 op=op_name, param=op_param, msgs=msgs_comp, inv=inv_comp)
        print('Wrote', fn)
    except Exception as e:
        print('Failed to write failure file:', e)
    return fn


def main():
    if not os.path.exists(NPZ_IN):
        print('Input NPZ not found:', NPZ_IN); sys.exit(1)
    data = np.load(NPZ_IN, allow_pickle=True)
    if 'pts_before' in data:
        pts = data['pts_before']
        tris = data['tris_before']
    elif 'points' in data and 'tris' in data:
        pts = data['points']
        tris = data['tris']
    else:
        print('Could not find pts_before/tris_before keys in', NPZ_IN); sys.exit(1)

    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    print('Loaded editor: npts=', len(editor.points), 'ntris=', len(editor.triangles))

    # Use unbound class methods to avoid double-binding issues when wrapping
    orig_add = PatchBasedMeshEditor.add_node
    orig_remove = PatchBasedMeshEditor.remove_node_with_patch
    orig_split = PatchBasedMeshEditor.split_edge
    orig_flip = PatchBasedMeshEditor.flip_edge

    import types

    def wrap(orig_name, orig_func):
        def wrapped(self, *args, **kwargs):
            # capture pre-commit compact and raw arrays
            pre_comp = compact_copy(editor)
            pre_raw_pts = editor.points.copy()
            pre_raw_tris = editor.triangles.copy()
            # call unbound orig_func with explicit self
            res = orig_func(self, *args, **kwargs)
            ok = False
            if isinstance(res, tuple):
                ok = bool(res[0])
            else:
                ok = bool(res)
            if ok:
                post_comp = compact_copy(editor)
                post_raw_pts = editor.points.copy()
                post_raw_tris = editor.triangles.copy()
                ok_comp, msgs_comp = check_mesh_conformity(post_comp[0], post_comp[1], allow_marked=False)
                inv_comp = find_inverted_triangles(post_comp[0], post_comp[1], eps=MIN_TRI_AREA)
                if (not ok_comp) or inv_comp:
                    # save both pre and post
                    fn = save_prepost(editor, orig_name, {'args': args, 'kwargs': kwargs}, pre_comp, pre_raw_pts, pre_raw_tris, post_comp, post_raw_pts, post_raw_tris, msgs_comp, inv_comp)
                    print('Detected compacted failure after', orig_name, '-> saved', fn)
                    sys.exit(0)
            return res
        return types.MethodType(wrapped, editor)

    editor.add_node = wrap('add_node', orig_add)
    editor.remove_node_with_patch = wrap('remove_node_with_patch', orig_remove)
    editor.split_edge = wrap('split_edge', orig_split)
    editor.flip_edge = wrap('flip_edge', orig_flip)

    try:
        print('Starting greedy_remesh with pre/post per-commit checks...')
        greedy_remesh(editor, max_vertex_passes=10, max_edge_passes=10, verbose=True)
        print('greedy_remesh finished; no compacted failure detected during run')
    except SystemExit:
        pass
    except Exception as e:
        print('greedy_remesh raised exception:', e)

if __name__ == '__main__':
    main()
