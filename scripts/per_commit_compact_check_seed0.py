"""Replay greedy_remesh from diagnostics/gfail_seed_0.npz and run a compact+strict check after every committing operation.
If a compacted failure is detected, save a diagnostic NPZ with the offending op and mesh and exit.
"""
import sys
import numpy as np
import os
import time

from sofia.sofia.remesh_driver import compact_copy, check_mesh_conformity, find_inverted_triangles, count_boundary_loops, MIN_TRI_AREA, greedy_remesh
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor

NPZ_IN = 'diagnostics/gfail_seed_0.npz'
OUT_PREFIX = 'diagnostics/per_commit_failure'


def save_failure(editor, op_name, op_param, extra_msgs=None):
    pts_comp, tris_comp, mapping, active_idx = compact_copy(editor)
    ok_comp, msgs_comp = check_mesh_conformity(pts_comp, tris_comp, allow_marked=False)
    inv_comp = find_inverted_triangles(pts_comp, tris_comp, eps=MIN_TRI_AREA)
    ts = int(time.time())
    fn = f"{OUT_PREFIX}_{op_name}_{ts}.npz"
    try:
        np.savez(fn, pts=pts_comp, tris=tris_comp, op=op_name, param=op_param, msgs=msgs_comp, inv=inv_comp)
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

    # helper check
    def check_and_dump(op_name, op_param):
        pts_comp, tris_comp, mapping, active_idx = compact_copy(editor)
        ok_comp, msgs_comp = check_mesh_conformity(pts_comp, tris_comp, allow_marked=False)
        inv_comp = find_inverted_triangles(pts_comp, tris_comp, eps=MIN_TRI_AREA)
        pre_pts, pre_tris, _, _ = compact_from_arrays(editor) if 'compact_from_arrays' in globals() else (None,None,{},[])
        # use simple condition: not ok_comp or inv_comp
        if (not ok_comp) or inv_comp:
            fn = save_failure(editor, op_name, op_param, extra_msgs=msgs_comp)
            print('Detected compacted failure after op', op_name, op_param)
            sys.exit(0)

    # We'll monkeypatch bound methods on the editor instance to run a compact+check after any successful mutation
    # Save originals
    orig_add = editor.add_node
    orig_remove = editor.remove_node_with_patch
    orig_split = editor.split_edge
    orig_flip = editor.flip_edge

    def wrap_call(name, orig, *args, **kwargs):
        try:
            res = orig(*args, **kwargs)
        except Exception as e:
            # Let exception propagate after printing
            print(f"Exception in {name}: {e}")
            raise
        ok = False
        if isinstance(res, tuple):
            if len(res) > 0:
                ok = bool(res[0])
        else:
            ok = bool(res)
        if ok:
            # do compact+strict check
            pts_comp, tris_comp, mapping, active_idx = compact_copy(editor)
            ok_comp, msgs_comp = check_mesh_conformity(pts_comp, tris_comp, allow_marked=False)
            inv_comp = find_inverted_triangles(pts_comp, tris_comp, eps=MIN_TRI_AREA)
            if (not ok_comp) or inv_comp:
                fn = save_failure(editor, name, {'args': args, 'kwargs': kwargs, 'res': res})
                print('Compacted failure detected after', name, 'args=', args, 'kwargs=', kwargs)
                sys.exit(0)
        return res

    # bind wrappers
    import types
    editor.add_node = types.MethodType(lambda self, *a, **k: wrap_call('add_node', orig_add, *a, **k), editor)
    editor.remove_node_with_patch = types.MethodType(lambda self, *a, **k: wrap_call('remove_node_with_patch', orig_remove, *a, **k), editor)
    editor.split_edge = types.MethodType(lambda self, *a, **k: wrap_call('split_edge', orig_split, *a, **k), editor)
    editor.flip_edge = types.MethodType(lambda self, *a, **k: wrap_call('flip_edge', orig_flip, *a, **k), editor)

    # Now run greedy_remesh with multiple passes; eventually the wrapper will exit on failure
    try:
        print('Starting greedy_remesh with per-commit compact checks...')
        greedy_remesh(editor, max_vertex_passes=10, max_edge_passes=10, verbose=True)
        print('greedy_remesh finished; no compacted failure detected during run')
    except SystemExit:
        # already handled in wrapper
        pass
    except Exception as e:
        print('greedy_remesh raised exception:', e)

if __name__ == '__main__':
    main()
