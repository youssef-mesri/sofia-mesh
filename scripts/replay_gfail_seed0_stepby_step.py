#!/usr/bin/env python3
"""Replay gfail_seed_0 operation trace step-by-step and check compacted conformity after each commit.
Saves a diagnostic NPZ with the offending op and mesh snapshot when found.
"""
import numpy as np
import os
import sys
# ensure project root is on sys.path so imports like `mesh_modifier2` resolve when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor, check_mesh_conformity

def main():
    p = 'diagnostics/trace_seed0.npz'
    trace = np.load(p, allow_pickle=True)['trace']
    # start from initial snapshot in trace[0]['before']
    initial = trace[0]['before']
    pts = initial['pts_c']
    tris = initial['tris_c']
    editor = PatchBasedMeshEditor(pts, tris)
    print('Starting replay with', len(trace), 'ops')
    out_dir = 'diagnostics/replay_step_debug'
    os.makedirs(out_dir, exist_ok=True)
    for i, entry in enumerate(trace):
        action = entry['action']
        kind = action[0]
        print(f"Step {i:03d}: {action}")
        committed = False
        reason = None
        try:
            if kind == 'add':
                # action format: ('add', tri_idx, success_bool, msg)
                _, tri_idx, ok, msg = action
                if ok:
                    res = editor.add_node(editor.points[0]*0 + 0, tri_idx=tri_idx)  # placeholder to get signature
                    # but trace's add entries include tri_idx only; use recorded 'before'/'after' snapshots instead
                    # safer: compare before/after and apply differential? For now re-run using recorded 'after' snapshot by
                    # applying node insertion by locating which vertex was added in 'after' snapshot.
                    # We'll use the trace's 'after' mesh snapshot for the subsequent state instead of calling editor.add_node.
                    pass
            elif kind == 'split':
                _, edge, ok, msg = action
                if ok:
                    succeeded, m, details = editor.split_edge(edge)
                    committed = succeeded
                    reason = m
            elif kind == 'remove':
                _, tri_idx, ok, msg = action
                if ok:
                    # remove_node_with_patch expects a vertex index; trace remove action used tri_idx (triangle center?)
                    # The trace's action used 'remove' with an integer likely being vertex index in that trace.
                    # We'll attempt to call editor.remove_node_with_patch with that index.
                    succeeded, m, details = editor.remove_node_with_patch(tri_idx)
                    committed = succeeded
                    reason = m
            elif kind == 'flip' or kind == 'flip_skipped':
                if kind == 'flip':
                    _, edge, ok, msg = action
                    if ok:
                        succeeded, m, details = editor.flip_edge(edge)
                        committed = succeeded
                        reason = m
                else:
                    # flip_skipped: no commit
                    committed = False
            elif kind == 'split_delaunay':
                _, edge, ok, msg = action
                if ok:
                    succeeded, m, details = editor.split_edge_delaunay(edge)
                    committed = succeeded
                    reason = m
            else:
                print('Unknown action kind:', kind)
        except Exception as e:
            print('Exception while applying op:', e)

        # After commit, if the trace recorded success, sync editor to the trace's saved 'after_issues' snapshot
        # (this contains the compacted points/triangles for the state after the operation) and check conformity.
        if entry['action'][2] is True:
            # prefer 'after_issues' snapshot (present in trace entries) which includes compacted pts/tris
            if 'after_issues' in entry and entry['after_issues']:
                st = entry['after_issues']
                # use returned compacted coords/triangles
                editor.points = np.asarray(st['pts_c'])
                editor.triangles = np.asarray(st['tris_c'])
                # rebuild maps to keep editor consistent
                try:
                    editor._update_maps()
                except Exception:
                    pass
            else:
                # fallback: compact current state
                try:
                    editor.compact_triangle_indices()
                except Exception as e:
                    print('compact failed', e)
            # now check compacted conformity (strict)
            ok_conf, msgs = check_mesh_conformity(editor.points, editor.triangles, verbose=False, allow_marked=False)
            if not ok_conf:
                print('Conformity failed after step', i, 'action=', action)
                fn = os.path.join(out_dir, f'first_offending_step_{i}_action_{kind}.npz')
                np.savez_compressed(fn,
                                     step=i,
                                     action=action,
                                     msgs=msgs,
                                     pts=editor.points,
                                     tris=editor.triangles)
                print('Wrote', fn)
                return 1
    print('Replay completed, no compacted conformity failure detected')
    return 0

if __name__ == '__main__':
    sys.exit(main())
