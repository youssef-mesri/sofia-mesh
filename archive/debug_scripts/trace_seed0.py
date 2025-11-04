#!/usr/bin/env python3
"""Trace the sequence of operations on seed=0 until the first duplication/non-manifold appears.

Saves diagnostics to diagnostics/trace_seed0.npz and prints a readable trace.
"""
import os
import numpy as np
from collections import defaultdict, Counter

from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor, check_mesh_conformity
from sofia.core.logging_utils import get_logger

logger = get_logger('sofia.scripts.trace_seed0')
import sofia.core.remesh_driver as debug_check
# disable flips for this trace run to observe behavior without flips
debug_check.ALLOW_FLIPS = False
from sofia.core.remesh_driver import compact_copy, tri_min_angle


OUT = 'diagnostics'
os.makedirs(OUT, exist_ok=True)


def detect_issues(editor):
    # compacted copy
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


def run_trace():
    pts, tris = build_random_delaunay(npts=60, seed=0)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

    # initial check
    init_issues = detect_issues(editor)
    logger.info('Initial conform_ok=%s dup_count=%d nm_edges=%d', init_issues['conform_ok'], init_issues['dup_count'], init_issues['nm_edge_count'])

    trace = []

    # We will reproduce the greedy_remesh two loops but instrument per-op
    # Loop 1: vertex-centric
    interior_vertices = [v for v in range(len(editor.points)) if not editor.edge_map and False]
    # Instead, reuse similar selection as greedy_remesh: interior vertices from editor.v_map
    interior_vertices = [v for v in range(len(editor.points)) if not any((v in e and len(editor.edge_map.get(e, []))==1) for e in editor.edge_map) and len(editor.v_map.get(v, [])) >= 3]

    # We'll iterate vertices and perform the same add/remove behavior until we find an issue.
    for v in interior_vertices:
        # snapshot
        before = snapshot_state(editor)
        tri_indices = sorted(set(editor.v_map.get(int(v), [])))
        if not tri_indices:
            continue
        # compute worst triangle
        worst_tri = None; worst_ang = 180.0
        for ti in tri_indices:
            try:
                ang = tri_min_angle(editor.points, editor.triangles[int(ti)])
                if ang < worst_ang:
                    worst_ang = ang; worst_tri = int(ti)
            except Exception:
                continue
        action = None
        # try remove if degree>6
        degree = len(editor.v_map.get(int(v), []))
        if degree > 6:
            ok, msg, info = editor.remove_node_with_patch(int(v), force_strict=False)
            action = ('remove', int(v), ok, msg)
            if ok:
                issues = detect_issues(editor)
                trace.append({'action': action, 'before': before, 'after_issues': issues})
                if issues['dup_count'] > 0 or issues['nm_edge_count'] > 0 or not issues['conform_ok']:
                    logger.warning('Issue after remove at v %s issues=%d %d', v, issues['dup_count'], issues['nm_edge_count'])
                    np.savez(os.path.join(OUT, 'trace_seed0.npz'), trace=trace)
                    return
                continue
            else:
                # revert not needed since remove_node_with_patch didn't commit on failure
                trace.append({'action': action, 'before': before, 'after_issues': detect_issues(editor)})
                continue

        # otherwise try add at worst triangle centroid if it meets size criteria
        if worst_tri is not None:
            pts_tri = editor.points[editor.triangles[worst_tri]]
            centroid = np.mean(pts_tri, axis=0)
            ok, msg, info = editor.add_node(centroid, tri_idx=worst_tri)
            action = ('add', worst_tri, ok, msg)
            trace.append({'action': action, 'before': before})
            if ok:
                issues = detect_issues(editor)
                trace[-1]['after_issues'] = issues
                if issues['dup_count'] > 0 or issues['nm_edge_count'] > 0 or not issues['conform_ok']:
                    logger.warning('Issue after add at tri %s issues=%d %d', worst_tri, issues['dup_count'], issues['nm_edge_count'])
                    np.savez(os.path.join(OUT, 'trace_seed0.npz'), trace=trace)
                    return
                continue
            else:
                trace[-1]['after_issues'] = detect_issues(editor)
                continue

    # Loop 2: edge-centric
    interior_edges = [e for e,s in editor.edge_map.items() if len(s) == 2]
    for e in interior_edges:
        before = snapshot_state(editor)
        tris_idx = sorted(list(editor.edge_map.get(e, [])))
        if len(tris_idx) != 2:
            continue
        # compute opposite angles
        try:
            tri0 = editor.triangles[int(tris_idx[0])]
            tri1 = editor.triangles[int(tris_idx[1])]
            opp0 = [v for v in tri0 if v not in e][0]
            opp1 = [v for v in tri1 if v not in e][0]
            angs0 = np.array(list(map(float, [np.nan, np.nan, np.nan])))
        except Exception:
            opp0 = opp1 = None

        # try split if opposite angle > 120
        # compute angles safely
        def compute_opposite_angle(tri, opp):
            try:
                p0 = editor.points[int(tri[0])]; p1 = editor.points[int(tri[1])]; p2 = editor.points[int(tri[2])]
                angs = []
                # reuse triangle_angles via mesh_modifier2? use simple law of cosines here
                from sofia.core.mesh_modifier2 import triangle_angles
                angs = triangle_angles(p0,p1,p2)
                idx = list(tri).index(opp)
                return angs[idx]
            except Exception:
                return 0.0

        opp_angle0 = compute_opposite_angle(editor.triangles[int(tris_idx[0])], opp0) if opp0 is not None else 0.0
        opp_angle1 = compute_opposite_angle(editor.triangles[int(tris_idx[1])], opp1) if opp1 is not None else 0.0
        action = None
        if opp_angle0 > 120.0 or opp_angle1 > 120.0:
            ok, msg, info = editor.split_edge(e)
            action = ('split', e, ok, msg)
            trace.append({'action': action, 'before': before})
            if ok:
                issues = detect_issues(editor)
                trace[-1]['after_issues'] = issues
                if issues['dup_count'] > 0 or issues['nm_edge_count'] > 0 or not issues['conform_ok']:
                    logger.warning('Issue after split on edge %s issues=%d %d', e, issues['dup_count'], issues['nm_edge_count'])
                    np.savez(os.path.join(OUT, 'trace_seed0.npz'), trace=trace)
                    return
                continue
            else:
                trace[-1]['after_issues'] = detect_issues(editor)
                continue

        # otherwise try flip (may be disabled via debug_check.ALLOW_FLIPS)
        if debug_check.ALLOW_FLIPS:
            ok, msg, _ = editor.flip_edge(e)
            action = ('flip', e, ok, msg)
            trace.append({'action': action, 'before': before})
            if ok:
                issues = detect_issues(editor)
                trace[-1]['after_issues'] = issues
                if issues['dup_count'] > 0 or issues['nm_edge_count'] > 0 or not issues['conform_ok']:
                    logger.warning('Issue after flip on edge %s issues=%d %d', e, issues['dup_count'], issues['nm_edge_count'])
                    np.savez(os.path.join(OUT, 'trace_seed0.npz'), trace=trace)
                    return
            else:
                trace[-1]['after_issues'] = detect_issues(editor)
        else:
            # flips disabled: record that we skipped the flip
            action = ('flip_skipped', e, None, 'flips_disabled')
            trace.append({'action': action, 'before': before, 'after_issues': detect_issues(editor)})

    # if reached here no issue found in these single passes
    logger.info('No issue detected in this single-pass trace')
    np.savez(os.path.join(OUT, 'trace_seed0.npz'), trace=trace)


if __name__ == '__main__':
    run_trace()
