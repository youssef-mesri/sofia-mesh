"""Local mesh operations (split, flip, remove, add, pocket fill)."""
from __future__ import annotations
import numpy as np
from .geometry import triangle_area, triangle_angles
from .constants import EPS_AREA, EPS_MIN_ANGLE_DEG
from .quality import worst_min_angle, non_worsening_quality
from .conformity import check_mesh_conformity, simulate_compaction_and_check
from .helpers import boundary_cycle_from_incident_tris

# NOTE: This is a placeholder extraction stub. The full migration from mesh_modifier2
# would proceed incrementally; current operations continue to live in the legacy file.
# Tests still import from mesh_modifier2 for now.

__all__ = [
    'local_quality_ok',
    'op_split_edge',
    'op_split_edge_delaunay',
    'op_remove_node_with_patch',
    'op_flip_edge',
    'op_add_node',
    'op_try_fill_pocket',
    'op_move_vertices_to_barycenter',
    'op_edge_collapse'
]

def local_quality_ok(pre_min, post_min):
    return not (post_min < pre_min - EPS_MIN_ANGLE_DEG)


# ------------------------
# Internal helpers
# ------------------------
def _simulate_preflight(editor, candidate_points, triangles_to_tombstone, new_tris,
                        stats=None, reject_msg_prefix="", eps_area=EPS_AREA):
    """Run optional simulated compaction preflight.

    Parameters
    ----------
    editor : PatchBasedMeshEditor
    candidate_points : ndarray
        Points array to use for simulation (may include new points).
    triangles_to_tombstone : Iterable[int]
        Indices of triangles that will be removed/tombstoned by the op.
    new_tris : Sequence[Sequence[int]]
        Newly created (already oriented) triangles to be appended.
    stats : OpStats or None
        Stats object for recording simulation_rejects / fail.
    reject_msg_prefix : str
        Prefix for rejection message context.
    eps_area : float
        Area epsilon forwarded to simulate_compaction_and_check.

    Returns (ok_bool, message_or_None)
    """
    if not getattr(editor, 'simulate_compaction_on_commit', False):
        return True, None
    # Amortize heavy simulation checks using editor heuristics
    change_mag = len(triangles_to_tombstone) + len(new_tris)
    should_check = True
    if hasattr(editor, '_should_run_simulation_check'):
        try:
            should_check = bool(editor._should_run_simulation_check(change_mag))
        except Exception:
            should_check = True
    if not should_check:
        return True, None
    try:
        cand_tris = editor.triangles.copy()
        for idx in triangles_to_tombstone:
            cand_tris[int(idx)] = [-1, -1, -1]
        cand_sim = cand_tris.tolist() + [list(t) for t in new_tris]
        ok_sim, sim_msgs, sim_inv = simulate_compaction_and_check(
            candidate_points, cand_sim,
            eps_area=eps_area,
            reject_boundary_loop_increase=getattr(editor,'reject_boundary_loop_increase',False),
            reject_any_boundary_loops=getattr(editor,'reject_any_boundary_loops',False),
            reject_crossing_edges=getattr(editor,'reject_crossing_edges',False))
        if not ok_sim:
            if stats:
                stats.simulation_rejects += 1; stats.fail += 1
            return False, f"{reject_msg_prefix}rejected by simulated compaction check: msgs={sim_msgs} inv={sim_inv}"
        return True, None
    except Exception as e:
        # Conservative: treat simulation error as rejection to avoid committing invalid state.
        if stats:
            stats.simulation_rejects += 1; stats.fail += 1
        return False, f"{reject_msg_prefix}simulation exception: {e}"


def _orient_tris(points, tris, eps=EPS_AREA):
    """Orient a sequence of triangles so each has positive (or > eps) signed area.

    Parameters
    ----------
    points : ndarray
    tris : iterable of index triplets (list/tuple)
    eps : float

    Returns list of oriented triplets (lists) (does not drop degenerates; caller filters earlier).
    """
    oriented = []
    for tri in tris:
        try:
            p0 = points[int(tri[0])]; p1 = points[int(tri[1])]; p2 = points[int(tri[2])]
            a_signed = triangle_area(p0, p1, p2)
            if a_signed <= eps:
                tri = [tri[0], tri[2], tri[1]]
        except Exception:
            # keep original ordering if anything goes wrong
            pass
        oriented.append(list(tri))
    return oriented


def _evaluate_quality_change(editor, old_tris, new_tris, stats, op_label,
                              candidate_points=None, before_kw='before', after_kw='post'):
    """Compare worst minimum angle pre/post and enforce non-worsening policy.

    Returns (ok_bool, fail_message_or_None). Increments stats on failure.
    Parameters allow customizing wording to preserve historical messages.
    """
    pts_old = editor.points if candidate_points is None else editor.points
    pts_new = editor.points if candidate_points is None else candidate_points
    try:
        pre_mina = worst_min_angle(pts_old, old_tris)
        post_mina = worst_min_angle(pts_new, new_tris)
        if not non_worsening_quality(pre_mina, post_mina):
            if stats:
                stats.quality_rejects += 1; stats.fail += 1
            return False, f"{op_label} would worsen worst-triangle ({before_kw}={pre_mina:.6f}deg {after_kw}={post_mina:.6f}deg)"
    except Exception as e:
        editor.logger.debug("%s quality eval error: %s", op_label.lower().replace(' ', '_'), e)
    return True, None


# ------------------------
# Delegated operations (extracted from legacy editor)
# Each function expects an editor instance as first argument.
# ------------------------

def op_split_edge_delaunay(editor, edge, strict_mode='centroid'):
    stats = getattr(editor, '_get_op_stats', None)
    stats = stats('split_delaunay') if stats else None
    if stats: stats.attempts += 1
    ok, msgs = editor_check(editor)
    if not ok:
        if stats: stats.fail += 1
        return False, "mesh not conforming before split", msgs
    a, b = tuple(sorted(edge))
    tri_indices = sorted(set(editor.v_map.get(int(a), []) + editor.v_map.get(int(b), [])))
    if len(tri_indices) == 0:
        return False, "no incident triangles", None
    from mesh_modifier2 import retriangulate_patch_strict  # lazy import to avoid cyclic
    new_pts, new_tris, success, appended_tris = retriangulate_patch_strict(
        editor.points, editor.triangles, tri_indices,
        new_point_coords=[0.5*(editor.points[a]+editor.points[b])], strict_mode=strict_mode)
    if not success:
        if stats: stats.fail += 1
        return False, "local retriangulation failed", None
    # local quality comparison
    ok_q, msg_q = _evaluate_quality_change(
        editor,
        [editor.triangles[int(ti)] for ti in tri_indices],
        appended_tris,
        stats,
        op_label="Split (delaunay)",
        candidate_points=new_pts,
        before_kw='before', after_kw='post')
    if not ok_q:
        return False, msg_q, None
    # orient appended triangles for consistency
    oriented_appended = _orient_tris(new_pts, appended_tris)
    ok_sim, sim_msg = _simulate_preflight(
        editor, new_pts, tri_indices, oriented_appended, stats,
        reject_msg_prefix="Split (delaunay) ")
    if not ok_sim:
        return False, sim_msg, None
    # commit: update points (new_pts may include appended vertex)
    editor.points = new_pts
    for idx in tri_indices:
        editor._remove_triangle_from_maps(idx)
        editor.triangles[idx] = [-1,-1,-1]
    if oriented_appended:
        start = len(editor.triangles)
        appended_arr = np.array(oriented_appended, dtype=np.int32)
        editor.triangles = np.ascontiguousarray(np.vstack([editor.triangles, appended_arr]).astype(np.int32))
        for i in range(start, len(editor.triangles)):
            editor._add_triangle_to_maps(i)
    # Register commit for amortization tracking
    if hasattr(editor, '_on_op_committed'):
        try:
            editor._on_op_committed(tombstoned=len(tri_indices), appended=len(oriented_appended))
        except Exception:
            pass
    if stats: stats.success += 1
    return True, "split successful", {'npts': len(editor.points), 'ntri': len(editor.triangles)}


def editor_check(editor):
    from .conformity import check_mesh_conformity
    return check_mesh_conformity(editor.points, editor.triangles)


def op_split_edge(editor, edge=None):
    stats = getattr(editor, '_get_op_stats', None)
    stats = stats('split_midpoint') if stats else None
    if stats: stats.attempts += 1
    ok, msgs = editor_check(editor)
    if not ok:
        if stats: stats.fail += 1
        return False, "mesh not conforming before add", msgs
    if edge is None:
        if stats: stats.fail += 1
        return False, "Specify edge", None
    edge = tuple(sorted(edge))
    tris_idx = list(editor.edge_map.get(edge, []))
    mid = 0.5*(editor.points[edge[0]] + editor.points[edge[1]])
    new_points_candidate = np.vstack([editor.points, mid])
    new_idx = len(editor.points)
    if len(tris_idx) == 2:
        t1, t2 = [editor.triangles[i] for i in tris_idx]
        opp1 = [v for v in t1 if v not in edge][0]
        opp2 = [v for v in t2 if v not in edge][0]
        new_tris = [
            [edge[0], new_idx, opp1],
            [edge[1], new_idx, opp1],
            [edge[0], new_idx, opp2],
            [edge[1], new_idx, opp2]
        ]
    elif len(tris_idx) == 1:
        # Boundary edge: split single adjacent triangle into two
        t = editor.triangles[tris_idx[0]]
        opp = [v for v in t if v not in edge][0]
        new_tris = [
            [edge[0], new_idx, opp],
            [edge[1], new_idx, opp]
        ]
    else:
        if stats: stats.fail += 1
        return False, "Edge is not splittable (no incident triangles)", None
    try:
        tmp_tri_full = [tuple(t) for i,t in enumerate(editor.triangles) if i not in tris_idx and not np.all(np.array(t) == -1)]
        tmp_tri_full.extend(new_tris)
        tmp_tri_full = np.array(tmp_tri_full, dtype=int)
        ok_sub, _ = check_mesh_conformity(new_points_candidate, tmp_tri_full, allow_marked=False)
    except Exception:
        ok_sub = False
    if not ok_sub:
        if stats: stats.fail += 1
        return False, "local retriangulation failed validation", None
    # orient & quality policy
    new_tris = _orient_tris(new_points_candidate, new_tris)
    enforce_q = getattr(editor, 'enforce_split_quality', True)
    if enforce_q:
        ok_q, msg_q = _evaluate_quality_change(
            editor,
            [editor.triangles[int(ti)] for ti in tris_idx],
            new_tris,
            stats,
            op_label="Split (midpoint)",
            candidate_points=new_points_candidate,
            before_kw='before', after_kw='post')
        if not ok_q:
            return False, msg_q, None
    else:
        # Refinement mode: do not gate on non-worsening; still compute for logs
        try:
            pre_list = [editor.triangles[int(ti)] for ti in tris_idx]
            pre_mina = worst_min_angle(editor.points, pre_list)
            post_mina = worst_min_angle(new_points_candidate, new_tris)
            editor.logger.debug("Split (midpoint) quality (before=%.6fdeg post=%.6fdeg)", pre_mina, post_mina)
        except Exception:
            pass
    # simulated compaction preflight
    ok_sim, sim_msg = _simulate_preflight(
        editor, new_points_candidate, tris_idx, new_tris, stats,
        reject_msg_prefix="Split (midpoint) ")
    if not ok_sim:
        return False, sim_msg, None
    # commit
    editor.points = new_points_candidate
    for idx in tris_idx:
        editor._remove_triangle_from_maps(idx)
        editor.triangles[idx] = [-1,-1,-1]
    start = len(editor.triangles)
    new_arr = np.array(new_tris, dtype=np.int32)
    editor.triangles = np.ascontiguousarray(np.vstack([editor.triangles, new_arr]).astype(np.int32))
    for i in range(start, len(editor.triangles)):
        editor._add_triangle_to_maps(i)
    if hasattr(editor, '_on_op_committed'):
        try:
            editor._on_op_committed(tombstoned=len(tris_idx), appended=len(new_tris))
        except Exception:
            pass
    if stats: stats.success += 1
    return True, "split successful", {'npts': len(editor.points), 'ntri': len(editor.triangles)}


def op_flip_edge(editor, edge):
    stats_fn = getattr(editor, '_get_op_stats', None)
    stats = stats_fn('flip') if stats_fn else None
    if stats: stats.attempts += 1
    edge = tuple(sorted(edge))
    tris_idx = list(editor.edge_map.get(edge, []))
    if len(tris_idx) != 2:
        if stats: stats.fail += 1
        return False, "Edge is not flippable (not shared by 2 triangles)", None
    # In virtual-boundary mode, prohibit flipping boundary edges (those with only one incident tri)
    if getattr(editor, 'virtual_boundary_mode', False):
        # A boundary edge would have len(incident)!=2; we already checked that above.
        # However, be conservative: if either of the opposite vertices is the virtual placeholder (id 0), reject.
        # Also, do not allow flipping if this edge lies on the outer boundary cycle (heuristic: any endpoint is boundary vertex)
        # Compute boundary flags once from edge_map
        boundary_vs = set()
        for e_key, ts in editor.edge_map.items():
            if len(ts) == 1:
                boundary_vs.add(int(e_key[0])); boundary_vs.add(int(e_key[1]))
        if int(edge[0]) in boundary_vs and int(edge[1]) in boundary_vs:
            if stats: stats.fail += 1
            return False, "Flip disabled on boundary edges in virtual-boundary mode", None
    if any(np.all(editor.triangles[i] == -1) for i in tris_idx):
        if stats: stats.fail += 1
        return False, "Edge is not flippable (one triangle tombstoned)", None
    t1, t2 = [editor.triangles[i] for i in tris_idx]
    opp1 = [v for v in t1 if v not in edge][0]
    opp2 = [v for v in t2 if v not in edge][0]
    new_tri1 = [opp1, opp2, edge[0]]
    new_tri2 = [opp1, opp2, edge[1]]
    try:
        a1 = triangle_area(editor.points[int(new_tri1[0])], editor.points[int(new_tri1[1])], editor.points[int(new_tri1[2])])
        a2 = triangle_area(editor.points[int(new_tri2[0])], editor.points[int(new_tri2[1])], editor.points[int(new_tri2[2])])
    except Exception:
        if stats: stats.fail += 1
        return False, "Flip would reference invalid vertex indices", None
    if abs(a1) <= EPS_AREA or abs(a2) <= EPS_AREA:
        if stats: stats.fail += 1
        return False, f"Flip would create inverted or near-zero-area triangle(s): signed areas {a1:.3e}, {a2:.3e}", None
    if a1 <= 0: new_tri1 = [new_tri1[0], new_tri1[2], new_tri1[1]]
    if a2 <= 0: new_tri2 = [new_tri2[0], new_tri2[2], new_tri2[1]]
    ok_q, msg_q = _evaluate_quality_change(
        editor,
        [t1, t2],
        [new_tri1, new_tri2],
        stats,
        op_label="Flip",
        before_kw='before', after_kw='after')
    if not ok_q:
        return False, msg_q, None
    ok_sim, sim_msg = _simulate_preflight(
        editor, editor.points.copy(), tris_idx, [new_tri1, new_tri2], stats,
        reject_msg_prefix="Flip ")
    if not ok_sim:
        return False, sim_msg, None
    for idx in tris_idx:
        editor._remove_triangle_from_maps(idx)
    try:
        editor.triangles[tris_idx[0]] = np.array(new_tri1, dtype=np.int32)
        editor.triangles[tris_idx[1]] = np.array(new_tri2, dtype=np.int32)
    except Exception:
        # revert originals and mark failure
        for idx, t in zip(tris_idx, [t1, t2]):
            editor.triangles[idx] = t
            editor._add_triangle_to_maps(idx)
        if stats: stats.fail += 1
        return False, "Flip failed to write new triangles", None
    # success path
    for idx in tris_idx:
        editor._add_triangle_to_maps(idx)
    if hasattr(editor, '_on_op_committed'):
        try:
            editor._on_op_committed(tombstoned=0, appended=0)
        except Exception:
            pass
    if stats: stats.success += 1
    return True, "Edge flipped (in-place)", None


def op_add_node(editor, point, tri_idx=None):
    stats_fn = getattr(editor, '_get_op_stats', None)
    stats = stats_fn('add_node') if stats_fn else None
    if stats: stats.attempts += 1
    if tri_idx is None:
        if stats: stats.fail += 1
        return False, "Specify tri_idx", None
    ok, _ = check_mesh_conformity(editor.points, editor.triangles)
    if not ok:
        if stats: stats.fail += 1
        return False, "mesh not conforming before add", None
    if tri_idx < 0 or tri_idx >= len(editor.triangles):
        if stats: stats.fail += 1
        return False, "tri_idx out of range", None
    tri = editor.triangles[tri_idx]
    pts_tri = [editor.points[tri[0]], editor.points[tri[1]], editor.points[tri[2]]]
    def sign(p1,p2,p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1 = sign(point, pts_tri[0], pts_tri[1])
    d2 = sign(point, pts_tri[1], pts_tri[2])
    d3 = sign(point, pts_tri[2], pts_tri[0])
    if not (((d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0)) and (d1!=0 and d2!=0 and d3!=0)):
        if stats: stats.fail += 1
        return False, "Point is not strictly inside the triangle (on edge or outside)", None
    new_tris = [
        [tri[0], tri[1], len(editor.points)],
        [tri[1], tri[2], len(editor.points)],
        [tri[2], tri[0], len(editor.points)]
    ]
    cand_pts = np.vstack([editor.points, point])
    new_tris = _orient_tris(cand_pts, new_tris)
    # For add_node we allow quality deterioration (refinement can introduce skinny tris) so we only log.
    # Optional quality observation (no stats mutation). We compute but ignore result.
    try:
        _ = worst_min_angle(editor.points, [tri])
        _ = worst_min_angle(cand_pts, new_tris)
    except Exception:
        pass
    ok_sim, sim_msg = _simulate_preflight(
        editor, cand_pts, [tri_idx], new_tris, stats,
        reject_msg_prefix="Add node ")
    if not ok_sim:
        return False, sim_msg, None
    editor.points = np.vstack([editor.points, point])
    editor._remove_triangle_from_maps(tri_idx)
    editor.triangles[tri_idx] = [-1,-1,-1]
    start = len(editor.triangles)
    new_arr = np.array(new_tris, dtype=np.int32)
    editor.triangles = np.ascontiguousarray(np.vstack([editor.triangles, new_arr]).astype(np.int32))
    for i in range(start, len(editor.triangles)):
        editor._add_triangle_to_maps(i)
    if hasattr(editor, '_on_op_committed'):
        try:
            editor._on_op_committed(tombstoned=1, appended=len(new_tris))
        except Exception:
            pass
    if stats: stats.success += 1
    return True, "Node added in triangle", None


def op_move_vertices_to_barycenter(editor, only_interior: bool = True) -> int:
    """Move vertices to the barycenter of their 1-ring neighbors.

    Parameters
    ----------
    editor : PatchBasedMeshEditor
        The mesh editor instance (points, triangles, maps)
    only_interior : bool, default True
        If True, skip boundary vertices (incident to exactly one triangle edge)

    Returns
    -------
    int
        Number of vertices whose position was updated.

    Notes
    -----
    - Uses a conservative guard: every incident triangle must remain positively
      oriented and above EPS_AREA. If not, the move is reverted for that vertex.
    - Maps (edge_map, v_map) are unchanged; only coordinates move.
    """
    # Build boundary vertex set if needed
    boundary_vs = set()
    if only_interior:
        for e, ts in editor.edge_map.items():
            ts_list = [int(t) for t in ts]
            if len(ts_list) == 1:
                boundary_vs.add(int(e[0])); boundary_vs.add(int(e[1]))
    moved = 0
    for v in range(len(editor.points)):
        if only_interior and v in boundary_vs:
            continue
        # Collect 1-ring neighbors from incident triangles
        tris = list(editor.v_map.get(int(v), [])) if hasattr(editor, 'v_map') else []
        if not tris:
            continue
        nbrs = set()
        for ti in tris:
            t = editor.triangles[int(ti)]
            if np.any(np.array(t) == -1):
                continue
            for u in t:
                uu = int(u)
                if uu != v:
                    nbrs.add(uu)
        if not nbrs:
            continue
        target = np.mean(editor.points[list(nbrs)], axis=0)
        # Guard: ensure incident triangles remain positively oriented and above EPS_AREA
        old = editor.points[int(v)].copy()
        ok = True
        editor.points[int(v)] = target
        for ti in tris:
            t = [int(x) for x in editor.triangles[int(ti)]]
            if -1 in t:
                continue
            pa, pb, pc = editor.points[t[0]], editor.points[t[1]], editor.points[t[2]]
            a = triangle_area(pa, pb, pc)
            if abs(a) <= EPS_AREA:
                ok = False; break
            # enforce positive orientation
            if a < 0:
                ok = False; break
        if ok:
            moved += 1
        else:
            editor.points[int(v)] = old
    return moved


def op_edge_collapse(editor, edge=None, position: str = 'midpoint'):
    """Collapse an edge by replacing its endpoints with a single vertex.

    Strategy: append a new vertex at the chosen position (midpoint), tombstone all
    triangles incident to either endpoint, and append modified copies where the
    endpoint is replaced with the new vertex. Degenerate candidates are dropped.

    This mirrors commit style used by other ops (no immediate compaction).

    Parameters
    ----------
    editor : PatchBasedMeshEditor
    edge : tuple[int,int]
        The edge to collapse (u,v). Must exist in the current mesh (boundary or interior).
    position : str, default 'midpoint'
        Currently supports 'midpoint'.

    Returns
    -------
    (bool, str, dict|None)
        Success flag, message, and info with counts on success.
    """
    stats_fn = getattr(editor, '_get_op_stats', None)
    stats = stats_fn('edge_collapse') if stats_fn else None
    if stats: stats.attempts += 1
    ok, msgs = editor_check(editor)
    if not ok:
        if stats: stats.fail += 1
        return False, "mesh not conforming before collapse", msgs
    if edge is None or len(edge) != 2:
        if stats: stats.fail += 1
        return False, "Specify a valid edge (u,v)", None
    u, v = int(edge[0]), int(edge[1])
    if u == v:
        if stats: stats.fail += 1
        return False, "Edge endpoints identical", None
    key = tuple(sorted((u, v)))
    incident = list(editor.edge_map.get(key, []))
    # Only allow collapsing interior edges (exactly two adjacent triangles)
    if len(incident) != 2:
        if stats: stats.fail += 1
        return False, "Edge is not collapsible (must be interior with exactly two adjacent triangles)", None
    # Decide new vertex position
    if position == 'midpoint':
        p_new = 0.5*(editor.points[u] + editor.points[v])
    else:
        if stats: stats.fail += 1
        return False, f"Unsupported position mode: {position}", None
    new_idx = len(editor.points)
    cand_pts = np.vstack([editor.points, p_new])
    # Gather all triangles incident to u or v (for rewrite), and explicitly mark the two adjacent
    # triangles to (u,v) from edge_map for removal using edge adjacency.
    tris_u = set(int(t) for t in editor.v_map.get(int(u), []))
    tris_v = set(int(t) for t in editor.v_map.get(int(v), []))
    edge_adjacent = set(int(t) for t in incident)  # exactly the two triangles sharing (u,v)
    touched = sorted(tris_u.union(tris_v))
    # Build replacement triangles (drop those containing both u and v and degenerates)
    new_tris = []
    for ti in touched:
        tri = [int(x) for x in editor.triangles[int(ti)]]
        if -1 in tri:
            continue
        has_u = u in tri
        has_v = v in tri
        if has_u and has_v:
            # triangle across the collapsing edge (directly from edge_map) -> tombstone only
            # no replacement; skip adding to new_tris
            continue
        if has_u or has_v:
            rep = [new_idx if x == u or x == v else x for x in tri]
            # Filter degenerate
            if len({int(rep[0]), int(rep[1]), int(rep[2])}) < 3:
                continue
            a = triangle_area(cand_pts[int(rep[0])], cand_pts[int(rep[1])], cand_pts[int(rep[2])])
            if abs(a) <= EPS_AREA:
                continue
            new_tris.append(rep)
    # Orient and quality check against previous local neighborhood
    new_tris = _orient_tris(cand_pts, new_tris)
    old_local = [editor.triangles[int(ti)] for ti in touched if not np.any(np.array(editor.triangles[int(ti)]) == -1)]
    ok_q, msg_q = _evaluate_quality_change(
        editor,
        old_local,
        new_tris,
        stats,
        op_label="Edge collapse",
        candidate_points=cand_pts,
        before_kw='before', after_kw='post')
    if not ok_q:
        return False, msg_q, None
    # Simulated compaction preflight: tombstone all touched, append new_tris
    ok_sim, sim_msg = _simulate_preflight(
        editor, cand_pts, touched, new_tris, stats,
        reject_msg_prefix="Edge collapse ")
    if not ok_sim:
        return False, sim_msg, None
    # Commit: append point, tombstone touched, append replacements
    editor.points = cand_pts
    for idx in touched:
        editor._remove_triangle_from_maps(idx)
        editor.triangles[int(idx)] = [-1, -1, -1]
    if new_tris:
        start = len(editor.triangles)
        new_arr = np.array(new_tris, dtype=np.int32)
        editor.triangles = np.ascontiguousarray(np.vstack([editor.triangles, new_arr]).astype(np.int32))
        for i in range(start, len(editor.triangles)):
            editor._add_triangle_to_maps(i)
    if hasattr(editor, '_on_op_committed'):
        try:
            editor._on_op_committed(tombstoned=len(touched), appended=len(new_tris))
        except Exception:
            pass
    if stats: stats.success += 1
    return True, "edge collapsed", { 'new_vertex': new_idx, 'tombstoned': len(touched), 'appended': len(new_tris) }


def op_remove_node_with_patch(editor, v_idx, force_strict=False):
    """Remove an interior node using the patch/cavity workflow with quality and simulation checks.
    Mirrors legacy implementation in mesh_modifier2.remove_node_with_patch but lives here for modularity.
    Returns (success, message, info_dict).
    """
    import numpy as np  # local to limit namespace pollution
    from .triangulation import optimal_star_triangulation, retriangulate_patch_strict, simplify_polygon_cycle  # extracted utilities

    cavity_tri_indices = sorted(set(editor.v_map.get(int(v_idx), [])))
    editor.logger.debug("[DEBUG] Triangles incidents à %s: %s", v_idx, cavity_tri_indices)
    for t in cavity_tri_indices:
        editor.logger.debug("[DEBUG] Triangle %d: %s", t, editor.triangles[t])
    if not cavity_tri_indices:
        return False, "vertex isolated", None

    cycle = boundary_cycle_from_incident_tris(editor.triangles, cavity_tri_indices, v_idx)
    editor.logger.debug("[DEBUG] Cycle reconstruit autour de %s: %s", v_idx, cycle)
    if cycle is None or len(cycle) < 3:
        neighbors = set()
        for t in cavity_tri_indices:
            for v in editor.triangles[t]:
                if v != v_idx:
                    neighbors.add(v)
        editor.logger.debug("[DEBUG] Voisins de %s: %s", v_idx, sorted(neighbors))
        # If virtual boundary mode is enabled, attempt to reconstruct the cavity boundary
        # by forming the patch boundary polygon as if the border was closed by a virtual node.
        if getattr(editor, 'virtual_boundary_mode', False):
            try:
                from .helpers import boundary_polygons_from_patch, select_outer_polygon
                polys = boundary_polygons_from_patch(editor.triangles, cavity_tri_indices)
                poly = select_outer_polygon(editor.points, polys)
                if poly and len(poly) >= 3:
                    cycle = poly
                    editor.logger.debug("[DEBUG] virtual-boundary: using patch boundary polygon as cycle size=%d", len(cycle))
                else:
                    return False, "cannot determine cavity boundary", None
            except Exception as e:
                editor.logger.debug("virtual-boundary fallback failed: %s", e)
                return False, "cannot determine cavity boundary", None
        else:
            adj = {}
            for t in cavity_tri_indices:
                tri = editor.triangles[t]
                others = [int(x) for x in tri if int(x) != int(v_idx)]
                if len(others) == 2:
                    a, b = others
                    adj.setdefault(a, set()).add(b)
                    adj.setdefault(b, set()).add(a)
            editor.logger.debug("[DEBUG] Adjacence locale: %s", adj)
            return False, "cannot determine cavity boundary", None

    # Metrics / flags (attach to editor lazily)
    # Stats tracking via dataclass on editor
    stats_fn = getattr(editor, '_get_op_stats', None)
    stats = stats_fn('remove_node') if stats_fn else None
    if stats: stats.attempts += 1

    enable_simplify = getattr(editor, 'enable_polygon_simplification', True)

    new_triangles = None
    used_fallback = False
    if not force_strict:
        try:
            # Avoid verbose stdout spam during automated tests / batch runs which previously caused
            # OSError: [Errno 28] No space left on device (excessive captured output).
            new_triangles = optimal_star_triangulation(editor.points, cycle, debug=False)
        except ValueError as e:
            # Handle newly introduced strict polygon validation (duplicates / self-intersections).
            # For internal remove operation we gracefully fallback instead of bubbling the exception.
            editor.logger.debug("optimal_star_triangulation raised ValueError (%s); considering simplification then fallback", e)
            if enable_simplify and stats:
                stats.simplify_attempted += 1
                simplified = simplify_polygon_cycle(editor.points, cycle, cavity_tri_indices, editor.triangles)
                if simplified and simplified != cycle:
                    # sampled logging: log first 5 occurrences, then every simplify_log_every-th
                    count_attempted = stats.simplify_attempted
                    freq = getattr(editor, 'simplify_log_every', 50)
                    if count_attempted <= 5 or (count_attempted % freq) == 0:
                        editor.logger.info("simplify_polygon_cycle adjusted cycle size %d->%d", len(cycle), len(simplified))
                    try:
                        new_triangles = optimal_star_triangulation(editor.points, simplified, debug=False)
                        if new_triangles and stats:
                            stats.simplify_helped += 1
                            cycle = simplified  # adopt simplified cycle for logging clarity
                    except ValueError:
                        new_triangles = None
            # If still None we proceed to fallback below
    if not new_triangles:  # fallback strict retriangulation
        editor.logger.debug("optimal_star_triangulation skipped/failed; trying strict retriangulation fallback")
        try:
            new_points, triangles_new, ok_sub, new_tri_list = retriangulate_patch_strict(
                editor.points, editor.triangles, cavity_tri_indices, new_point_coords=None, strict_mode='centroid')
        except Exception as e:
            editor.logger.debug("Fallback retriangulation raised exception: %s", e)
            return False, "retriangulation failed", None
        if not ok_sub or not new_tri_list:
            return False, "retriangulation failed", None
        new_triangles = [list(t) for t in new_tri_list]
        used_fallback = True
        if stats: stats.fallback_used += 1
    else:
        if stats: stats.star_success += 1

    # Validate candidate locally
    try:
        tmp_tri_full = [tuple(t) for i, t in enumerate(editor.triangles) if i not in cavity_tri_indices and not np.all(np.array(t) == -1)]
        tmp_tri_full.extend(new_triangles)
        tmp_tri_full = np.array(tmp_tri_full, dtype=int) if tmp_tri_full else np.empty((0,3), dtype=int)
        ok_sub, msgs = check_mesh_conformity(editor.points, tmp_tri_full, allow_marked=False)
    except Exception as e:
        return False, f"retriangulation validation error: {e}", None
    if not ok_sub:
        return False, f"retriangulation failed local conformity: {msgs}", None

    # Quality gating (non-worsening worst min-angle)
    ok_q, msg_q = _evaluate_quality_change(
        editor,
        [editor.triangles[int(ti)] for ti in cavity_tri_indices],
        new_triangles,
        stats,
        op_label="retriangulation",
        before_kw='before', after_kw='post')
    if not ok_q:
        editor.logger.debug('remove_node_with_patch: rejecting candidate: %s', msg_q)
        return False, msg_q, None

    # Simulated compaction preflight
    ok_sim, sim_msg = _simulate_preflight(
        editor, editor.points.copy(), cavity_tri_indices, new_triangles, stats,
        reject_msg_prefix="retriangulation ")
    if not ok_sim:
        editor.logger.debug('remove_node_with_patch: simulated compaction rejected candidate: %s', sim_msg)
        return False, sim_msg, None

    # Commit (tombstone cavity, append new triangles)
    for idx in cavity_tri_indices:
        editor._remove_triangle_from_maps(idx)
        editor.triangles[idx] = [-1, -1, -1]
    start = len(editor.triangles)
    new_arr = np.array(new_triangles, dtype=np.int32)
    editor.triangles = np.ascontiguousarray(np.vstack([editor.triangles, new_arr]).astype(np.int32))
    for idx in range(start, len(editor.triangles)):
        editor._add_triangle_to_maps(idx)
    if hasattr(editor, '_on_op_committed'):
        try:
            editor._on_op_committed(tombstoned=len(cavity_tri_indices), appended=len(new_triangles))
        except Exception:
            pass
    if used_fallback:
        editor.logger.debug("remove_node_with_patch: used strict retriangulation fallback; added %d triangles", len(new_triangles))
    if stats: stats.success += 1
    return True, "remove successful (patch)", {'npts': len(editor.points), 'ntri': len(editor.triangles)}


def op_try_fill_pocket(editor, verts, min_tri_area=EPS_AREA, reject_min_angle_deg=None):
    stats_fn = getattr(editor, '_get_op_stats', None)
    stats = stats_fn('fill_pocket') if stats_fn else None
    if stats: stats.attempts += 1
    """Fill a polygonal pocket using (quad | steiner | earclip) strategies.
    Returns (ok, details_dict). Mirrors legacy semantics.
    """
    from .triangulation import fill_pocket_quad, fill_pocket_steiner, fill_pocket_earclip
    details_accum = []
    try:
        if not verts or len(verts) < 3:
            d = {'method': None, 'triangles': [], 'new_point_idx': None, 'failure_reasons': ['invalid_input: less than 3 vertices']}
            if stats: stats.fail += 1
            return False, d
        # Strategy order
        if len(verts) == 4:
            if stats: stats.pocket_quad_attempts += 1
            ok, d = fill_pocket_quad(editor, verts, min_tri_area, reject_min_angle_deg)
            if ok:
                editor.logger.info('try_fill_pocket: filled quad verts=%s via quad diag', verts)
                if stats:
                    stats.success += 1; stats.pocket_quad_success += 1
                if hasattr(editor, '_on_op_committed'):
                    try:
                        # quad produces 2 triangles after tombstoning pocket boundary adjacents in helper
                        editor._on_op_committed(tombstoned=0, appended=len(d.get('triangles', [])))
                    except Exception:
                        pass
                return True, d
            details_accum.append(d)
        if len(verts) > 4:
            if stats: stats.pocket_steiner_attempts += 1
            ok, d = fill_pocket_steiner(editor, verts, min_tri_area, reject_min_angle_deg)
            if ok:
                editor.logger.info('try_fill_pocket: filled polygon verts=%s via Steiner point (fan)', verts)
                if stats:
                    stats.success += 1; stats.pocket_steiner_success += 1
                if hasattr(editor, '_on_op_committed'):
                    try:
                        editor._on_op_committed(tombstoned=0, appended=len(d.get('triangles', [])))
                    except Exception:
                        pass
                return True, d
            details_accum.append(d)
        # Earclip fallback
        if stats: stats.pocket_earclip_attempts += 1
        ok, d = fill_pocket_earclip(editor, verts, min_tri_area, reject_min_angle_deg)
        if ok:
            editor.logger.info('try_fill_pocket: filled polygon verts=%s via earclip', verts)
            if stats:
                stats.success += 1; stats.pocket_earclip_success += 1
            if hasattr(editor, '_on_op_committed'):
                try:
                    editor._on_op_committed(tombstoned=0, appended=len(d.get('triangles', [])))
                except Exception:
                    pass
            return True, d
        details_accum.append(d)
        # Aggregate failure reasons
        agg = {'method': None, 'triangles': [], 'new_point_idx': None,
               'failure_reasons': sum([x.get('failure_reasons', []) for x in details_accum], [])}
        if stats: stats.fail += 1
        return False, agg
    except Exception as e:
        if stats: stats.fail += 1
        editor.logger.debug('try_fill_pocket exception: %s', e)
        return False, {'method': None, 'triangles': [], 'new_point_idx': None, 'failure_reasons': [f'exception: {e}']}

