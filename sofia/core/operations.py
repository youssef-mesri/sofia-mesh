"""Local mesh operations (split, flip, remove, add, pocket fill)."""
from __future__ import annotations
import numpy as np
from .geometry import triangle_area, triangle_angles
from .constants import EPS_AREA, EPS_MIN_ANGLE_DEG
from .quality import worst_min_angle, non_worsening_quality, _triangle_qualities_norm
from .conformity import check_mesh_conformity, simulate_compaction_and_check
import time
from .helpers import boundary_cycle_from_incident_tris

# NOTE: This is a placeholder extraction stub. The full migration from mesh_modifier2
# would proceed incrementally; current operations continue to live in the legacy file.
# Tests still import from mesh_modifier2 for now.

__all__ = [
    'local_quality_ok',
    'op_split_edge',
    'op_split_edge_delaunay',
    'op_remove_node_with_patch',
    'op_remove_node_with_patch2',  # Refactored version with virtual_boundary_mode
    'try_remove_node_strategically',  # New strategy-based removal
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
        ok_sim, sim_msgs = simulate_compaction_and_check(
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
    # Use normalized triangle quality metric (area / sum(edge_len^2) normalized)
    # Compute average quality pre/post and enforce non-worsening within an epsilon.
    pts_old = editor.points
    pts_new = editor.points if candidate_points is None else candidate_points
    try:
        import numpy as _np
        # Build arrays of triangles (may be empty)
        old_arr = _np.array(old_tris, dtype=_np.int32) if old_tris else _np.empty((0,3), dtype=_np.int32)
        new_arr = _np.array(new_tris, dtype=_np.int32) if new_tris else _np.empty((0,3), dtype=_np.int32)
        # If there are no triangles, treat quality as maximal
        if old_arr.size:
            q_old = float(_np.mean(_triangle_qualities_norm(pts_old, old_arr)))
        else:
            q_old = 1.0
        if new_arr.size:
            q_new = float(_np.mean(_triangle_qualities_norm(pts_new, new_arr)))
        else:
            q_new = 1.0
        # Allowed degradation epsilon (in quality units [0,1]) configurable on editor
        eps_q = float(getattr(editor, 'quality_metric_eps', 0.02))
        if q_new < q_old - eps_q:
            if stats:
                stats.quality_rejects += 1; stats.fail += 1
            return False, f"{op_label} would worsen avg-quality ({before_kw}={q_old:.6f} {after_kw}={q_new:.6f})"
    except Exception as e:
        editor.logger.debug("%s quality eval error (fallback to angle): %s", op_label.lower().replace(' ', '_'), e)
        # Conservative fallback to previous worst-min-angle policy
        try:
            pre_mina = worst_min_angle(editor.points, old_tris)
            post_mina = worst_min_angle(pts_new, new_tris)
            if not non_worsening_quality(pre_mina, post_mina):
                if stats:
                    stats.quality_rejects += 1; stats.fail += 1
                return False, f"{op_label} would worsen worst-triangle ({before_kw}={pre_mina:.6f}deg {after_kw}={post_mina:.6f}deg)"
        except Exception:
            pass
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
        appended_arr = np.asarray(oriented_appended, dtype=np.int32)
        # preallocate new array to avoid intermediate cast/copies
        new_tris = np.empty((start + appended_arr.shape[0], 3), dtype=np.int32)
        new_tris[:start, :] = editor.triangles
        new_tris[start:, :] = appended_arr
        editor.triangles = np.ascontiguousarray(new_tris)
        for i in range(start, start + appended_arr.shape[0]):
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


def op_split_edge(editor, edge=None, skip_preflight_check=False, skip_simulation=False):
    """Split an edge by inserting a vertex at its midpoint.
    
    Parameters
    ----------
    editor : PatchBasedMeshEditor
        The mesh editor
    edge : tuple, optional
        The edge to split (u, v)
    skip_preflight_check : bool, default=False
        If True, skip initial conformity check (faster for batch operations)
    skip_simulation : bool, default=False
        If True, skip simulation preflight (faster for batch operations)
        
    Returns
    -------
    (bool, str, dict|None)
        Success flag, message, and info dict
    """
    stats = getattr(editor, '_get_op_stats', None)
    stats = stats('split_midpoint') if stats else None
    if stats: stats.attempts += 1
    
    if not skip_preflight_check:
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
    # Note: Conformity validation is performed later by _simulate_preflight (line ~318)
    # which calls simulate_compaction_and_check. No need to check twice.
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
    if not skip_simulation:
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
    new_arr = np.asarray(new_tris, dtype=np.int32)
    total = len(editor.triangles) + new_arr.shape[0]
    out = np.empty((total, 3), dtype=np.int32)
    out[:len(editor.triangles), :] = editor.triangles
    out[len(editor.triangles):, :] = new_arr
    editor.triangles = np.ascontiguousarray(out)
    for i in range(start, total):
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
    - Only operates on vertices with convex 1-ring patches to prevent creating
      invalid meshes (vertices with non-convex patches are skipped).
    - Uses a conservative guard: every incident triangle must remain positively
      oriented and above EPS_AREA. If not, the move is reverted for that vertex.
    - Maps (edge_map, v_map) are unchanged; only coordinates move.
    """
    def _is_patch_convex(vertex_idx, neighbor_indices, points):
        """Check if the 1-ring patch around a vertex forms a convex polygon.
        
        For interior vertices, the 1-ring neighbors should form a convex polygon
        when ordered angularly around the vertex.
        """
        if len(neighbor_indices) < 3:
            return False
        
        # Order neighbors angularly around the vertex
        center = points[vertex_idx]
        neighbor_pts = points[neighbor_indices]
        angles = np.arctan2(neighbor_pts[:, 1] - center[1], neighbor_pts[:, 0] - center[0])
        sorted_indices = neighbor_indices[np.argsort(angles)]
        sorted_pts = points[sorted_indices]
        
        # Check if the ordered neighbors form a convex polygon
        # Use cross product sign consistency
        n = len(sorted_pts)
        for i in range(n):
            p1 = sorted_pts[i]
            p2 = sorted_pts[(i + 1) % n]
            p3 = sorted_pts[(i + 2) % n]
            # Cross product of (p2-p1) and (p3-p2)
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            if cross < 0:  # Non-convex turn detected
                return False
        return True
    
    # Vectorized boundary detection
    n_pts = len(editor.points)
    boundary_vs = set()
    if only_interior:
        edges = np.array(list(editor.edge_map.keys()), dtype=np.int32)
        edge_tris = [editor.edge_map[tuple(e)] for e in edges]
        is_boundary = np.array([len(ts) == 1 for ts in edge_tris])
        boundary_vs = set(edges[is_boundary].flatten())
    # Precompute neighbors for all vertices using NumPy arrays
    neighbors = [[] for _ in range(n_pts)]
    for ti, tri in enumerate(editor.triangles):
        if -1 in tri: continue
        for i in range(3):
            v = int(tri[i])
            nbrs = [int(tri[j]) for j in range(3) if j != i]
            neighbors[v].extend(nbrs)
    # Convert neighbor lists to NumPy arrays for fast mean computation
    neighbors_np = [np.array(list(set(nbrs)), dtype=np.int32) if nbrs else np.array([], dtype=np.int32) for nbrs in neighbors]
    
    # Filter eligible vertices: must have neighbors and convex 1-ring patch
    eligible = []
    for v in range(n_pts):
        if only_interior and v in boundary_vs:
            continue
        if neighbors_np[v].size == 0:
            continue
        # Check convexity of 1-ring patch
        if not _is_patch_convex(v, neighbors_np[v], editor.points):
            continue
        eligible.append(v)
    eligible = np.array(eligible, dtype=np.int32)
    
    targets = np.copy(editor.points)
    for v in eligible:
        targets[v] = np.mean(editor.points[neighbors_np[v]], axis=0)
    # Batch triangle validity checks using NumPy
    moved = 0
    old_points = editor.points.copy()
    editor.points[eligible] = targets[eligible]
    # Collect all affected triangles and build mapping from vertex to incident triangles
    affected_tris = set()
    v_to_tris = [[] for _ in range(n_pts)]
    for v in eligible:
        tris = list(editor.v_map.get(int(v), [])) if hasattr(editor, 'v_map') else []
        v_to_tris[v] = tris
        affected_tris.update(tris)
    affected_tris = np.array(sorted(affected_tris), dtype=np.int32)
    # Compute areas for all affected triangles
    from sofia.core.geometry import triangles_signed_areas
    tris_arr = np.array([editor.triangles[int(ti)] for ti in affected_tris if not np.any(np.array(editor.triangles[int(ti)]) == -1)], dtype=np.int32)
    valid_mask = np.array([not np.any(tri == -1) for tri in tris_arr], dtype=bool)
    areas = triangles_signed_areas(editor.points, tris_arr)
    # Map triangle index to area
    tri_idx_to_area = dict(zip([ti for i, ti in enumerate(affected_tris) if valid_mask[i]], areas))
    # For each eligible vertex, check if any incident triangle is invalid
    revert_mask = np.zeros(eligible.shape, dtype=bool)
    for idx, v in enumerate(eligible):
        tris = v_to_tris[v]
        for ti in tris:
            if ti not in tri_idx_to_area:
                continue
            a = tri_idx_to_area[ti]
            if abs(a) <= EPS_AREA or a < 0:
                revert_mask[idx] = True; break
    # Batch revert invalid vertices
    editor.points[eligible[revert_mask]] = old_points[eligible[revert_mask]]
    moved = int(np.sum(~revert_mask))
    return moved


def op_edge_collapse(editor, edge=None, position: str = 'midpoint', skip_preflight_check=False, skip_simulation=False):
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
    skip_preflight_check : bool, default=False
        If True, skip initial conformity check (faster for batch operations)
    skip_simulation : bool, default=False
        If True, skip simulation preflight (faster for batch operations)

    Returns
    -------
    (bool, str, dict|None)
        Success flag, message, and info with counts on success.
    """
    stats_fn = getattr(editor, '_get_op_stats', None)
    stats = stats_fn('edge_collapse') if stats_fn else None
    if stats: stats.attempts += 1
    
    if not skip_preflight_check:
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
    
    # Explicit edge crossing check (optional, controlled by editor flag)
    if getattr(editor, 'reject_crossing_edges', False):
        try:
            # Build test mesh with tombstoned triangles removed and new triangles added
            # Optimized: use numpy boolean indexing
            mask = np.ones(len(editor.triangles), dtype=bool)
            if touched:
                mask[list(touched)] = False
            mask &= (editor.triangles[:, 0] != -1)
            active_tris = editor.triangles[mask]
            test_tris_arr = np.vstack([active_tris, new_tris]) if mask.any() and len(new_tris) > 0 else (
                active_tris if mask.any() else (np.array(new_tris, dtype=np.int32) if len(new_tris) > 0 else np.empty((0, 3), dtype=np.int32))
            )
        
            # Check for edge crossings using simulation
            ok_cross, cross_msgs = simulate_compaction_and_check(
                cand_pts, test_tris_arr, reject_crossing_edges=True
            )
            if not ok_cross:
                if stats:
                    stats.fail += 1
                crossing_msgs = [m for m in cross_msgs if 'crossing' in m.lower()] or cross_msgs
                return False, f"edge collapse would create crossing edges: {crossing_msgs}", None
        except Exception as e:
            if stats:
                stats.fail += 1
            return False, f"edge crossing validation error: {e}", None
    
    # Simulated compaction preflight: tombstone all touched, append new_tris
    if not skip_simulation:
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


def try_remove_node_strategically(editor, v_idx, config=None):
    """Attempt to remove a node using the strategy pattern from Phase 4.
    
    This is a cleaner, more maintainable version of node removal that uses
    the extracted components from the refactoring phases:
    - Phase 1: CavityInfo and extract_removal_cavity()
    - Phase 2: AreaPreservationChecker  
    - Phase 4: RemovalTriangulationStrategy classes
    
    Args:
        editor: PatchBasedMeshEditor instance
        v_idx: Vertex index to remove
        config: Optional BoundaryRemoveConfig
        
    Returns:
        tuple: (success: bool, message: str, info: dict or None)
        
    Note: This is a demonstration of the refactored approach. The full
    integration into op_remove_node_with_patch would happen in a later phase.
    """
    from .helpers import extract_removal_cavity, filter_cycle_vertex
    from .quality import AreaPreservationChecker
    from .triangulation import (
        OptimalStarStrategy,
        SimplifyAndRetryStrategy,
        AreaPreservingStarStrategy,
        QualityStarStrategy,
        EarClipStrategy,
        ChainedStrategy
    )
    from .config import BoundaryRemoveConfig
    
    # Use config or default
    if config is None:
        config = BoundaryRemoveConfig()
    
    # Phase 1: Extract cavity info
    cavity_info = extract_removal_cavity(editor, v_idx)
    if not cavity_info.ok:
        return False, cavity_info.error, None
    
    # Filter out the removed vertex from the cycle
    cycle = filter_cycle_vertex(cavity_info.cycle, v_idx)
    if len(cycle) < 3:
        return False, "boundary cycle too small after filtering", None
    
    # Phase 2: Check area preservation requirements
    area_checker = AreaPreservationChecker(config)
    
    # Phase 4: Build strategy chain based on config
    strategies = []
    
    # Try optimal star first (fast and usually good)
    strategies.append(OptimalStarStrategy())
    
    # Try simplified polygon with optimal star
    if getattr(editor, 'enable_polygon_simplification', True):
        strategies.append(SimplifyAndRetryStrategy(OptimalStarStrategy()))
    
    # Try area-preserving star if required/preferred
    if config.prefer_area_preserving_star:
        strategies.append(AreaPreservingStarStrategy())
    
    # Try quality star if preferred
    if config.prefer_worst_angle_star:
        strategies.append(QualityStarStrategy())
    
    # Ear clip as last resort
    strategies.append(EarClipStrategy())
    
    # Chain all strategies
    strategy = ChainedStrategy(strategies)
    
    # Try triangulation
    success, triangles, error = strategy.try_triangulate(
        editor.points, cycle, config
    )
    
    if not success:
        return False, f"all triangulation strategies failed: {error}", None
    
    # Check area preservation if required
    if config.require_area_preservation and cavity_info.removed_area is not None:
        from .geometry import compute_triangulation_area
        candidate_area = compute_triangulation_area(
            editor.points, np.array(triangles), list(range(len(triangles)))
        )
        ok, area_error = area_checker.check(cavity_info.removed_area, candidate_area)
        if not ok:
            return False, f"area preservation failed: {area_error}", None
    
    # Validate conformity
    try:
        # Temporarily remove old triangles and add new ones
        # Optimized: use numpy boolean indexing
        mask = np.ones(len(editor.triangles), dtype=bool)
        if cavity_info.cavity_indices:
            mask[list(cavity_info.cavity_indices)] = False
        mask &= (editor.triangles[:, 0] != -1)
        test_tris_arr = np.vstack([editor.triangles[mask], triangles]) if mask.any() and len(triangles) > 0 else (
            editor.triangles[mask] if mask.any() else (np.array(triangles, dtype=np.int32) if len(triangles) > 0 else np.empty((0, 3), dtype=np.int32))
        )
        
        from .conformity import check_mesh_conformity
        ok, msgs = check_mesh_conformity(editor.points, test_tris_arr, allow_marked=False)
        if not ok:
            return False, f"conformity check failed: {msgs}", None
    except Exception as e:
        return False, f"conformity validation error: {e}", None
    
    # Commit the changes
    try:
        # Remove old triangles
        for idx in cavity_info.cavity_indices:
            editor._remove_triangle_from_maps(idx)
            editor.triangles[idx] = [-1, -1, -1]
        
        # Add new triangles
        start_idx = len(editor.triangles)
        tri_array = np.array(triangles, dtype=np.int32)
        editor.triangles = np.ascontiguousarray(
            np.vstack([editor.triangles, tri_array]).astype(np.int32)
        )
        
        for idx in range(start_idx, len(editor.triangles)):
            editor._add_triangle_to_maps(idx)
        
        # Notify editor
        if hasattr(editor, '_on_op_committed'):
            try:
                editor._on_op_committed(
                    tombstoned=len(cavity_info.cavity_indices),
                    appended=len(triangles)
                )
            except Exception:
                pass
        
        return True, "node removed successfully", {
            'removed_vertex': v_idx,
            'cavity_size': len(cavity_info.cavity_indices),
            'new_triangles': len(triangles)
        }
    except Exception as e:
        return False, f"failed to commit changes: {e}", None


def _handle_virtual_boundary_mode(editor, v_idx, cavity_info, stats=None):
    """Handle special virtual boundary mode logic.
    
    In virtual_boundary_mode, boundary vertices can be removed. This function:
    1. Attempts alternate cavity extraction if standard extraction fails for boundary vertices
    2. Sanitizes the cycle by removing v_idx from it
    3. Handles degenerate boundary corners (< 3 neighbors) with deletion-only
    
    Args:
        editor: PatchBasedMeshEditor instance
        v_idx: Vertex index to remove
        cavity_info: CavityInfo from extract_removal_cavity()
        stats: Optional stats object for tracking
    
    Returns:
        tuple: (ok, cavity_info_or_none, error_message)
            - If ok=True: Returns (True, updated_cavity_info, "")
            - If ok=False: Returns (False, None, error_message)
            - If deletion_only: Returns (True, None, "deletion_only") with committed changes
    """
    from .helpers import CavityInfo, boundary_polygons_from_patch, select_outer_polygon, filter_cycle_vertex
    from .geometry import triangle_area
    from .triangulation import polygon_signed_area
    from .conformity import check_mesh_conformity
    
    # If cavity extraction failed but we're in virtual_boundary_mode, try alternate approach
    if not cavity_info.ok and "boundary" in cavity_info.error.lower():
        # For boundary vertices in virtual mode, manually build the cavity
        editor.logger.debug("[DEBUG] virtual-boundary: standard extraction failed, using alternate approach")
        
        cavity_tri_indices = sorted(set(editor.v_map.get(int(v_idx), [])))
        if not cavity_tri_indices:
            if stats:
                stats.remove_early_rejects += 1
            return False, None, "vertex isolated"
        
        # Try to build boundary polygon from patch
        try:
            polys = boundary_polygons_from_patch(editor.triangles, cavity_tri_indices)
            cycle = select_outer_polygon(editor.points, polys)
            
            if not cycle or len(cycle) < 3:
                # Fallback: construct neighbor cycle manually
                neighbors = []
                for t in cavity_tri_indices:
                    tri = [int(x) for x in editor.triangles[int(t)] if int(x) != int(v_idx) and int(x) >= 0]
                    for nv in tri:
                        if nv not in neighbors:
                            neighbors.append(nv)
                
                if len(neighbors) < 2:
                    if stats:
                        stats.remove_early_rejects += 1
                        stats.remove_early_boundary += 1
                    return False, None, "cannot determine cavity boundary"
                
                # Order neighbors angularly
                pts_neighbors = editor.points[neighbors]
                centroid = pts_neighbors.mean(axis=0)
                angles = np.arctan2(pts_neighbors[:,1]-centroid[1], pts_neighbors[:,0]-centroid[0])
                cycle = [nbr for _, nbr in sorted(zip(angles, neighbors))]
            
            # Compute removed area
            removed_area = 0.0
            for ti in cavity_tri_indices:
                tri = editor.triangles[int(ti)]
                p0, p1, p2 = editor.points[tri[0]], editor.points[tri[1]], editor.points[tri[2]]
                removed_area += abs(triangle_area(p0, p1, p2))
            
            # Build synthetic CavityInfo
            cavity_info = CavityInfo(
                ok=True,
                cavity_indices=cavity_tri_indices,
                cycle=cycle,
                removed_area=removed_area,
                error=""
            )
            editor.logger.debug("[DEBUG] virtual-boundary: alternate extraction succeeded, cycle size=%d", len(cycle))
        
        except Exception as e:
            editor.logger.debug("virtual-boundary alternate approach failed: %s", e)
            if stats:
                stats.remove_early_rejects += 1
                stats.remove_early_exception += 1
                stats.remove_early_boundary += 1
            return False, None, f"boundary extraction failed: {e}"
    
    # Standard failure path
    if not cavity_info.ok:
        return False, None, cavity_info.error
    
    # In virtual boundary mode, remove v_idx from the cycle and treat it as a closed polygon
    editor.logger.debug("[DEBUG] virtual-boundary mode: sanitizing cycle")
    
    # Filter out the removed vertex from cycle
    cycle = filter_cycle_vertex(cavity_info.cycle, v_idx)
    
    if len(cycle) < 3:
        # Degenerate case: boundary corner with < 3 neighbors after filtering
        editor.logger.debug("[DEBUG] virtual-boundary: degenerate corner (cycle size %d). Deletion only.", len(cycle))
        
        # Validate conformity with just deletions (no new triangles)
        try:
            # Optimized: use numpy boolean indexing
            mask = np.ones(len(editor.triangles), dtype=bool)
            if cavity_info.cavity_indices:
                mask[list(cavity_info.cavity_indices)] = False
            mask &= (editor.triangles[:, 0] != -1)
            tmp_tri_full = editor.triangles[mask] if mask.any() else np.empty((0, 3), dtype=np.int32)
            ok_sub, msgs = check_mesh_conformity(editor.points, tmp_tri_full, allow_marked=False)
        except Exception as e:
            if stats:
                stats.remove_early_rejects += 1
            return False, None, f"conformity validation error: {e}"
        
        if not ok_sub:
            if stats:
                stats.remove_early_rejects += 1
            return False, None, f"deletion-only failed conformity: {msgs}"
        
        # Commit deletion-only
        for idx in cavity_info.cavity_indices:
            editor._remove_triangle_from_maps(idx)
            editor.triangles[idx] = [-1, -1, -1]
        
        if hasattr(editor, '_on_op_committed'):
            try:
                editor._on_op_committed(tombstoned=len(cavity_info.cavity_indices), appended=0)
            except Exception:
                pass
        
        if stats:
            stats.success += 1
        
        # Return special marker indicating deletion-only was committed
        return True, None, "deletion_only"
    
    # Update cavity_info with sanitized cycle
    cavity_info = CavityInfo(
        ok=True,
        cavity_indices=cavity_info.cavity_indices,
        cycle=cycle,
        removed_area=cavity_info.removed_area,
        error=""
    )
    
    # Compute polygon area for logging/diagnostics
    try:
        poly_target_area = abs(polygon_signed_area([editor.points[int(v)] for v in cycle]))
        editor.logger.debug("[DEBUG] virtual-boundary: sanitized polygon area=%.6e", poly_target_area)
    except Exception:
        pass
    
    return True, cavity_info, ""


def op_remove_node_with_patch2(editor, v_idx, force_strict=False):
    """Remove an interior node using refactored components with virtual_boundary_mode support.
    
    This is an improved version of op_remove_node_with_patch that uses the extracted
    components from the refactoring phases while maintaining full compatibility with
    virtual_boundary_mode and other advanced features.
    
    Args:
        editor: PatchBasedMeshEditor instance
        v_idx: Vertex index to remove
        force_strict: If True, force strict retriangulation fallback
        
    Returns:
        tuple: (success: bool, message: str, info: dict or None)
    """
    import numpy as np
    from .helpers import (
        extract_removal_cavity, 
        filter_cycle_vertex,
        boundary_polygons_from_patch,
        select_outer_polygon
    )
    from .quality import AreaPreservationChecker
    from .geometry import compute_triangulation_area
    from .triangulation import (
        OptimalStarStrategy,
        SimplifyAndRetryStrategy,
        AreaPreservingStarStrategy,
        QualityStarStrategy,
        EarClipStrategy,
        ChainedStrategy,
        polygon_signed_area
    )
    from .config import BoundaryRemoveConfig
    from .conformity import check_mesh_conformity
    
    # Get stats tracking
    stats_fn = getattr(editor, '_get_op_stats', None)
    stats = stats_fn('remove_node') if stats_fn else None
    
    # Get configuration
    config = getattr(editor, 'boundary_remove_config', None)
    if config is None:
        config = BoundaryRemoveConfig()
    
    # Check if virtual_boundary_mode is enabled
    virtual_boundary_mode = getattr(editor, 'virtual_boundary_mode', False)
    
    # Phase 1: Extract cavity info
    cavity_info = extract_removal_cavity(editor, v_idx)
    
    # Phase 2: Handle virtual boundary mode if enabled
    if virtual_boundary_mode:
        vb_ok, vb_cavity_info, vb_error = _handle_virtual_boundary_mode(editor, v_idx, cavity_info, stats)
        
        if not vb_ok:
            # Virtual boundary handling failed
            if stats:
                stats.remove_early_rejects += 1
                if "isolated" in vb_error.lower():
                    pass  # No specific counter for isolated
                elif "boundary" in vb_error.lower():
                    stats.remove_early_boundary += 1
            return False, vb_error, None
        
        if vb_error == "deletion_only":
            # Deletion-only case was handled and committed inside the helper
            return True, "remove successful (deletion-only boundary corner)", {
                'npts': len(editor.points),
                'ntri': len(editor.triangles),
                'removed_vertex': v_idx,
                'cavity_size': len(cavity_info.cavity_indices),
                'new_triangles': 0
            }
        
        # Use updated cavity_info from virtual boundary handling
        cavity_info = vb_cavity_info
    else:
        # Standard non-virtual mode: check cavity extraction
        if not cavity_info.ok:
            if stats:
                stats.remove_early_rejects += 1
                if "isolated" in cavity_info.error.lower():
                    pass  # No specific counter for isolated
                elif "boundary" in cavity_info.error.lower():
                    stats.remove_early_boundary += 1
            return False, cavity_info.error, None
    
    if stats:
        stats.attempts += 1
    
    # Extract cycle and area
    cycle = cavity_info.cycle
    removed_area = cavity_info.removed_area
    
    editor.logger.debug("[DEBUG] Cavity extraction: %d triangles, cycle size %d", 
                       len(cavity_info.cavity_indices), len(cycle) if cycle else 0)
    
    # Check if cycle is valid
    if cycle is None or len(cycle) < 3:
        if stats:
            stats.remove_early_rejects += 1
            stats.remove_early_boundary += 1
        return False, "boundary cycle too small", None
    
    # Phase 2: Setup area preservation checker
    area_checker = AreaPreservationChecker(config)
    
    # Phase 4: Build triangulation strategy chain
    strategies = []
    
    # Try optimal star first
    strategies.append(OptimalStarStrategy())
    
    # Try with polygon simplification if enabled
    if getattr(editor, 'enable_polygon_simplification', True):
        strategies.append(SimplifyAndRetryStrategy(OptimalStarStrategy()))
    
    # Try area-preserving star if configured
    if getattr(config, 'prefer_area_preserving_star', False):
        strategies.append(AreaPreservingStarStrategy())
    
    # Try quality star if configured
    if getattr(config, 'prefer_worst_angle_star', False):
        strategies.append(QualityStarStrategy())
    
    # Ear clip as last resort
    strategies.append(EarClipStrategy())
    
    strategy = ChainedStrategy(strategies)
    
    # Try triangulation
    success, triangles, error = strategy.try_triangulate(editor.points, cycle, config)
    
    if not success:
        if stats:
            stats.remove_early_rejects += 1
        return False, f"all triangulation strategies failed: {error}", None
    
    # Phase 3: Check area preservation if required
    if config.require_area_preservation and removed_area is not None:
        candidate_area = compute_triangulation_area(
            editor.points, np.array(triangles), list(range(len(triangles)))
        )
        ok, area_error = area_checker.check(removed_area, candidate_area)
        if not ok:
            if stats:
                stats.remove_early_rejects += 1
            return False, f"area preservation failed: {area_error}", None
    
    # Validate conformity including edge crossings
    try:
        from .conformity import simulate_compaction_and_check
        
        # Optimized: use numpy boolean indexing
        mask = np.ones(len(editor.triangles), dtype=bool)
        if cavity_info.cavity_indices:
            mask[list(cavity_info.cavity_indices)] = False
        mask &= (editor.triangles[:, 0] != -1)
        test_tris_arr = np.vstack([editor.triangles[mask], triangles]) if mask.any() and len(triangles) > 0 else (
            editor.triangles[mask] if mask.any() else (np.array(triangles, dtype=np.int32) if len(triangles) > 0 else np.empty((0, 3), dtype=np.int32))
        )
        
        # First check basic conformity
        ok, msgs = check_mesh_conformity(editor.points, test_tris_arr, allow_marked=False)
        if not ok:
            if stats:
                stats.remove_early_rejects += 1
            return False, f"conformity check failed: {msgs}", None
        
        # Then check for edge crossings using simulation (optional, controlled by editor flag)
        if getattr(editor, 'reject_crossing_edges', False):
            ok_sim, sim_msgs = simulate_compaction_and_check(
                editor.points, test_tris_arr, reject_crossing_edges=True
            )
            if not ok_sim:
                if stats:
                    stats.remove_early_rejects += 1
                    stats.remove_early_validation += 1
                # Filter for crossing messages
                crossing_msgs = [m for m in sim_msgs if 'crossing' in m.lower()] or sim_msgs
                return False, f"edge crossing detected: {crossing_msgs}", None
    except Exception as e:
        if stats:
            stats.remove_early_rejects += 1
            stats.remove_early_exception += 1
        return False, f"conformity validation error: {e}", None
    
    # Commit the changes
    try:
        # Remove old triangles
        for idx in cavity_info.cavity_indices:
            editor._remove_triangle_from_maps(idx)
            editor.triangles[idx] = [-1, -1, -1]
        
        # Add new triangles
        start_idx = len(editor.triangles)
        tri_array = np.array(triangles, dtype=np.int32)
        editor.triangles = np.ascontiguousarray(
            np.vstack([editor.triangles, tri_array]).astype(np.int32)
        )
        
        for idx in range(start_idx, len(editor.triangles)):
            editor._add_triangle_to_maps(idx)
        
        # Notify editor
        if hasattr(editor, '_on_op_committed'):
            try:
                editor._on_op_committed(
                    tombstoned=len(cavity_info.cavity_indices),
                    appended=len(triangles)
                )
            except Exception:
                pass
        
        if stats:
            stats.success += 1
        
        return True, "node removed successfully", {
            'removed_vertex': v_idx,
            'cavity_size': len(cavity_info.cavity_indices),
            'new_triangles': len(triangles),
            'npts': len(editor.points),
            'ntri': len(editor.triangles)
        }
    except Exception as e:
        if stats:
            stats.remove_early_exception += 1
        return False, f"failed to commit changes: {e}", None


def op_remove_node_with_patch(editor, v_idx, force_strict=False):
    """Remove an interior node using the patch/cavity workflow with quality and simulation checks.
    Mirrors legacy implementation in mesh_modifier2.remove_node_with_patch but lives here for modularity.
    Returns (success, message, info_dict).
    """
    import numpy as np  # local to limit namespace pollution
    from .triangulation import optimal_star_triangulation, best_star_triangulation_by_min_angle, ear_clip_triangulation, retriangulate_patch_strict, simplify_polygon_cycle  # extracted utilities

    cavity_tri_indices = sorted(set(editor.v_map.get(int(v_idx), [])))
    editor.logger.debug("[DEBUG] Triangles incidents à %s: %s", v_idx, cavity_tri_indices)
    for t in cavity_tri_indices:
        editor.logger.debug("[DEBUG] Triangle %d: %s", t, editor.triangles[t])
    # Metrics / flags (attach to editor lazily) - define before any early returns/branches use it
    stats_fn = getattr(editor, '_get_op_stats', None)
    stats = stats_fn('remove_node') if stats_fn else None
    if not cavity_tri_indices:
        if stats:
            stats.remove_early_rejects += 1
        return False, "vertex isolated", None

    cycle = boundary_cycle_from_incident_tris(editor.triangles, cavity_tri_indices, v_idx)
    editor.logger.debug("[DEBUG] Cycle reconstruit autour de %s: %s", v_idx, cycle)

    if stats: stats.attempts += 1
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
                    # Fallback: construct neighbor cycle manually (generic boundary removal)
                    neighbors = []
                    for t in cavity_tri_indices:
                        tri = [int(x) for x in editor.triangles[int(t)] if int(x) != int(v_idx) and int(x) >= 0]
                        for nv in tri:
                            if nv not in neighbors:
                                neighbors.append(nv)
                    if len(neighbors) < 2:
                        if stats:
                            stats.remove_early_rejects += 1
                            stats.remove_early_boundary += 1
                        return False, "cannot determine cavity boundary", None
                    # Order neighbors angularly around centroid for polygon fan
                    pts_neighbors = editor.points[neighbors]
                    centroid = pts_neighbors.mean(axis=0)
                    angles = np.arctan2(pts_neighbors[:,1]-centroid[1], pts_neighbors[:,0]-centroid[0])
                    ordered = [nbr for _, nbr in sorted(zip(angles, neighbors))]
                    if len(ordered) >= 3:
                        cycle = ordered
                        editor.logger.debug("[DEBUG] virtual-boundary: constructed neighbor cycle size=%d", len(cycle))
                    else:
                        if stats:
                            stats.remove_early_rejects += 1
                            stats.remove_early_boundary += 1
                        return False, "cannot determine cavity boundary", None
            except Exception as e:
                editor.logger.debug("virtual-boundary fallback failed: %s", e)
                if stats:
                    stats.remove_early_rejects += 1
                    stats.remove_early_exception += 1
                    stats.remove_early_boundary += 1
                return False, "cannot determine cavity boundary", None

    # Pre-compute area of cavity (sum of absolute areas of triangles to be removed)
    try:
        removed_area = 0.0
        for ti in cavity_tri_indices:
            tri = editor.triangles[int(ti)]
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            p0, p1, p2 = editor.points[a], editor.points[b], editor.points[c]
            removed_area += abs(triangle_area(p0, p1, p2))
    except Exception:
        removed_area = None

    # Early robust area-preservation guard: build the patch boundary polygon around v_idx
    # and compare its sanitized area (excluding v_idx) to the removed cavity area. If they
    # differ and area preservation is required, reject immediately.
    # Note: In virtual_boundary_mode, skip this check because the polygon area and cavity area
    # represent different regions (the polygon is the sanitized boundary, while the cavity includes
    # triangles incident to the removed vertex). The second area check after virtual-boundary
    # filtering will handle this case correctly.
    try:
        if not getattr(editor, 'virtual_boundary_mode', False):
            cfg0 = getattr(editor, 'boundary_remove_config', None)
            # Default to requiring area preservation when config is None
            require_area_pres0 = True if (cfg0 is None) else bool(getattr(cfg0, 'require_area_preservation', True))
            if require_area_pres0:
                from .helpers import boundary_polygons_from_patch, select_outer_polygon
                from .triangulation import polygon_signed_area as _poly_area0
                polys0 = boundary_polygons_from_patch(editor.triangles, cavity_tri_indices)
                cyc0 = select_outer_polygon(editor.points, polys0)
                if cyc0 and removed_area is not None:
                    filtered0 = [int(v) for v in cyc0 if int(v) != int(v_idx)]
                    if len(filtered0) >= 2 and filtered0[0] == filtered0[-1]:
                        filtered0 = filtered0[:-1]
                    if len(filtered0) >= 3:
                        poly_area0 = abs(_poly_area0([editor.points[int(v)] for v in filtered0]))
                        from .constants import EPS_TINY, EPS_AREA
                        tol_rel0 = float(getattr(cfg0, 'area_tol_rel', EPS_TINY)) if cfg0 is not None else float(EPS_TINY)
                        tol_abs0 = (float(getattr(cfg0, 'area_tol_abs_factor', 4.0)) if cfg0 is not None else 4.0) * float(EPS_AREA)
                        if abs(poly_area0 - removed_area) > max(tol_abs0, tol_rel0*max(1.0, removed_area)):
                            if stats:
                                stats.remove_early_rejects += 1
                            return False, (
                                f"cavity area changed: poly={poly_area0:.6e} cavity={removed_area:.6e}"
                            ), None
    except Exception:
        pass

    # In virtual boundary mode we conceptually remove v_idx and connect its neighbors (treat border as closed by virtual node)
    if getattr(editor, 'virtual_boundary_mode', False) and cycle:
        filtered = []
        for c in cycle:
            if int(c) == int(v_idx):
                continue
            if filtered and filtered[-1] == c:
                continue
            filtered.append(int(c))
        if len(filtered) >= 2 and filtered[0] == filtered[-1]:
            filtered = filtered[:-1]
        if len(filtered) >= 3:
            editor.logger.debug("[DEBUG] virtual-boundary: sanitized cycle removed vertex %s -> size %d", v_idx, len(filtered))
            cycle = filtered
            # Pre-compute polygon target area for diagnostics (fallback only)
            try:
                from .triangulation import polygon_signed_area as _poly_area
                poly_target_area = abs(_poly_area([editor.points[int(v)] for v in cycle]))
            except Exception:
                poly_target_area = None
            # Note: We intentionally do NOT perform an early area-preservation check here in virtual_boundary_mode,
            # because poly_target_area (the sanitized boundary polygon area) and removed_area (the cavity triangle area)
            # represent different geometric regions and should not be expected to match. The proper area check happens
            # later when we compare the area of the new candidate triangulation with removed_area.
        else:
            # Degenerate boundary corner removal: after excluding v_idx, there are fewer than 3 neighbors.
            # In this case, simply remove incident triangles and do not add any new triangles.
            editor.logger.debug("[DEBUG] virtual-boundary: degenerate boundary cavity after removing %s (cycle size %d). Proceeding with deletion only.", v_idx, len(filtered))
            cycle = filtered  # keep for logging; will yield zero new triangles
            new_triangles = []
            used_fallback = False
            # Validate local conformity of the mesh with incident triangles tombstoned
            try:
                # Optimized: use numpy boolean indexing
                mask = np.ones(len(editor.triangles), dtype=bool)
                if cavity_tri_indices:
                    mask[list(cavity_tri_indices)] = False
                mask &= (editor.triangles[:, 0] != -1)
                tmp_tri_full = editor.triangles[mask] if mask.any() else np.empty((0, 3), dtype=np.int32)
                ok_sub, msgs = check_mesh_conformity(editor.points, tmp_tri_full, allow_marked=False)
            except Exception as e:
                return False, f"retriangulation validation error: {e}", None
            if not ok_sub:
                if stats:
                    stats.remove_early_rejects += 1
                return False, f"retriangulation failed local conformity: {msgs}", None
            # Simulated compaction preflight (no new tris)
            ok_sim, sim_msg = _simulate_preflight(
                editor, editor.points, cavity_tri_indices, [], stats,
                reject_msg_prefix="retriangulation ")
            if not ok_sim:
                editor.logger.debug('remove_node_with_patch: simulated compaction rejected deletion-only candidate: %s', sim_msg)
                # simulation_rejects already counted inside _simulate_preflight
                return False, sim_msg, None
            # Commit: tombstone cavity, append nothing
            for idx in cavity_tri_indices:
                editor._remove_triangle_from_maps(idx)
                editor.triangles[idx] = [-1, -1, -1]
            if hasattr(editor, '_on_op_committed'):
                try:
                    editor._on_op_committed(tombstoned=len(cavity_tri_indices), appended=0)
                except Exception:
                    pass
            if stats: stats.success += 1
            return True, "remove successful (deletion-only boundary corner)", {'npts': len(editor.points), 'ntri': len(editor.triangles)}

    # Metrics / flags were initialized above

    enable_simplify = getattr(editor, 'enable_polygon_simplification', True)

    new_triangles = None
    used_fallback = False
    from collections import OrderedDict
    from hashlib import blake2b
    # Configurable caps (per-operation, can be tuned on editor)
    _CAND_CACHE_MAX = int(getattr(editor, 'remove_candidate_cache_max', 128))
    _PREFLIGHT_CACHE_MAX = int(getattr(editor, 'remove_preflight_cache_max', 256))
    # LRU caches (kept local to this op invocation)
    _candidate_edges_cache = OrderedDict()
    _preflight_cross_cache = OrderedDict()

    def _hash_edges_iter(edges_iter):
        """Compute compact hash for an iterable of (a,b) edges (already normalized).

        Returns hex digest string (8-byte blake2b) or empty string for no edges.
        """
        try:
            it = list(edges_iter)
            if not it:
                return ''
            s = '|'.join(f"{int(a)},{int(b)}" for a, b in it)
            return blake2b(s.encode('ascii'), digest_size=8).hexdigest()
        except Exception:
            # fallback: use Python hash of tuple
            try:
                return str(hash(tuple(edges_iter)))
            except Exception:
                return ''
    # Normalized triangle quality metric: q = (area / sum(edge_len^2)) normalized to [0,1]
    def _triangle_qualities_norm(points_arr, tris_arr):
        """Return array of normalized qualities for each triangle in tris_arr.

        points_arr: (N,2) array-like; tris_arr: (M,3) int array
        Quality = (area) / (sum(edge_len^2)) scaled so equilateral -> 1.
        """
        try:
            pts = _np.asarray(points_arr, dtype=_np.float64)
            tris = _np.asarray(tris_arr, dtype=_np.int32)
            if tris.size == 0:
                return _np.empty((0,), dtype=_np.float64)
            p0 = pts[tris[:, 0]]
            p1 = pts[tris[:, 1]]
            p2 = pts[tris[:, 2]]
            # signed areas
            a = 0.5 * _np.abs((p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0]))
            # squared edge lengths
            e0 = _np.sum((p1 - p0)**2, axis=1)
            e1 = _np.sum((p2 - p1)**2, axis=1)
            e2 = _np.sum((p0 - p2)**2, axis=1)
            denom = e0 + e1 + e2
            # avoid div by zero
            safe = denom > 0
            q = _np.zeros(tris.shape[0], dtype=_np.float64)
            q[safe] = a[safe] / denom[safe]
            # normalize so equilateral triangle has quality 1
            # For equilateral: a = sqrt(3)/4 * s^2, denom = 3*s^2 -> ratio = sqrt(3)/12
            norm_factor = 12.0 / (_np.sqrt(3.0))
            q = q * norm_factor
            # clamp to [0,1]
            q = _np.clip(q, 0.0, 1.0)
            return q
        except Exception:
            # fallback scalar
            out = []
            for tri in tris_arr:
                try:
                    p0 = points_arr[int(tri[0])]; p1 = points_arr[int(tri[1])]; p2 = points_arr[int(tri[2])]
                    a = abs(triangle_area(p0, p1, p2))
                    e0 = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
                    e1 = (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
                    e2 = (p0[0]-p2[0])**2 + (p0[1]-p2[1])**2
                    denom = e0 + e1 + e2
                    q = 0.0
                    if denom > 0:
                        q = (a / denom) * (12.0 / (3.0**0.5))
                        if q > 1.0: q = 1.0
                    out.append(q)
                except Exception:
                    out.append(0.0)
            return _np.array(out, dtype=_np.float64)
    # Triangle/polygon quality metric helper
    def _triangle_quality(p0, p1, p2):
        import math
        try:
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            # area
            a = abs((x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0)) * 0.5
            # sum of squared edge lengths
            l01 = (x1 - x0)**2 + (y1 - y0)**2
            l12 = (x2 - x1)**2 + (y2 - y1)**2
            l20 = (x0 - x2)**2 + (y0 - y2)**2
            s = l01 + l12 + l20
            if s <= 0:
                return 0.0
            # normalize so equilateral -> 1. normalization factor = 12 / sqrt(3)
            norm = 12.0 / math.sqrt(3.0)
            q = (a / s) * norm
            if q < 0.0: q = 0.0
            if q > 1.0: q = 1.0
            return q
        except Exception:
            return 0.0

    def _patch_quality(tris, pts):
        try:
            if not tris:
                return 1.0
            vals = []
            for t in tris:
                p0 = pts[int(t[0])]; p1 = pts[int(t[1])]; p2 = pts[int(t[2])]
                vals.append(_triangle_quality(p0, p1, p2))
            return float(sum(vals) / len(vals)) if vals else 0.0
        except Exception:
            return 0.0
    # threshold to skip heavy crossing/simulation when patch is high-quality
    _SIM_QUALITY_THRESH = float(getattr(editor, 'remove_simulate_quality_thresh', 0.25))
    _t_op_start = time.perf_counter()
    if not force_strict and not new_triangles:
        try:
            # Avoid verbose stdout spam during automated tests / batch runs which previously caused
            # OSError: [Errno 28] No space left on device (excessive captured output).
            if stats:
                stats.remove_candidate_attempts += 1
            _t_triang_start = time.perf_counter()
            new_triangles = optimal_star_triangulation(editor.points, cycle, debug=False)
            _t_triang_end = time.perf_counter()
            print(f"[remove_node] optimal_star_triang_ms={( _t_triang_end - _t_triang_start)*1000:.3f}")
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
                        if stats:
                            stats.remove_candidate_attempts += 1
                        new_triangles = optimal_star_triangulation(editor.points, simplified, debug=False)
                        if new_triangles and stats:
                            stats.simplify_helped += 1
                            cycle = simplified  # adopt simplified cycle for logging clarity
                    except ValueError:
                        new_triangles = None
            # If still None we proceed to fallback below
    # If optimal star failed, try a fast-path quadrilateral triangulation before broader fallbacks
    if (not new_triangles
        and getattr(editor, 'virtual_boundary_mode', False)
        and getattr(editor, 'enable_remove_quad_fastpath', False)
        and cycle and len(cycle) == 4):
        v0, v1, v2, v3 = [int(x) for x in cycle]
        quads = [
            [[v0, v1, v2], [v0, v2, v3]],  # diagonal (v0,v2)
            [[v1, v2, v3], [v1, v3, v0]],  # diagonal (v1,v3)
        ]
        # Score by worst-min-angle improvement; try best first
        pre_list = [editor.triangles[int(ti)] for ti in cavity_tri_indices]
        from .quality import worst_min_angle as _wmin
        try:
            pre_mina = _wmin(editor.points, pre_list)
        except Exception:
            pre_mina = None
        cand_infos = []
        for cand in quads:
            if pre_mina is not None:
                try:
                    post_mina = _wmin(editor.points, cand)
                    cand_infos.append((cand, post_mina - pre_mina))
                except Exception:
                    cand_infos.append((cand, 0.0))
            else:
                cand_infos.append((cand, 0.0))
        cand_infos.sort(key=lambda x: x[1], reverse=True)
        if cand_infos:
            if stats:
                stats.remove_candidate_attempts += 1
            chosen = cand_infos[0][0]
            new_triangles = [list(chosen[0]), list(chosen[1])]

    # If optimal star failed and (possibly) quad fast-path didn’t help, prefer polygon-based fallbacks
    if not new_triangles and getattr(editor, 'virtual_boundary_mode', False) and cycle:
        # Collect multiple polygon-based candidate triangulations and rank them by
        # average normalized triangle quality. Prefer candidates that preserve area
        # when the config requests it.
        cfg = getattr(editor, 'boundary_remove_config', None)
        # Default to requiring area preservation when config is None
        require_area = True if (cfg is None) else bool(getattr(cfg, 'require_area_preservation', True))
        try:
            from .constants import EPS_AREA
            area_tol_rel = float(getattr(cfg, 'area_tol_rel', None)) if cfg is not None and getattr(cfg, 'area_tol_rel', None) is not None else None
            area_abs = float(getattr(cfg, 'area_tol_abs_factor', 4.0)) * float(EPS_AREA) if cfg is not None else float(4.0 * EPS_AREA)
        except Exception:
            area_tol_rel = None; area_abs = None

        candidate_options = []  # list of (tri_list, source_label)

        # Try area-preserving best-star (may return None)
        try:
            from .triangulation import best_star_triangulation_area_preserving, best_star_triangulation_by_min_angle
            try:
                if area_tol_rel is None or area_abs is None:
                    cand = best_star_triangulation_area_preserving(editor.points, cycle, debug=False)
                else:
                    cand = best_star_triangulation_area_preserving(editor.points, cycle, area_tol_rel=area_tol_rel, area_tol_abs=area_abs, debug=False)
                if cand:
                    candidate_options.append((cand, 'area_star'))
            except ValueError as e:
                editor.logger.debug("best_star_triangulation_area_preserving raised ValueError: %s", e)
        except Exception:
            # triangulation function not available or failed; continue
            pass

        # Try worst-angle best star
        try:
            try:
                cand = best_star_triangulation_by_min_angle(editor.points, cycle, debug=False)
                if cand:
                    candidate_options.append((cand, 'worst_star'))
            except ValueError as e:
                editor.logger.debug("best_star_triangulation_by_min_angle raised ValueError: %s", e)
        except Exception:
            pass

        # Ear-clip fallback (always try, may produce candidate)
        try:
            if stats:
                stats.remove_candidate_attempts += 1
            _t_ear_start = time.perf_counter()
            cand = ear_clip_triangulation(editor.points, cycle)
            _t_ear_end = time.perf_counter()
            print(f"[remove_node] ear_clip_ms={( _t_ear_end - _t_ear_start)*1000:.3f}")
            if cand:
                candidate_options.append((cand, 'ear_clip'))
        except ValueError as e:
            editor.logger.debug("ear_clip_triangulation raised ValueError: %s", e)

        # If we collected multiple candidates, rank & pick best according to avg quality
        if candidate_options:
            try:
                import numpy as _np
                scored = []
                for cand, src in candidate_options:
                    tris_np = _np.array(cand, dtype=_np.int32) if cand else _np.empty((0,3), dtype=_np.int32)
                    try:
                        q_arr = _triangle_qualities_norm(editor.points, tris_np)
                        avg_q = float(_np.mean(q_arr)) if q_arr.size else 0.0
                    except Exception:
                        # Fallback scalar mean
                        avg_q = _patch_quality(cand, editor.points)
                    # compute candidate area for area-preservation checks
                    try:
                        cand_area = 0.0
                        for t in cand:
                            p0 = editor.points[int(t[0])]; p1 = editor.points[int(t[1])]; p2 = editor.points[int(t[2])]
                            cand_area += abs(triangle_area(p0, p1, p2))
                    except Exception:
                        cand_area = None
                    scored.append((avg_q, cand_area, cand, src))
                # prefer area-preserving candidates when required
                # sort by avg_q desc
                scored.sort(key=lambda x: x[0], reverse=True)
                chosen = None
                if require_area and removed_area is not None:
                    for avg_q, cand_area, cand, src in scored:
                        if cand_area is None:
                            continue
                        tol_rel = area_tol_rel if area_tol_rel is not None else (getattr(cfg, 'area_tol_rel', None) if cfg is not None else None) or 0.0
                        tol_abs = area_abs
                        if abs(cand_area - removed_area) <= max(tol_abs, tol_rel*max(1.0, removed_area)):
                            chosen = cand
                            break
                if chosen is None:
                    # either area not required or none matched; pick highest-quality candidate
                    chosen = scored[0][2]
                new_triangles = chosen
            except Exception:
                # On error, fall back to first successful candidate if any
                try:
                    new_triangles = candidate_options[0][0]
                except Exception:
                    new_triangles = None
        else:
            new_triangles = None

    if not new_triangles:  # fallback strict retriangulation (generic)
        editor.logger.debug("optimal_star_triangulation skipped/failed; trying strict retriangulation fallback")
        try:
            if stats:
                stats.remove_candidate_attempts += 1
            _t_retr_start = time.perf_counter()
            new_points, triangles_new, ok_sub, new_tri_list = retriangulate_patch_strict(
                editor.points, editor.triangles, cavity_tri_indices, new_point_coords=None, strict_mode='centroid')
            _t_retr_end = time.perf_counter()
            print(f"[remove_node] retriangulate_patch_strict_ms={( _t_retr_end - _t_retr_start)*1000:.3f}")
        except Exception as e:
            editor.logger.debug("Fallback retriangulation raised exception: %s", e)
            if stats:
                stats.remove_early_rejects += 1
            return False, "retriangulation failed", None
        if not ok_sub or not new_tri_list:
            if stats:
                stats.remove_early_rejects += 1
            return False, "retriangulation failed", None
        # In virtual-boundary mode, drop any triangles that still reference the removed vertex
        cand_tris = [list(t) for t in new_tri_list]
        if getattr(editor, 'virtual_boundary_mode', False):
            cand_tris = [list(t) for t in cand_tris if int(v_idx) not in (int(t[0]), int(t[1]), int(t[2]))]
        new_triangles = cand_tris
        used_fallback = True
        if stats:
            stats.fallback_used += 1
            # Count the strict-retriangulation candidate as an attempt as well
            if cand_tris:
                stats.remove_candidate_attempts += 1
    else:
        if stats: stats.star_success += 1

    # Safety: in virtual boundary mode, ensure the removed vertex does not appear in candidate triangles
    if getattr(editor, 'virtual_boundary_mode', False) and new_triangles:
        if any(int(v_idx) in [int(t[0]), int(t[1]), int(t[2])] for t in new_triangles):
            editor.logger.debug("virtual-boundary: new triangulation contains removed vertex %s; filtering and retrying polygonal fallback", v_idx)
            # First, filter out any triangles referencing v_idx
            new_triangles = [list(t) for t in new_triangles if int(v_idx) not in (int(t[0]), int(t[1]), int(t[2]))]
            # If filtering emptied the set, try polygon-based fill again (ear-clip)
            if not new_triangles and cycle:
                try:
                    new_triangles = ear_clip_triangulation(editor.points, cycle)
                except ValueError as e:
                    editor.logger.debug("ear_clip_triangulation retry failed: %s", e)
                    return False, "retriangulation failed", None

    # Vectorized candidate triangle filtering
    try:
        _t_vec_start = time.perf_counter()
        import numpy as _np
        from collections import defaultdict
        # Cache adjacency maps if not present
        if not hasattr(editor, 'edge_map') or not hasattr(editor, 'v_map'):
            editor._update_maps()
        # Existing triangles kept (outside the cavity)
        tris_keep = [tuple(t) for i, t in enumerate(editor.triangles)
                     if i not in cavity_tri_indices and not _np.all(_np.array(t) == -1)]
        # Build kept-edges list once (from editor.edge_map) and compute the subset touching the cavity
        kept_edges_all = list(getattr(editor, 'edge_map', {}).keys()) if hasattr(editor, 'edge_map') else kept_edges
        # local_kept_edges: only kept edges that touch the cavity boundary (cycle)
        # Exclude edges that are fully internal to the cavity (both endpoints in cycle),
        # as these will be removed during retriangulation and should not be crossing-checked.
        if cycle is not None:
            cycle_set = set(int(x) for x in cycle)
            # local_kept_edges: edges that touch the cavity boundary but are not being removed.
            # Exclude:
            # 1. Edges fully internal to the cavity (both endpoints in cycle)
            # 2. Edges incident to the vertex being removed (v_idx)
            local_kept_edges = [e for e in kept_edges_all 
                               if ((int(e[0]) in cycle_set) or (int(e[1]) in cycle_set))
                               and not ((int(e[0]) in cycle_set) and (int(e[1]) in cycle_set))
                               and not (int(e[0]) == int(v_idx) or int(e[1]) == int(v_idx))]
        else:
            local_kept_edges = kept_edges_all
        # expose kept_edges list and compute edge degrees once for reuse
        kept_edges = kept_edges_all
        try:
            edge_deg = {tuple(e): len(s) for e, s in getattr(editor, 'edge_map', {}).items()}
        except Exception:
            edge_deg = {}
        tris_keep_arr = _np.array(tris_keep, dtype=_np.int32) if tris_keep else _np.empty((0,3), dtype=_np.int32)
        # convert local_kept_edges to numpy array once for reuse in crossing checks
        kept_edge_arr = _np.array(local_kept_edges, dtype=_np.int32) if local_kept_edges else _np.empty((0,2), dtype=_np.int32)
        # Build a reusable spatial grid index for kept edges to avoid rebuilding per-validate
        try:
            from sofia.core.conformity import build_kept_edge_grid
            kept_grid = build_kept_edge_grid(editor.points, kept_edge_arr) if kept_edge_arr.shape[0] else None
        except Exception:
            kept_grid = None
        # Precompute helper arrays for kept-edges to allow reuse in fallback/ear-clip paths
        # (_ke_arr alias, endpoint coordinate arrays and per-edge bboxes)
        if kept_edge_arr.shape[0]:
            pts_arr = _np.asarray(editor.points, dtype=_np.float64)
            _ke_arr = kept_edge_arr
            try:
                _ke_pa = pts_arr[_ke_arr[:, 0]]
                _ke_pb = pts_arr[_ke_arr[:, 1]]
                _ke_minx = _np.minimum(_ke_pa[:, 0], _ke_pb[:, 0])
                _ke_maxx = _np.maximum(_ke_pa[:, 0], _ke_pb[:, 0])
                _ke_miny = _np.minimum(_ke_pa[:, 1], _ke_pb[:, 1])
                _ke_maxy = _np.maximum(_ke_pa[:, 1], _ke_pb[:, 1])
            except Exception:
                # Fallback: create empty arrays if any indexing fails
                _ke_pa = _np.empty((0,2), dtype=_np.float64)
                _ke_pb = _np.empty((0,2), dtype=_np.float64)
                _ke_minx = _np.empty((0,), dtype=_np.float64)
                _ke_maxx = _np.empty((0,), dtype=_np.float64)
                _ke_miny = _np.empty((0,), dtype=_np.float64)
                _ke_maxy = _np.empty((0,), dtype=_np.float64)
        else:
            _ke_arr = _np.empty((0,2), dtype=_np.int32)
            _ke_pa = _np.empty((0,2), dtype=_np.float64)
            _ke_pb = _np.empty((0,2), dtype=_np.float64)
            _ke_minx = _np.empty((0,), dtype=_np.float64)
            _ke_maxx = _np.empty((0,), dtype=_np.float64)
            _ke_miny = _np.empty((0,), dtype=_np.float64)
            _ke_maxy = _np.empty((0,), dtype=_np.float64)
        # Track duplicate triangles (by sorted vertex set)
        keep_sorted = _np.sort(tris_keep_arr, axis=1) if tris_keep_arr.shape[0] else _np.empty((0,3), dtype=_np.int32)
        # Candidate triangles as array
        cand_arr = _np.array(new_triangles, dtype=_np.int32) if new_triangles else _np.empty((0,3), dtype=_np.int32)
        # Area and orientation check (vectorized)
        from sofia.core.geometry import triangles_signed_areas
        areas = triangles_signed_areas(editor.points, cand_arr)
        valid_mask = (areas > float(_EPS))
        cand_arr = cand_arr[valid_mask]
        # Remove duplicates: sort vertices and use np.unique
        cand_sorted = _np.sort(cand_arr, axis=1)
        all_tris = _np.vstack([keep_sorted, cand_sorted]) if keep_sorted.shape[0] else cand_sorted
        _, unique_idx = _np.unique(all_tris, axis=0, return_index=True)
        # Only keep candidate triangles that are unique (not in keep)
        unique_mask = unique_idx >= keep_sorted.shape[0] if keep_sorted.shape[0] else _np.ones(cand_sorted.shape[0], dtype=bool)
        cand_arr = cand_arr[unique_mask]
        # Vectorized non-manifold and edge crossing checks
        if cand_arr.shape[0]:
            # Extract candidate edges (each triangle: 3 edges)
            edges_a = cand_arr[:, [0,1,2]].reshape(-1)
            edges_b = cand_arr[:, [1,2,0]].reshape(-1)
            cand_edges = _np.stack([edges_a, edges_b], axis=1)
            cand_edges = _np.sort(cand_edges, axis=1)
            # Non-manifold check: vectorized attempt using edge degree arrays
            try:
                edge_deg_map = edge_deg
                if edge_deg_map:
                    # Build sorted edge keys and degree array
                    ke_keys = _np.array([list(k) for k in edge_deg_map.keys()], dtype=_np.int32)
                    ke_keys.sort(axis=1)
                    dtype = _np.dtype([('a', _np.int32), ('b', _np.int32)])
                    ke_struct = ke_keys.view(dtype).ravel()
                    deg_arr = _np.array([edge_deg_map.get((int(k[0]), int(k[1])), 0) for k in ke_keys], dtype=_np.int32)
                    order = _np.argsort(ke_struct)
                    ke_sorted = ke_struct[order]
                    deg_sorted = deg_arr[order]
                    # Candidate triangle edges (3 per triangle)
                    ce0 = cand_arr[:, [0,1]]; ce1 = cand_arr[:, [1,2]]; ce2 = cand_arr[:, [2,0]]
                    ce_all = _np.vstack([ce0, ce1, ce2])
                    ce_all_sorted = _np.sort(ce_all, axis=1)
                    ce_struct = ce_all_sorted.view(dtype).ravel()
                    idxs = _np.searchsorted(ke_sorted, ce_struct)
                    degrees_found = _np.zeros(ce_all_sorted.shape[0], dtype=_np.int32)
                    mask_valid = (idxs < ke_sorted.shape[0]) & (ke_sorted[idxs] == ce_struct)
                    if _np.any(mask_valid):
                        degrees_found[mask_valid] = deg_sorted[idxs[mask_valid]]
                    degrees_tri = degrees_found.reshape(3, cand_arr.shape[0]).T
                    tri_ok_mask = _np.all(degrees_tri < 2, axis=1)
                    cand_arr = cand_arr[tri_ok_mask]
                else:
                    # no existing edges -> all candidates pass
                    pass
            except Exception:
                # fallback to scalar loop on error
                tri_ok = _np.ones(cand_arr.shape[0], dtype=bool)
                for i in range(cand_arr.shape[0]):
                    tri_edges = [tuple(sorted([int(cand_arr[i,0]), int(cand_arr[i,1])])),
                                 tuple(sorted([int(cand_arr[i,1]), int(cand_arr[i,2])])),
                                 tuple(sorted([int(cand_arr[i,2]), int(cand_arr[i,0])]))]
                    if any(edge_deg.get(e,0) >= 2 for e in tri_edges):
                        tri_ok[i] = False
                cand_arr = cand_arr[tri_ok]
            # Edge crossing check: use helper in conformity.py
            if cand_arr.shape[0] and kept_edge_arr.shape[0]:
                from sofia.core.conformity import filter_crossing_candidate_edges
                cand_edges = _np.sort(_np.stack([
                    cand_arr[:, [0,1]], cand_arr[:, [1,2]], cand_arr[:, [2,0]]
                ], axis=1).reshape(-1,2), axis=1)
                _t_cross_start = time.perf_counter()
                crosses = filter_crossing_candidate_edges(editor.points, kept_edge_arr, cand_edges, kept_grid=kept_grid)
                _t_cross_end = time.perf_counter()
                # remember we already preflighted crossings for these candidates
                preflight_crossings_checked = True
                preflight_crosses_any = bool(_np.any(crosses))
                print(f"[remove_node] crossing_helper_ms={( _t_cross_end - _t_cross_start)*1000:.3f}")
                crosses_tri = crosses.reshape(-1,3).any(axis=1)
                n_cross_reject = int(_np.sum(crosses_tri))
                if stats and n_cross_reject > 0:
                    stats.remove_early_rejects += n_cross_reject
                    stats.remove_early_validation += n_cross_reject
                cand_arr = cand_arr[~crosses_tri]
            # Accept all remaining candidates
            if cand_arr.shape[0]:
                new_triangles = cand_arr.tolist()
        _t_vec_end = time.perf_counter()
        editor.logger.info("remove_node_with_patch: vectorized_filtering_ms=%.3f", 1000.0*(_t_vec_end - _t_vec_start))
        try:
            print(f"[operations] vectorized_filtering_ms={1000.0*(_t_vec_end - _t_vec_start):.3f}")
        except Exception:
            pass
    except Exception:
        # If anything goes wrong, proceed with original set and let validation catch issues
        pass

    # Validate candidate locally
    def _validate_candidate(tris_cand):
        _t_val_start = time.perf_counter()
        t_build0 = time.perf_counter()
        # Optimized: use numpy boolean indexing
        mask = np.ones(len(editor.triangles), dtype=bool)
        if cavity_tri_indices:
            mask[list(cavity_tri_indices)] = False
        mask &= (editor.triangles[:, 0] != -1)
        arr = np.vstack([editor.triangles[mask], tris_cand]) if mask.any() and len(tris_cand) > 0 else (
            editor.triangles[mask] if mask.any() else (np.array(tris_cand, dtype=np.int32) if len(tris_cand) > 0 else np.empty((0, 3), dtype=np.int32))
        )
        t_build1 = time.perf_counter()
        # For boundary removal with a polygon cycle, optionally enforce area preservation
        if getattr(editor, 'virtual_boundary_mode', False) and tris_cand:
            try:
                cand_area = 0.0
                for t in tris_cand:
                    p0 = editor.points[int(t[0])]; p1 = editor.points[int(t[1])]; p2 = editor.points[int(t[2])]
                    cand_area += abs(triangle_area(p0, p1, p2))
                # allow small tolerance; compare against local removed area when available, else polygon area
                from .constants import EPS_TINY, EPS_AREA
                cfg = getattr(editor, 'boundary_remove_config', None)
                tol_rel = EPS_TINY if cfg is None else float(getattr(cfg, 'area_tol_rel', EPS_TINY))
                tol_abs = 4.0*EPS_AREA if cfg is None else float(getattr(cfg, 'area_tol_abs_factor', 4.0)) * float(EPS_AREA)
                # Area preservation compares candidate area against removed cavity area (sum of incident triangles).
                # This ensures the total mesh area doesn't change: area removed = area added.
                target_area = removed_area
                if target_area is None:
                    try:
                        target_area = poly_target_area
                    except NameError:
                        target_area = None
                if target_area is not None and abs(cand_area - target_area) > max(tol_abs, tol_rel*max(1.0, target_area)):
                    # If strict area preservation required, reject; otherwise allow but log
                    # Default to requiring area preservation when cfg is None
                    require_area_pres = True if (cfg is None) else bool(getattr(cfg, 'require_area_preservation', True))
                    if require_area_pres:
                        return False, [f"cavity area changed: appended={cand_area:.6e} target={target_area:.6e}"]
                    else:
                        editor.logger.debug("area changed within relaxed policy: appended=%.6e target=%.6e", cand_area, target_area)
            except Exception:
                pass
        t_check0 = time.perf_counter()
        ok_c, msgs_c = check_mesh_conformity(editor.points, arr, allow_marked=False)
        t_check1 = time.perf_counter()
        try:
            print(f"[remove_node] validate_candidate_ms_total={(t_check1 - _t_val_start)*1000:.3f} build_ms={(t_build1 - t_build0)*1000:.3f} check_ms={(t_check1 - t_check0)*1000:.3f}")
        except Exception:
            pass
        if not ok_c:
            return ok_c, msgs_c
        # Attempt local-only validation (fast path): if the candidate patch area matches
        # the removed cavity area (within tolerances) and cheap local checks pass
        # (no non-manifold edges and no crossings with nearby kept edges), accept.
        try:
            local_ok = False
            # Compute candidate patch area
            cand_area = 0.0
            for t in tris_cand:
                p0 = editor.points[int(t[0])]; p1 = editor.points[int(t[1])]; p2 = editor.points[int(t[2])]
                cand_area += abs(triangle_area(p0, p1, p2))
            # Use configured tolerances if available
            cfg = getattr(editor, 'boundary_remove_config', None)
            from .constants import EPS_TINY
            tol_rel = EPS_TINY if cfg is None else float(getattr(cfg, 'area_tol_rel', EPS_TINY))
            tol_abs = 4.0*EPS_AREA if cfg is None else float(getattr(cfg, 'area_tol_abs_factor', 4.0)) * float(EPS_AREA)
            # Only try local accept when removed_area is known
            if removed_area is not None and abs(cand_area - removed_area) <= max(tol_abs, tol_rel*max(1.0, removed_area)):
                # Non-manifold check against existing edges (cheap)
                try:
                    # Build edge degree map from editor.edge_map
                    edge_deg_local = {tuple(e): len(s) for e, s in getattr(editor, 'edge_map', {}).items()}
                except Exception:
                    edge_deg_local = {}
                nm_violation = False
                for t in tris_cand:
                    a, b, c = int(t[0]), int(t[1]), int(t[2])
                    for e in (tuple(sorted((a,b))), tuple(sorted((b,c))), tuple(sorted((c,a)))):
                        if edge_deg_local.get(e, 0) >= 2:
                            nm_violation = True
                            break
                    if nm_violation:
                        break
                if not nm_violation:
                    # Local crossing check: only test kept-edges incident to the cavity boundary (cycle)
                    try:
                        from .conformity import filter_crossing_candidate_edges
                        import numpy as _np
                        # build candidate edge normalized list (deduped) and memoize
                        ce_list = []
                        for t in tris_cand:
                            a, b, c = int(t[0]), int(t[1]), int(t[2])
                            ce_list.extend([(a,b),(b,c),(c,a)])
                        ce_norm = [tuple(sorted(e)) for e in ce_list]
                        # compute compact hash key for candidate-edge set
                        ce_key_hash = _hash_edges_iter(sorted(set(ce_norm)))
                        if ce_key_hash in _candidate_edges_cache:
                            # LRU: move to end
                            ce_arr = _candidate_edges_cache.pop(ce_key_hash)
                            _candidate_edges_cache[ce_key_hash] = ce_arr
                        else:
                            ce_arr = _np.array(list(sorted(set(ce_norm))), dtype=_np.int32) if ce_norm else _np.empty((0,2), dtype=_np.int32)
                            _candidate_edges_cache[ce_key_hash] = ce_arr
                            if len(_candidate_edges_cache) > _CAND_CACHE_MAX:
                                _candidate_edges_cache.popitem(last=False)
                        # kept edges limited to those touching the cycle vertices
                        cycle_vs = set(cycle) if cycle is not None else set()
                        if kept_edge_arr.shape[0]:
                            # filter kept_edge_arr rows where either vertex is in cycle_vs
                            mask0 = _np.isin(kept_edge_arr[:,0], _np.array(list(cycle_vs), dtype=_np.int32))
                            mask1 = _np.isin(kept_edge_arr[:,1], _np.array(list(cycle_vs), dtype=_np.int32))
                            mask = mask0 | mask1
                            ke_arr = kept_edge_arr[mask]
                        else:
                            ke_arr = _np.empty((0,2), dtype=_np.int32)
                        if ce_arr.shape[0] and ke_arr.shape[0]:
                            try:
                                ke_list_sorted = sorted(map(tuple, ke_arr.tolist()))
                            except Exception:
                                ke_list_sorted = []
                            ke_key_hash = _hash_edges_iter(ke_list_sorted)
                            pre_key = (ke_key_hash, ce_key_hash)
                            if pre_key in _preflight_cross_cache:
                                crosses = _preflight_cross_cache.pop(pre_key)
                                _preflight_cross_cache[pre_key] = crosses
                            else:
                                crosses = filter_crossing_candidate_edges(editor.points, ke_arr, ce_arr, kept_grid=kept_grid)
                                _preflight_cross_cache[pre_key] = crosses
                                if len(_preflight_cross_cache) > _PREFLIGHT_CACHE_MAX:
                                    _preflight_cross_cache.popitem(last=False)
                            if not _np.any(crosses):
                                local_ok = True
                        else:
                            # no local kept edges to conflict with -> accept
                            local_ok = True
                    except Exception:
                        local_ok = False
            if local_ok:
                # local validation passed — accept candidate without full simulation
                return True, []
        except Exception:
            # any error in local fast-checks: fall back to full simulation below
            pass
        # Cheap preflight: use the fast grid-based helper to detect crossings between
        # candidate edges and kept edges. If no crossings are detected by this fast
        # helper (and we already checked earlier), skip the heavy simulate_compaction_and_check call.
        try:
            from .conformity import filter_crossing_candidate_edges
            import numpy as _np
            # Build kept edges list from editor.edge_map if available
            # reuse cached kept_edges (kept_edges variable) and its numpy array kept_edge_arr
            kept_edges_local = kept_edges if 'kept_edges' in locals() else ([e for e in editor.edge_map.keys()] if hasattr(editor, 'edge_map') else [])
            # Build candidate edge array (deduped) and reuse cache when possible
            ce_list = []
            for t in tris_cand:
                a, b, c = int(t[0]), int(t[1]), int(t[2])
                ce_list.extend([(a,b),(b,c),(c,a)])
            if ce_list and kept_edges_local:
                ce_norm = [tuple(sorted(e)) for e in ce_list]
                ce_key_hash = _hash_edges_iter(sorted(set(ce_norm)))
                if ce_key_hash in _candidate_edges_cache:
                    ce_arr = _candidate_edges_cache.pop(ce_key_hash)
                    _candidate_edges_cache[ce_key_hash] = ce_arr
                else:
                    ce_arr = _np.array(list(sorted(set(ce_norm))), dtype=_np.int32)
                    _candidate_edges_cache[ce_key_hash] = ce_arr
                    if len(_candidate_edges_cache) > _CAND_CACHE_MAX:
                        _candidate_edges_cache.popitem(last=False)
                ke_arr = _np.array(kept_edges_local, dtype=_np.int32) if kept_edges_local else _np.empty((0,2), dtype=_np.int32)
                # If we already ran the preflight earlier for the same set of
                # candidate edges and kept edges, reuse that result to avoid redundant work.
                try:
                    if preflight_crossings_checked:
                        must_simulate = preflight_crosses_any
                    else:
                        try:
                            ke_list_sorted = sorted(map(tuple, ke_arr.tolist()))
                        except Exception:
                            ke_list_sorted = []
                        ke_key_hash = _hash_edges_iter(ke_list_sorted)
                        pre_key = (ke_key_hash, ce_key_hash)
                        if pre_key in _preflight_cross_cache:
                            crosses_mask = _preflight_cross_cache.pop(pre_key)
                            _preflight_cross_cache[pre_key] = crosses_mask
                        else:
                            crosses_mask = filter_crossing_candidate_edges(editor.points, ke_arr, ce_arr, kept_grid=kept_grid)
                            _preflight_cross_cache[pre_key] = crosses_mask
                            if len(_preflight_cross_cache) > _PREFLIGHT_CACHE_MAX:
                                _preflight_cross_cache.popitem(last=False)
                        must_simulate = bool(_np.any(crosses_mask))
                except NameError:
                    # preflight flags not present, run helper
                    crosses_mask = filter_crossing_candidate_edges(editor.points, ke_arr, ce_arr, kept_grid=kept_grid)
                    must_simulate = bool(_np.any(crosses_mask))
            else:
                must_simulate = False
        except Exception:
            must_simulate = True
        # Compute average normalized triangle quality (vectorized) and possibly skip
        # heavy simulation when patch quality is high. Use _triangle_qualities_norm
        # (area / sum(edge_len^2) normalized) for a robust score in [0,1].
        try:
            import numpy as _np
            tris_np = _np.array(tris_cand, dtype=_np.int32) if tris_cand else _np.empty((0,3), dtype=_np.int32)
            q_arr = _triangle_qualities_norm(editor.points, tris_np)
            avg_q = float(_np.mean(q_arr)) if q_arr.size else 0.0
        except Exception:
            try:
                # Fallback to scalar patch quality if vectorized helper fails
                avg_q = _patch_quality(tris_cand if 'tris_cand' in locals() else tris_cand, editor.points)
            except Exception:
                avg_q = 0.0
        if avg_q >= _SIM_QUALITY_THRESH:
            must_simulate = False

        if not getattr(editor, 'simulate_compaction_on_commit', False) and must_simulate:
            try:
                from .conformity import simulate_compaction_and_check
                t_sim0 = time.perf_counter()
                ok_sim, sim_msgs = simulate_compaction_and_check(editor.points, arr, reject_crossing_edges=True)
                t_sim1 = time.perf_counter()
                if not ok_sim:
                    try:
                        print(f"[validate_candidate] simulate_compaction_ms={(t_sim1-t_sim0)*1000:.3f}")
                    except Exception:
                        pass
                    msgs_x = [m for m in sim_msgs if 'crossing edges detected' in m] or sim_msgs
                    return False, msgs_x
            except Exception:
                pass
        # Also reject if any edge crossings are detected after adding candidates.
        # If final preflight simulation is enabled, defer heavy simulation to that step
        # to avoid duplicate work; otherwise do a quick simulation here for safety.
        # IMPORTANT: Always check for edge crossings even when quality is high, as crossings
        # can occur independently of triangle quality (BUG FIX: was incorrectly bypassing this check).
        if not getattr(editor, 'simulate_compaction_on_commit', False):
            try:
                from .conformity import simulate_compaction_and_check
                t_sim0 = time.perf_counter()
                ok_sim, sim_msgs = simulate_compaction_and_check(editor.points, arr, reject_crossing_edges=True)
                t_sim1 = time.perf_counter()
                if not ok_sim:
                    try:
                        print(f"[validate_candidate] simulate_compaction_ms={(t_sim1-t_sim0)*1000:.3f}")
                    except Exception:
                        pass
                    # only keep crossing messages for brevity
                    msgs_x = [m for m in sim_msgs if 'crossing edges detected' in m] or sim_msgs
                    return False, msgs_x
            except Exception:
                pass
        return True, []
    try:
        _t_validate_call = time.perf_counter()
        ok_sub, msgs = _validate_candidate(new_triangles)
        _t_validate_call_end = time.perf_counter()
        print(f"[remove_node] validate_call_ms={( _t_validate_call_end - _t_validate_call)*1000:.3f}")
    except Exception as e:
        if stats:
            stats.remove_early_rejects += 1
        return False, f"retriangulation validation error: {e}", None
    if not ok_sub and getattr(editor, 'virtual_boundary_mode', False) and cycle:
        # Retry by rotating/reversing the polygon and using ear-clip again, with non-manifold filtering
        tried = 0
        best = None
        from .triangulation import ear_clip_triangulation
        # Collect ear-clip candidates across rotations/reversals and score by patch quality
        candidates_scored = []
        for rev in (False, True):
            cyc_seq = list(cycle)[::-1] if rev else list(cycle)
            for k in range(len(cyc_seq)):
                cyc2 = cyc_seq[k:] + cyc_seq[:k]
                try:
                    if stats:
                        stats.remove_candidate_attempts += 1
                    cand = ear_clip_triangulation(editor.points, cyc2)
                except Exception:
                    continue
                try:
                    import numpy as _np
                    # quick sanitize: drop triangles referencing removed vertex if any
                    cand_sanitized = [list(t) for t in cand if int(v_idx) not in (int(t[0]), int(t[1]), int(t[2]))]
                    if not cand_sanitized:
                        continue
                    q = _patch_quality(cand_sanitized, editor.points)
                    candidates_scored.append((q, cand_sanitized))
                except Exception:
                    continue
        # Try candidates best-first by quality
        candidates_scored.sort(key=lambda x: x[0], reverse=True)
        for q, cand in candidates_scored:
            tried += 1
            try:
                # Apply filtering same as before
                import numpy as _np
                exist_sets = existing_tri_sets
                edge_deg_r = edge_deg.copy()
                filtered = []
                fsets = set()
                fedges = set()
                def _area2(p0, p1, p2):
                    return (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
                from .constants import EPS_AREA as _EPS
                from .geometry import seg_intersect as _seg_intersect

                def _edge_crosses_reuse(u, v):
                    pu, pv = editor.points[int(u)], editor.points[int(v)]
                    if _ke_arr.shape[0]:
                        min_uvx = pu[0] if pu[0] < pv[0] else pv[0]
                        max_uvx = pu[0] if pu[0] > pv[0] else pv[0]
                        min_uvy = pu[1] if pu[1] < pv[1] else pv[1]
                        max_uvy = pu[1] if pu[1] > pv[1] else pv[1]
                        shared = (_ke_arr[:,0] == int(u)) | (_ke_arr[:,0] == int(v)) | (_ke_arr[:,1] == int(u)) | (_ke_arr[:,1] == int(v))
                        from .geometry import bbox_overlap, orient_vectorized
                        bb_ok = (~shared) & bbox_overlap(_ke_minx, _ke_maxx, _ke_miny, _ke_maxy,
                                                        min_uvx, max_uvx, min_uvy, max_uvy)
                        if _np.any(bb_ok):
                            pa = _ke_pa[bb_ok]
                            pb = _ke_pb[bb_ok]
                            o1 = orient_vectorized(pa, pb, pu)
                            o2 = orient_vectorized(pa, pb, pv)
                            o3 = (pv[0]-pu[0])*(pa[:,1]-pu[1]) - (pv[1]-pu[1])*(pa[:,0]-pu[0])
                            o4 = (pv[0]-pu[0])*(pb[:,1]-pu[1]) - (pv[1]-pu[1])*(pb[:,0]-pu[0])
                            from .constants import EPS_TINY as _ET
                            colinear = (_np.abs(o1) <= _ET) & (_np.abs(o2) <= _ET) & (_np.abs(o3) <= _ET) & (_np.abs(o4) <= _ET)
                            crosses = (o1*o2 < 0) & (o3*o4 < 0) & (~colinear)
                            if _np.any(crosses):
                                return True
                    for a, b in fedges:
                        if a in (u, v) or b in (u, v):
                            continue
                        pa, pb = editor.points[int(a)], editor.points[int(b)]
                        min_abx = min(pa[0], pb[0]); max_abx = max(pa[0], pb[0])
                        min_aby = min(pa[1], pb[1]); max_aby = max(pa[1], pb[1])
                        min_uvx = min(pu[0], pv[0]); max_uvx = max(pu[0], pv[0])
                        min_uvy = min(pu[1], pv[1]); max_uvy = max(pu[1], pv[1])
                        if (max_abx < min_uvx) or (max_uvx < min_abx) or (max_aby < min_uvy) or (max_uvy < min_aby):
                            pass
                        else:
                            if _seg_intersect(pa, pb, pu, pv):
                                return True
                    return False
                for t in cand:
                    a, b, c = int(t[0]), int(t[1]), int(t[2])
                    s = frozenset((a, b, c))
                    if s in exist_sets or s in fsets:
                        continue
                    p0, p1, p2 = editor.points[a], editor.points[b], editor.points[c]
                    if abs(_area2(p0, p1, p2)) <= float(_EPS):
                        continue
                    ea = tuple(sorted((a, b))); eb = tuple(sorted((b, c))); ec = tuple(sorted((c, a)))
                    if edge_deg_r[ea] >= 2 or edge_deg_r[eb] >= 2 or edge_deg_r[ec] >= 2:
                        continue
                    if _edge_crosses_reuse(a, b) or _edge_crosses_reuse(b, c) or _edge_crosses_reuse(c, a):
                        continue
                    filtered.append([a, b, c])
                    fsets.add(s)
                    edge_deg_r[ea] += 1; edge_deg_r[eb] += 1; edge_deg_r[ec] += 1
                    fedges.add(ea); fedges.add(eb); fedges.add(ec)
                if filtered:
                    ok_sub2, msgs2 = _validate_candidate(filtered)
                    if ok_sub2:
                        best = filtered
                        break
            except Exception:
                continue
        if best is not None:
            new_triangles = best
            ok_sub = True
        else:
            if stats:
                stats.remove_early_rejects += 1
                stats.remove_early_validation += 1
            return False, f"retriangulation failed local conformity: {msgs}", None
        if best is not None:
            new_triangles = best
            ok_sub = True
        else:
            if stats:
                stats.remove_early_rejects += 1
                stats.remove_early_validation += 1
            return False, f"retriangulation failed local conformity: {msgs}", None
    elif not ok_sub:
        if stats:
            stats.remove_early_rejects += 1
            stats.remove_early_validation += 1
        return False, f"retriangulation failed local conformity: {msgs}", None

    # Quality gating (non-worsening worst min-angle) unless disabled
    enforce_remove_q = getattr(editor, 'enforce_remove_quality', True)
    if enforce_remove_q:
        if stats:
            stats.remove_quality_checks += 1
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
    else:
        # Log pre/post worst min-angle for visibility but do not reject
        try:
            pre_list = [editor.triangles[int(ti)] for ti in cavity_tri_indices]
            pre_mina = worst_min_angle(editor.points, pre_list)
            post_mina = worst_min_angle(editor.points, new_triangles)
            editor.logger.debug("remove_node (relaxed quality): before=%.6fdeg post=%.6fdeg", pre_mina, post_mina)
        except Exception:
            pass

    # Final hard area-preservation gate (redundant with validation, ensures no bypass)
    try:
        if getattr(editor, 'virtual_boundary_mode', False) and new_triangles:
            cand_area = 0.0
            for t in new_triangles:
                p0 = editor.points[int(t[0])]; p1 = editor.points[int(t[1])]; p2 = editor.points[int(t[2])]
                cand_area += abs(triangle_area(p0, p1, p2))
            cfg = getattr(editor, 'boundary_remove_config', None)
            if cfg is not None and bool(getattr(cfg, 'require_area_preservation', False)):
                # Prefer cavity area; fallback to polygon area computed earlier
                tgt = removed_area
                if tgt is None:
                    try:
                        tgt = poly_target_area
                    except NameError:
                        tgt = None
                if tgt is not None:
                    from .constants import EPS_TINY
                    tol_rel = float(getattr(cfg, 'area_tol_rel', EPS_TINY))
                    tol_abs = float(getattr(cfg, 'area_tol_abs_factor', 4.0)) * float(EPS_AREA)
                    if abs(cand_area - tgt) > max(tol_abs, tol_rel*max(1.0, tgt)):
                        if stats:
                            stats.remove_early_rejects += 1
                            stats.remove_early_area += 1
                        return False, f"cavity area changed: appended={cand_area:.6e} target={tgt:.6e}", None
    except Exception:
        pass

    # Simulated compaction preflight
    ok_sim, sim_msg = _simulate_preflight(
        editor, editor.points, cavity_tri_indices, new_triangles, stats,
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

