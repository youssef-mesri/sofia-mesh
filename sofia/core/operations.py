"""Local mesh operations (split, flip, remove, add, pocket fill)."""
from __future__ import annotations
import numpy as np
from .geometry import triangle_area, triangle_angles
from .constants import EPS_AREA, EPS_MIN_ANGLE_DEG
from .quality import worst_min_angle, non_worsening_quality, _triangle_qualities_norm
from .conformity import check_mesh_conformity, simulate_compaction_and_check
import time
from .helpers import boundary_cycle_from_incident_tris


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
    tri_indices = sorted(set(editor.v_map.get(int(a), set())) | set(editor.v_map.get(int(b), set())))
    if len(tri_indices) == 0:
        return False, "no incident triangles", None
    from sofia.core.mesh_modifier2 import retriangulate_patch_strict  # lazy import to avoid cyclic
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

    Strategy: append a new vertex at the chosen position (midpoint by default, or
    automatically chooses a boundary vertex if one or both endpoints are on the boundary).
    Tombstone all triangles incident to either endpoint, and append modified copies where
    the endpoint is replaced with the new vertex. Degenerate candidates are dropped.

    This mirrors commit style used by other ops (no immediate compaction).

    Parameters
    ----------
    editor : PatchBasedMeshEditor
    edge : tuple[int,int]
        The edge to collapse (u,v). Must exist in the current mesh (boundary or interior).
    position : str, default 'midpoint'
        Position strategy: 'midpoint' (default), or 'auto' (chooses boundary vertex if present).
        When 'midpoint' is specified and one endpoint is on the boundary, it automatically
        collapses to the boundary vertex instead.
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
    
    # Check if either endpoint is on the boundary
    from .conformity import is_boundary_vertex_from_maps
    u_is_boundary = is_boundary_vertex_from_maps(u, editor.edge_map)
    v_is_boundary = is_boundary_vertex_from_maps(v, editor.edge_map)
    
    # Decide new vertex position based on boundary status
    if u_is_boundary and v_is_boundary:
        # Both on boundary: collapse to first vertex (keep boundary intact)
        p_new = editor.points[u].copy()
        collapse_to_existing = u
    elif u_is_boundary:
        # Only u on boundary: collapse to u
        p_new = editor.points[u].copy()
        collapse_to_existing = u
    elif v_is_boundary:
        # Only v on boundary: collapse to v
        p_new = editor.points[v].copy()
        collapse_to_existing = v
    elif position == 'midpoint' or position == 'auto':
        # Neither on boundary: use midpoint
        p_new = 0.5*(editor.points[u] + editor.points[v])
        collapse_to_existing = None
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
            # Filter degenerate, TODO maybe not needed due to earlier checks ? 
            if len({int(rep[0]), int(rep[1]), int(rep[2])}) < 3:
                continue
            # reject inverted or near-zero-area triangles
            a = triangle_area(cand_pts[int(rep[0])], cand_pts[int(rep[1])], cand_pts[int(rep[2])])
            if abs(a) <= EPS_AREA:
                if stats:
                    stats.fail += 1
                return False, "edge collapse would create inverted triangle", None
            #if abs(a) <= EPS_AREA:
            #    continue
            new_tris.append(rep)
    # Orient and quality check against previous local neighborhood
    new_tris = _orient_tris(cand_pts, new_tris)
    old_local = [editor.triangles[int(ti)] for ti in touched if not np.any(np.array(editor.triangles[int(ti)]) == -1)]
    
    # Quality check (can be disabled for anisotropic remeshing where elongated triangles are desired)
    enforce_collapse_quality = getattr(editor, 'enforce_collapse_quality', True)
    if enforce_collapse_quality:
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
            
            # Special case: if the result is an empty mesh (all triangles deleted), accept it
            if tmp_tri_full.size == 0:
                editor.logger.debug("[DEBUG] virtual-boundary: deletion results in empty mesh (acceptable)")
                ok_sub = True
                msgs = []
            else:
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
        ok, msgs = check_mesh_conformity(
            editor.points, test_tris_arr, allow_marked=False,
            reject_boundary_loops=getattr(editor, 'reject_any_boundary_loops', False)
        )
        if not ok:
            if stats:
                stats.remove_early_rejects += 1
            return False, f"conformity check failed: {msgs}", None
        
        # Then check for edge crossings and boundary loops using simulation (controlled by editor flags)
        if getattr(editor, 'reject_crossing_edges', False) or getattr(editor, 'reject_any_boundary_loops', False):
            ok_sim, sim_msgs = simulate_compaction_and_check(
                editor.points, test_tris_arr, 
                reject_crossing_edges=getattr(editor, 'reject_crossing_edges', False),
                reject_any_boundary_loops=getattr(editor, 'reject_any_boundary_loops', False)
            )
            if not ok_sim:
                if stats:
                    stats.remove_early_rejects += 1
                    stats.remove_early_validation += 1
                # Filter for specific messages
                filter_msgs = [m for m in sim_msgs if 'crossing' in m.lower() or 'boundary' in m.lower()] or sim_msgs
                return False, f"validation failed: {filter_msgs}", None
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

