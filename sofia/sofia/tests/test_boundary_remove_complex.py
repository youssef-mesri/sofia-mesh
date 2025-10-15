import numpy as np

from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor
from sofia.sofia.conformity import check_mesh_conformity
from sofia.sofia.config import BoundaryRemoveConfig
from sofia.sofia.helpers import boundary_polygons_from_patch, select_outer_polygon
from sofia.sofia.triangulation import polygon_signed_area


def mesh_with_boundary_vertex_degree_three():
    # Square corners with two interior points near the bottom-right corner (vertex 1)
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1 (boundary, target to remove)
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.3],  # 4 interior near bottom
        [0.7, 0.6],  # 5 interior near right
    ], dtype=float)
    # Triangulation that makes vertex 1 incident to three triangles
    # Use interior-first ordering to encourage positive orientation
    tris = np.array([
        [4, 0, 1],  # involves 1
        [4, 1, 5],  # involves 1
        [5, 1, 2],  # involves 1
        [4, 0, 3],
        [4, 3, 2],
        [5, 2, 4],
    ], dtype=int)
    return pts, tris


def test_remove_boundary_vertex_degree_three_virtual_mode():
    pts, tris = mesh_with_boundary_vertex_degree_three()
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True)
    # Relax quality to focus on topology and generic retriangulation
    try:
        editor.enforce_remove_quality = False
    except Exception:
        pass

    ok, msg, _ = editor.remove_node_with_patch(1)
    assert ok, f"remove_node_with_patch failed on boundary degree-3 vertex: {msg}"

    ok1, msgs1 = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok1, f"Mesh not conforming after removal (pre-compact): {msgs1}"
    # Pre-compaction: ensure no active triangle still references the removed vertex index
    assert not any(1 in [int(t[0]), int(t[1]), int(t[2])] for t in editor.triangles if not np.all(np.array(t) == -1)), "Removed vertex index still present in active triangles pre-compaction"
    old_to_new = editor.compact_triangle_indices()
    ok2, msgs2 = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok2, f"Mesh not conforming after compaction: {msgs2}"
    # After compaction, use mapping to ensure old vertex 1 does not exist anymore
    assert old_to_new[1] == -1, "Removed vertex still present after compaction"


def square_with_center():
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1 (corner to remove)
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.5],  # 4 center
    ], dtype=float)
    tris = np.array([
        [4, 0, 1],
        [4, 1, 2],
        [4, 2, 3],
        [4, 3, 0],
    ], dtype=int)
    return pts, tris


def test_remove_boundary_corner_after_adjacent_splits():
    # Start with square+center
    pts, tris = square_with_center()
    # Enforce area preservation in boundary removal by passing a config
    br_cfg = BoundaryRemoveConfig(
        prefer_area_preserving_star=True,
        prefer_worst_angle_star=True,
        require_area_preservation=True,
    )
    editor = PatchBasedMeshEditor(
        pts.copy(),
        tris.copy(),
        virtual_boundary_mode=True,
        enforce_split_quality=False,
        boundary_remove_config=br_cfg,
    )

    # Split two adjacent boundary edges around corner 1: (0,1) and (1,2)
    ok_a, msg_a, _ = editor.split_edge((0, 1))
    assert ok_a, f"split_edge(0,1) failed: {msg_a}"
    ok_b, msg_b, _ = editor.split_edge((1, 2))
    assert ok_b, f"split_edge(1,2) failed: {msg_b}"

    # Compute sanitized polygon cycle area around vertex 1 before removal (exclude the vertex itself)
    incident = [i for i, t in enumerate(editor.triangles) if 1 in [int(t[0]), int(t[1]), int(t[2])]]
    polys = boundary_polygons_from_patch(editor.triangles, incident)
    cyc = select_outer_polygon(editor.points, polys)
    # Remove the vertex 1 from the cycle to form the target polygon to be filled
    if cyc is None:
        raise AssertionError("Could not build boundary polygon around vertex 1")
    filtered = [int(v) for v in cyc if int(v) != 1]
    if len(filtered) >= 2 and filtered[0] == filtered[-1]:
        filtered = filtered[:-1]
    assert len(filtered) >= 3, "Sanitized boundary cycle has fewer than 3 vertices"
    poly_area = abs(polygon_signed_area([editor.points[int(v)] for v in filtered]))

    # Now remove the boundary corner vertex 1
    try:
        editor.enforce_remove_quality = False
    except Exception:
        pass
    pre_len = len(editor.triangles)
    ok_r, msg_r, _ = editor.remove_node_with_patch(1)
    assert ok_r, f"remove_node_with_patch failed on boundary corner with adjacent splits: {msg_r}"

    ok1, msgs1 = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok1, f"Mesh not conforming after removal (pre-compact): {msgs1}"
    # Pre-compaction: ensure no active triangle still references the removed vertex index
    assert not any(1 in [int(t[0]), int(t[1]), int(t[2])] for t in editor.triangles if not np.all(np.array(t) == -1)), "Removed vertex index still present in active triangles pre-compaction"

    # Explicit area-preservation check BEFORE compaction: appended triangles equal sanitized polygon area
    appended = [t for t in editor.triangles[pre_len:] if not np.all(np.array(t) == -1)]
    appended_area = 0.0
    for t in appended:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        p0, p1, p2 = editor.points[a], editor.points[b], editor.points[c]
        appended_area += abs(
            (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
        ) * 0.5
    # Use a tight tolerance consistent with config defaults
    from sofia.sofia.constants import EPS_AREA, EPS_TINY
    assert abs(appended_area - poly_area) <= max(4.0*EPS_AREA, EPS_TINY*max(1.0, poly_area)), \
        f"Polygon area changed: appended={appended_area:.6e}, poly={poly_area:.6e}"

    # Now compact and finalize checks
    old_to_new = editor.compact_triangle_indices()
    ok2, msgs2 = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok2, f"Mesh not conforming after compaction: {msgs2}"
    # After compaction, use mapping to ensure old vertex 1 does not exist anymore
    assert old_to_new[1] == -1, "Removed vertex still present after compaction"


def test_default_rejects_non_preserving_corner_after_splits():
    """By default, boundary removals must preserve cavity area. The corner-with-splits
    case shrinks the cavity by half if accepted, so the operation must be rejected
    without passing any explicit BoundaryRemoveConfig.
    """
    pts, tris = square_with_center()
    # Default editor (no explicit boundary_remove_config) should enforce area preservation
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True, enforce_split_quality=False)
    # Apply splits around corner 1: (0,1) and (1,2)
    ok_a, msg_a, _ = editor.split_edge((0, 1))
    assert ok_a, f"split_edge(0,1) failed: {msg_a}"
    ok_b, msg_b, _ = editor.split_edge((1, 2))
    assert ok_b, f"split_edge(1,2) failed: {msg_b}"
    # Attempt removal; expect rejection due to cavity-area preservation
    try:
        editor.enforce_remove_quality = False
    except Exception:
        pass
    ok_r, msg_r, _ = editor.remove_node_with_patch(1)
    assert not ok_r, "Removal should be rejected by default when cavity area is not preserved"
    msg_str = str(msg_r)
    assert ('patch area changed' in msg_str) or ('polygon fill would change cavity area' in msg_str), \
        f"Unexpected rejection reason: {msg_r}"
