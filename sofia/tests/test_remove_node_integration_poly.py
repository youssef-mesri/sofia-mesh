import numpy as np
from types import SimpleNamespace
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.conformity import check_mesh_conformity
from sofia.core.constants import EPS_AREA
from sofia.core.geometry import triangle_area


def make_regular_polygon_star(n_vertices=5, radius=1.0):
    # center at origin
    angles = np.linspace(0.0, 2*np.pi, n_vertices, endpoint=False)
    verts = np.column_stack([np.cos(angles)*radius, np.sin(angles)*radius])
    center = np.array([[0.0, 0.0]])
    points = np.vstack([verts, center])
    center_idx = len(points) - 1
    tris = []
    for i in range(n_vertices):
        tris.append([i, (i+1) % n_vertices, center_idx])
    return points.astype(float), np.array(tris, dtype=int), center_idx


def sum_tris_area(points, tris):
    a = 0.0
    for t in tris:
        p0 = points[int(t[0])]; p1 = points[int(t[1])]; p2 = points[int(t[2])]
        a += abs(triangle_area(p0, p1, p2))
    return a


def test_remove_node_pentagon_requires_area_preservation():
    pts, tris, center = make_regular_polygon_star(5, radius=1.0)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True)
    # allow removal even if local worst-min-angle would worsen (tests control quality expectations)
    editor.enforce_remove_quality = False
    # compute removed cavity area
    removed_area = sum_tris_area(editor.points, [editor.triangles[i] for i in range(len(tris))])

    # configure area-preserving removal
    br_cfg = SimpleNamespace()
    br_cfg.require_area_preservation = True
    br_cfg.area_tol_rel = 1e-6
    br_cfg.area_tol_abs_factor = 4.0
    editor.boundary_remove_config = br_cfg

    ok, msg, info = editor.remove_node_with_patch(int(center))
    assert ok, f"remove_node_with_patch failed: {msg}"

    # active triangles should form a triangulation of the pentagon: n-2 triangles
    n = 5
    active_tris = [t for t in editor.triangles if not np.all(np.array(t) == -1)]
    assert len(active_tris) == n - 2, f"expected {n-2} active tris, got {len(active_tris)}"

    # none of the active triangles should reference the removed vertex
    assert all(int(center) not in [int(x) for x in t] for t in active_tris), "Committed triangles reference removed vertex"

    # area should be preserved within tolerances
    new_area = sum_tris_area(editor.points, active_tris)
    tol_rel = br_cfg.area_tol_rel
    tol_abs = br_cfg.area_tol_abs_factor * EPS_AREA
    assert abs(new_area - removed_area) <= max(tol_abs, tol_rel * max(1.0, removed_area)), (
        f"Area not preserved: removed={removed_area:.6e} new={new_area:.6e}")

    ok_conf, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok_conf, f"mesh not conforming after removal: {msgs}"


def test_remove_node_pentagon_relaxed_allows_non_area_preserving():
    pts, tris, center = make_regular_polygon_star(6, radius=1.0)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True)
    # allow removal even if local worst-min-angle would worsen (tests control quality expectations)
    editor.enforce_remove_quality = False
    # compute removed cavity area
    removed_area = sum_tris_area(editor.points, [editor.triangles[i] for i in range(len(tris))])

    # relaxed config: do not require strict area preservation
    br_cfg = SimpleNamespace()
    br_cfg.require_area_preservation = False
    br_cfg.area_tol_rel = 1e-6
    br_cfg.area_tol_abs_factor = 4.0
    editor.boundary_remove_config = br_cfg

    ok, msg, info = editor.remove_node_with_patch(int(center))
    assert ok, f"remove_node_with_patch failed (relaxed): {msg}"

    active_tris = [t for t in editor.triangles if not np.all(np.array(t) == -1)]
    n = 6
    assert len(active_tris) == n - 2, f"expected {n-2} active tris, got {len(active_tris)}"
    assert all(int(center) not in [int(x) for x in t] for t in active_tris), "Committed triangles reference removed vertex"

    # area may differ (we only assert conformity)
    ok_conf, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok_conf, f"mesh not conforming after relaxed removal: {msgs}"
