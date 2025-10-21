#!/usr/bin/env python3
"""
Generate before/after plots for the complex boundary removal scenarios from
sofia/sofia/tests/test_boundary_remove_complex.py.

Outputs PNGs into demos/output/.
"""
import os
from typing import Optional
import numpy as np
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.visualization import plot_mesh
from sofia.core.constants import EPS_AREA
from sofia.core.diagnostics import compact_copy, extract_boundary_loops
from sofia.core.config import BoundaryRemoveConfig
from sofia.core.helpers import boundary_polygons_from_patch, select_outer_polygon
from sofia.core.triangulation import polygon_signed_area


OUTDIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTDIR, exist_ok=True)


def _enforce_ccw(points: np.ndarray, tris: np.ndarray, eps: float = None) -> np.ndarray:
    """Return tris with triangles oriented CCW, using a tolerance to avoid flipping near-degenerate cases.

    Only flips when the signed area is less than -2*EPS_AREA (significantly clockwise). Zero/near-zero area triangles are left as-is.
    """
    T = tris.copy()
    thr = 2.0 * (float(eps) if eps is not None else float(EPS_AREA))
    for i, t in enumerate(T):
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        p0, p1, p2 = points[a], points[b], points[c]
        area2 = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
        if area2 < -thr:
            T[i] = np.array([a, c, b], dtype=T.dtype)
    return T


def mesh_with_boundary_vertex_degree_three():
    # Square corners with two interior points; triangulation chosen so vertex 1 has degree 3 and the mesh is conforming
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1 (boundary, target to remove)
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.3],  # 4 interior near bottom
        [0.8, 0.6],  # 5 interior near right
    ], dtype=float)
    # Triangulation matching the test case; removal of vertex 1 should eliminate it from the mesh
    tris = np.array([
        [4, 0, 1],  # involves 1
        [4, 1, 5],  # involves 1
        [5, 1, 2],  # involves 1
        [4, 0, 3],
        [4, 3, 2],
        [5, 2, 4],
    ], dtype=int)
    return pts, tris


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


def plot_degree_three_case():
    pts, tris = mesh_with_boundary_vertex_degree_three()
    tris = _enforce_ccw(pts, tris)
    # Keep a pristine copy for side-by-side plotting
    br_cfg = BoundaryRemoveConfig(
        prefer_area_preserving_star=True,
        prefer_worst_angle_star=True,
        require_area_preservation=True,
    )
    editor_before = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True, boundary_remove_config=br_cfg)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True, boundary_remove_config=br_cfg)
    try:
        editor.enforce_remove_quality = False
    except Exception:
        pass
    # Before
    plot_mesh(editor_before, outname=os.path.join(OUTDIR, 'boundary_degree3_before.png'))
    # Area targets before removal
    v_rm = 1
    incident = [i for i, t in enumerate(editor_before.triangles) if v_rm in [int(t[0]), int(t[1]), int(t[2])]]
    polys = boundary_polygons_from_patch(editor_before.triangles, incident)
    cyc = select_outer_polygon(editor_before.points, polys)
    poly_area = None
    cavity_area = 0.0
    for ti in incident:
        t = editor_before.triangles[int(ti)]
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        p0, p1, p2 = editor_before.points[a], editor_before.points[b], editor_before.points[c]
        cavity_area += abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])) * 0.5
    if cyc is not None:
        filtered = [int(v) for v in cyc if int(v) != v_rm]
        if len(filtered) >= 2 and filtered[0] == filtered[-1]:
            filtered = filtered[:-1]
        if len(filtered) >= 3:
            poly_area = abs(polygon_signed_area([editor_before.points[int(v)] for v in filtered]))
    # Remove vertex 1
    pre_len = len(editor.triangles)
    ok, msg, _ = editor.remove_node_with_patch(v_rm)
    if not ok:
        print('[WARN] remove_node_with_patch(1) failed in degree-3 case:', msg)
    else:
        # Compute appended area
        appended = [t for t in editor.triangles[pre_len:] if not np.all(np.array(t) == -1)]
        appended_area = 0.0
        for t in appended:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            p0, p1, p2 = editor.points[a], editor.points[b], editor.points[c]
            appended_area += abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])) * 0.5
        tol_abs = 4.0*float(EPS_AREA); tol_rel = 1e-12
        ok_poly = (poly_area is not None) and (abs(appended_area - poly_area) <= max(tol_abs, tol_rel*max(1.0, poly_area)))
        ok_cavity = abs(appended_area - cavity_area) <= max(tol_abs, tol_rel*max(1.0, cavity_area))
        print(f"[degree3] area check: appended={appended_area:.6e} poly={poly_area if poly_area is not None else float('nan'):.6e} ({'OK' if ok_poly else 'X'}) cavity={cavity_area:.6e} ({'OK' if ok_cavity else 'VIOLATION'})")
    # After
    plot_mesh(editor, outname=os.path.join(OUTDIR, 'boundary_degree3_after.png'))
    # Side-by-side boundary comparison for clarity
    _plot_boundary_side_by_side(editor_before, editor, os.path.join(OUTDIR, 'boundary_degree3_side_by_side.png'), removed_vertex=1)


def plot_corner_with_splits_case():
    pts, tris = square_with_center()
    tris = _enforce_ccw(pts, tris)
    # Keep a pristine copy for side-by-side plotting
    br_cfg = BoundaryRemoveConfig(
        prefer_area_preserving_star=True,
        prefer_worst_angle_star=True,
        require_area_preservation=True,
    )
    editor_before = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True, enforce_split_quality=False, boundary_remove_config=br_cfg)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True, enforce_split_quality=False, boundary_remove_config=br_cfg)
    # Apply splits around corner 1
    ok_a, msg_a, _ = editor.split_edge((0, 1))
    if not ok_a:
        print('[WARN] split_edge(0,1) failed:', msg_a)
    ok_b, msg_b, _ = editor.split_edge((1, 2))
    if not ok_b:
        print('[WARN] split_edge(1,2) failed:', msg_b)
    # Apply same splits to before-editor for consistent before snapshot
    ok_a2, msg_a2, _ = editor_before.split_edge((0, 1))
    ok_b2, msg_b2, _ = editor_before.split_edge((1, 2))
    # Before removal (after splits)
    plot_mesh(editor_before, outname=os.path.join(OUTDIR, 'boundary_corner_splits_before.png'))
    # Remove vertex 1
    try:
        editor.enforce_remove_quality = False
    except Exception:
        pass
    # Compute targets before removal
    v_rm = 1
    incident = [i for i, t in enumerate(editor_before.triangles) if v_rm in [int(t[0]), int(t[1]), int(t[2])]]
    polys = boundary_polygons_from_patch(editor_before.triangles, incident)
    cyc = select_outer_polygon(editor_before.points, polys)
    poly_area = None
    cavity_area = 0.0
    for ti in incident:
        t = editor_before.triangles[int(ti)]
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        p0, p1, p2 = editor_before.points[a], editor_before.points[b], editor_before.points[c]
        cavity_area += abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])) * 0.5
    if cyc is not None:
        filtered = [int(v) for v in cyc if int(v) != v_rm]
        if len(filtered) >= 2 and filtered[0] == filtered[-1]:
            filtered = filtered[:-1]
        if len(filtered) >= 3:
            poly_area = abs(polygon_signed_area([editor_before.points[int(v)] for v in filtered]))
    pre_len = len(editor.triangles)
    ok_r, msg_r, _ = editor.remove_node_with_patch(v_rm)
    if not ok_r:
        print('[WARN] remove_node_with_patch(1) failed in corner-with-splits case:', msg_r)
    else:
        # Compute appended area
        appended = [t for t in editor.triangles[pre_len:] if not np.all(np.array(t) == -1)]
        appended_area = 0.0
        for t in appended:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            p0, p1, p2 = editor.points[a], editor.points[b], editor.points[c]
            appended_area += abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])) * 0.5
        tol_abs = 4.0*float(EPS_AREA); tol_rel = 1e-12
        ok_poly = (poly_area is not None) and (abs(appended_area - poly_area) <= max(tol_abs, tol_rel*max(1.0, poly_area)))
        ok_cavity = abs(appended_area - cavity_area) <= max(tol_abs, tol_rel*max(1.0, cavity_area))
        print(f"[corner+splits] area check: appended={appended_area:.6e} poly={poly_area if poly_area is not None else float('nan'):.6e} ({'OK' if ok_poly else 'X'}) cavity={cavity_area:.6e} ({'OK' if ok_cavity else 'VIOLATION'})")
        # If cavity area is violated, annotate and save a violation image instead of normal 'after'
        if not ok_cavity:
            from sofia.core.visualization import plot_mesh as _plot
            out_violation = os.path.join(OUTDIR, 'boundary_corner_splits_after_VIOLATION.png')
            _plot(editor, outname=out_violation)
            return
    # After (only when success and no early return)
    plot_mesh(editor, outname=os.path.join(OUTDIR, 'boundary_corner_splits_after.png'))
    # Side-by-side boundary comparison
    _plot_boundary_side_by_side(editor_before, editor, os.path.join(OUTDIR, 'boundary_corner_splits_side_by_side.png'), removed_vertex=1)


def _plot_boundary_side_by_side(editor_before: PatchBasedMeshEditor, editor_after: PatchBasedMeshEditor, outname: str, removed_vertex: Optional[int] = None):
    """Save a side-by-side figure with boundary overlays before/after.

    - Highlights boundary loops in thick lines.
    - Marks the removed vertex in the 'before' plot if provided.
    - Highlights new boundary edges that appear only in 'after' in green.
    """
    import matplotlib.pyplot as plt
    # Compact both states
    pts_b, tris_b, map_b, _ = compact_copy(editor_before)
    pts_a, tris_a, map_a, _ = compact_copy(editor_after)
    # Extract boundary loops
    loops_b = extract_boundary_loops(pts_b, tris_b)
    loops_a = extract_boundary_loops(pts_a, tris_a)
    # Build boundary edge sets in compacted index space
    def loops_to_edges(loops):
        edges = set()
        for lp in loops:
            if len(lp) < 2: continue
            for i in range(len(lp)):
                u = int(lp[i]); v = int(lp[(i+1) % len(lp)])
                edges.add(tuple(sorted((u, v))))
        return edges
    edges_b_c = loops_to_edges(loops_b)
    edges_a_c = loops_to_edges(loops_a)
    # Map compacted edges back to original vertex ids for a meaningful comparison
    import numpy as _np
    def old_index_edges(edges_c, new_to_old):
        out = set()
        for (u,v) in edges_c:
            ou = int(new_to_old[int(u)])
            ov = int(new_to_old[int(v)])
            if ou >= 0 and ov >= 0:
                out.add(tuple(sorted((ou, ov))))
        return out
    # build new_to_old arrays
    nnb = int(_np.max(map_b)) + 1 if map_b.size and _np.any(map_b >= 0) else 0
    new_to_old_b = _np.full((nnb,), -1, dtype=_np.int32)
    if nnb:
        mask_b = map_b >= 0
        new_to_old_b[map_b[mask_b]] = _np.arange(map_b.shape[0], dtype=_np.int32)[mask_b]
    nna = int(_np.max(map_a)) + 1 if map_a.size and _np.any(map_a >= 0) else 0
    new_to_old_a = _np.full((nna,), -1, dtype=_np.int32)
    if nna:
        mask_a = map_a >= 0
        new_to_old_a[map_a[mask_a]] = _np.arange(map_a.shape[0], dtype=_np.int32)[mask_a]
    edges_b_old = old_index_edges(edges_b_c, new_to_old_b)
    edges_a_old = old_index_edges(edges_a_c, new_to_old_a)
    new_edges_old = edges_a_old - edges_b_old
    # Make figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Before subplot
    ax = axes[0]
    if tris_b.size:
        ax.triplot(pts_b[:,0], pts_b[:,1], tris_b, lw=0.6, color='gray')
    ax.scatter(pts_b[:,0], pts_b[:,1], s=6, color='black')
    for lp in loops_b:
        xs = [pts_b[int(v)][0] for v in lp] + [pts_b[int(lp[0])][0]]
        ys = [pts_b[int(v)][1] for v in lp] + [pts_b[int(lp[0])][1]]
        ax.plot(xs, ys, color=(0.85,0.2,0.2), linewidth=2.0)
    if removed_vertex is not None:
        # Map original index to compact before index to mark
        if removed_vertex < map_b.shape[0] and map_b[int(removed_vertex)] >= 0:
            cv = int(map_b[int(removed_vertex)])
            p = pts_b[cv]
            ax.plot([p[0]], [p[1]], marker='o', markersize=6.0, color='red')
    ax.set_aspect('equal'); ax.set_title('Before (boundary in red)')
    # After subplot
    ax = axes[1]
    if tris_a.size:
        ax.triplot(pts_a[:,0], pts_a[:,1], tris_a, lw=0.6, color='gray')
    ax.scatter(pts_a[:,0], pts_a[:,1], s=6, color='black')
    # Draw after boundary
    for lp in loops_a:
        xs = [pts_a[int(v)][0] for v in lp] + [pts_a[int(lp[0])][0]]
        ys = [pts_a[int(v)][1] for v in lp] + [pts_a[int(lp[0])][1]]
        ax.plot(xs, ys, color=(0.85,0.2,0.2), linewidth=2.0)
    # Highlight new boundary edges in green (convert old indices to after-compact indices for plotting)
    # Build old->new (after) map to locate coordinates
    old_to_new_after = map_a
    for (uo, vo) in new_edges_old:
        # map to after-compact indices; skip if either missing (e.g., removed vertex)
        cu = int(old_to_new_after[uo]) if (uo < old_to_new_after.shape[0]) else -1
        cv = int(old_to_new_after[vo]) if (vo < old_to_new_after.shape[0]) else -1
        if cu < 0 or cv < 0: continue
        pu, pv = pts_a[cu], pts_a[cv]
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color=(0.1,0.7,0.2), linewidth=3.0)
    ax.set_aspect('equal'); ax.set_title('After (new boundary edges in green)')
    plt.tight_layout()
    fig.savefig(outname, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    plot_degree_three_case()
    plot_corner_with_splits_case()
    print('Saved plots to', OUTDIR)
