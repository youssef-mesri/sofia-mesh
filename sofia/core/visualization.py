"""Visualization helpers for the remeshing driver.

Separated from `remesh_driver` to keep the core driver logic lean.
"""
from __future__ import annotations

import logging
import os as _os
import matplotlib as _mpl
# Ensure a non-interactive backend in headless environments before importing pyplot
if not _os.environ.get('MPLBACKEND'):
    try:
        _mpl.use('Agg')
    except Exception:
        pass
import numpy as np
import matplotlib.pyplot as plt

from .mesh_modifier2 import check_mesh_conformity
from .conformity import simulate_compaction_and_check
from .diagnostics import extract_boundary_loops
from .logging_utils import get_logger

logger = get_logger('sofia.viz')


def plot_mesh(
    editor,
    outname="mesh.png",
    annotate_bad=None,
    annotate_rejected=None,
    highlight_crossings=True,
    highlight_boundary_loops=True,
    loop_color_mode: str = "per-loop",
    loop_vertex_labels: bool = False,
):
    """Plot a compacted view of the mesh with optional overlays.

    Args:
        editor: PatchBasedMeshEditor-like with .points and .triangles
        outname: output image path
        annotate_bad: iterable of original triangle indices to fill in orange
        annotate_rejected: iterable of original triangle indices to fill in red
        highlight_crossings: if True, annotate crossing count (via simulation)
        highlight_boundary_loops: if True, draw boundary loops overlay
        loop_color_mode: 'per-loop' to color each loop differently, or 'uniform' for a single color
        loop_vertex_labels: if True, label loop vertices with their order index in the loop
    """
    tris = np.array(editor.triangles)
    pts = np.array(editor.points)
    active_mask = ~np.all(tris == -1, axis=1)
    active_tris = tris[active_mask]
    if active_tris.size == 0:
        plt.figure(figsize=(6, 6))
        pts = np.array(editor.points)
        if pts.shape[0] >= 2:
            # Draw the polygon as a closed polyline
            xs = list(pts[:, 0]) + [pts[0, 0]]
            ys = list(pts[:, 1]) + [pts[0, 1]]
            plt.plot(xs, ys, color=(0.85, 0.2, 0.2), linewidth=1.8)
            # Scale marker size down for dense point sets so points don't dominate
            npts = max(1, pts.shape[0])
            s = max(0.6, min(12.0, 200.0 / float(npts)))
            plt.scatter(pts[:, 0], pts[:, 1], s=s, color='black')
        plt.title('empty mesh (polygon only)')
        plt.gca().set_aspect('equal')
        plt.savefig(outname, dpi=150)
        plt.close()
        return
    used_verts = sorted(set(active_tris.flatten().tolist()))
    used_verts = [v for v in used_verts if v >= 0]
    mapping = {old: new for new, old in enumerate(used_verts)}
    new_points = pts[used_verts]
    new_tris = []
    for t in active_tris:
        if np.any(t < 0):
            continue
        try:
            new_tris.append([mapping[int(t[0])], mapping[int(t[1])], mapping[int(t[2])]])
        except KeyError:
            continue
    new_tris = np.array(new_tris, dtype=int)
    ok_conf, msgs = check_mesh_conformity(new_points, new_tris, allow_marked=False)
    crossings = []
    if highlight_crossings:
        ok_sim, sim_msgs, _ = simulate_compaction_and_check(editor.points, editor.triangles, reject_crossing_edges=True)
        for m in sim_msgs:
            if 'crossing edges detected' in m:
                try:
                    part = m.split('detected:')[1].strip()
                    crossings.append(part)
                except Exception:
                    pass
    # Boundary loops (using consolidated helper)
    loops = extract_boundary_loops(new_points, new_tris) if highlight_boundary_loops else []
    if not ok_conf:
        logger.warning('Plotting mesh: compacted copy NOT conforming: %s', msgs)
    plt.figure(figsize=(6, 6))
    plt.triplot(new_points[:, 0], new_points[:, 1], new_tris, lw=0.6)
    # scale markers by vertex count
    npts = max(1, new_points.shape[0])
    s = max(0.6, min(8.0, 200.0 / float(npts)))
    plt.scatter(new_points[:, 0], new_points[:, 1], s=s)
    tris_all = np.array(editor.triangles)
    active_mask = ~np.all(tris_all == -1, axis=1)
    active_idx = np.nonzero(active_mask)[0].tolist()
    if annotate_bad:
        for tidx in annotate_bad:
            if tidx in active_idx:
                local = active_idx.index(tidx)
                tri = new_tris[local]
                pts_local = new_points[[int(tri[0]), int(tri[1]), int(tri[2])]]
                plt.fill(pts_local[:, 0], pts_local[:, 1], facecolor='orange', alpha=0.4, edgecolor='none')
    if annotate_rejected:
        for tidx in annotate_rejected:
            if tidx in active_idx:
                local = active_idx.index(tidx)
                tri = new_tris[local]
                pts_local = new_points[[int(tri[0]), int(tri[1]), int(tri[2])]]
                plt.fill(pts_local[:, 0], pts_local[:, 1], facecolor='red', alpha=0.5, edgecolor='none')
    plt.gca().set_aspect('equal')
    if highlight_boundary_loops and loops:
        # choose colors per loop or uniformly
        palette = [(0.85,0.2,0.2), (0.2,0.6,0.8), (0.2,0.8,0.3), (0.75,0.5,0.2), (0.6,0.2,0.7)]
        uniform_color = (0.85, 0.2, 0.2)
        for i, loop in enumerate(loops):
            col = palette[i % len(palette)] if loop_color_mode == "per-loop" else uniform_color
            # draw as a polyline (close it visually)
            if len(loop) >= 2:
                xs = [new_points[int(v)][0] for v in loop] + [new_points[int(loop[0])][0]]
                ys = [new_points[int(v)][1] for v in loop] + [new_points[int(loop[0])][1]]
                plt.plot(xs, ys, color=col, linewidth=1.4)
            # optional small markers
            for v in loop:
                p = new_points[int(v)]
                plt.plot([p[0]],[p[1]], marker='o', markersize=2.5, color=col)
            # optional labels with vertex order within the loop
            if loop_vertex_labels:
                for j, v in enumerate(loop):
                    pv = new_points[int(v)]
                    plt.text(pv[0], pv[1], str(j), fontsize=7, color=col)
    # annotate crossings textually
    if crossings:
        plt.text(0.02, 0.98, f"crossings: {len(crossings)}", transform=plt.gca().transAxes,
                 va='top', ha='left', fontsize=8, color='crimson')
    plt.title(outname)
    plt.savefig(outname, dpi=150)
    plt.close()


from matplotlib.patches import Patch as _MPatch


def plot_mesh_by_tri_groups(
    editor,
    tri_groups,
    outname="mesh.png",
    alpha: float = 0.35,
    palette=None,
    highlight_boundary_loops: bool = False,
    annotate_vertices=None,
    annotate_color=(0.85, 0.2, 0.2),
    annotate_size: float = 24.0,
    annotate_labels: bool = False,
):
    """Plot the mesh and fill specified triangle groups with distinct colors.

    Args:
        editor: PatchBasedMeshEditor-like with .points and .triangles
        tri_groups: dict(group_key -> iterable of original triangle indices)
        outname: output image path
        alpha: face transparency for filled groups
        palette: optional list of RGB tuples to cycle; defaults to tab20-ish set
        highlight_boundary_loops: if True, overlay boundary loops on top
    """
    tris = np.array(editor.triangles)
    pts = np.array(editor.points)
    active_mask = ~np.all(tris == -1, axis=1)
    active_tris = tris[active_mask]
    if active_tris.size == 0:
        plt.figure(figsize=(6, 6))
        plt.title('empty mesh')
        plt.savefig(outname, dpi=150)
        plt.close()
        return
    used_verts = sorted({int(v) for t in active_tris for v in t if int(v) >= 0})
    mapping = {old: new for new, old in enumerate(used_verts)}
    new_points = pts[used_verts]
    # map original active tri index -> local index after compaction
    tris_all = np.array(editor.triangles)
    active_idx = np.nonzero(~np.all(tris_all == -1, axis=1))[0].tolist()
    local_by_orig = {orig: i for i, orig in enumerate(active_idx)}
    new_tris = []
    for t in tris_all[active_idx]:
        if np.any(t < 0):
            new_tris.append([-1, -1, -1])
            continue
        try:
            new_tris.append([mapping[int(t[0])], mapping[int(t[1])], mapping[int(t[2])]])
        except KeyError:
            new_tris.append([-1, -1, -1])
    new_tris = np.array(new_tris, dtype=int)

    if palette is None:
        palette = [
            (0.121, 0.466, 0.705), (1.0, 0.498, 0.054), (0.172, 0.627, 0.172), (0.839, 0.153, 0.157),
            (0.580, 0.404, 0.741), (0.549, 0.337, 0.294), (0.890, 0.467, 0.761), (0.498, 0.498, 0.498),
            (0.737, 0.741, 0.133), (0.090, 0.745, 0.811)
        ]

    plt.figure(figsize=(6, 6))
    # base wireframe
    # filter out invalid rows
    valid_rows = np.all(new_tris >= 0, axis=1)
    tri_plot = new_tris[valid_rows]
    plt.triplot(new_points[:, 0], new_points[:, 1], tri_plot, lw=0.6)
    # scale markers by vertex count
    npts = max(1, new_points.shape[0])
    s = max(0.6, min(8.0, 200.0 / float(npts)))
    plt.scatter(new_points[:, 0], new_points[:, 1], s=s)

    # paint groups
    legend_patches = []
    for gi, (gk, tri_list) in enumerate(tri_groups.items()):
        col = palette[gi % len(palette)]
        count = 0
        for tidx in tri_list:
            local = local_by_orig.get(int(tidx))
            if local is None:
                continue
            tri = new_tris[local]
            if np.any(np.array(tri) < 0):
                continue
            pts_local = new_points[[int(tri[0]), int(tri[1]), int(tri[2])]]
            plt.fill(pts_local[:, 0], pts_local[:, 1], facecolor=col, alpha=alpha, edgecolor='none')
            count += 1
        legend_patches.append(_MPatch(facecolor=col, edgecolor='none', alpha=alpha, label=f'part {gk} (n={count})'))

    if legend_patches:
        plt.legend(handles=legend_patches, loc='upper right', fontsize=8, frameon=True)

    # optional boundary loop overlay
    if highlight_boundary_loops:
        loops = extract_boundary_loops(new_points, tri_plot)
        loop_color = (0.1, 0.1, 0.1)
        for loop in loops:
            if len(loop) >= 2:
                xs = [new_points[int(v)][0] for v in loop] + [new_points[int(loop[0])][0]]
                ys = [new_points[int(v)][1] for v in loop] + [new_points[int(loop[0])][1]]
                plt.plot(xs, ys, color=loop_color, linewidth=1.0)

    # Optional overlay of specific vertices (old/original indices)
    if annotate_vertices is not None:
        try:
            to_annot = [int(v) for v in annotate_vertices]
        except Exception:
            to_annot = []
        for v_old in to_annot:
            if v_old in mapping:
                v_local = mapping[int(v_old)]
                p = new_points[int(v_local)]
                plt.plot([p[0]],[p[1]], marker='o', markersize=max(2.0, annotate_size/3.0),
                         markeredgecolor='k', markerfacecolor=annotate_color, linewidth=0.0)
                if annotate_labels:
                    plt.text(p[0], p[1], str(int(v_old)), fontsize=7, color=annotate_color)

    plt.gca().set_aspect('equal')
    plt.title(outname)
    plt.savefig(outname, dpi=150)
    plt.close()


__all__ = ['plot_mesh', 'plot_mesh_by_tri_groups']

