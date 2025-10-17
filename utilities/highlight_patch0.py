#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.helpers import patch_nodes_for_triangles
from sofia.sofia.geometry import point_in_polygon
from sofia.sofia.patch_batching import build_patches_from_metrics_strict
from sofia.sofia.logging_utils import get_logger

logger = get_logger('sofia.utilities.highlight_patch0')

def main():  # pragma: no cover
    pts, tris = build_random_delaunay(npts=40, seed=7)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=12, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    p0 = next((p for p in patches if p.get('id') == 0), None)
    if p0 is None:
        print('patch 0 not found'); return 1
    tris_idx = sorted(list(p0['tris']))
    boundary = p0.get('boundary', [])
    poly = boundary[0] if boundary else []
    poly_coords = [tuple(editor.points[int(x)]) for x in poly]
    patch_nodes = patch_nodes_for_triangles(editor.triangles, tris_idx)
    interior_nodes = []
    on_boundary_nodes = []
    for v in patch_nodes:
        if v in poly:
            onb = True
        else:
            onb = False
        if not onb:
            inside_flag = point_in_polygon(float(editor.points[int(v)][0]), float(editor.points[int(v)][1]), poly_coords)
            if inside_flag:
                interior_nodes.append(v)
            else:
                on_boundary_nodes.append(v)
        else:
            on_boundary_nodes.append(v)
    all_verts = set()
    for t in tris_idx:
        tri = editor.triangles[int(t)]
        all_verts.update([int(x) for x in tri])
    coords = editor.points[list(all_verts)]
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    padx = (xmax - xmin) * 0.6 + 1e-6
    pady = (ymax - ymin) * 0.6 + 1e-6
    plt.figure(figsize=(5,5))
    plt.triplot(editor.points[:,0], editor.points[:,1], editor.triangles, color='lightgray', lw=0.6)
    from matplotlib.patches import Polygon
    for t_idx in tris_idx:
        tri = editor.triangles[int(t_idx)]
        tri_coords = editor.points[[int(tri[0]), int(tri[1]), int(tri[2])]]
        poly_patch = Polygon(tri_coords, facecolor='tab:cyan', edgecolor='tab:cyan', alpha=0.25)
        plt.gca().add_patch(poly_patch)
    if poly:
        bx = [editor.points[int(v)][0] for v in poly] + [editor.points[int(poly[0])][0]]
        by = [editor.points[int(v)][1] for v in poly] + [editor.points[int(poly[0])][1]]
        plt.plot(bx, by, color='black', linewidth=2.2)
        plt.plot(bx, by, color='red', linewidth=1.6)
    for v in on_boundary_nodes:
        p = editor.points[int(v)]
        plt.scatter([p[0]], [p[1]], color='white', edgecolor='k', s=60, zorder=5)
    for v in interior_nodes:
        p = editor.points[int(v)]
        plt.scatter([p[0]], [p[1]], color='green', edgecolor='k', s=80, zorder=6)
    seed = int(p0.get('seed'))
    ps = editor.points[seed]
    plt.scatter([ps[0]], [ps[1]], marker='*', color='yellow', edgecolor='k', s=220, zorder=10)
    plt.xlim(xmin - padx, xmax + padx)
    plt.ylim(ymin - pady, ymax + pady)
    plt.gca().set_aspect('equal')
    plt.title('Patch 0 inspection: boundary(black/red), interior(green), boundary nodes(white), seed(*)')
    plt.savefig('patch0_highlight.png', dpi=200)
    logger.info('Wrote patch0_highlight.png')
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
