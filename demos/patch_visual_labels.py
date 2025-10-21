"""Draw patches with IDs and per-patch zooms (moved from patch_visual_labels.py)."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.patch_batching import build_patches_from_metrics_strict
from sofia.core.logging_utils import get_logger

logger = get_logger('sofia.demos.patch_visual_labels')

def run_patch_visual_labels(npts=40, seed=7, node_top_k=12):  # pragma: no cover
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=node_top_k, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    cmap = plt.get_cmap('tab20')
    plt.figure(figsize=(10,10))
    plt.triplot(editor.points[:,0], editor.points[:,1], editor.triangles, color='lightgray', lw=0.6)
    plt.scatter(editor.points[:,0], editor.points[:,1], s=6, color='k')
    for i,p in enumerate(patches):
        if p.get('type') != 'node':
            continue
        color = cmap(i % 20)
        tris_idx = sorted(p.get('tris', []))
        for t_idx in tris_idx:
            tri = editor.triangles[int(t_idx)]
            coords = editor.points[[int(tri[0]), int(tri[1]), int(tri[2])]]
            poly = Polygon(coords, facecolor=color, edgecolor=color, alpha=0.45, linewidth=0.6)
            plt.gca().add_patch(poly)
        for loop in (p.get('boundary') or []):
            for j in range(len(loop)):
                a = int(loop[j]); b = int(loop[(j+1)%len(loop)])
                pa = editor.points[a]; pb = editor.points[b]
                plt.plot([pa[0], pb[0]], [pa[1], pb[1]], color='red', lw=2.5)
        tris_idx = sorted(p.get('tris', []))
        if tris_idx:
            centroids = []
            for t in tris_idx:
                tri = editor.triangles[int(t)]
                pts_tri = editor.points[[int(tri[0]), int(tri[1]), int(tri[2])]]
                centroids.append(np.mean(pts_tri, axis=0))
            cent = np.mean(np.vstack(centroids), axis=0)
            plt.text(cent[0], cent[1], str(p.get('id')), color='k', fontsize=12, fontweight='bold', ha='center', va='center')
        else:
            seed_v = p.get('seed')
            spt = editor.points[int(seed_v)]
            plt.text(spt[0], spt[1], str(p.get('id')), color='k', fontsize=12, fontweight='bold', ha='center', va='center')
    plt.gca().set_aspect('equal')
    plt.title('Patches (labeled by id)')
    plt.savefig('patch_boundaries_labeled.png', dpi=200)
    logger.info('Wrote patch_boundaries_labeled.png')
    # per-patch zooms
    for p in patches:
        pid = p.get('id'); tris_idx = sorted(p.get('tris', []))
        if tris_idx:
            verts = set()
            for t in tris_idx:
                tri = editor.triangles[int(t)]
                verts.update([int(x) for x in tri])
            coords = editor.points[list(verts)]
            xmin, ymin = coords.min(axis=0); xmax, ymax = coords.max(axis=0)
            padx = (xmax - xmin) * 0.4 + 1e-6; pady = (ymax - ymin) * 0.4 + 1e-6
            plt.figure(figsize=(4,4))
            plt.triplot(editor.points[:,0], editor.points[:,1], editor.triangles, color='lightgray', lw=0.6)
            color = cmap(pid % 20)
            for t_idx in tris_idx:
                tri = editor.triangles[int(t_idx)]
                coords_tri = editor.points[[int(tri[0]), int(tri[1]), int(tri[2])]]
                poly = Polygon(coords_tri, facecolor=color, edgecolor=color, alpha=0.6, linewidth=0.6)
                plt.gca().add_patch(poly)
            plt.xlim(xmin - padx, xmax + padx); plt.ylim(ymin - pady, ymax + pady)
            plt.gca().set_aspect('equal'); plt.title(f'Patch {pid} zoom')
            fname = f'patch_zoom_{pid}.png'; plt.savefig(fname, dpi=200)
            logger.info('Wrote %s', fname)
        else:
            seed_v = int(p.get('seed'))
            sxy = editor.points[seed_v]
            plt.figure(figsize=(4,4))
            plt.triplot(editor.points[:,0], editor.points[:,1], editor.triangles, color='lightgray', lw=0.6)
            plt.xlim(sxy[0]-0.05, sxy[0]+0.05); plt.ylim(sxy[1]-0.05, sxy[1]+0.05)
            plt.gca().set_aspect('equal')
            fname = f'patch_zoom_{pid}.png'; plt.savefig(fname, dpi=200)
            logger.info('Wrote %s', fname)
    return patches

if __name__ == '__main__':  # pragma: no cover
    run_patch_visual_labels()