"""Visualize patch batches (node-centered) with filled boundary polygons.

Originally `demo_patch_batches.py`; relocated under `demos/`.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from sofia.core.logging_utils import get_logger, configure_logging
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.patch_batching import build_patches_from_metrics_strict

logger = get_logger('sofia.demos.patch_batches')

def run_patch_batches(npts=40, seed=7, node_top_k=12, edge_top_k=6):  # pragma: no cover
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=node_top_k, edge_top_k=edge_top_k,
                                                radius=1, disjoint_on='tri', allow_overlap=False)
    plt.figure(figsize=(8,8))
    plt.triplot(editor.points[:,0], editor.points[:,1], editor.triangles, color='gray', lw=0.6)
    plt.scatter(editor.points[:,0], editor.points[:,1], s=6)
    cmap_node = plt.get_cmap('tab20')
    node_count = 0
    for p in patches:
        if p.get('type') != 'node':
            continue
        color = cmap_node(node_count % 20)
        node_count += 1
        for poly in p.get('boundary', []):
            coords = np.array([editor.points[v] for v in poly])
            plt.fill(coords[:,0], coords[:,1], color=color, alpha=0.35, edgecolor='k')
        tris_local = [editor.triangles[t] for t in sorted(p['tris'])]
        for t in tris_local:
            coords = editor.points[list(map(int,t))]
            xs = [coords[0,0], coords[1,0], coords[2,0], coords[0,0]]
            ys = [coords[0,1], coords[1,1], coords[2,1], coords[0,1]]
            plt.plot(xs, ys, color='k', lw=0.5)
        v = int(p.get('seed'))
        pt = editor.points[v]
        plt.scatter([pt[0]], [pt[1]], s=40, marker='o', facecolor='k', edgecolor='k')
    plt.gca().set_aspect('equal')
    plt.title('Patch partition (node-centered)')
    out = 'patch_batches.png'
    plt.savefig(out, dpi=200)
    logger.info('Wrote %s', out)

# backward compatible alias
main = run_patch_batches

if __name__ == '__main__':  # pragma: no cover
    configure_logging('INFO')
    run_patch_batches()