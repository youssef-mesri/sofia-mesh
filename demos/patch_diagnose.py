"""Diagnostic visualization for mesh patches.

Generates an image highlighting patch boundary loops with vertex order labels.
Originally `demo_patch_diagnose.py` at repo root; relocated under `demos/`.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from sofia.sofia.logging_utils import get_logger, configure_logging
from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.patch_batching import build_patches_from_metrics_strict

logger = get_logger('sofia.demos.patch_diagnose')

def run_patch_diagnose(npts=40, seed=7, node_top_k=12):  # pragma: no cover (visual demo)
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=node_top_k, edge_top_k=0,
                                                radius=1, disjoint_on='tri', allow_overlap=False)
    plt.figure(figsize=(10,10))
    plt.triplot(editor.points[:,0], editor.points[:,1], editor.triangles, color='lightgray', lw=0.6)
    plt.scatter(editor.points[:,0], editor.points[:,1], s=6, color='k')
    cmap = plt.get_cmap('tab20')
    for i,p in enumerate(patches):
        if p.get('type') != 'node':
            continue
        color = cmap(i % 20)
        tris_idx = sorted(p['tris'])
        for t_idx in tris_idx:
            t = editor.triangles[int(t_idx)]
            coords = editor.points[[int(t[0]), int(t[1]), int(t[2]), int(t[0])]]
            plt.plot(coords[:,0], coords[:,1], color=color, lw=0.8)
        for loop in p.get('boundary', []):
            for j in range(len(loop)):
                a = int(loop[j]); b = int(loop[(j+1)%len(loop)])
                pa = editor.points[a]; pb = editor.points[b]
                plt.plot([pa[0], pb[0]], [pa[1], pb[1]], color='red', lw=2.5)
            for idx_v, v in enumerate(loop):
                pv = editor.points[int(v)]
                plt.text(pv[0], pv[1], f'{i}:{idx_v}', color='blue', fontsize=8)
        v = int(p.get('seed'))
        pv = editor.points[v]
        plt.scatter([pv[0]],[pv[1]], s=80, facecolor='none', edgecolor='k', linewidth=1.2)
    plt.gca().set_aspect('equal')
    plt.title('Patch diagnostics: boundary edges in red, boundary vertex labels i:pos')
    out = 'patch_diagnose.png'
    plt.savefig(out, dpi=200)
    logger.info('Wrote %s', out)

# backward compatible alias
main = run_patch_diagnose

if __name__ == '__main__':  # pragma: no cover
    configure_logging('INFO')
    run_patch_diagnose()
