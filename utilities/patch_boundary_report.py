#!/usr/bin/env python3
"""Diagnostic: report patch boundary extraction and draw explicit boundary edges."""
import csv
import matplotlib.pyplot as plt
import numpy as np
from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.patch_batching import build_patches_from_metrics_strict

def main():  # pragma: no cover
    pts, tris = build_random_delaunay(npts=40, seed=7)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=12, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    rows = []
    for p in patches:
        pid = p.get('id')
        ptype = p.get('type')
        seed = p.get('seed')
        ntris = len(p.get('tris', []))
        b = p.get('boundary')
        nloops = len(b) if b is not None else 0
        loop_lens = [len(loop) for loop in (b or [])]
        rows.append((pid, ptype, seed, ntris, nloops, loop_lens))
    print(f"Found {len(patches)} patches")
    for r in rows:
        print(f"patch id={r[0]} type={r[1]} seed={r[2]} ntris={r[3]} nloops={r[4]} loop_lens={r[5]}")
    with open('patch_boundary_report.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','type','seed','ntris','nloops','loop_lens'])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], ';'.join(map(str, r[5]))])
    plt.figure(figsize=(8,8))
    plt.triplot(editor.points[:,0], editor.points[:,1], editor.triangles, color='lightgray', lw=0.6)
    plt.scatter(editor.points[:,0], editor.points[:,1], s=6, color='k')
    from matplotlib.patches import Polygon
    cmap = plt.get_cmap('tab20')
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
        b = p.get('boundary') or []
        for loop in b:
            for j in range(len(loop)):
                a = int(loop[j]); bb = int(loop[(j+1)%len(loop)])
                pa = editor.points[a]; pb = editor.points[bb]
                plt.plot([pa[0], pb[0]], [pa[1], pb[1]], color='red', lw=2.5)
        for loop in b:
            for v in loop:
                pv = editor.points[int(v)]
                plt.scatter([pv[0]],[pv[1]], color='white', edgecolor='k', s=28, zorder=3)
    plt.gca().set_aspect('equal')
    plt.title('Explicit patch boundary edges (red) and vertices (white)')
    plt.savefig('patch_boundaries_explicit.png', dpi=180)
    plt.savefig('patch_boundaries_colored.png', dpi=180)
    print('Wrote patch_boundary_report.csv, patch_boundaries_explicit.png and patch_boundaries_colored.png')
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
