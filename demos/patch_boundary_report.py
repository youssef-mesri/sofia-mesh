"""Generate CSV + PNGs summarizing patch boundaries (moved from patch_boundary_report.py)."""
from __future__ import annotations
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sofia.sofia.logging_utils import get_logger, configure_logging
from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.patch_batching import build_patches_from_metrics_strict

logger = get_logger('sofia.demos.patch_boundary_report')

def run_patch_boundary_report(npts=40, seed=7, node_top_k=12):  # pragma: no cover
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=node_top_k, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    rows = []
    for p in patches:
        pid = p.get('id'); ptype = p.get('type'); s = p.get('seed')
        ntris = len(p.get('tris', []))
        loops = p.get('boundary') or []
        rows.append((pid, ptype, s, ntris, len(loops), [len(l) for l in loops]))
    logger.info("Found %d patches", len(patches))
    for r in rows:
        logger.info("patch id=%s type=%s seed=%s ntris=%s nloops=%s loop_lens=%s", r[0], r[1], r[2], r[3], r[4], r[5])
    with open('patch_boundary_report.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','type','seed','ntris','nloops','loop_lens'])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], ';'.join(map(str, r[5]))])
    plt.figure(figsize=(8,8))
    plt.triplot(editor.points[:,0], editor.points[:,1], editor.triangles, color='lightgray', lw=0.6)
    plt.scatter(editor.points[:,0], editor.points[:,1], s=6, color='k')
    cmap = plt.get_cmap('tab20')
    for i,p in enumerate(patches):
        if p.get('type') != 'node':
            continue
        color = cmap(i % 20)
        for t_idx in sorted(p.get('tris', [])):
            tri = editor.triangles[int(t_idx)]
            coords = editor.points[[int(tri[0]), int(tri[1]), int(tri[2])]]
            poly = Polygon(coords, facecolor=color, edgecolor=color, alpha=0.45, linewidth=0.6)
            plt.gca().add_patch(poly)
        for loop in (p.get('boundary') or []):
            for j in range(len(loop)):
                a = int(loop[j]); b = int(loop[(j+1)%len(loop)])
                pa = editor.points[a]; pb = editor.points[b]
                plt.plot([pa[0], pb[0]], [pa[1], pb[1]], color='red', lw=2.5)
            for v in loop:
                pv = editor.points[int(v)]
                plt.scatter([pv[0]],[pv[1]], color='white', edgecolor='k', s=28, zorder=3)
    plt.gca().set_aspect('equal')
    plt.title('Explicit patch boundary edges (red)')
    plt.savefig('patch_boundaries_explicit.png', dpi=180)
    plt.savefig('patch_boundaries_colored.png', dpi=180)
    logger.info('Wrote patch_boundary_report.csv, patch_boundaries_explicit.png and patch_boundaries_colored.png')
    return rows

if __name__ == '__main__':  # pragma: no cover
    configure_logging('INFO')
    run_patch_boundary_report()