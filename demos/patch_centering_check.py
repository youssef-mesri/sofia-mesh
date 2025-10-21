"""Check that each node-centered patch seed lies within its patch polygon and triangles.

Moved from `check_patch_centering.py`.
"""
from __future__ import annotations
import numpy as np
from sofia.core.logging_utils import get_logger, configure_logging
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor, point_in_polygon
from sofia.core.patch_batching import build_patches_from_metrics_strict

logger = get_logger('sofia.demos.patch_centering_check')

def run_patch_centering_check(npts=40, seed=7, node_top_k=12):  # pragma: no cover
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=node_top_k, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    bad = []
    for p in patches:
        if p.get('type') != 'node':
            continue
        pid = p.get('id'); seed_v = int(p.get('seed'))
        tris_set = p.get('tris', set())
        verts = p.get('verts', set())
        boundary = p.get('boundary', [])
        in_verts = seed_v in verts
        in_tri = any(seed_v in [int(x) for x in editor.triangles[int(t)]] for t in tris_set)
        if boundary:
            poly = boundary[0]
            coords = [tuple(editor.points[int(v)]) for v in poly]
            in_poly = point_in_polygon(editor.points[seed_v][0], editor.points[seed_v][1], coords)
        else:
            in_poly = False
        dist_centroid = None
        if tris_set:
            centroids = []
            for t in tris_set:
                tri = editor.triangles[int(t)]
                pts_tri = editor.points[[int(tri[0]), int(tri[1]), int(tri[2])]]
                centroids.append(np.mean(pts_tri, axis=0))
            cent = np.mean(np.vstack(centroids), axis=0)
            dist_centroid = float(np.linalg.norm(editor.points[seed_v] - cent))
        logger.info('patch %s seed=%s in_verts=%s in_tri=%s in_poly=%s dist_centroid=%s', pid, seed_v, in_verts, in_tri, in_poly, dist_centroid)
        if not (in_verts and in_tri and in_poly):
            bad.append((pid, seed_v, in_verts, in_tri, in_poly, dist_centroid))
    logger.info('Summary: patches failing centering checks: %d', len(bad))
    for b in bad:
        logger.info('%s', b)
    return bad

if __name__ == '__main__':  # pragma: no cover
    configure_logging('INFO')
    run_patch_centering_check()