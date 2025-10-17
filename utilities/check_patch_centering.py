#!/usr/bin/env python3
import numpy as np
from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor, point_in_polygon
from sofia.sofia.patch_batching import build_patches_from_metrics_strict
from sofia.sofia.logging_utils import get_logger

logger = get_logger('sofia.utilities.check_patch_centering')

def main():  # pragma: no cover
    pts, tris = build_random_delaunay(npts=40, seed=7)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=12, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    bad = []
    for p in patches:
        if p.get('type') != 'node':
            continue
        pid = p.get('id')
        seed = int(p.get('seed'))
        tris = p.get('tris', set())
        boundary = p.get('boundary', [])
        in_verts = seed in (p.get('verts') or set())
        in_tri = any(seed in [int(x) for x in editor.triangles[int(t)]] for t in tris)
        if boundary:
            poly = boundary[0]
            coords = [tuple(editor.points[int(v)]) for v in poly]
            in_poly = point_in_polygon(editor.points[seed][0], editor.points[seed][1], coords)
        else:
            in_poly = False
        dist_centroid = None
        if tris:
            centroids = []
            for t in tris:
                tri = editor.triangles[int(t)]
                pts_tri = editor.points[[int(tri[0]), int(tri[1]), int(tri[2])]]
                centroids.append(np.mean(pts_tri, axis=0))
            cent = np.mean(np.vstack(centroids), axis=0)
            dist_centroid = np.linalg.norm(editor.points[seed] - cent)
        logger.info('patch %s seed=%s in_verts=%s in_tri=%s in_poly=%s dist_centroid=%s', pid, seed, in_verts, in_tri, in_poly, dist_centroid)
        if not (in_verts and in_tri and in_poly):
            bad.append((pid, seed, in_verts, in_tri, in_poly, dist_centroid))
    logger.info('\nSummary: patches failing centering checks: %d', len(bad))
    for b in bad:
        logger.info('%s', b)
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
