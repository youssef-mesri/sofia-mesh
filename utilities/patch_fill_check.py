#!/usr/bin/env python3
import numpy as np
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor, point_in_polygon, triangle_area
from sofia.core.patch_batching import build_patches_from_metrics_strict
from sofia.core.logging_utils import get_logger

logger = get_logger('sofia.utilities.patch_fill_check')

def main():  # pragma: no cover
    pts, tris = build_random_delaunay(npts=40, seed=7)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=12, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    suspicious = []
    for p in patches:
        pid = p['id']
        tris = sorted(list(p['tris']))
        b = p.get('boundary')
        if not b:
            logger.info('patch %s no boundary', pid)
            suspicious.append((pid, 'no_boundary'))
            continue
        poly = b[0]
        poly_coords = [tuple(editor.points[int(x)]) for x in poly]
        cent_in = 0
        areas_sum = 0.0
        for t in tris:
            tri = editor.triangles[int(t)]
            coords = editor.points[[int(tri[0]), int(tri[1]), int(tri[2])]]
            centroid = np.mean(coords, axis=0)
            inside = point_in_polygon(centroid[0], centroid[1], poly_coords)
            if inside:
                cent_in += 1
                areas_sum += abs(triangle_area(coords[0], coords[1], coords[2]))
            logger.info('patch %s: ntris=%d centroids_inside=%d total_tri_area=%0.6f poly_vertices=%d', pid, len(tris), cent_in, areas_sum, len(poly))
            if cent_in == 0:
                suspicious.append((pid, 'no_tri_centroid_inside'))
    logger.info('\nSuspicious patches: %s', suspicious)
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
