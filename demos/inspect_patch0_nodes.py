"""Inspect nodes of patch 0 (moved from inspect_patch0_nodes.py)."""
from __future__ import annotations
from sofia.core.logging_utils import get_logger, configure_logging
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.helpers import patch_nodes_for_triangles
from sofia.core.geometry import point_in_polygon
from sofia.core.patch_batching import build_patches_from_metrics_strict

logger = get_logger('sofia.demos.inspect_patch0_nodes')

def run_inspect_patch0_nodes(npts=40, seed=7, node_top_k=12):  # pragma: no cover
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=node_top_k, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    p0 = None
    for p in patches:
        if p.get('id') == 0:
            p0 = p; break
    if p0 is None:
        logger.warning('patch 0 not found')
        return None
    tris = sorted(list(p0['tris']))
    boundary = p0.get('boundary', [])
    poly = boundary[0] if boundary else None
    logger.info('patch 0: %s', p0)
    logger.info('tris: %s', tris)
    logger.info('boundary poly: %s', boundary)
    poly_coords = [tuple(editor.points[int(x)]) for x in poly] if poly else []
    logger.info('poly coords: %s', poly_coords)
    patch_nodes = patch_nodes_for_triangles(editor.triangles, tris)
    logger.info('patch_nodes: %s', patch_nodes)
    inside = []
    for v in patch_nodes:
        if poly:
            inside_flag = point_in_polygon(float(editor.points[int(v)][0]), float(editor.points[int(v)][1]), poly_coords)
        else:
            inside_flag = False
        inside.append((v, inside_flag))
        status = 'on boundary' if poly and v in poly else ('inside' if inside_flag else 'outside')
        logger.info('vertex %s: coord=%s status=%s', v, tuple(editor.points[int(v)]), status)
    logger.info('Summary interior tests:')
    logger.info('any strictly inside? %s', any(flag for (_,flag) in inside))
    logger.info('inside list: %s', [v for v,flag in inside if flag])
    return p0

if __name__ == '__main__':  # pragma: no cover  
    configure_logging('INFO')
    run_inspect_patch0_nodes()