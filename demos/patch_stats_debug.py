"""Print size/metric summaries for generated patches (moved from debug_patch_stats.py)."""
from __future__ import annotations
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.patch_batching import triangle_to_vertex_metric, vertex_to_edge_metric, build_patches_from_metrics_strict
from sofia.core.logging_utils import get_logger

logger = get_logger('sofia.demos.patch_stats_debug')

def run_patch_stats_debug(npts=40, seed=7, node_top_k=12):  # pragma: no cover
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    logger.info('npts=%d', len(editor.points))
    active = [i for i,t in enumerate(editor.triangles) if not all(int(x)==-1 for x in t)]
    logger.info('active triangles=%d', len(active))
    logger.info('edge_map entries=%d', len(getattr(editor,'edge_map',{})))
    logger.info('v_map entries=%d', len(getattr(editor,'v_map',{})))
    v_metric = triangle_to_vertex_metric(editor)
    logger.info('v_metric size=%d', len(v_metric))
    if v_metric:
        some = sorted(v_metric.items(), key=lambda kv: kv[1], reverse=True)[:10]
        logger.info('top v_metric: %s', some)
    e_metric = vertex_to_edge_metric(editor, v_metric)
    logger.info('e_metric size=%d', len(e_metric))
    patches = build_patches_from_metrics_strict(editor, node_top_k=node_top_k, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    logger.info('patches returned=%d', len(patches))
    for p in patches:
        logger.info('patch %s seed=%s ntris=%d', p.get('id'), p.get('seed'), len(p.get('tris')))
    return patches

if __name__ == '__main__':  # pragma: no cover
    run_patch_stats_debug()