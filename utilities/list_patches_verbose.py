#!/usr/bin/env python3
from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.core.patch_batching import build_patches_from_metrics_strict
from sofia.core.logging_utils import get_logger

logger = get_logger('sofia.utilities.list_patches_verbose')

def main():  # pragma: no cover
    pts, tris = build_random_delaunay(npts=40, seed=7)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=12, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    logger.info('Total patches: %d', len(patches))
    for p in patches:
        pid = p.get('id'); ptype = p.get('type'); seed = p.get('seed')
        tris = p.get('tris'); verts = p.get('verts'); boundary = p.get('boundary')
        logger.info('---')
        logger.info('id %s type %s seed %s', pid, ptype, seed)
        logger.info('ntris %s nverts %s', len(tris) if tris is not None else 'None', len(verts) if verts is not None else 'None')
        logger.info('tris sample %s', sorted(list(tris))[:10] if tris else [])
        logger.info('boundary %s', boundary)
        if not tris:
            logger.info('*** EMPTY tris for patch %s', pid)
        if not boundary:
            logger.info('*** EMPTY boundary for patch %s', pid)
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
