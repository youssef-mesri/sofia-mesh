#!/usr/bin/env python3
from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.patch_batching import build_patches_from_metrics_strict

def main():  # pragma: no cover
    pts, tris = build_random_delaunay(npts=40, seed=7)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    patches = build_patches_from_metrics_strict(editor, node_top_k=12, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    print('Total patches:', len(patches))
    for p in patches:
        pid = p.get('id'); ptype = p.get('type'); seed = p.get('seed')
        tris = p.get('tris'); verts = p.get('verts'); boundary = p.get('boundary')
        print('---')
        print('id', pid, 'type', ptype, 'seed', seed)
        print('ntris', len(tris) if tris is not None else 'None', 'nverts', len(verts) if verts is not None else 'None')
        print('tris sample', sorted(list(tris))[:10] if tris else [])
        print('boundary', boundary)
        if not tris:
            print('*** EMPTY tris for patch', pid)
        if not boundary:
            print('*** EMPTY boundary for patch', pid)
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
