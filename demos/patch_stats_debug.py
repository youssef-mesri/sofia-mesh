"""Print size/metric summaries for generated patches (moved from debug_patch_stats.py)."""
from __future__ import annotations
from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.patch_batching import triangle_to_vertex_metric, vertex_to_edge_metric, build_patches_from_metrics_strict

def run_patch_stats_debug(npts=40, seed=7, node_top_k=12):  # pragma: no cover
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    print('npts=', len(editor.points))
    active = [i for i,t in enumerate(editor.triangles) if not all(int(x)==-1 for x in t)]
    print('active triangles=', len(active))
    print('edge_map entries=', len(getattr(editor,'edge_map',{})))
    print('v_map entries=', len(getattr(editor,'v_map',{})))
    v_metric = triangle_to_vertex_metric(editor)
    print('v_metric size=', len(v_metric))
    if v_metric:
        some = sorted(v_metric.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print('top v_metric:', some)
    e_metric = vertex_to_edge_metric(editor, v_metric)
    print('e_metric size=', len(e_metric))
    patches = build_patches_from_metrics_strict(editor, node_top_k=node_top_k, edge_top_k=0, radius=1, disjoint_on='tri', allow_overlap=False)
    print('patches returned=', len(patches))
    for p in patches:
        print('patch', p.get('id'), 'seed', p.get('seed'), 'ntris', len(p.get('tris')))
    return patches

if __name__ == '__main__':  # pragma: no cover
    run_patch_stats_debug()