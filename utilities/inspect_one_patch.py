from sofia.sofia.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor
from sofia.sofia.patch_batching import vertex_patch, patch_boundary_loops
import numpy as np

pts, tris = build_random_delaunay(npts=40, seed=7)
editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

# pick top vertex from v_metric produced earlier (41) but we compute anew
nodes = sorted([v for v in editor.v_map.keys()], key=lambda v: len(editor.v_map.get(v,[])), reverse=True)
print('candidate nodes (by valence):', nodes[:10])
seed = nodes[0]
print('seed:', seed)
tris_set = set(vertex_patch(editor, seed, radius=1))
print('vertex_patch tris:', tris_set)

# boundary edges via editor.edge_map counting
edge_map = editor.edge_map
boundary_edges_emap = []
for e, tri_ids in edge_map.items():
    inside = tris_set.intersection(set(int(x) for x in tri_ids))
    if len(inside) == 1:
        boundary_edges_emap.append(e)
print('boundary_edges via edge_map:', boundary_edges_emap)

# compute by counting edges in tri list
edge_count = {}
for t_idx in tris_set:
    tri = editor.triangles[int(t_idx)]
    for i in range(3):
        a = int(tri[i]); b = int(tri[(i+1)%3])
        key = tuple(sorted((a,b)))
        edge_count[key] = edge_count.get(key, 0) + 1
boundary_edges_count = [e for e,c in edge_count.items() if c==1]
print('boundary_edges via recount:', boundary_edges_count)

polys = patch_boundary_loops(tris_set, editor)
print('patch_boundary_loops returned:', polys)

# show local triangles coordinates
for t in sorted(tris_set):
    t = editor.triangles[int(t)]
    coords = editor.points[[int(t[0]),int(t[1]),int(t[2]),int(t[0])]]
    print('tri', coords.tolist())
