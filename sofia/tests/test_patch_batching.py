import os
import numpy as np
import matplotlib.pyplot as plt

from sofia.core.mesh_modifier2 import build_random_delaunay, PatchBasedMeshEditor, triangle_angles, is_active_triangle
from sofia.core.patch_batching import build_patches_from_triangles, partition_batches, patch_boundary_loops


def color_map_for_batches(n_batches):
    cmap = plt.get_cmap('tab20')
    return [cmap(i % 20) for i in range(n_batches)]


def test_partition_and_visualize(tmp_path):
    # Write visualization to a stable, easy-to-inspect path under the repo root.
    out_dir = os.path.join(os.getcwd(), 'test-outputs')
    os.makedirs(out_dir, exist_ok=True)
    pts, tris = build_random_delaunay(npts=80, seed=123)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

    # compute per-triangle min-angle and take top-K worst
    tri_mins = []
    for tidx, tri in enumerate(editor.triangles):
        if np.all(np.array(tri) == -1):
            tri_mins.append((tidx, 180.0))
            continue
        angs = triangle_angles(editor.points[int(tri[0])], editor.points[int(tri[1])], editor.points[int(tri[2])])
        tri_mins.append((tidx, min(angs)))

    tri_mins.sort(key=lambda x: x[1])
    worst = [t for t,_ in tri_mins[:30]]

    # request many seeds (top_k) and maximize number of patches by enforcing triangle-disjointness
    patches = build_patches_from_triangles(editor, worst, mode='vertex', radius=1, disjoint_on='tri', top_k=80, allow_overlap=False)
    batches = partition_batches(patches)

    # Assert independence: patches in a batch do not share triangle indices (we requested tri-disjoint)
    pid_to_tris = {p['id']: p['tris'] for p in patches}
    for batch in batches:
        all_tris = set()
        for pid in batch:
            assert pid in pid_to_tris
            assert all_tris.isdisjoint(pid_to_tris[pid])
            all_tris.update(pid_to_tris[pid])

    # Visualization: color triangles according to batch membership
    tri_to_color = {}
    colors = color_map_for_batches(len(batches))
    for b_idx, batch in enumerate(batches):
        for pid in batch:
            for t in pid_to_tris[pid]:
                tri_to_color[int(t)] = colors[b_idx]

    fig, ax = plt.subplots(figsize=(6,6))
    # draw all triangles in light gray
    for tidx, tri in enumerate(editor.triangles):
        if np.all(np.array(tri) == -1):
            continue
        coords = editor.points[[int(tri[0]), int(tri[1]), int(tri[2]), int(tri[0])]]
        if tidx in tri_to_color:
            ax.plot(coords[:,0], coords[:,1], color=tri_to_color[tidx], lw=1.2)
            ax.fill(coords[:,0], coords[:,1], color=tri_to_color[tidx], alpha=0.12)
        else:
            ax.plot(coords[:,0], coords[:,1], color='lightgray', lw=0.6)

    ax.scatter(editor.points[:,0], editor.points[:,1], s=6, color='k')
    ax.set_aspect('equal')
    # draw patch boundaries as closed loops and assert each patch has at least one loop
    for p in patches:
        loops = patch_boundary_loops(p['tris'], editor)
        assert loops, f'Patch {p["id"]} has no boundary loops'
        # draw first loop for each patch in darker color
        # find which batch this patch belongs to
        b_idx = None
        for bi, batch in enumerate(batches):
            if p['id'] in batch:
                b_idx = bi
                break
        color = colors[b_idx] if b_idx is not None else (0.2,0.2,0.2,1.0)
        for loop in loops:
            coords = editor.points[np.array(loop + [loop[0]])]
            ax.plot(coords[:,0], coords[:,1], color=color, lw=1.6)

    out_path = os.path.join(out_dir, 'patch_batches.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    assert os.path.exists(out_path)
