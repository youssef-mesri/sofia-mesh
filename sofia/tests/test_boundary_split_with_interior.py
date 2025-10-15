import numpy as np

from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor
from sofia.sofia.conformity import check_mesh_conformity
from sofia.sofia.visualization import plot_mesh
import matplotlib.pyplot as plt


def square_with_center():
    # Square corners + interior center
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.5],  # 4 interior
    ], dtype=float)
    # Fan around center (4): four triangles
    tris = np.array([
        [4, 0, 1],
        [4, 1, 2],
        [4, 2, 3],
        [4, 3, 0],
    ], dtype=int)
    return pts, tris


def test_split_boundary_edge_connects_new_vertex_and_preserves_conformity():
    pts, tris = square_with_center()
    # Use relaxed split-quality to focus on topology/connectivity
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), enforce_split_quality=False)

    # Split boundary edge (0,1), which belongs to triangle [4,0,1]
    edge = (0, 1)
    ok, msg, _ = editor.split_edge(edge)
    assert ok, f"split_edge on boundary should succeed: {msg}"

    # New point should be appended at the end
    new_idx = len(editor.points) - 1

    # The original boundary edge (0,1) should be gone; replaced by (0,new) and (new,1)
    assert tuple(sorted(edge)) not in editor.edge_map, "Old boundary edge should be removed from edge_map"
    assert (min(0, new_idx), max(0, new_idx)) in editor.edge_map, "Edge (0,new) should exist"
    assert (min(1, new_idx), max(1, new_idx)) in editor.edge_map, "Edge (1,new) should exist"

    # The new vertex should be connected to exactly two new triangles (before compaction)
    assert new_idx in editor.v_map, "New vertex should be present in v_map"
    tri_refs = [ti for ti in editor.v_map[new_idx] if not np.all(editor.triangles[int(ti)] == -1)]
    assert len(tri_refs) == 2, f"New vertex should be referenced by 2 active triangles, got {len(tri_refs)}"

    # Conformity should hold while allowing tombstones before compaction
    ok1, msgs1 = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok1, f"Mesh not conforming after boundary split (pre-compact): {msgs1}"

    # After compaction, strict conformity should also hold
    editor.compact_triangle_indices()
    ok2, msgs2 = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok2, f"Mesh not conforming after compaction: {msgs2}"


def test_remove_node_after_boundary_split_midpoint():
    """Use the 'after' state of boundary split (equivalent to boundary_demo_after.png)
    and remove the newly inserted boundary midpoint vertex. Validate conformity.
    """
    pts, tris = square_with_center()
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True, enforce_split_quality=False)

    # First, split a boundary edge to create the midpoint vertex
    edge = (0, 1)
    ok_s, msg_s, _ = editor.split_edge(edge)
    assert ok_s, f"Failed to split boundary edge {edge}: {msg_s}"
    new_idx = len(editor.points) - 1

    # Now remove that boundary midpoint vertex in virtual boundary mode
    # Relax remove quality to focus on topological correctness
    try:
        editor.enforce_remove_quality = False
    except Exception:
        pass
    # Plot state before removal (after split). Highlight the soon-to-be removed vertex.
    before_pts = editor.points.copy()
    before_tris = editor.triangles.copy()
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    # Basic wireframe
    active_mask = ~np.all(before_tris == -1, axis=1)
    tri_plot = before_tris[active_mask]
    ax.triplot(before_pts[:,0], before_pts[:,1], tri_plot, lw=0.6, color='gray')
    ax.scatter(before_pts[:,0], before_pts[:,1], s=20, color='black')
    # Highlight candidate removal vertex
    ax.scatter([before_pts[new_idx,0]],[before_pts[new_idx,1]], s=80, color='red', zorder=5, label='to remove')
    ax.set_aspect('equal')
    ax.set_title('before removal (midpoint highlighted)')
    ax.legend(loc='upper right', fontsize=8)
    fig.savefig('boundary_remove_before.png', dpi=150)
    plt.close(fig)

    ok_r, msg_r, _ = editor.remove_node_with_patch(int(new_idx))
    assert ok_r, f"remove_node_with_patch should succeed on split boundary midpoint: {msg_r}"

    # Conformity checks pre/post compaction
    ok_c1, msgs1 = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok_c1, f"Mesh not conforming after removal (pre-compact): {msgs1}"
    npts_after_remove_pre_compact = len(editor.points)
    old_to_new = editor.compact_triangle_indices()
    after_pts = editor.points.copy()
    after_tris = editor.triangles.copy()
    fig2 = plt.figure(figsize=(4,4))
    ax2 = fig2.add_subplot(111)
    active_mask2 = ~np.all(after_tris == -1, axis=1)
    tri_plot2 = after_tris[active_mask2]
    ax2.triplot(after_pts[:,0], after_pts[:,1], tri_plot2, lw=0.6, color='gray')
    ax2.scatter(after_pts[:,0], after_pts[:,1], s=20, color='black')
    ax2.set_aspect('equal')
    ax2.set_title('after removal (midpoint gone)')
    fig2.savefig('boundary_remove_after.png', dpi=150)
    plt.close(fig2)

    # The removal retriangulation keeps existing points (no compaction of unused vertex indices until a global pass).
    # Instead of asserting a vertex count decrease, assert the removed vertex index is no longer referenced by any active triangle.
    assert before_pts.shape[0] == 6, f"Expected 6 points before removal, got {before_pts.shape[0]}"
    # Confirm midpoint index not present in v_map (or contains only tombstoned triangles)
    star = editor.v_map.get(new_idx, set())
    active_refs = [ti for ti in star if not np.all(editor.triangles[int(ti)] == -1)] if star else []
    assert len(active_refs) == 0, "Removed midpoint vertex should not belong to any active triangle"
    ok_c2, msgs2 = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok_c2, f"Mesh not conforming after compaction: {msgs2}"
    # The removed midpoint old index should not appear in the compaction vertex map
    assert old_to_new[new_idx] == -1, "Removed midpoint still present after compaction"
