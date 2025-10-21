import numpy as np

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor  # canonical import
from sofia.core.conformity import build_edge_to_tri_map, check_mesh_conformity


def test_compact_triangle_indices_remaps_vertices():
    # Build a simple mesh with 6 vertices and 3 triangles
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [2.0, 0.0],  # 4
        [2.0, 1.0],  # 5
    ], dtype=float)

    tris = np.array([
        [0, 1, 2],
        [2, 3, 0],
        [4, 5, 2],
    ], dtype=int)

    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())

    # Mark the middle triangle as deleted (tombstone) to simulate an in-progress operation
    editor.triangles[1] = np.array([-1, -1, -1], dtype=int)

    # Now compact: this should remove unused vertices (vertex 3 only appears in tombstoned tri)
    editor.compact_triangle_indices()

    # Expected active triangles after compaction are the original tri0 and tri2 remapped
    # Active original triangles reference vertices {0,1,2,4,5} in that sorted order
    expected_used = [0, 1, 2, 4, 5]
    expected_pts = pts[expected_used]
    old_to_new = {old: new for new, old in enumerate(expected_used)}
    expected_tris = np.array([
        [old_to_new[0], old_to_new[1], old_to_new[2]],
        [old_to_new[4], old_to_new[5], old_to_new[2]],
    ], dtype=int)

    assert editor.points.shape == expected_pts.shape
    assert np.allclose(editor.points, expected_pts)
    # triangles should be exactly the expected remapped triangles
    assert editor.triangles.shape[0] == expected_tris.shape[0]
    assert np.array_equal(editor.triangles, expected_tris)

    # No tombstoned triangles should remain in the compacted result
    assert not np.any(np.all(editor.triangles == -1, axis=1))

    # Edge map should reference only active triangle indices and be manifold
    emap = build_edge_to_tri_map(editor.triangles)
    for e, s in emap.items():
        assert len(s) <= 2

    # Strict conformity should hold on the compacted mesh
    ok, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok, f"Compacted mesh not conforming: {msgs}"
