import numpy as np
import pytest

from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor
from sofia.sofia.operations import op_remove_node_with_patch, op_split_edge
from sofia.sofia.conformity import check_mesh_conformity


def square_mesh():
    # 4 points in a square, split into two triangles along the diagonal (0-3)
    pts = np.array([
        [0.0, 0.0],  # 0 bottom-left (boundary)
        [1.0, 0.0],  # 1 bottom-right (boundary)
        [1.0, 1.0],  # 2 top-right (boundary)
        [0.0, 1.0],  # 3 top-left (boundary)
    ], dtype=float)
    tris = np.array([
        [0, 1, 3],
        [1, 2, 3],
    ], dtype=int)
    return pts, tris


def test_remove_node_on_boundary_virtual_mode():
    pts, tris = square_mesh()
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True)

    # Pick boundary vertex with degree 2 (corner)
    v = 1  # vertex 1 is on boundary
    ok, msg, _ = op_remove_node_with_patch(editor, v)
    assert ok, f"remove_node should work on boundary in virtual mode: {msg}"

    # Mesh should still be conforming (allowing tombstones before compaction)
    ok_c, _ = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok_c, "Mesh must remain conforming after boundary remove"
    # After compaction, strict conformity should also hold
    editor.compact_triangle_indices()
    ok_c2, _ = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok_c2, "Mesh must be conforming after compaction"


def test_split_edge_on_boundary():
    pts, tris = square_mesh()
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), virtual_boundary_mode=True, enforce_split_quality=False)

    # Split boundary edge (0,1)
    edge = tuple(sorted((0, 1)))
    ok, msg, _ = op_split_edge(editor, edge)
    assert ok, f"split_edge should work on boundary edges: {msg}"

    # Should add one point and increase triangle count by 1 (1 tombstoned, 2 appended)
    assert len(editor.points) == 5
    ok_c, _ = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    assert ok_c, "Mesh must remain conforming after boundary edge split"
    # After compaction, strict conformity should hold
    editor.compact_triangle_indices()
    ok_c2, _ = check_mesh_conformity(editor.points, editor.triangles, allow_marked=False)
    assert ok_c2, "Mesh must be conforming after compaction"
