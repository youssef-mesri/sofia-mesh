import numpy as np
import pytest

from sofia.sofia.diagnostics import count_boundary_loops, extract_boundary_loops

def test_no_boundary_in_closed_mesh():
    # Two triangles forming a square (0,1,2) and (0,2,3) with all edges used twice except boundary
    pts = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=float)
    tris = np.array([[0,1,2],[0,2,3]], dtype=int)
    # This mesh is a solid square: one boundary loop
    assert count_boundary_loops(pts, tris) == 1
    loops = extract_boundary_loops(pts, tris)
    assert len(loops) == 1
    assert set(loops[0]) == {0,1,2,3}


def test_two_disjoint_components():
    # Two separate triangles far apart: each triangle contributes a 3-vertex boundary loop
    pts = np.array([[0,0],[1,0],[0,1], [10,10],[11,10],[10,11]], dtype=float)
    tris = np.array([[0,1,2],[3,4,5]], dtype=int)
    assert count_boundary_loops(pts, tris) == 2
    loops = extract_boundary_loops(pts, tris)
    assert len(loops) == 2
    # Expect two loops of size 3
    assert sorted([len(l) for l in loops]) == [3,3]


def test_hole_component_counts_as_boundary():
    # Create a ring: outer square triangulated as before, remove a central quad (simulate hole)
    # For boundary detection based on single-use edges, we'll simulate by not including inner tris.
    pts = np.array([[0,0],[2,0],[2,2],[0,2], [0.8,0.8],[1.2,0.8],[1.2,1.2],[0.8,1.2]], dtype=float)
    # Triangulate outer square: (0,1,2),(0,2,3)
    tris = np.array([[0,1,2],[0,2,3]], dtype=int)
    # No inner triangles -> only outer boundary loop
    assert count_boundary_loops(pts, tris) == 1
    loops = extract_boundary_loops(pts, tris)
    assert len(loops) == 1
    assert set(loops[0]) == {0,1,2,3}
