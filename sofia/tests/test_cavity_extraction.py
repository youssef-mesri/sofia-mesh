"""Unit tests for cavity extraction in helpers.py"""
import numpy as np
import pytest
from sofia.core.helpers import (
    CavityInfo,
    extract_removal_cavity,
    filter_cycle_vertex
)
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor


def test_cavity_info_dataclass():
    """Test CavityInfo dataclass creation."""
    # Success case
    info = CavityInfo(
        ok=True,
        cavity_indices=[0, 1, 2],
        cycle=[5, 6, 7],
        removed_area=0.5,
        error=None
    )
    assert info.ok
    assert info.cavity_indices == [0, 1, 2]
    assert info.cycle == [5, 6, 7]
    assert info.removed_area == 0.5
    assert info.error is None
    
    # Error case
    info_error = CavityInfo(
        ok=False,
        cavity_indices=None,
        cycle=None,
        removed_area=None,
        error="test error"
    )
    assert not info_error.ok
    assert info_error.error == "test error"


def test_extract_removal_cavity_simple():
    """Test cavity extraction on a simple star mesh."""
    # Create a simple star: center vertex 4 connected to boundary vertices 0,1,2,3
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.5],  # 4 center (to remove)
    ], dtype=float)
    tris = np.array([
        [4, 0, 1],  # 0
        [4, 1, 2],  # 1
        [4, 2, 3],  # 2
        [4, 3, 0],  # 3
    ], dtype=int)
    
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    
    # Extract cavity for center vertex
    cavity = extract_removal_cavity(editor, 4)
    
    assert cavity.ok
    assert set(cavity.cavity_indices) == {0, 1, 2, 3}
    assert len(cavity.cycle) == 4
    assert set(cavity.cycle) == {0, 1, 2, 3}
    assert cavity.removed_area is not None
    assert cavity.removed_area > 0.9  # Should be close to 1.0 (unit square area)
    assert cavity.error is None


def test_extract_removal_cavity_isolated_vertex():
    """Test cavity extraction on isolated vertex (no incident triangles)."""
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [0.5, 1.0],  # 2
        [2.0, 2.0],  # 3 isolated
    ], dtype=float)
    tris = np.array([
        [0, 1, 2],
    ], dtype=int)
    
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    
    cavity = extract_removal_cavity(editor, 3)
    
    assert not cavity.ok
    assert cavity.error == "vertex isolated"
    assert cavity.cavity_indices is None
    assert cavity.cycle is None


def test_extract_removal_cavity_boundary_vertex():
    """Test cavity extraction on a boundary vertex - boundary cycle extraction will fail."""
    pts = np.array([
        [0.0, 0.0],  # 0 boundary (to remove)
        [1.0, 0.0],  # 1
        [0.5, 1.0],  # 2
    ], dtype=float)
    tris = np.array([
        [0, 1, 2],
    ], dtype=int)
    
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    
    cavity = extract_removal_cavity(editor, 0)
    
    # For a single triangle, the boundary cycle extraction will fail
    # because the neighbors [1, 2] don't form a proper cycle (each has degree 1, not 2)
    assert not cavity.ok
    assert cavity.error == "boundary cycle extraction failed"
    assert cavity.cavity_indices == [0]


def test_filter_cycle_vertex_basic():
    """Test removing a vertex from a cycle."""
    cycle = [0, 1, 2, 3, 4]
    v_idx = 2
    
    filtered = filter_cycle_vertex(cycle, v_idx)
    
    assert filtered == [0, 1, 3, 4]
    assert 2 not in filtered


def test_filter_cycle_vertex_with_duplicates():
    """Test filtering with consecutive duplicates."""
    cycle = [0, 1, 1, 2, 3, 3, 3, 4]
    v_idx = 5  # Not in cycle
    
    filtered = filter_cycle_vertex(cycle, v_idx)
    
    # Should remove consecutive duplicates
    assert filtered == [0, 1, 2, 3, 4]


def test_filter_cycle_vertex_closing_duplicate():
    """Test filtering with closing duplicate (first == last)."""
    cycle = [0, 1, 2, 3, 0]  # Closing loop
    v_idx = 5  # Not in cycle
    
    filtered = filter_cycle_vertex(cycle, v_idx)
    
    # Should remove closing duplicate
    assert filtered == [0, 1, 2, 3]
    assert filtered[0] != filtered[-1]


def test_filter_cycle_vertex_remove_and_clean():
    """Test removing vertex and cleaning duplicates in one pass."""
    cycle = [0, 1, 1, 2, 2, 3, 4, 4]
    v_idx = 2
    
    filtered = filter_cycle_vertex(cycle, v_idx)
    
    # Should remove 2 and clean duplicates
    assert 2 not in filtered
    assert filtered == [0, 1, 3, 4]


def test_filter_cycle_vertex_empty():
    """Test filtering empty cycle."""
    filtered = filter_cycle_vertex([], 5)
    assert filtered == []


def test_filter_cycle_vertex_single_element():
    """Test filtering cycle with single element."""
    cycle = [5]
    filtered = filter_cycle_vertex(cycle, 5)
    assert filtered == []
    
    filtered2 = filter_cycle_vertex(cycle, 3)
    assert filtered2 == [5]


def test_extract_removal_cavity_with_splits():
    """Test cavity extraction after edge splits (more complex topology)."""
    # Start with simple star
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1 corner (to remove after splits)
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.5],  # 4 center
    ], dtype=float)
    tris = np.array([
        [4, 0, 1],
        [4, 1, 2],
        [4, 2, 3],
        [4, 3, 0],
    ], dtype=int)
    
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy(), 
                                  virtual_boundary_mode=True, 
                                  enforce_split_quality=False)
    
    # Split edges around vertex 1
    ok_a, _, _ = editor.split_edge((0, 1))
    ok_b, _, _ = editor.split_edge((1, 2))
    
    assert ok_a and ok_b
    
    # Now extract cavity for vertex 1
    cavity = extract_removal_cavity(editor, 1)
    
    # Cavity extraction should at least identify the incident triangles
    # The cycle extraction may fail for boundary vertices, which is OK
    assert len(cavity.cavity_indices) > 0  # Has incident triangles
    # Note: cavity.ok may be False if boundary cycle fails, which is expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
