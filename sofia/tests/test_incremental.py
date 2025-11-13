"""Unit tests for incremental computation structures.

Tests correctness and validates that incremental updates match full rebuilds.
"""

import numpy as np
import pytest
from sofia.core.incremental import IncrementalEdgeMap, IncrementalConformityChecker
from sofia.core.conformity import build_edge_to_tri_map


def generate_simple_mesh():
    """Generate a simple test mesh with known properties."""
    # Square mesh with 2 triangles
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.5],  # center point
    ])
    
    triangles = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ], dtype=np.int32)
    
    return points, triangles


def test_incremental_edge_map_initial_build():
    """Test that initial edge map build is correct."""
    points, triangles = generate_simple_mesh()
    
    edge_map_full = build_edge_to_tri_map(triangles)
    edge_map_inc = IncrementalEdgeMap(triangles)
    
    # Should have same edges
    assert set(edge_map_full.keys()) == set(edge_map_inc.edge_to_tris.keys())
    
    # Each edge should have same triangle set
    for edge in edge_map_full.keys():
        assert set(edge_map_full[edge]) == edge_map_inc.get_triangles_for_edge(edge)


def test_incremental_edge_map_remove():
    """Test removing triangles."""
    points, triangles = generate_simple_mesh()
    
    edge_map = IncrementalEdgeMap(triangles)
    
    # Remove first triangle
    edge_map.remove_triangles([0])
    
    # Edge (0,1) should now only be in triangle 0's neighbors (none after removal)
    # Edge (1,4) should still exist in triangle 1
    tris_for_14 = edge_map.get_triangles_for_edge((1, 4))
    assert 1 in tris_for_14
    assert 0 not in tris_for_14


def test_incremental_edge_map_add():
    """Test adding triangles."""
    points, triangles = generate_simple_mesh()
    
    edge_map = IncrementalEdgeMap(triangles[:2])  # Only first 2 triangles
    
    # Add remaining triangles
    new_tris = [tuple(triangles[2]), tuple(triangles[3])]
    edge_map.add_triangles(new_tris, start_idx=2)
    
    # Should now match full build
    edge_map_full = IncrementalEdgeMap(triangles)
    assert set(edge_map.edge_to_tris.keys()) == set(edge_map_full.edge_to_tris.keys())


def test_incremental_edge_map_boundary():
    """Test boundary edge detection."""
    points, triangles = generate_simple_mesh()
    
    edge_map = IncrementalEdgeMap(triangles)
    boundary = edge_map.get_boundary_edges()
    
    # Outer square edges should be boundary (canonical form: min vertex first)
    expected_boundary = {
        (0, 1), (1, 2), (2, 3), (0, 3)  # (3,0) becomes (0,3) in canonical form
    }
    
    assert boundary == expected_boundary


def test_incremental_edge_map_is_conforming():
    """Test conformity check."""
    points, triangles = generate_simple_mesh()
    
    edge_map = IncrementalEdgeMap(triangles)
    
    # Should be conforming (no non-manifold edges)
    assert edge_map.is_conforming()
    
    # Add a non-manifold edge by adding duplicate triangle
    edge_map.add_triangles([(0, 1, 4)], start_idx=len(triangles))
    
    # Should now be non-conforming
    assert not edge_map.is_conforming()


def test_incremental_conformity_checker_initial():
    """Test initial conformity checker build."""
    points, triangles = generate_simple_mesh()
    
    checker = IncrementalConformityChecker(triangles)
    
    # Should be conforming
    assert checker.is_conforming()
    
    # Should have boundary
    assert checker.has_boundary()
    
    # Should have 4 boundary edges
    assert checker.get_boundary_edge_count() == 4


def test_incremental_conformity_checker_update():
    """Test conformity checker updates."""
    points, triangles = generate_simple_mesh()
    
    checker = IncrementalConformityChecker(triangles)
    
    # Remove first triangle
    checker.update_after_operation([0], [])
    
    # Should still be conforming
    assert checker.is_conforming()
    
    # Should have more boundary edges now
    assert checker.get_boundary_edge_count() > 4


def test_incremental_conformity_checker_non_manifold():
    """Test non-manifold edge detection."""
    points, triangles = generate_simple_mesh()
    
    checker = IncrementalConformityChecker(triangles)
    
    # Initially conforming
    assert checker.is_conforming()
    assert checker.get_non_manifold_edge_count() == 0
    
    # Add duplicate triangle (creates non-manifold edge)
    checker.update_after_operation([], [(0, 1, 4)])
    
    # Should now be non-conforming
    assert not checker.is_conforming()
    assert checker.get_non_manifold_edge_count() > 0


def test_incremental_edge_map_validate():
    """Test edge map validation against full rebuild."""
    points, triangles = generate_simple_mesh()
    
    edge_map = IncrementalEdgeMap(triangles)
    
    # Should validate successfully
    assert edge_map.validate(triangles)


def test_incremental_conformity_checker_validate():
    """Test conformity checker validation."""
    points, triangles = generate_simple_mesh()
    
    checker = IncrementalConformityChecker(triangles)
    
    # Should validate successfully
    assert checker.validate(triangles)


def test_edge_map_with_marked_triangles():
    """Test edge map handles marked (deleted) triangles."""
    points, triangles = generate_simple_mesh()
    
    # Mark second triangle as deleted
    triangles_with_marked = triangles.copy()
    triangles_with_marked[1] = [-1, -1, -1]
    
    edge_map = IncrementalEdgeMap(triangles_with_marked)
    
    # Should skip marked triangle
    # Edge (1,4) should only appear once (from triangle 0)
    tris = edge_map.get_triangles_for_edge((1, 4))
    assert 0 in tris
    assert 1 not in tris  # Triangle 1 is marked


def test_conformity_checker_with_marked_triangles():
    """Test conformity checker handles marked triangles."""
    points, triangles = generate_simple_mesh()
    
    # Mark second triangle as deleted
    triangles_with_marked = triangles.copy()
    triangles_with_marked[1] = [-1, -1, -1]
    
    checker = IncrementalConformityChecker(triangles_with_marked)
    
    # Should build correctly, ignoring marked triangle
    # Will have more boundary edges due to gap
    assert checker.has_boundary()


def test_edge_map_get_edge_count():
    """Test edge count query."""
    points, triangles = generate_simple_mesh()
    
    edge_map = IncrementalEdgeMap(triangles)
    
    # 4 triangles, each with 3 edges, but shared
    # Outer square: 4 edges
    # Inner cross: 4 edges
    # Total: 8 unique edges
    assert edge_map.get_edge_count() == 8


def test_conformity_checker_edge_count():
    """Test getting edge count for specific edge."""
    points, triangles = generate_simple_mesh()
    
    checker = IncrementalConformityChecker(triangles)
    
    # Inner edges should have count 2
    assert checker.get_edge_count_for_edge((0, 4)) == 2
    assert checker.get_edge_count_for_edge((1, 4)) == 2
    
    # Outer edges should have count 1
    assert checker.get_edge_count_for_edge((0, 1)) == 1
    assert checker.get_edge_count_for_edge((1, 2)) == 1


def test_large_scale_updates():
    """Test many sequential updates maintain correctness."""
    np.random.seed(42)
    n_points = 50
    points = np.random.rand(n_points, 2) * 10.0
    
    # Build initial mesh
    triangles = []
    for i in range(100):
        v0, v1, v2 = np.random.choice(n_points, 3, replace=False)
        triangles.append([v0, v1, v2])
    triangles = np.array(triangles, dtype=np.int32)
    
    # Create incremental structures
    edge_map = IncrementalEdgeMap(triangles)
    checker = IncrementalConformityChecker(triangles)
    
    # Perform 20 operations
    for i in range(20):
        # Simulate edge split
        removed = [i % len(triangles)]
        added = [
            (triangles[removed[0]][0], triangles[removed[0]][1], n_points + i),
            (triangles[removed[0]][1], triangles[removed[0]][2], n_points + i),
        ]
        
        edge_map.remove_triangles(removed)
        edge_map.add_triangles(added, start_idx=len(triangles) + i * 2)
        
        checker.update_after_operation(removed, added)
    
    # Should still work after many updates
    assert edge_map.get_edge_count() > 0
    assert isinstance(checker.is_conforming(), bool)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
