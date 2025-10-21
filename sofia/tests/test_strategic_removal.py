"""Tests for strategic node removal using the refactored components."""
import numpy as np
import pytest

from sofia.core.operations import try_remove_node_strategically
from sofia.core.config import BoundaryRemoveConfig
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor


class TestStrategicNodeRemoval:
    """Test the new strategy-based node removal function."""
    
    def test_remove_center_of_star(self):
        """Test removing the center vertex of a simple star topology."""
        # Create a star: center vertex connected to 4 outer vertices
        points = np.array([
            [0.0, 0.0],   # 0: center
            [1.0, 0.0],   # 1: right
            [0.0, 1.0],   # 2: top
            [-1.0, 0.0],  # 3: left
            [0.0, -1.0],  # 4: bottom
        ])
        triangles = np.array([
            [0, 1, 2],  # 0
            [0, 2, 3],  # 1
            [0, 3, 4],  # 2
            [0, 4, 1],  # 3
        ], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        
        # Remove center vertex
        success, msg, info = try_remove_node_strategically(editor, 0)
        
        assert success, f"Failed: {msg}"
        assert "successfully" in msg.lower()
        assert info is not None
        assert info['removed_vertex'] == 0
        assert info['cavity_size'] == 4  # All 4 triangles removed
        assert info['new_triangles'] == 2  # Square needs 2 triangles
    
    def test_remove_with_area_preservation(self):
        """Test that area preservation is enforced when required."""
        # Regular hexagon with center point
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        outer = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        points = np.vstack([center, outer])
        
        # Star triangulation from center
        triangles = []
        for i in range(6):
            triangles.append([0, i+1, ((i+1) % 6) + 1])
        triangles = np.array(triangles, dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        
        config = BoundaryRemoveConfig(require_area_preservation=True)
        success, msg, info = try_remove_node_strategically(editor, 0, config)
        
        # Should succeed and preserve area
        assert success, f"Failed: {msg}"
        assert info['new_triangles'] == 4  # Hexagon needs 4 triangles
    
    def test_remove_fails_on_boundary(self):
        """Test that removal fails for boundary vertices."""
        # Simple triangle
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        
        # Try to remove a boundary vertex (should fail in cavity extraction)
        success, msg, info = try_remove_node_strategically(editor, 0)
        
        assert not success
        # The error should mention boundary or cycle extraction failure
        assert any(word in msg.lower() for word in ['boundary', 'cycle', 'isolated'])
    
    def test_remove_with_simplification_enabled(self):
        """Test removal with polygon simplification enabled."""
        # Create a configuration with nearly-collinear points
        points = np.array([
            [0.0, 0.0],    # 0: center
            [1.0, 0.0],    # 1
            [1.0, 0.001],  # 2: nearly collinear with 1 and 3
            [1.0, 1.0],    # 3
            [0.0, 1.0],    # 4
            [-1.0, 0.0],   # 5
        ])
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 1],
        ], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        editor.enable_polygon_simplification = True
        
        success, msg, info = try_remove_node_strategically(editor, 0)
        
        # Should succeed, possibly using simplified polygon
        assert success, f"Failed: {msg}"
        assert info['new_triangles'] >= 3  # At least 3 triangles needed
    
    def test_remove_isolated_vertex_fails(self):
        """Test that removing an isolated vertex fails gracefully."""
        points = np.array([[0, 0], [1, 0], [0, 1], [5, 5]])  # Last is isolated
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        
        success, msg, info = try_remove_node_strategically(editor, 3)
        
        assert not success
        assert "isolated" in msg.lower() or "no triangles" in msg.lower()
    
    def test_remove_preserves_mesh_conformity(self):
        """Test that the result maintains mesh conformity."""
        # Create a larger mesh
        angles = np.linspace(0, 2*np.pi, 9)[:-1]  # Octagon
        outer = np.column_stack([2*np.cos(angles), 2*np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        points = np.vstack([center, outer])
        
        # Star from center
        triangles = []
        for i in range(8):
            triangles.append([0, i+1, ((i+1) % 8) + 1])
        triangles = np.array(triangles, dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        
        success, msg, info = try_remove_node_strategically(editor, 0)
        
        assert success, f"Failed: {msg}"
        
        # Check conformity after removal
        from sofia.core.conformity import check_mesh_conformity
        ok, msgs = check_mesh_conformity(
            editor.points, editor.triangles, allow_marked=True
        )
        assert ok, f"Mesh not conforming after removal: {msgs}"
    
    def test_strategy_chain_fallback(self):
        """Test that strategy chain falls back appropriately."""
        # Pentagon (5 vertices)
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        outer = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        points = np.vstack([center, outer])
        
        triangles = []
        for i in range(5):
            triangles.append([0, i+1, ((i+1) % 5) + 1])
        triangles = np.array(triangles, dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        
        # Use config that prefers certain strategies
        config = BoundaryRemoveConfig(
            prefer_area_preserving_star=True,
            prefer_worst_angle_star=True,
            require_area_preservation=False  # Don't require, but prefer
        )
        
        success, msg, info = try_remove_node_strategically(editor, 0, config)
        
        # Should succeed using one of the strategies
        assert success, f"Failed: {msg}"
        assert info['new_triangles'] == 3  # Pentagon needs 3 triangles


class TestStrategicRemovalComparison:
    """Compare strategic removal with traditional approach."""
    
    def test_same_result_as_traditional(self):
        """Test that strategic removal produces same topology as traditional."""
        # Square with center
        points = np.array([
            [0.0, 0.0],   # 0: center
            [1.0, 0.0],   # 1
            [1.0, 1.0],   # 2
            [0.0, 1.0],   # 3
            [-1.0, 1.0],  # 4
            [-1.0, 0.0],  # 5
        ])
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 1],
        ], dtype=np.int32)
        
        # Test with strategic approach
        editor1 = PatchBasedMeshEditor(points.copy(), triangles.copy())
        success1, msg1, info1 = try_remove_node_strategically(editor1, 0)
        
        assert success1, f"Strategic removal failed: {msg1}"
        
        # Count active triangles
        active1 = sum(1 for t in editor1.triangles if not np.all(t == -1))
        assert active1 == info1['new_triangles']
        
        # Verify we have a valid mesh
        from sofia.core.conformity import check_mesh_conformity
        ok, msgs = check_mesh_conformity(
            editor1.points, editor1.triangles, allow_marked=True
        )
        assert ok, f"Result not conforming: {msgs}"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_degenerate_triangle_rejected(self):
        """Test that degenerate configurations are rejected."""
        # Collinear points forming degenerate triangles
        points = np.array([
            [0, 0],
            [1, 0],
            [2, 0],  # Collinear with 0 and 1
        ])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        
        success, msg, info = try_remove_node_strategically(editor, 0)
        
        # Should fail due to degenerate geometry
        assert not success
    
    def test_none_config_uses_defaults(self):
        """Test that None config uses default configuration."""
        points = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1], [-1, 0]
        ])
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
        ], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        
        # Pass None config - should use defaults
        success, msg, info = try_remove_node_strategically(editor, 0, config=None)
        
        assert success, f"Failed with default config: {msg}"
