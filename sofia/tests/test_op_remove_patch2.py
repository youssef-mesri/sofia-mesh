"""Tests for op_remove_node_with_patch2 with virtual_boundary_mode support."""
import numpy as np
import pytest

from sofia.core.operations import op_remove_node_with_patch2
from sofia.core.config import BoundaryRemoveConfig
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor


class TestOpRemoveNodeWithPatch2:
    """Test the refactored op_remove_node_with_patch2 function."""
    
    def test_interior_star_removal(self):
        """Test removing interior vertex in star topology."""
        # Star: center connected to 4 outer vertices
        points = np.array([
            [0.0, 0.0],   # 0: center
            [1.0, 0.0],   # 1: right
            [0.0, 1.0],   # 2: top
            [-1.0, 0.0],  # 3: left
            [0.0, -1.0],  # 4: bottom
        ])
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
        ], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        
        success, msg, info = op_remove_node_with_patch2(editor, 0)
        
        assert success, f"Failed: {msg}"
        assert info['removed_vertex'] == 0
        assert info['cavity_size'] == 4
        assert info['new_triangles'] == 2  # Square needs 2 triangles
    
    def test_with_area_preservation_required(self):
        """Test area preservation enforcement."""
        # Hexagon with center
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        outer = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        points = np.vstack([center, outer])
        
        triangles = []
        for i in range(6):
            triangles.append([0, i+1, ((i+1) % 6) + 1])
        triangles = np.array(triangles, dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        config = BoundaryRemoveConfig(require_area_preservation=True)
        editor.boundary_remove_config = config
        
        success, msg, info = op_remove_node_with_patch2(editor, 0)
        
        assert success, f"Failed: {msg}"
        assert info['new_triangles'] == 4


class TestVirtualBoundaryMode:
    """Test virtual_boundary_mode support in op_remove_node_with_patch2."""
    
    def test_virtual_boundary_simple_removal(self):
        """Test removing boundary vertex with virtual_boundary_mode."""
        # Simple mesh with boundary
        points = np.array([
            [0.0, 0.0],  # 0: boundary
            [1.0, 0.0],  # 1: boundary
            [0.5, 1.0],  # 2: interior
        ])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles, virtual_boundary_mode=True)
        
        # In virtual boundary mode, we can remove boundary vertices
        success, msg, info = op_remove_node_with_patch2(editor, 0)
        
        assert success, f"Failed: {msg}"
        assert info['removed_vertex'] == 0
    
    def test_virtual_boundary_corner_degenerate(self):
        """Test degenerate boundary corner removal (deletion-only)."""
        # Triangle where removing a vertex leaves < 3 neighbors
        points = np.array([
            [0.0, 0.0],  # 0: corner to remove
            [1.0, 0.0],  # 1
            [0.5, 1.0],  # 2
        ])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles, virtual_boundary_mode=True)
        
        success, msg, info = op_remove_node_with_patch2(editor, 0)
        
        # Should succeed with deletion-only
        assert success, f"Failed: {msg}"
        assert "deletion-only" in msg.lower() or info['new_triangles'] == 0
    
    def test_virtual_boundary_interior_like_removal(self):
        """Test boundary vertex removal with virtual boundary preserving cavity area."""
        # Mesh with OPEN star (3 triangles) around boundary vertex 0
        # Virtual boundary mode treats this as if cavity was closed with virtual node
        # The discrete cavity area (sum of triangle areas) IS preserved: 1.5 → 1.5
        # This validates that we're remeshing the same geometry
        points = np.array([
            [0.0, 0.0],    # 0: boundary vertex to remove
            [1.0, 0.0],    # 1: boundary
            [0.5, 1.0],    # 2: boundary
            [-0.5, 1.0],   # 3: boundary
            [-1.0, 0.0],   # 4: boundary
        ])
        # Three triangles forming an OPEN star (missing edge 1-4)
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
        ], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles, virtual_boundary_mode=True)
        config = BoundaryRemoveConfig(require_area_preservation=True)
        editor.boundary_remove_config = config
        
        success, msg, info = op_remove_node_with_patch2(editor, 0)
        
        # Should SUCCEED because discrete cavity area is preserved
        assert success, f"Failed: {msg}"
        assert info['removed_vertex'] == 0
        assert info['new_triangles'] >= 2
    
    def test_virtual_boundary_with_area_preservation(self):
        """Test virtual boundary mode with configuration - removing an interior-like vertex."""
        # Pentagon-like boundary with center
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        points = np.vstack([points, center])
        
        # Star from center (index 5, not 6!)
        center_idx = len(points) - 1  # Last point is the center
        triangles = []
        for i in range(5):
            triangles.append([center_idx, i, (i+1) % 5])
        triangles = np.array(triangles, dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles, virtual_boundary_mode=True)
        config = BoundaryRemoveConfig(require_area_preservation=True)
        editor.boundary_remove_config = config
        
        # Remove the CENTER vertex (not a boundary vertex) which has a closed cycle
        # This should preserve area since we're removing 5 triangles and recreating a pentagon
        success, msg, info = op_remove_node_with_patch2(editor, center_idx)
        
        assert success, f"Failed: {msg}"
    
    def test_virtual_boundary_disabled_rejects_boundary(self):
        """Test that boundary removal fails when virtual_boundary_mode is disabled."""
        points = np.array([
            [0.0, 0.0],  # 0: boundary
            [1.0, 0.0],  # 1: boundary
            [0.5, 1.0],  # 2: interior
        ])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        # virtual_boundary_mode = False (default)
        editor = PatchBasedMeshEditor(points, triangles, virtual_boundary_mode=False)
        
        success, msg, info = op_remove_node_with_patch2(editor, 0)
        
        # Should fail for boundary vertex
        assert not success
        assert "boundary" in msg.lower() or "cycle" in msg.lower()


class TestConfigurationIntegration:
    """Test configuration integration with op_remove_node_with_patch2."""
    
    def test_custom_config_respected(self):
        """Test that custom configuration is properly used."""
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
        
        # Custom config with specific preferences
        config = BoundaryRemoveConfig(
            prefer_area_preserving_star=True,
            prefer_worst_angle_star=False,
            require_area_preservation=False
        )
        editor.boundary_remove_config = config
        
        success, msg, info = op_remove_node_with_patch2(editor, 0)
        
        assert success, f"Failed: {msg}"
    
    def test_no_config_uses_defaults(self):
        """Test that missing config uses defaults."""
        points = np.array([
            [0, 0], [1, 0], [0.5, 1]
        ])
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        # Don't set boundary_remove_config
        
        # Should work with defaults but fail on this boundary case
        success, msg, info = op_remove_node_with_patch2(editor, 1)
        
        # Expect failure for boundary (not a good test case)
        # Let's use interior instead
        points2 = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ])
        triangles2 = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)
        editor2 = PatchBasedMeshEditor(points2, triangles2)
        
        # Add a center vertex
        center_idx = len(editor2.points)
        editor2.points = np.vstack([editor2.points, [[0.5, 0.5]]])
        
        # Reconnect with center
        new_tris = []
        for i in range(4):
            new_tris.append([center_idx, i, (i+1) % 4])
        editor2.triangles = np.array(new_tris, dtype=np.int32)
        
        # Rebuild maps using the proper method (not manual reconstruction with wrong types)
        editor2._update_maps(force=True)
        
        success, msg, info = op_remove_node_with_patch2(editor2, center_idx)
        assert success, f"Failed with default config: {msg}"


class TestConformityPreservation:
    """Test that mesh conformity is preserved."""
    
    def test_conformity_after_removal(self):
        """Test that mesh remains conforming after removal."""
        # Octagon with center
        angles = np.linspace(0, 2*np.pi, 9)[:-1]
        outer = 2 * np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        points = np.vstack([center, outer])
        
        triangles = []
        for i in range(8):
            triangles.append([0, i+1, ((i+1) % 8) + 1])
        triangles = np.array(triangles, dtype=np.int32)
        
        editor = PatchBasedMeshEditor(points, triangles)
        
        success, msg, info = op_remove_node_with_patch2(editor, 0)
        
        assert success, f"Failed: {msg}"
        
        # Check conformity
        from sofia.core.conformity import check_mesh_conformity
        ok, msgs = check_mesh_conformity(
            editor.points, editor.triangles, allow_marked=True
        )
        assert ok, f"Mesh not conforming after removal: {msgs}"


class TestComparisonWithOriginal:
    """Compare op_remove_node_with_patch2 with try_remove_node_strategically."""
    
    def test_similar_results(self):
        """Test that both functions produce similar results."""
        from sofia.core.operations import try_remove_node_strategically
        
        # Pentagon with center
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        outer = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        points = np.vstack([center, outer])
        
        triangles = []
        for i in range(5):
            triangles.append([0, i+1, ((i+1) % 5) + 1])
        triangles = np.array(triangles, dtype=np.int32)
        
        # Test with op_remove_node_with_patch2
        editor1 = PatchBasedMeshEditor(points.copy(), triangles.copy())
        success1, msg1, info1 = op_remove_node_with_patch2(editor1, 0)
        
        # Test with try_remove_node_strategically
        editor2 = PatchBasedMeshEditor(points.copy(), triangles.copy())
        success2, msg2, info2 = try_remove_node_strategically(editor2, 0)
        
        # Both should succeed
        assert success1, f"op_remove_node_with_patch2 failed: {msg1}"
        assert success2, f"try_remove_node_strategically failed: {msg2}"
        
        # Should produce same number of triangles
        assert info1['new_triangles'] == info2['new_triangles']
