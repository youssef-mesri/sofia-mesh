"""Unit tests for geometry utility functions."""
import numpy as np
import pytest

from sofia.sofia.geometry import compute_triangulation_area, normalize_edge, triangle_area


class TestComputeTriangulationArea:
    """Test compute_triangulation_area function."""
    
    def test_compute_triangulation_area_empty(self):
        """Test that empty triangle list returns zero area."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        triangles = np.array([[0, 1, 2]])
        
        area = compute_triangulation_area(points, triangles, [])
        assert area == 0.0
    
    def test_compute_triangulation_area_single_triangle(self):
        """Test area computation for a single triangle."""
        # Right triangle with base=1, height=1, area=0.5
        points = np.array([[0, 0], [1, 0], [0, 1]])
        triangles = np.array([[0, 1, 2]])
        
        area = compute_triangulation_area(points, triangles, [0])
        assert abs(area - 0.5) < 1e-10
    
    def test_compute_triangulation_area_multiple_triangles(self):
        """Test area computation for multiple triangles."""
        # Two triangles forming a square: total area = 1
        points = np.array([
            [0, 0],  # 0
            [1, 0],  # 1
            [1, 1],  # 2
            [0, 1]   # 3
        ])
        triangles = np.array([
            [0, 1, 2],  # Lower-right triangle, area=0.5
            [0, 2, 3]   # Upper-left triangle, area=0.5
        ])
        
        area = compute_triangulation_area(points, triangles, [0, 1])
        assert abs(area - 1.0) < 1e-10
    
    def test_compute_triangulation_area_subset(self):
        """Test area computation for a subset of triangles."""
        points = np.array([
            [0, 0],  # 0
            [1, 0],  # 1
            [1, 1],  # 2
            [0, 1]   # 3
        ])
        triangles = np.array([
            [0, 1, 2],  # area=0.5
            [0, 2, 3]   # area=0.5
        ])
        
        # Only first triangle
        area = compute_triangulation_area(points, triangles, [0])
        assert abs(area - 0.5) < 1e-10
        
        # Only second triangle
        area = compute_triangulation_area(points, triangles, [1])
        assert abs(area - 0.5) < 1e-10
    
    def test_compute_triangulation_area_negative_orientation(self):
        """Test that negative orientation (CW) triangles have positive area sum."""
        # Triangle with negative (clockwise) orientation
        points = np.array([[0, 0], [0, 1], [1, 0]])  # CW order
        triangles = np.array([[0, 1, 2]])
        
        # Should return absolute value
        area = compute_triangulation_area(points, triangles, [0])
        assert area > 0
        assert abs(area - 0.5) < 1e-10
    
    def test_compute_triangulation_area_large_mesh(self):
        """Test area computation for a larger mesh."""
        # Create a simple regular grid (2x2 squares = 8 triangles)
        points = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1],
            [0, 2], [1, 2], [2, 2]
        ], dtype=float)
        
        triangles = np.array([
            [0, 1, 4], [0, 4, 3],  # Bottom-left square
            [1, 2, 5], [1, 5, 4],  # Bottom-right square
            [3, 4, 7], [3, 7, 6],  # Top-left square
            [4, 5, 8], [4, 8, 7]   # Top-right square
        ])
        
        # Total area should be 4 (2x2 square)
        all_indices = list(range(len(triangles)))
        area = compute_triangulation_area(points, triangles, all_indices)
        assert abs(area - 4.0) < 1e-10


class TestNormalizeEdge:
    """Test normalize_edge function."""
    
    def test_normalize_edge_already_ordered(self):
        """Test edge that is already in canonical form."""
        edge = normalize_edge(1, 5)
        assert edge == (1, 5)
    
    def test_normalize_edge_reversed(self):
        """Test edge that needs to be reversed."""
        edge = normalize_edge(5, 1)
        assert edge == (1, 5)
    
    def test_normalize_edge_equal_vertices(self):
        """Test edge with equal vertices (degenerate edge)."""
        edge = normalize_edge(3, 3)
        assert edge == (3, 3)
    
    def test_normalize_edge_zero_vertex(self):
        """Test edge involving vertex 0."""
        assert normalize_edge(0, 5) == (0, 5)
        assert normalize_edge(5, 0) == (0, 5)
    
    def test_normalize_edge_consistency(self):
        """Test that (u, v) and (v, u) produce the same result."""
        u, v = 7, 3
        edge1 = normalize_edge(u, v)
        edge2 = normalize_edge(v, u)
        assert edge1 == edge2
        assert edge1 == (3, 7)
    
    def test_normalize_edge_as_dict_key(self):
        """Test that normalized edges work as dictionary keys."""
        edge_data = {}
        
        # Add edge (5, 2)
        edge_data[normalize_edge(5, 2)] = "edge_A"
        
        # Try to access via reversed edge (2, 5)
        key = normalize_edge(2, 5)
        assert key in edge_data
        assert edge_data[key] == "edge_A"
