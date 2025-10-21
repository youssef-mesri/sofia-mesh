"""
Example: Phase 1 Refactoring - Extract Quality Metrics

This file shows how to extract the nested quality computation functions
from op_remove_node_with_patch into a reusable module.
"""

import numpy as np
from .constants import EPS_AREA


class TriangleQualityMetrics:
    """Compute various quality metrics for triangles."""
    
    @staticmethod
    def normalized_quality(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Compute normalized quality metrics for triangles.
        
        Quality = (area) / (sum(edge_len^2)) scaled so equilateral -> 1.
        
        Args:
            points: (N, 2) array of vertex coordinates
            triangles: (M, 3) array of triangle vertex indices
            
        Returns:
            (M,) array of quality values in [0, 1], where 1 is equilateral
        """
        pts = np.asarray(points, dtype=np.float64)
        tris = np.asarray(triangles, dtype=np.int32)
        
        if tris.size == 0:
            return np.empty((0,), dtype=np.float64)
        
        # Get triangle vertices
        p0 = pts[tris[:, 0]]
        p1 = pts[tris[:, 1]]
        p2 = pts[tris[:, 2]]
        
        # Compute signed areas
        areas = 0.5 * np.abs(
            (p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - 
            (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0])
        )
        
        # Compute edge lengths squared
        e01_sq = np.sum((p1 - p0)**2, axis=1)
        e12_sq = np.sum((p2 - p1)**2, axis=1)
        e20_sq = np.sum((p0 - p2)**2, axis=1)
        edge_sum_sq = e01_sq + e12_sq + e20_sq
        
        # Avoid division by zero
        safe_denom = np.maximum(edge_sum_sq, 1e-30)
        
        # Raw quality: area / sum(edge_len^2)
        raw_q = areas / safe_denom
        
        # Normalization factor: for equilateral triangle, 
        # area = (sqrt(3)/4) * side^2, sum(edges^2) = 3 * side^2
        # So raw_q = (sqrt(3)/4) / 3 = sqrt(3)/12
        # To normalize: multiply by 12/sqrt(3) = 4*sqrt(3) ≈ 6.928
        norm_factor = 4.0 * np.sqrt(3.0)
        normalized_q = raw_q * norm_factor
        
        # Clamp to [0, 1] to handle numerical errors
        return np.clip(normalized_q, 0.0, 1.0)
    
    @staticmethod
    def min_angles_degrees(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Compute minimum interior angle in degrees for each triangle.
        
        Args:
            points: (N, 2) array of vertex coordinates
            triangles: (M, 3) array of triangle vertex indices
            
        Returns:
            (M,) array of minimum angles in degrees [0, 180]
        """
        pts = np.asarray(points, dtype=np.float64)
        tris = np.asarray(triangles, dtype=np.int32)
        
        if tris.size == 0:
            return np.empty((0,), dtype=np.float64)
        
        # Get triangle vertices
        p0 = pts[tris[:, 0]]
        p1 = pts[tris[:, 1]]
        p2 = pts[tris[:, 2]]
        
        # Compute edge vectors
        e01 = p1 - p0
        e12 = p2 - p1
        e20 = p0 - p2
        
        # Edge lengths
        len_e01 = np.linalg.norm(e01, axis=1)
        len_e12 = np.linalg.norm(e12, axis=1)
        len_e20 = np.linalg.norm(e20, axis=1)
        
        # Avoid division by zero
        len_e01 = np.maximum(len_e01, 1e-30)
        len_e12 = np.maximum(len_e12, 1e-30)
        len_e20 = np.maximum(len_e20, 1e-30)
        
        # Compute angles using dot product
        # Angle at p0: between e01 and -e20
        cos_a0 = np.sum(e01 * (-e20), axis=1) / (len_e01 * len_e20)
        # Angle at p1: between e12 and -e01
        cos_a1 = np.sum(e12 * (-e01), axis=1) / (len_e12 * len_e01)
        # Angle at p2: between e20 and -e12
        cos_a2 = np.sum(e20 * (-e12), axis=1) / (len_e20 * len_e12)
        
        # Clamp to [-1, 1] to handle numerical errors
        cos_a0 = np.clip(cos_a0, -1.0, 1.0)
        cos_a1 = np.clip(cos_a1, -1.0, 1.0)
        cos_a2 = np.clip(cos_a2, -1.0, 1.0)
        
        # Convert to angles in radians, then degrees
        angles_0 = np.arccos(cos_a0) * (180.0 / np.pi)
        angles_1 = np.arccos(cos_a1) * (180.0 / np.pi)
        angles_2 = np.arccos(cos_a2) * (180.0 / np.pi)
        
        # Stack and find minimum per triangle
        angles_stacked = np.column_stack([angles_0, angles_1, angles_2])
        min_angles = np.min(angles_stacked, axis=1)
        
        return min_angles
    
    @staticmethod
    def aspect_ratio(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Compute aspect ratio (longest edge / shortest edge) for triangles.
        
        Args:
            points: (N, 2) array of vertex coordinates
            triangles: (M, 3) array of triangle vertex indices
            
        Returns:
            (M,) array of aspect ratios >= 1.0 (1.0 = equilateral)
        """
        pts = np.asarray(points, dtype=np.float64)
        tris = np.asarray(triangles, dtype=np.int32)
        
        if tris.size == 0:
            return np.empty((0,), dtype=np.float64)
        
        p0 = pts[tris[:, 0]]
        p1 = pts[tris[:, 1]]
        p2 = pts[tris[:, 2]]
        
        len_e01 = np.linalg.norm(p1 - p0, axis=1)
        len_e12 = np.linalg.norm(p2 - p1, axis=1)
        len_e20 = np.linalg.norm(p0 - p2, axis=1)
        
        edges_stacked = np.column_stack([len_e01, len_e12, len_e20])
        longest = np.max(edges_stacked, axis=1)
        shortest = np.min(edges_stacked, axis=1)
        
        # Avoid division by zero
        shortest = np.maximum(shortest, 1e-30)
        
        return longest / shortest
    
    @staticmethod
    def is_degenerate(points: np.ndarray, triangles: np.ndarray, 
                     area_threshold: float = EPS_AREA) -> np.ndarray:
        """Check if triangles are degenerate (near-zero area).
        
        Args:
            points: (N, 2) array of vertex coordinates
            triangles: (M, 3) array of triangle vertex indices
            area_threshold: minimum area threshold
            
        Returns:
            (M,) boolean array, True for degenerate triangles
        """
        pts = np.asarray(points, dtype=np.float64)
        tris = np.asarray(triangles, dtype=np.int32)
        
        if tris.size == 0:
            return np.empty((0,), dtype=bool)
        
        p0 = pts[tris[:, 0]]
        p1 = pts[tris[:, 1]]
        p2 = pts[tris[:, 2]]
        
        areas = 0.5 * np.abs(
            (p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - 
            (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0])
        )
        
        return areas < area_threshold


# Example usage in operations.py after refactoring:
"""
from .quality_metrics import TriangleQualityMetrics

def op_remove_node_with_patch(editor, v_idx, force_strict=False):
    ...
    # Instead of embedded _triangle_qualities_norm function:
    quality_metrics = TriangleQualityMetrics()
    
    # Compute quality for old triangles
    old_qualities = quality_metrics.normalized_quality(
        editor.points, 
        editor.triangles[cavity_tri_indices]
    )
    
    # Compute quality for candidate triangulation
    cand_qualities = quality_metrics.normalized_quality(
        editor.points,
        candidate_triangles
    )
    
    # Compute min angles
    cand_min_angles = quality_metrics.min_angles_degrees(
        editor.points,
        candidate_triangles
    )
    ...
"""
