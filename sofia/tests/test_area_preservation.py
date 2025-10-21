"""Unit tests for AreaPreservationChecker in quality.py"""
import numpy as np
import pytest
from sofia.sofia.quality import AreaPreservationChecker
from sofia.sofia.config import BoundaryRemoveConfig
from sofia.sofia.mesh_modifier2 import PatchBasedMeshEditor


def test_area_preservation_checker_default():
    """Test AreaPreservationChecker with default (strict) configuration."""
    checker = AreaPreservationChecker()
    
    assert checker.require_preservation is True
    assert checker.rel_tolerance > 0
    assert checker.abs_tolerance > 0


def test_area_preservation_checker_with_config():
    """Test AreaPreservationChecker with custom BoundaryRemoveConfig."""
    config = BoundaryRemoveConfig(require_area_preservation=False)
    checker = AreaPreservationChecker(config)
    
    assert checker.require_preservation is False


def test_area_preservation_check_exact_match():
    """Test that exact area match passes check."""
    checker = AreaPreservationChecker()
    
    removed_area = 1.0
    candidate_area = 1.0
    
    ok, msg = checker.check(removed_area, candidate_area)
    
    assert ok
    assert msg is None


def test_area_preservation_check_small_difference():
    """Test that small difference within tolerance passes."""
    checker = AreaPreservationChecker()
    
    removed_area = 1.0
    candidate_area = 1.0 + 1e-14  # Very small difference
    
    ok, msg = checker.check(removed_area, candidate_area)
    
    assert ok
    assert msg is None


def test_area_preservation_check_large_difference():
    """Test that large difference fails check."""
    checker = AreaPreservationChecker()
    
    removed_area = 1.0
    candidate_area = 1.5  # 50% difference
    
    ok, msg = checker.check(removed_area, candidate_area)
    
    assert not ok
    assert msg is not None
    assert "area not preserved" in msg
    assert "1.000000e+00" in msg  # removed area
    assert "1.500000e+00" in msg  # candidate area


def test_area_preservation_check_disabled():
    """Test that disabled preservation always passes."""
    config = BoundaryRemoveConfig(require_area_preservation=False)
    checker = AreaPreservationChecker(config)
    
    removed_area = 1.0
    candidate_area = 100.0  # Huge difference
    
    ok, msg = checker.check(removed_area, candidate_area)
    
    assert ok  # Should pass because preservation is disabled
    assert msg is None


def test_area_preservation_check_none_values():
    """Test that None values are handled gracefully."""
    checker = AreaPreservationChecker()
    
    # Both None
    ok, msg = checker.check(None, None)
    assert ok
    
    # Removed is None
    ok, msg = checker.check(None, 1.0)
    assert ok
    
    # Candidate is None
    ok, msg = checker.check(1.0, None)
    assert ok


def test_compute_cavity_area_simple():
    """Test computing area of a simple cavity."""
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.5],  # 4 center
    ], dtype=float)
    tris = np.array([
        [4, 0, 1],  # 0
        [4, 1, 2],  # 1
        [4, 2, 3],  # 2
        [4, 3, 0],  # 3
    ], dtype=int)
    
    checker = AreaPreservationChecker()
    
    # Compute area of all triangles (should be unit square = 1.0)
    area = checker.compute_cavity_area(pts, tris, [0, 1, 2, 3])
    
    assert area == pytest.approx(1.0, abs=1e-10)


def test_compute_cavity_area_single_triangle():
    """Test computing area of single triangle."""
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=float)
    tris = np.array([
        [0, 1, 2],
    ], dtype=int)
    
    checker = AreaPreservationChecker()
    area = checker.compute_cavity_area(pts, tris, [0])
    
    # Triangle with base=1, height=1: area = 0.5
    assert area == pytest.approx(0.5, abs=1e-10)


def test_compute_cavity_area_subset():
    """Test computing area of triangle subset."""
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.5],  # 4 center
    ], dtype=float)
    tris = np.array([
        [4, 0, 1],  # 0
        [4, 1, 2],  # 1
        [4, 2, 3],  # 2
        [4, 3, 0],  # 3
    ], dtype=int)
    
    checker = AreaPreservationChecker()
    
    # Compute area of only first two triangles (half of square = 0.5)
    area = checker.compute_cavity_area(pts, tris, [0, 1])
    
    assert area == pytest.approx(0.5, abs=1e-10)


def test_area_preservation_integration():
    """Integration test with actual mesh editor."""
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
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
    
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    
    # Create checker with editor's config
    config = getattr(editor, 'boundary_remove_config', None)
    checker = AreaPreservationChecker(config)
    
    # Compute cavity area (all 4 triangles)
    cavity_area = checker.compute_cavity_area(
        editor.points, 
        editor.triangles, 
        [0, 1, 2, 3]
    )
    
    assert cavity_area == pytest.approx(1.0, abs=1e-10)
    
    # Check area preservation (same area)
    ok, msg = checker.check(cavity_area, 1.0)
    assert ok


def test_area_preservation_tolerances():
    """Test that tolerances work correctly."""
    from sofia.sofia.constants import EPS_TINY, EPS_AREA
    
    checker = AreaPreservationChecker()
    
    # Check default tolerances are set
    assert checker.rel_tolerance == EPS_TINY
    assert checker.abs_tolerance == 4.0 * EPS_AREA
    
    # Test that difference within tolerance passes
    removed_area = 1.0
    candidate_area = 1.0 + EPS_TINY * 0.5 * removed_area  # Within relative tolerance
    
    ok, msg = checker.check(removed_area, candidate_area)
    assert ok


def test_area_preservation_custom_config():
    """Test custom configuration with relaxed tolerances."""
    # Create config with custom tolerances
    config = BoundaryRemoveConfig(
        require_area_preservation=True,
        area_tol_rel=1e-6,
        area_tol_abs_factor=10.0
    )
    
    checker = AreaPreservationChecker(config)
    
    assert checker.require_preservation is True
    assert checker.rel_tolerance == 1e-6
    # abs_tolerance = 10.0 * EPS_AREA


def test_compute_cavity_area_empty():
    """Test computing area of empty cavity."""
    pts = np.array([[0.0, 0.0]], dtype=float)
    tris = np.array([[0, 0, 0]], dtype=int)
    
    checker = AreaPreservationChecker()
    area = checker.compute_cavity_area(pts, tris, [])
    
    assert area == 0.0


def test_area_preservation_negative_areas():
    """Test that absolute values are used (negative areas handled)."""
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=float)
    # Reversed winding order gives negative area
    tris = np.array([
        [0, 2, 1],  # Reversed from [0, 1, 2]
    ], dtype=int)
    
    checker = AreaPreservationChecker()
    area = checker.compute_cavity_area(pts, tris, [0])
    
    # Should be positive due to abs()
    assert area == pytest.approx(0.5, abs=1e-10)
    assert area > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
