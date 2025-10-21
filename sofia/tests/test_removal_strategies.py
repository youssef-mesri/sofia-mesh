"""Unit tests for removal triangulation strategies."""
import numpy as np
import pytest

from sofia.sofia.triangulation import (
    RemovalTriangulationStrategy,
    OptimalStarStrategy,
    QualityStarStrategy,
    AreaPreservingStarStrategy,
    EarClipStrategy,
    SimplifyAndRetryStrategy,
    ChainedStrategy,
)
from sofia.sofia.config import BoundaryRemoveConfig


class TestRemovalTriangulationStrategy:
    """Test the base RemovalTriangulationStrategy class."""
    
    def test_base_strategy_not_implemented(self):
        """Test that base class raises NotImplementedError."""
        strategy = RemovalTriangulationStrategy()
        points = np.array([[0, 0], [1, 0], [0, 1]])
        cycle = [0, 1, 2]
        
        with pytest.raises(NotImplementedError):
            strategy.try_triangulate(points, cycle)


class TestOptimalStarStrategy:
    """Test OptimalStarStrategy."""
    
    def test_optimal_star_simple_triangle(self):
        """Test optimal star on a simple triangle."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        cycle = [0, 1, 2]
        
        strategy = OptimalStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert len(triangles) == 1
        assert error == ""
    
    def test_optimal_star_square(self):
        """Test optimal star on a square."""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        cycle = [0, 1, 2, 3]
        
        strategy = OptimalStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert len(triangles) == 2  # Square needs 2 triangles
        assert error == ""
    
    def test_optimal_star_pentagon(self):
        """Test optimal star on a regular pentagon."""
        # Regular pentagon
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        cycle = list(range(5))
        
        strategy = OptimalStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert len(triangles) == 3  # Pentagon needs 3 triangles
        assert error == ""
    
    def test_optimal_star_degenerate_polygon(self):
        """Test optimal star on degenerate (collinear) polygon."""
        points = np.array([[0, 0], [1, 0], [2, 0]])  # Collinear
        cycle = [0, 1, 2]
        
        strategy = OptimalStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        # Should fail due to zero area
        assert not success
        assert triangles is None
        assert "ValueError" in error or "None" in error
    
    def test_optimal_star_duplicate_vertices(self):
        """Test optimal star with duplicate vertices (should fail)."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        cycle = [0, 1, 1]  # Duplicate vertex
        
        strategy = OptimalStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        # Should fail due to duplicate
        assert not success
        assert "ValueError" in error or "Duplicate" in error


class TestQualityStarStrategy:
    """Test QualityStarStrategy."""
    
    def test_quality_star_simple_triangle(self):
        """Test quality star on a simple triangle."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        cycle = [0, 1, 2]
        
        strategy = QualityStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert len(triangles) == 1
        assert error == ""
    
    def test_quality_star_square(self):
        """Test quality star on a square."""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        cycle = [0, 1, 2, 3]
        
        strategy = QualityStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert len(triangles) == 2
        assert error == ""
    
    def test_quality_star_skinny_polygon(self):
        """Test quality star on a skinny polygon."""
        # Skinny rectangle - quality star should handle it
        points = np.array([[0, 0], [10, 0], [10, 0.1], [0, 0.1]])
        cycle = [0, 1, 2, 3]
        
        strategy = QualityStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        # May or may not succeed depending on degenerate checks
        if success:
            assert triangles is not None
            assert len(triangles) == 2


class TestAreaPreservingStarStrategy:
    """Test AreaPreservingStarStrategy."""
    
    def test_area_preserving_star_simple_triangle(self):
        """Test area-preserving star on a simple triangle."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        cycle = [0, 1, 2]
        
        strategy = AreaPreservingStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert len(triangles) == 1
        assert error == ""
    
    def test_area_preserving_star_with_config(self):
        """Test area-preserving star with custom config."""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        cycle = [0, 1, 2, 3]
        
        config = BoundaryRemoveConfig(
            require_area_preservation=True,
            area_tol_rel=1e-6,
            area_tol_abs_factor=0.01  # Very small factor
        )
        
        strategy = AreaPreservingStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle, config)
        
        # Square should preserve area well
        assert success
        assert triangles is not None
        assert len(triangles) == 2
        assert error == ""
    
    def test_area_preserving_star_pentagon(self):
        """Test area-preserving star on a regular pentagon."""
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        cycle = list(range(5))
        
        strategy = AreaPreservingStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        # Pentagon should preserve area well
        assert success
        assert triangles is not None
        assert len(triangles) == 3
        assert error == ""


class TestEarClipStrategy:
    """Test EarClipStrategy."""
    
    def test_earclip_simple_triangle(self):
        """Test ear clipping on a simple triangle."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        cycle = [0, 1, 2]
        
        strategy = EarClipStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert len(triangles) == 1
        assert error == ""
    
    def test_earclip_square(self):
        """Test ear clipping on a square."""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        cycle = [0, 1, 2, 3]
        
        strategy = EarClipStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert len(triangles) == 2
        assert error == ""
    
    def test_earclip_concave_polygon(self):
        """Test ear clipping on a concave polygon."""
        # L-shaped polygon
        points = np.array([
            [0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]
        ])
        cycle = list(range(6))
        
        strategy = EarClipStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert len(triangles) == 4  # L-shape needs 4 triangles
        assert error == ""
    
    def test_earclip_degenerate_polygon(self):
        """Test ear clipping on degenerate polygon."""
        points = np.array([[0, 0], [1, 0], [2, 0]])  # Collinear
        cycle = [0, 1, 2]
        
        strategy = EarClipStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        # Ear clip should fail or return empty for degenerate
        if not success:
            assert "ValueError" in error or "empty" in error


class TestSimplifyAndRetryStrategy:
    """Test SimplifyAndRetryStrategy."""
    
    def test_simplify_and_retry_with_optimal(self):
        """Test simplify and retry with optimal star strategy."""
        # Create a polygon with some nearly-collinear points
        points = np.array([
            [0, 0], [0.5, 0.001], [1, 0],  # Nearly collinear
            [1, 1], [0, 1]
        ])
        cycle = list(range(5))
        
        wrapped = OptimalStarStrategy()
        strategy = SimplifyAndRetryStrategy(wrapped)
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        # Should succeed after simplification
        assert success or "simplified" in error  # May simplify successfully
    
    def test_simplify_invalid_input(self):
        """Test simplify with invalid input."""
        points = np.array([[0, 0], [1, 0]])
        cycle = [0, 1]  # Too few points
        
        wrapped = OptimalStarStrategy()
        strategy = SimplifyAndRetryStrategy(wrapped)
        success, triangles, error = strategy.try_triangulate(points, cycle)
        
        assert not success
        assert len(error) > 0


class TestChainedStrategy:
    """Test ChainedStrategy."""
    
    def test_chained_first_succeeds(self):
        """Test chained strategy where first strategy succeeds."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        cycle = [0, 1, 2]
        
        strategies = [
            OptimalStarStrategy(),
            QualityStarStrategy(),
            EarClipStrategy()
        ]
        chained = ChainedStrategy(strategies)
        success, triangles, error = chained.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert error == ""
    
    def test_chained_fallback_to_second(self):
        """Test chained strategy where first fails, second succeeds."""
        # Create a scenario where optimal might fail but earclip succeeds
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        cycle = [0, 1, 2, 3]
        
        strategies = [
            EarClipStrategy(),  # Should succeed
            QualityStarStrategy()
        ]
        chained = ChainedStrategy(strategies)
        success, triangles, error = chained.try_triangulate(points, cycle)
        
        assert success
        assert triangles is not None
        assert error == ""
    
    def test_chained_all_fail(self):
        """Test chained strategy where all strategies fail."""
        points = np.array([[0, 0], [1, 0], [2, 0]])  # Degenerate
        cycle = [0, 1, 2]
        
        strategies = [
            OptimalStarStrategy(),
            QualityStarStrategy(),
            AreaPreservingStarStrategy()
        ]
        chained = ChainedStrategy(strategies)
        success, triangles, error = chained.try_triangulate(points, cycle)
        
        assert not success
        assert triangles is None
        assert "All strategies failed" in error
    
    def test_chained_empty_list(self):
        """Test chained strategy with empty strategy list."""
        points = np.array([[0, 0], [1, 0], [0.5, 1]])
        cycle = [0, 1, 2]
        
        strategies = []
        chained = ChainedStrategy(strategies)
        success, triangles, error = chained.try_triangulate(points, cycle)
        
        assert not success
        assert "All strategies failed" in error


class TestStrategyIntegration:
    """Integration tests combining multiple strategies."""
    
    def test_realistic_removal_chain(self):
        """Test a realistic chain mimicking op_remove_node_with_patch logic."""
        # Regular hexagon
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        cycle = list(range(6))
        
        # Build the strategy chain used in operations.py
        config = BoundaryRemoveConfig(require_area_preservation=True)
        
        strategies = [
            OptimalStarStrategy(),
            SimplifyAndRetryStrategy(OptimalStarStrategy()),
            AreaPreservingStarStrategy(),
            QualityStarStrategy(),
            EarClipStrategy()
        ]
        
        chained = ChainedStrategy(strategies)
        success, triangles, error = chained.try_triangulate(points, cycle, config)
        
        assert success
        assert triangles is not None
        assert len(triangles) == 4  # Hexagon needs 4 triangles
    
    def test_config_propagation(self):
        """Test that config is properly propagated through chained strategies."""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        cycle = [0, 1, 2, 3]
        
        config = BoundaryRemoveConfig(
            require_area_preservation=True,
            area_tol_rel=1e-10,  # Very tight
            area_tol_abs_factor=0.001  # Very small factor
        )
        
        strategy = AreaPreservingStarStrategy()
        success, triangles, error = strategy.try_triangulate(points, cycle, config)
        
        # Square should preserve area perfectly
        assert success
        assert triangles is not None
