#!/usr/bin/env python3
"""
benchmark comparing performance with and without Numba.

This script tests the same operations twice:
1. With Numba enabled (if available)
2. With Numba disabled (forced fallback to NumPy)
"""

import time
import numpy as np
import sys

def benchmark_geometry_operations(use_numba=True):
    """Benchmark geometry.py operations."""
    # Import/reload to respect HAS_NUMBA changes
    if 'sofia.core.geometry' in sys.modules:
        del sys.modules['sofia.core.geometry']
    
    # Temporarily disable Numba if requested
    if not use_numba:
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'numba' or name.startswith('numba.'):
                raise ImportError("Numba disabled for benchmark")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = mock_import
    
    try:
        from sofia.core import geometry
        
        # Verify Numba state
        actual_has_numba = geometry.HAS_NUMBA
        
        results = {
            'has_numba': actual_has_numba,
            'triangles_signed_areas': None,
            'triangles_min_angles': None,
            'compute_triangulation_area': None
        }
        
        # 1. triangles_signed_areas
        n_tris = 10000
        points = np.random.rand(n_tris * 3, 2) * 100
        tris = np.arange(n_tris * 3).reshape(n_tris, 3)
        
        # Warm up
        _ = geometry.triangles_signed_areas(points, tris)
        
        # Benchmark
        n_iters = 100
        start = time.time()
        for _ in range(n_iters):
            _ = geometry.triangles_signed_areas(points, tris)
        elapsed = time.time() - start
        results['triangles_signed_areas'] = elapsed / n_iters
        
        # 2. triangles_min_angles
        # Warm up
        _ = geometry.triangles_min_angles(points, tris)
        
        # Benchmark
        n_iters = 100
        start = time.time()
        for _ in range(n_iters):
            _ = geometry.triangles_min_angles(points, tris)
        elapsed = time.time() - start
        results['triangles_min_angles'] = elapsed / n_iters
        
        # 3. compute_triangulation_area
        n_tris_small = 1000
        points_small = np.random.rand(n_tris_small * 3, 2) * 100
        tris_small = np.arange(n_tris_small * 3).reshape(n_tris_small, 3)
        indices = np.arange(n_tris_small)
        
        # Warm up
        _ = geometry.compute_triangulation_area(points_small, tris_small, indices)
        
        # Benchmark
        n_iters = 500
        start = time.time()
        for _ in range(n_iters):
            _ = geometry.compute_triangulation_area(points_small, tris_small, indices)
        elapsed = time.time() - start
        results['compute_triangulation_area'] = elapsed / n_iters
        
        return results
        
    finally:
        # Restore original import if we mocked it
        if not use_numba:
            import builtins
            builtins.__import__ = original_import


def benchmark_conformity_operations(use_numba=True):
    """Benchmark conformity.py operations."""
    # Import/reload to respect HAS_NUMBA changes
    if 'sofia.core.conformity' in sys.modules:
        del sys.modules['sofia.core.conformity']
    if 'sofia.core.geometry' in sys.modules:
        del sys.modules['sofia.core.geometry']
    
    # Temporarily disable Numba if requested
    if not use_numba:
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'numba' or name.startswith('numba.'):
                raise ImportError("Numba disabled for benchmark")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = mock_import
    
    try:
        from sofia.core import conformity
        
        # Verify Numba state
        actual_has_numba = conformity.HAS_NUMBA
        
        results = {
            'has_numba': actual_has_numba,
            'build_kept_edge_grid_1k': None,
            'build_kept_edge_grid_5k': None,
            'filter_crossing_candidate_edges': None,
            'simulate_compaction': None
        }
        
        # 1. build_kept_edge_grid with 1k edges
        n_points = 1000
        points = np.random.rand(n_points, 2) * 100
        edges_1k = [(i, (i+1) % n_points) for i in range(1000)]
        
        # Warm up
        _ = conformity.build_kept_edge_grid(points, edges_1k)
        
        # Benchmark
        n_iters = 100
        start = time.time()
        for _ in range(n_iters):
            _ = conformity.build_kept_edge_grid(points, edges_1k)
        elapsed = time.time() - start
        results['build_kept_edge_grid_1k'] = elapsed / n_iters
        
        # 2. build_kept_edge_grid with 5k edges
        n_points = 5000
        points_5k = np.random.rand(n_points, 2) * 100
        edges_5k = [(i, (i+1) % n_points) for i in range(5000)]
        
        # Warm up
        _ = conformity.build_kept_edge_grid(points_5k, edges_5k)
        
        # Benchmark
        n_iters = 50
        start = time.time()
        for _ in range(n_iters):
            _ = conformity.build_kept_edge_grid(points_5k, edges_5k)
        elapsed = time.time() - start
        results['build_kept_edge_grid_5k'] = elapsed / n_iters
        
        # 3. filter_crossing_candidate_edges
        n_points = 700
        points_filter = np.random.rand(n_points, 2) * 100
        kept_edges = [(i, (i+1) % 500) for i in range(500)]
        cand_edges = [(i, (i+50) % 700) for i in range(500, 700)]
        kept_grid = conformity.build_kept_edge_grid(points_filter, kept_edges)
        
        # Warm up
        _ = conformity.filter_crossing_candidate_edges(points_filter, kept_edges, cand_edges, kept_grid)
        
        # Benchmark
        n_iters = 50
        start = time.time()
        for _ in range(n_iters):
            _ = conformity.filter_crossing_candidate_edges(points_filter, kept_edges, cand_edges, kept_grid)
        elapsed = time.time() - start
        results['filter_crossing_candidate_edges'] = elapsed / n_iters
        
        # 4. simulate_compaction_and_check with crossing detection
        n_tris = 500
        n_points = n_tris * 2
        points_sim = np.random.rand(n_points, 2) * 100
        triangles = []
        for i in range(n_tris):
            a = np.random.randint(0, n_points - 2)
            b = a + 1
            c = a + 2
            triangles.append([a, b, c])
        triangles = np.array(triangles, dtype=np.int32)
        
        # Warm up
        _ = conformity.simulate_compaction_and_check(points_sim, triangles, reject_crossing_edges=True)
        
        # Benchmark
        n_iters = 20
        start = time.time()
        for _ in range(n_iters):
            _ = conformity.simulate_compaction_and_check(points_sim, triangles, reject_crossing_edges=True)
        elapsed = time.time() - start
        results['simulate_compaction'] = elapsed / n_iters
        
        return results
        
    finally:
        # Restore original import if we mocked it
        if not use_numba:
            import builtins
            builtins.__import__ = original_import


def print_comparison(with_numba, without_numba, category_name):
    """Print comparison table for a category."""
    print(f"\n{'='*80}")
    print(f"{category_name.upper()}")
    print(f"{'='*80}")
    print(f"{'Operation':<40} {'With Numba':<15} {'Without Numba':<15} {'Speedup':<10}")
    print(f"{'-'*80}")
    
    for key in with_numba.keys():
        if key == 'has_numba':
            continue
        
        time_with = with_numba[key]
        time_without = without_numba[key]
        
        if time_with is not None and time_without is not None:
            speedup = time_without / time_with
            print(f"{key:<40} {time_with*1000:>10.3f} ms {time_without*1000:>12.3f} ms {speedup:>8.2f}x")
    
    print(f"{'-'*80}")


if __name__ == "__main__":
    print("="*80)
    print("NUMBA PERFORMANCE COMPARISON BENCHMARK")
    print("="*80)
    print("\nThis benchmark compares performance with Numba enabled vs disabled.")
    print("Each operation is tested with identical inputs in both modes.\n")
    
    # Test geometry operations
    print("\n[1/4] Testing GEOMETRY operations WITH Numba...")
    geo_with = benchmark_geometry_operations(use_numba=True)
    print(f"      Numba status: {geo_with['has_numba']}")
    
    print("\n[2/4] Testing GEOMETRY operations WITHOUT Numba...")
    geo_without = benchmark_geometry_operations(use_numba=False)
    print(f"      Numba status: {geo_without['has_numba']}")
    
    # Test conformity operations
    print("\n[3/4] Testing CONFORMITY operations WITH Numba...")
    conf_with = benchmark_conformity_operations(use_numba=True)
    print(f"      Numba status: {conf_with['has_numba']}")
    
    print("\n[4/4] Testing CONFORMITY operations WITHOUT Numba...")
    conf_without = benchmark_conformity_operations(use_numba=False)
    print(f"      Numba status: {conf_without['has_numba']}")
    
    # Print results
    print_comparison(geo_with, geo_without, "Geometry Operations (geometry.py)")
    print_comparison(conf_with, conf_without, "Grid Operations (conformity.py)")
    
    # Calculate average speedups
    geo_speedups = []
    for key in geo_with.keys():
        if key != 'has_numba' and geo_with[key] and geo_without[key]:
            geo_speedups.append(geo_without[key] / geo_with[key])
    
    conf_speedups = []
    for key in conf_with.keys():
        if key != 'has_numba' and conf_with[key] and conf_without[key]:
            conf_speedups.append(conf_without[key] / conf_with[key])
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Geometry operations average speedup:   {np.mean(geo_speedups):.2f}x")
    print(f"Grid operations average speedup:       {np.mean(conf_speedups):.2f}x")
    print(f"Overall average speedup:               {np.mean(geo_speedups + conf_speedups):.2f}x")
    print("="*80)
    
    print("\nConclusion:")
    if np.mean(geo_speedups + conf_speedups) > 3:
        print("  Numba provides SIGNIFICANT performance improvements (>3x)")
    elif np.mean(geo_speedups + conf_speedups) > 1.5:
        print("  Numba provides MODERATE performance improvements (1.5-3x)")
    else:
        print("  Numba provides minimal performance improvements (<1.5x)")
    print("="*80)
