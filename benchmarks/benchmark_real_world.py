#!/usr/bin/env python3
"""
Real-world end-to-end performance comparison.

Tests actual mesh remeshing operations with and without Numba to show
the practical performance impact on typical workflows.
"""

import time
import numpy as np
import sys
import os

# Save original state
original_env = os.environ.copy()

def run_with_numba_state(enable_numba):
    """Run test with Numba enabled or disabled."""
    # Clear module cache
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('sofia')]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    # Set environment to disable/enable Numba
    if not enable_numba:
        os.environ['NUMBA_DISABLE_JIT'] = '1'
    else:
        os.environ.pop('NUMBA_DISABLE_JIT', None)
    
    # Import after environment change
    from sofia.core import conformity, geometry
    from sofia import PatchBasedMeshEditor
    
    return conformity.HAS_NUMBA, geometry.HAS_NUMBA


def benchmark_real_world_operations(enable_numba=True):
    """Benchmark real-world mesh operations."""
    # Set environment
    if not enable_numba:
        os.environ['NUMBA_DISABLE_JIT'] = '1'
    else:
        os.environ.pop('NUMBA_DISABLE_JIT', None)
    
    # Clear module cache
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('sofia')]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    # Import fresh
    from sofia.core import conformity, geometry
    from sofia import PatchBasedMeshEditor
    
    results = {
        'has_numba_conf': conformity.HAS_NUMBA,
        'has_numba_geom': geometry.HAS_NUMBA,
        'crossing_detection_small': None,
        'crossing_detection_medium': None,
        'crossing_detection_large': None,
        'edge_filtering': None,
    }
    
    # 1. Small mesh crossing detection (100 triangles)
    print(f"   Testing small mesh (100 tris)...", end='', flush=True)
    n_points = 200
    points = np.random.rand(n_points, 2) * 100
    triangles = []
    for i in range(100):
        a = np.random.randint(0, n_points - 2)
        b = a + 1
        c = a + 2
        triangles.append([a, b, c])
    triangles = np.array(triangles, dtype=np.int32)
    
    # Warm up
    _ = conformity.simulate_compaction_and_check(points, triangles, reject_crossing_edges=True)
    
    # Benchmark
    n_iters = 50
    start = time.time()
    for _ in range(n_iters):
        _ = conformity.simulate_compaction_and_check(points, triangles, reject_crossing_edges=True)
    elapsed = (time.time() - start) / n_iters
    results['crossing_detection_small'] = elapsed
    print(f" {elapsed*1000:.1f}ms")
    
    # 2. Medium mesh crossing detection (500 triangles)
    print(f"   Testing medium mesh (500 tris)...", end='', flush=True)
    n_points = 1000
    points = np.random.rand(n_points, 2) * 100
    triangles = []
    for i in range(500):
        a = np.random.randint(0, n_points - 2)
        b = a + 1
        c = a + 2
        triangles.append([a, b, c])
    triangles = np.array(triangles, dtype=np.int32)
    
    # Warm up
    _ = conformity.simulate_compaction_and_check(points, triangles, reject_crossing_edges=True)
    
    # Benchmark
    n_iters = 20
    start = time.time()
    for _ in range(n_iters):
        _ = conformity.simulate_compaction_and_check(points, triangles, reject_crossing_edges=True)
    elapsed = (time.time() - start) / n_iters
    results['crossing_detection_medium'] = elapsed
    print(f" {elapsed*1000:.1f}ms")
    
    # 3. Large mesh crossing detection (2000 triangles)
    print(f"   Testing large mesh (2000 tris)...", end='', flush=True)
    n_points = 4000
    points = np.random.rand(n_points, 2) * 100
    triangles = []
    for i in range(2000):
        a = np.random.randint(0, n_points - 2)
        b = a + 1
        c = a + 2
        triangles.append([a, b, c])
    triangles = np.array(triangles, dtype=np.int32)
    
    # Warm up
    _ = conformity.simulate_compaction_and_check(points, triangles, reject_crossing_edges=True)
    
    # Benchmark
    n_iters = 5
    start = time.time()
    for _ in range(n_iters):
        _ = conformity.simulate_compaction_and_check(points, triangles, reject_crossing_edges=True)
    elapsed = (time.time() - start) / n_iters
    results['crossing_detection_large'] = elapsed
    print(f" {elapsed*1000:.1f}ms")
    
    # 4. Edge filtering (realistic workload)
    print(f"   Testing edge filtering (1000 edges)...", end='', flush=True)
    n_points = 1200
    points = np.random.rand(n_points, 2) * 100
    kept_edges = [(i, (i+1) % 1000) for i in range(1000)]
    cand_edges = [(i, (i+50) % 1200) for i in range(1000, 1200)]
    
    # Build grid once
    kept_grid = conformity.build_kept_edge_grid(points, kept_edges)
    
    # Warm up
    _ = conformity.filter_crossing_candidate_edges(points, kept_edges, cand_edges, kept_grid)
    
    # Benchmark
    n_iters = 50
    start = time.time()
    for _ in range(n_iters):
        _ = conformity.filter_crossing_candidate_edges(points, kept_edges, cand_edges, kept_grid)
    elapsed = (time.time() - start) / n_iters
    results['edge_filtering'] = elapsed
    print(f" {elapsed*1000:.1f}ms")
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("REAL-WORLD PERFORMANCE COMPARISON: WITH vs WITHOUT NUMBA")
    print("="*80)
    print("\nThis benchmark measures end-to-end performance on typical mesh operations.")
    print("Results show the actual time savings you get in production workflows.\n")
    
    # Test WITH Numba
    print("Running benchmarks WITH Numba enabled...")
    print("-" * 80)
    with_numba = benchmark_real_world_operations(enable_numba=True)
    print(f"\nNumba status: conformity.HAS_NUMBA={with_numba['has_numba_conf']}, "
          f"geometry.HAS_NUMBA={with_numba['has_numba_geom']}")
    
    # Test WITHOUT Numba
    print("\n" + "="*80)
    print("Running benchmarks WITHOUT Numba (pure NumPy/Python)...")
    print("-" * 80)
    without_numba = benchmark_real_world_operations(enable_numba=False)
    print(f"\nNumba status: conformity.HAS_NUMBA={without_numba['has_numba_conf']}, "
          f"geometry.HAS_NUMBA={without_numba['has_numba_geom']}")
    
    # Comparison table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Operation':<40} {'With Numba':<15} {'Without':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for key in ['crossing_detection_small', 'crossing_detection_medium', 
                'crossing_detection_large', 'edge_filtering']:
        t_with = with_numba[key]
        t_without = without_numba[key]
        if t_with and t_without:
            speedup = t_without / t_with
            name = key.replace('_', ' ').title()
            print(f"{name:<40} {t_with*1000:>10.1f} ms {t_without*1000:>12.1f} ms {speedup:>8.2f}x")
    
    print("-" * 80)
    
    # Calculate time saved
    small_saved = without_numba['crossing_detection_small'] - with_numba['crossing_detection_small']
    medium_saved = without_numba['crossing_detection_medium'] - with_numba['crossing_detection_medium']
    large_saved = without_numba['crossing_detection_large'] - with_numba['crossing_detection_large']
    
    print("\n" + "="*80)
    print("TIME SAVINGS ANALYSIS")
    print("="*80)
    print(f"\nFor a typical remeshing session with:")
    print(f"  - 10 small mesh operations  (100 tris):   {small_saved*10*1000:>8.1f} ms saved")
    print(f"  - 5 medium mesh operations (500 tris):    {medium_saved*5*1000:>8.1f} ms saved")
    print(f"  - 2 large mesh operations (2000 tris):    {large_saved*2*1000:>8.1f} ms saved")
    print(f"  {'='*45}")
    total_saved = (small_saved*10 + medium_saved*5 + large_saved*2)
    print(f"  Total time saved per session:             {total_saved*1000:>8.1f} ms")
    print(f"  That's {total_saved:.2f} seconds or {total_saved/60:.1f} minutes!")
    
    # Calculate average speedup
    speedups = []
    for key in ['crossing_detection_small', 'crossing_detection_medium', 
                'crossing_detection_large', 'edge_filtering']:
        t_with = with_numba[key]
        t_without = without_numba[key]
        if t_with and t_without:
            speedups.append(t_without / t_with)
    
    avg_speedup = np.mean(speedups)
    print(f"\n{'Average speedup:':<45} {avg_speedup:>8.2f}x")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if avg_speedup > 10:
        print("  Numba provides DRAMATIC performance improvements (>10x)")
        print("      Highly recommended for production use!")
    elif avg_speedup > 3:
        print("  Numba provides SIGNIFICANT performance improvements (3-10x)")
        print("     Recommended for all users working with meshes >500 triangles")
    elif avg_speedup > 1.5:
        print("  Numba provides MODERATE performance improvements (1.5-3x)")
        print("    Beneficial for users working with large meshes")
    else:
        print("  Numba provides minimal performance improvements (<1.5x)")
        print("    May not be worth the dependency for small meshes")
    
    print("="*80)
    
    # Restore environment
    os.environ.clear()
    os.environ.update(original_env)
