#!/usr/bin/env python3
"""Benchmark to measure Numba JIT performance improvements."""

import time
import numpy as np
from sofia.core import geometry

def benchmark_triangles_signed_areas(n_triangles=10000):
    """Benchmark triangles_signed_areas function."""
    # Create random triangulation
    points = np.random.rand(n_triangles * 3, 2) * 100
    tris = np.arange(n_triangles * 3).reshape(n_triangles, 3)
    
    # Warm up (JIT compilation)
    _ = geometry.triangles_signed_areas(points, tris)
    
    # Benchmark
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        _ = geometry.triangles_signed_areas(points, tris)
    elapsed = time.time() - start
    
    return elapsed / n_iters

def benchmark_triangulation_area(n_triangles=1000):
    """Benchmark compute_triangulation_area function."""
    # Create random triangulation
    points = np.random.rand(n_triangles * 3, 2) * 100
    tris = np.arange(n_triangles * 3).reshape(n_triangles, 3)
    indices = np.arange(n_triangles)
    
    # Warm up
    _ = geometry.compute_triangulation_area(points, tris, indices)
    
    # Benchmark
    n_iters = 1000
    start = time.time()
    for _ in range(n_iters):
        _ = geometry.compute_triangulation_area(points, tris, indices)
    elapsed = time.time() - start
    
    return elapsed / n_iters

def benchmark_triangles_min_angles(n_triangles=10000):
    """Benchmark triangles_min_angles function."""
    # Create random triangulation
    points = np.random.rand(n_triangles * 3, 2) * 100
    tris = np.arange(n_triangles * 3).reshape(n_triangles, 3)
    
    # Warm up
    _ = geometry.triangles_min_angles(points, tris)
    
    # Benchmark
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        _ = geometry.triangles_min_angles(points, tris)
    elapsed = time.time() - start
    
    return elapsed / n_iters

if __name__ == "__main__":
    print("=" * 70)
    print("NUMBA JIT PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Numba enabled: {geometry.HAS_NUMBA}")
    print()
    
    print("Benchmarking triangles_signed_areas (10k triangles)...")
    t1 = benchmark_triangles_signed_areas()
    print(f"  Average time: {t1*1000:.3f} ms")
    print()
    
    print("Benchmarking triangles_min_angles (10k triangles)...")
    t2 = benchmark_triangles_min_angles()
    print(f"  Average time: {t2*1000:.3f} ms")
    print()
    
    print("Benchmarking compute_triangulation_area (1k triangles)...")
    t3 = benchmark_triangulation_area()
    print(f"  Average time: {t3*1000:.3f} ms")
    print()
    
    print("=" * 70)
    if geometry.HAS_NUMBA:
        print("Numba JIT acceleration is ACTIVE")
        print("  Expected speedup: 3-10x on large meshes")
    else:
        print("Numba JIT is NOT available - using NumPy fallback")
        print("  Install numba>=0.57 for acceleration")
    print("=" * 70)
