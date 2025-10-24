#!/usr/bin/env python3
"""
Direct comparison: Numba-accelerated vs NumPy-only implementations.

This script directly tests the Numba functions vs their fallback implementations.
"""

import time
import numpy as np
from sofia.core import geometry, conformity

def benchmark_geometry_direct():
    """Directly compare Numba vs NumPy implementations in geometry.py."""
    print("\n" + "="*80)
    print("GEOMETRY MODULE COMPARISON")
    print("="*80)
    
    n_tris = 10000
    points = np.random.rand(n_tris * 3, 2).astype(np.float64)
    tris = np.arange(n_tris * 3).reshape(n_tris, 3).astype(np.int32)
    
    results = []
    
    # 1. triangles_signed_areas
    print("\n1. triangles_signed_areas (10k triangles)")
    print("   " + "-"*70)
    
    # Test if Numba version exists and is being used
    if hasattr(geometry, '_triangles_signed_areas_numba') and geometry.HAS_NUMBA:
        # Warm up Numba
        _ = geometry._triangles_signed_areas_numba(points, tris)
        
        # Benchmark Numba version
        n_iters = 100
        start = time.time()
        for _ in range(n_iters):
            _ = geometry._triangles_signed_areas_numba(points, tris)
        time_numba = (time.time() - start) / n_iters
        print(f"   Numba version:     {time_numba*1000:>10.3f} ms")
    else:
        time_numba = None
        print(f"   Numba version:     NOT AVAILABLE")
    
    # Benchmark NumPy fallback (direct computation)
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        p0 = points[tris[:, 0]]
        p1 = points[tris[:, 1]]
        p2 = points[tris[:, 2]]
        _ = 0.5 * np.cross(p1 - p0, p2 - p0)
    time_numpy = (time.time() - start) / n_iters
    print(f"   NumPy version:     {time_numpy*1000:>10.3f} ms")
    
    if time_numba:
        speedup = time_numpy / time_numba
        print(f"   Speedup:           {speedup:>10.2f}x")
        results.append(('triangles_signed_areas', speedup))
    
    # 2. triangles_min_angles
    print("\n2. triangles_min_angles (10k triangles)")
    print("   " + "-"*70)
    
    if hasattr(geometry, '_triangles_min_angles_numba') and geometry.HAS_NUMBA:
        # Warm up Numba
        _ = geometry._triangles_min_angles_numba(points, tris)
        
        # Benchmark Numba version
        n_iters = 100
        start = time.time()
        for _ in range(n_iters):
            _ = geometry._triangles_min_angles_numba(points, tris)
        time_numba = (time.time() - start) / n_iters
        print(f"   Numba version:     {time_numba*1000:>10.3f} ms")
    else:
        time_numba = None
        print(f"   Numba version:     NOT AVAILABLE")
    
    # Benchmark NumPy fallback (direct computation)
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        p0 = points[tris[:, 0]]
        p1 = points[tris[:, 1]]
        p2 = points[tris[:, 2]]
        a = np.linalg.norm(p1 - p2, axis=1)
        b = np.linalg.norm(p0 - p2, axis=1)
        c = np.linalg.norm(p0 - p1, axis=1)
        cosA = (b*b + c*c - a*a) / (2*b*c + 1e-20)
        cosB = (a*a + c*c - b*b) / (2*a*c + 1e-20)
        cosC = (a*a + b*b - c*c) / (2*a*b + 1e-20)
        angA = np.degrees(np.arccos(np.clip(cosA, -1.0, 1.0)))
        angB = np.degrees(np.arccos(np.clip(cosB, -1.0, 1.0)))
        angC = np.degrees(np.arccos(np.clip(cosC, -1.0, 1.0)))
        _ = np.minimum(angA, np.minimum(angB, angC))
    time_numpy = (time.time() - start) / n_iters
    print(f"   NumPy version:     {time_numpy*1000:>10.3f} ms")
    
    if time_numba:
        speedup = time_numpy / time_numba
        print(f"   Speedup:           {speedup:>10.2f}x")
        results.append(('triangles_min_angles', speedup))
    
    return results


def benchmark_conformity_direct():
    """Directly compare Numba vs Python implementations in conformity.py."""
    print("\n" + "="*80)
    print("CONFORMITY MODULE COMPARISON")
    print("="*80)
    
    results = []
    
    # 1. Grid cell assignment
    print("\n1. Grid cell assignment (5k edges)")
    print("   " + "-"*70)
    
    n_edges = 5000
    minx = np.random.rand(n_edges) * 100
    maxx = minx + np.random.rand(n_edges) * 10
    miny = np.random.rand(n_edges) * 100
    maxy = miny + np.random.rand(n_edges) * 10
    gx0, gy0 = 0.0, 0.0
    cell = 5.0
    N = 70
    
    if hasattr(conformity, '_assign_edges_to_cells_numba') and conformity.HAS_NUMBA:
        # Warm up Numba
        _ = conformity._assign_edges_to_cells_numba(minx, maxx, miny, maxy, gx0, gy0, cell, N)
        
        # Benchmark Numba version
        n_iters = 50
        start = time.time()
        for _ in range(n_iters):
            _ = conformity._assign_edges_to_cells_numba(minx, maxx, miny, maxy, gx0, gy0, cell, N)
        time_numba = (time.time() - start) / n_iters
        print(f"   Numba version:     {time_numba*1000:>10.3f} ms")
    else:
        time_numba = None
        print(f"   Numba version:     NOT AVAILABLE")
    
    # Benchmark Python fallback
    n_iters = 50
    start = time.time()
    for _ in range(n_iters):
        cells = {}
        E = len(minx)
        for idx in range(E):
            i0 = int((minx[idx] - gx0) / cell)
            i1 = int((maxx[idx] - gx0) / cell)
            j0 = int((miny[idx] - gy0) / cell)
            j1 = int((maxy[idx] - gy0) / cell)
            if i1 < i0: i0, i1 = i1, i0
            if j1 < j0: j0, j1 = j1, j0
            for ii in range(max(0, i0), min(N, i1 + 1)):
                for jj in range(max(0, j0), min(N, j1 + 1)):
                    cells.setdefault((ii, jj), []).append(idx)
    time_python = (time.time() - start) / n_iters
    print(f"   Python version:    {time_python*1000:>10.3f} ms")
    
    if time_numba:
        speedup = time_python / time_numba
        print(f"   Speedup:           {speedup:>10.2f}x")
        results.append(('grid_cell_assignment', speedup))
    
    # 2. Bbox overlap batch test
    print("\n2. Bbox overlap batch test (10k pairs)")
    print("   " + "-"*70)
    
    n_pairs = 10000
    minx1 = np.random.rand(n_pairs) * 100
    maxx1 = minx1 + np.random.rand(n_pairs) * 10
    miny1 = np.random.rand(n_pairs) * 100
    maxy1 = miny1 + np.random.rand(n_pairs) * 10
    minx2 = np.random.rand(n_pairs) * 100
    maxx2 = minx2 + np.random.rand(n_pairs) * 10
    miny2 = np.random.rand(n_pairs) * 100
    maxy2 = miny2 + np.random.rand(n_pairs) * 10
    
    if hasattr(conformity, '_bbox_overlap_batch_numba') and conformity.HAS_NUMBA:
        # Warm up Numba
        _ = conformity._bbox_overlap_batch_numba(minx1, maxx1, miny1, maxy1, minx2, maxx2, miny2, maxy2)
        
        # Benchmark Numba version
        n_iters = 100
        start = time.time()
        for _ in range(n_iters):
            _ = conformity._bbox_overlap_batch_numba(minx1, maxx1, miny1, maxy1, minx2, maxx2, miny2, maxy2)
        time_numba = (time.time() - start) / n_iters
        print(f"   Numba version:     {time_numba*1000:>10.3f} ms")
    else:
        time_numba = None
        print(f"   Numba version:     NOT AVAILABLE")
    
    # Benchmark NumPy fallback
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        _ = ~((maxx1 < minx2) | (maxx2 < minx1) | (maxy1 < miny2) | (maxy2 < miny1))
    time_numpy = (time.time() - start) / n_iters
    print(f"   NumPy version:     {time_numpy*1000:>10.3f} ms")
    
    if time_numba:
        speedup = time_numpy / time_numba
        print(f"   Speedup:           {speedup:>10.2f}x")
        results.append(('bbox_overlap_batch', speedup))
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("DIRECT NUMBA VS NUMPY COMPARISON")
    print("="*80)
    print(f"\nNumba available: {geometry.HAS_NUMBA}")
    print("Testing individual optimized functions against their fallback implementations\n")
    
    if not geometry.HAS_NUMBA:
        print("ERROR: Numba is not available. Install numba>=0.57 to run this benchmark.")
        exit(1)
    
    # Run benchmarks
    geo_results = benchmark_geometry_direct()
    conf_results = benchmark_conformity_direct()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_speedups = [s for _, s in geo_results + conf_results]
    
    print(f"\nGeometry module optimizations:")
    for name, speedup in geo_results:
        print(f"  {name:<35} {speedup:>6.2f}x speedup")
    
    print(f"\nConformity module optimizations:")
    for name, speedup in conf_results:
        print(f"  {name:<35} {speedup:>6.2f}x speedup")
    
    if all_speedups:
        avg_speedup = np.mean(all_speedups)
        print(f"\n{'Average speedup across all functions:':<40} {avg_speedup:>6.2f}x")
        print(f"{'Minimum speedup:':<40} {min(all_speedups):>6.2f}x")
        print(f"{'Maximum speedup:':<40} {max(all_speedups):>6.2f}x")
    
    print("="*80)
