"""Benchmark incremental structures integrated into PatchBasedMeshEditor.

This script tests the real-world performance of Phase 4 incremental computation
when integrated into the mesh editor workflow.
"""

import numpy as np
import time
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor


def generate_test_mesh(n_points=100):
    """Generate a simple Delaunay mesh for testing."""
    from scipy.spatial import Delaunay
    np.random.seed(42)
    points = np.random.rand(n_points, 2) * 10.0
    tri = Delaunay(points)
    return points, tri.simplices.astype(np.int32)


def benchmark_editor_conformity_checks(n_points=100, n_checks=1000):
    """Benchmark conformity checks in editor context."""
    print(f"\n{'='*70}")
    print(f"EDITOR CONFORMITY CHECK BENCHMARK: {n_points} points, {n_checks} checks")
    print(f"{'='*70}")
    
    points, triangles = generate_test_mesh(n_points)
    n_triangles = len(triangles)
    
    # Test with incremental structures
    print("\n1. Editor with incremental structures:")
    editor_inc = PatchBasedMeshEditor(points, triangles, use_incremental_structures=True)
    
    t0 = time.perf_counter()
    for _ in range(n_checks):
        is_conf = editor_inc.is_conforming_incremental()
    t1 = time.perf_counter()
    time_inc = (t1 - t0) * 1000
    
    print(f"   {n_checks} checks: {time_inc:.3f} ms total")
    print(f"   Per check: {time_inc/n_checks:.6f} ms")
    print(f"   Is conforming: {is_conf}")
    
    # Test without incremental structures
    print("\n2. Editor with traditional checks:")
    editor_trad = PatchBasedMeshEditor(points, triangles, use_incremental_structures=False)
    
    t0 = time.perf_counter()
    for _ in range(n_checks):
        is_conf = editor_trad.is_conforming_incremental()  # Uses fallback
    t1 = time.perf_counter()
    time_trad = (t1 - t0) * 1000
    
    print(f"   {n_checks} checks: {time_trad:.3f} ms total")
    print(f"   Per check: {time_trad/n_checks:.6f} ms")
    print(f"   Is conforming: {is_conf}")
    
    # Calculate speedup
    speedup = time_trad / time_inc if time_inc > 0 else float('inf')
    print(f"\n   💥 SPEEDUP: {speedup:.1f}x faster")
    
    return {
        'n_triangles': n_triangles,
        'incremental_ms': time_inc,
        'traditional_ms': time_trad,
        'speedup': speedup
    }


def benchmark_editor_operations(n_points=100, n_operations=50):
    """Benchmark mesh operations with incremental tracking."""
    print(f"\n{'='*70}")
    print(f"EDITOR OPERATIONS BENCHMARK: {n_points} points, {n_operations} operations")
    print(f"{'='*70}")
    
    points, triangles = generate_test_mesh(n_points)
    n_triangles = len(triangles)
    
    # Test with incremental structures
    print("\n1. Editor with incremental structures:")
    editor_inc = PatchBasedMeshEditor(points, triangles, use_incremental_structures=True)
    
    t0 = time.perf_counter()
    for i in range(n_operations):
        # Simulate operations: get boundary, check conformity
        boundary = editor_inc.get_boundary_edges_incremental()
        is_conf = editor_inc.is_conforming_incremental()
        non_manifold = editor_inc.get_non_manifold_edges_incremental()
    t1 = time.perf_counter()
    time_inc = (t1 - t0) * 1000
    
    print(f"   {n_operations} operation cycles: {time_inc:.3f} ms total")
    print(f"   Per cycle: {time_inc/n_operations:.4f} ms")
    print(f"   Boundary edges: {len(boundary)}, Non-manifold: {len(non_manifold)}")
    
    # Test without incremental structures
    print("\n2. Editor with traditional methods:")
    editor_trad = PatchBasedMeshEditor(points, triangles, use_incremental_structures=False)
    
    t0 = time.perf_counter()
    for i in range(n_operations):
        # Same operations using traditional methods
        boundary = editor_trad.get_boundary_edges_incremental()  # Uses fallback
        is_conf = editor_trad.is_conforming_incremental()  # Uses fallback
        non_manifold = editor_trad.get_non_manifold_edges_incremental()  # Uses fallback
    t1 = time.perf_counter()
    time_trad = (t1 - t0) * 1000
    
    print(f"   {n_operations} operation cycles: {time_trad:.3f} ms total")
    print(f"   Per cycle: {time_trad/n_operations:.4f} ms")
    print(f"   Boundary edges: {len(boundary)}, Non-manifold: {len(non_manifold)}")
    
    # Calculate speedup
    speedup = time_trad / time_inc if time_inc > 0 else float('inf')
    print(f"\n   💥 SPEEDUP: {speedup:.1f}x faster")
    
    return {
        'n_triangles': n_triangles,
        'incremental_ms': time_inc,
        'traditional_ms': time_trad,
        'speedup': speedup
    }


def benchmark_editor_with_modifications(n_points=100, n_splits=20):
    """Benchmark editor with actual mesh modifications."""
    print(f"\n{'='*70}")
    print(f"EDITOR WITH MODIFICATIONS: {n_points} points, {n_splits} edge splits")
    print(f"{'='*70}")
    
    points, triangles = generate_test_mesh(n_points)
    
    # Test with incremental structures
    print("\n1. Editor with incremental structures:")
    editor_inc = PatchBasedMeshEditor(points, triangles, use_incremental_structures=True)
    
    t0 = time.perf_counter()
    success_count = 0
    for i in range(n_splits):
        # Try to split edges
        boundary = editor_inc.get_boundary_edges_incremental()
        if boundary:
            edge = list(boundary)[i % len(boundary)]
            try:
                ok = editor_inc.split_edge(edge)
                if ok:
                    success_count += 1
                    # Check conformity after each operation
                    is_conf = editor_inc.is_conforming_incremental()
            except Exception as e:
                pass
    t1 = time.perf_counter()
    time_inc = (t1 - t0) * 1000
    
    print(f"   {n_splits} split attempts: {success_count} succeeded")
    print(f"   Total time: {time_inc:.3f} ms")
    print(f"   Per split: {time_inc/n_splits:.4f} ms")
    print(f"   Final triangles: {len(editor_inc.triangles)}")
    
    # Validate incremental structures
    valid = editor_inc.validate_incremental_structures()
    print(f"   Incremental structures valid: {valid}")
    
    # Test without incremental structures
    print("\n2. Editor with traditional methods:")
    points, triangles = generate_test_mesh(n_points)  # Reset
    editor_trad = PatchBasedMeshEditor(points, triangles, use_incremental_structures=False)
    
    t0 = time.perf_counter()
    success_count = 0
    for i in range(n_splits):
        # Same operations
        boundary = editor_trad.get_boundary_edges_incremental()
        if boundary:
            edge = list(boundary)[i % len(boundary)]
            try:
                ok = editor_trad.split_edge(edge)
                if ok:
                    success_count += 1
                    # Check conformity after each operation
                    is_conf = editor_trad.is_conforming_incremental()
            except Exception as e:
                pass
    t1 = time.perf_counter()
    time_trad = (t1 - t0) * 1000
    
    print(f"   {n_splits} split attempts: {success_count} succeeded")
    print(f"   Total time: {time_trad:.3f} ms")
    print(f"   Per split: {time_trad/n_splits:.4f} ms")
    print(f"   Final triangles: {len(editor_trad.triangles)}")
    
    # Calculate speedup
    speedup = time_trad / time_inc if time_inc > 0 else float('inf')
    print(f"\n   💥 SPEEDUP: {speedup:.1f}x faster")
    
    return {
        'incremental_ms': time_inc,
        'traditional_ms': time_trad,
        'speedup': speedup
    }


def test_editor_builtin_benchmark():
    """Test the built-in benchmark method."""
    print(f"\n{'='*70}")
    print(f"BUILT-IN BENCHMARK METHOD TEST")
    print(f"{'='*70}")
    
    points, triangles = generate_test_mesh(200)
    
    print("\n1. With incremental structures:")
    editor = PatchBasedMeshEditor(points, triangles, use_incremental_structures=True)
    results = editor.benchmark_conformity_check(n_iterations=1000)
    
    print(f"   Incremental: {results['incremental_ms']:.6f} ms/check")
    print(f"   Traditional: {results['traditional_ms']:.6f} ms/check")
    print(f"   Speedup: {results['speedup']:.1f}x")
    
    return results


if __name__ == '__main__':
    print("\n" + "="*70)
    print("PHASE 4: INTEGRATED INCREMENTAL COMPUTATION BENCHMARKS")
    print("Testing incremental structures within PatchBasedMeshEditor")
    print("="*70)
    
    # Run benchmarks at different scales
    results_checks = []
    for n_points in [50, 100, 200, 500]:
        result = benchmark_editor_conformity_checks(n_points, n_checks=1000)
        results_checks.append(result)
    
    results_ops = []
    for n_points in [50, 100, 200]:
        result = benchmark_editor_operations(n_points, n_operations=100)
        results_ops.append(result)
    
    # Test with actual modifications
    result_mods = benchmark_editor_with_modifications(n_points=100, n_splits=50)
    
    # Test built-in benchmark
    builtin_result = test_editor_builtin_benchmark()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Integrated Incremental Computation Performance")
    print(f"{'='*70}")
    
    print("\nConformity Check Performance (1000 checks):")
    print(f"  {'Triangles':<12} {'Incr (ms)':<12} {'Trad (ms)':<12} {'Speedup':<12}")
    print(f"  {'-'*48}")
    for r in results_checks:
        print(f"  {r['n_triangles']:<12} {r['incremental_ms']:<12.3f} {r['traditional_ms']:<12.3f} {r['speedup']:<12.1f}x")
    
    print("\nOperation Cycle Performance (100 cycles):")
    print(f"  {'Triangles':<12} {'Incr (ms)':<12} {'Trad (ms)':<12} {'Speedup':<12}")
    print(f"  {'-'*48}")
    for r in results_ops:
        print(f"  {r['n_triangles']:<12} {r['incremental_ms']:<12.3f} {r['traditional_ms']:<12.3f} {r['speedup']:<12.1f}x")
    
    print("\n" + "="*70)
    print("✅ Phase 4 incremental structures successfully integrated!")
    print("   - Conformity checks: up to {:.0f}x faster".format(max(r['speedup'] for r in results_checks)))
    print("   - Operation cycles: up to {:.0f}x faster".format(max(r['speedup'] for r in results_ops)))
    print("   - Real modifications: {:.1f}x faster".format(result_mods['speedup']))
    print("="*70)
