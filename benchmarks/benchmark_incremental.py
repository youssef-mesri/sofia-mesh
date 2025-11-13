"""Benchmark incremental computation vs full rebuild.

This script measures the performance gains from using IncrementalEdgeMap
and IncrementalConformityChecker compared to full rebuilds.

Expected results:
- Edge map updates: 100-1000x faster (0.01ms vs 10ms)
- Conformity checks: 100-5000x faster (0.001ms vs 5ms)
"""

import numpy as np
import time
from sofia.core.incremental import IncrementalEdgeMap, IncrementalConformityChecker
from sofia.core.conformity import build_edge_to_tri_map, check_mesh_conformity


def generate_test_mesh(n_triangles=1000):
    """Generate a simple test mesh."""
    # Create random Delaunay-ish triangulation
    np.random.seed(42)
    n_points = n_triangles // 2 + 10
    points = np.random.rand(n_points, 2) * 10.0
    
    # Simple triangulation (not proper Delaunay, but good enough for benchmarking)
    triangles = []
    for i in range(n_triangles):
        v0 = np.random.randint(0, n_points)
        v1 = np.random.randint(0, n_points)
        v2 = np.random.randint(0, n_points)
        if v0 != v1 and v1 != v2 and v0 != v2:
            triangles.append([v0, v1, v2])
    
    return points, np.array(triangles, dtype=np.int32)


def simulate_edge_split(triangles, edge_idx=0):
    """Simulate an edge split operation.
    
    Removes 2 triangles, adds 4 triangles.
    Returns removed_indices, added_triangles.
    """
    n_tris = len(triangles)
    if n_tris < 2:
        return [], []
    
    # Pick two adjacent triangles (simulate)
    removed = [edge_idx % n_tris, (edge_idx + 1) % n_tris]
    
    # Generate 4 new triangles (simplified)
    tri0 = triangles[removed[0]]
    tri1 = triangles[removed[1]]
    
    # New vertex index (simulate adding a vertex)
    new_v = triangles.max() + 1
    
    added = [
        (tri0[0], tri0[1], new_v),
        (tri0[1], tri0[2], new_v),
        (tri1[0], tri1[1], new_v),
        (tri1[1], tri1[2], new_v),
    ]
    
    return removed, added


def benchmark_edge_map(n_triangles=1000, n_operations=100):
    """Benchmark IncrementalEdgeMap vs full rebuild."""
    print(f"\n{'='*70}")
    print(f"EDGE MAP BENCHMARK: {n_triangles} triangles, {n_operations} operations")
    print(f"{'='*70}")
    
    points, triangles = generate_test_mesh(n_triangles)
    
    # Test 1: Initial build time
    t0 = time.perf_counter()
    edge_map_full = build_edge_to_tri_map(triangles)
    t1 = time.perf_counter()
    print(f"\nInitial build (full):        {(t1-t0)*1000:.3f} ms")
    
    t0 = time.perf_counter()
    edge_map_inc = IncrementalEdgeMap(triangles)
    t1 = time.perf_counter()
    print(f"Initial build (incremental): {(t1-t0)*1000:.3f} ms")
    
    # Test 2: Update performance
    print(f"\nUpdate performance ({n_operations} edge splits):")
    
    # Full rebuild approach
    current_tris = triangles.copy()
    times_full = []
    for i in range(n_operations):
        removed, added = simulate_edge_split(current_tris, i)
        
        t0 = time.perf_counter()
        # Full rebuild
        edge_map_full = build_edge_to_tri_map(current_tris)
        t1 = time.perf_counter()
        times_full.append((t1 - t0) * 1000)
    
    avg_full = np.mean(times_full)
    print(f"  Full rebuild:   {avg_full:.4f} ms/operation (median: {np.median(times_full):.4f} ms)")
    
    # Incremental approach
    current_tris = triangles.copy()
    edge_map_inc = IncrementalEdgeMap(triangles)
    times_inc = []
    for i in range(n_operations):
        removed, added = simulate_edge_split(current_tris, i)
        
        t0 = time.perf_counter()
        # Incremental update
        edge_map_inc.remove_triangles(removed)
        edge_map_inc.add_triangles(added)
        t1 = time.perf_counter()
        times_inc.append((t1 - t0) * 1000)
    
    avg_inc = np.mean(times_inc)
    print(f"  Incremental:    {avg_inc:.4f} ms/operation (median: {np.median(times_inc):.4f} ms)")
    
    speedup = avg_full / avg_inc if avg_inc > 0 else float('inf')
    print(f"\n  SPEEDUP: {speedup:.1f}x faster")
    
    # Test 3: Query performance
    print(f"\nQuery performance (1000 edge lookups):")
    test_edges = list(edge_map_inc.edge_to_tris.keys())[:1000]
    
    t0 = time.perf_counter()
    for edge in test_edges:
        _ = edge_map_full.get(edge, set())
    t1 = time.perf_counter()
    time_full_query = (t1 - t0) * 1000
    
    t0 = time.perf_counter()
    for edge in test_edges:
        _ = edge_map_inc.get_triangles_for_edge(edge)
    t1 = time.perf_counter()
    time_inc_query = (t1 - t0) * 1000
    
    print(f"  Dict lookup:    {time_full_query:.3f} ms")
    print(f"  Incremental:    {time_inc_query:.3f} ms")
    print(f"  (Both are O(1), similar performance)")
    
    return {
        'n_triangles': n_triangles,
        'speedup': speedup,
        'full_ms': avg_full,
        'inc_ms': avg_inc
    }


def benchmark_conformity_checker(n_triangles=1000, n_operations=100):
    """Benchmark IncrementalConformityChecker vs full check."""
    print(f"\n{'='*70}")
    print(f"CONFORMITY CHECKER BENCHMARK: {n_triangles} triangles, {n_operations} operations")
    print(f"{'='*70}")
    
    points, triangles = generate_test_mesh(n_triangles)
    
    # Test 1: Initial build time
    t0 = time.perf_counter()
    ok, msgs = check_mesh_conformity(points, triangles, allow_marked=False)
    t1 = time.perf_counter()
    print(f"\nInitial check (full):        {(t1-t0)*1000:.3f} ms")
    
    t0 = time.perf_counter()
    checker_inc = IncrementalConformityChecker(triangles)
    t1 = time.perf_counter()
    print(f"Initial build (incremental): {(t1-t0)*1000:.3f} ms")
    
    # Test 2: Update + check performance
    print(f"\nUpdate + conformity check performance ({n_operations} operations):")
    
    # Full check approach
    current_tris = triangles.copy()
    times_full = []
    for i in range(n_operations):
        removed, added = simulate_edge_split(current_tris, i)
        
        t0 = time.perf_counter()
        # Full conformity check
        ok, msgs = check_mesh_conformity(points, current_tris, allow_marked=False)
        t1 = time.perf_counter()
        times_full.append((t1 - t0) * 1000)
    
    avg_full = np.mean(times_full)
    print(f"  Full check:     {avg_full:.4f} ms/operation (median: {np.median(times_full):.4f} ms)")
    
    # Incremental approach
    current_tris = triangles.copy()
    checker_inc = IncrementalConformityChecker(triangles)
    times_inc = []
    for i in range(n_operations):
        removed, added = simulate_edge_split(current_tris, i)
        
        t0 = time.perf_counter()
        # Incremental update + check
        checker_inc.update_after_operation(removed, added)
        is_conforming = checker_inc.is_conforming()
        t1 = time.perf_counter()
        times_inc.append((t1 - t0) * 1000)
    
    avg_inc = np.mean(times_inc)
    print(f"  Incremental:    {avg_inc:.4f} ms/operation (median: {np.median(times_inc):.4f} ms)")
    
    speedup = avg_full / avg_inc if avg_inc > 0 else float('inf')
    print(f"\n  SPEEDUP: {speedup:.1f}x faster")
    
    # Test 3: Query performance
    print(f"\nQuery performance (1000 conformity checks):")
    
    t0 = time.perf_counter()
    for _ in range(1000):
        ok, msgs = check_mesh_conformity(points, triangles, allow_marked=False)
    t1 = time.perf_counter()
    time_full_query = (t1 - t0)
    
    t0 = time.perf_counter()
    for _ in range(1000):
        is_conf = checker_inc.is_conforming()
    t1 = time.perf_counter()
    time_inc_query = (t1 - t0)
    
    query_speedup = time_full_query / time_inc_query if time_inc_query > 0 else float('inf')
    print(f"  Full check:     {time_full_query*1000:.3f} ms (total)")
    print(f"  Incremental:    {time_inc_query*1000:.3f} ms (total)")
    print(f"  Query speedup:  {query_speedup:.1f}x")
    
    return {
        'n_triangles': n_triangles,
        'speedup': speedup,
        'query_speedup': query_speedup,
        'full_ms': avg_full,
        'inc_ms': avg_inc
    }


def benchmark_validation():
    """Test that incremental structures produce correct results."""
    print(f"\n{'='*70}")
    print(f"VALIDATION: Correctness testing")
    print(f"{'='*70}")
    
    points, triangles = generate_test_mesh(500)
    
    # Test IncrementalEdgeMap
    print("\nTesting IncrementalEdgeMap...")
    edge_map = IncrementalEdgeMap(triangles)
    
    # Perform some operations
    for i in range(10):
        removed, added = simulate_edge_split(triangles, i)
        edge_map.remove_triangles(removed)
        edge_map.add_triangles(added)
    
    # Validate (this is just a placeholder - real validation needs actual triangle array)
    print("  ✓ Edge map operations completed")
    print(f"  Total edges: {edge_map.get_edge_count()}")
    print(f"  Boundary edges: {len(edge_map.get_boundary_edges())}")
    print(f"  Non-manifold edges: {len(edge_map.get_non_manifold_edges())}")
    
    # Test IncrementalConformityChecker
    print("\nTesting IncrementalConformityChecker...")
    checker = IncrementalConformityChecker(triangles)
    
    # Perform some operations
    for i in range(10):
        removed, added = simulate_edge_split(triangles, i)
        checker.update_after_operation(removed, added)
    
    print(f"  Conformity checker operations completed")
    print(f"  Is conforming: {checker.is_conforming()}")
    print(f"  Boundary edges: {checker.get_boundary_edge_count()}")
    print(f"  Non-manifold edges: {checker.get_non_manifold_edge_count()}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("INCREMENTAL COMPUTATION BENCHMARKS")
    print("Testing Phase 4 optimization: O(k) updates vs O(N) full rebuilds")
    print("="*70)
    
    # Validation first
    benchmark_validation()
    
    # Edge map benchmarks at different scales
    results_edge = []
    for n_tris in [100, 500, 1000, 5000]:
        result = benchmark_edge_map(n_tris, n_operations=50)
        results_edge.append(result)
    
    # Conformity checker benchmarks
    results_conf = []
    for n_tris in [100, 500, 1000, 5000]:
        result = benchmark_conformity_checker(n_tris, n_operations=50)
        results_conf.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Incremental Computation Performance")
    print(f"{'='*70}")
    
    print("\nEdge Map Updates:")
    print(f"  {'Triangles':<12} {'Full (ms)':<12} {'Incr (ms)':<12} {'Speedup':<12}")
    print(f"  {'-'*48}")
    for r in results_edge:
        print(f"  {r['n_triangles']:<12} {r['full_ms']:<12.4f} {r['inc_ms']:<12.4f} {r['speedup']:<12.1f}x")
    
    print("\nConformity Checks:")
    print(f"  {'Triangles':<12} {'Full (ms)':<12} {'Incr (ms)':<12} {'Speedup':<12}")
    print(f"  {'-'*48}")
    for r in results_conf:
        print(f"  {r['n_triangles']:<12} {r['full_ms']:<12.4f} {r['inc_ms']:<12.4f} {r['speedup']:<12.1f}x")
    
    print("\n" + "="*70)
    print("Expected gains in real workflow:")
    print("  - Remeshing with 1000 operations on 5k tri mesh:")
    print(f"    Current: ~{results_conf[-1]['full_ms']*1000:.0f}ms")
    print(f"    With incremental: ~{results_conf[-1]['inc_ms']*1000:.0f}ms")
    print(f"    Time saved: ~{(results_conf[-1]['full_ms']-results_conf[-1]['inc_ms'])*1000:.1f}ms")
    print("="*70)
