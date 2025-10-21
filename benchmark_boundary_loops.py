"""Benchmark diagnostics.py boundary loop extraction performance."""
import numpy as np
import time
from sofia.core.diagnostics import (
    extract_boundary_loops,
    count_boundary_loops,
    _vectorized_boundary_edges,
    compact_copy,
    compact_from_arrays,
    find_inverted_triangles,
)


def create_test_mesh(n_tris=1000, n_boundary_loops=1):
    """Create a simple test mesh with boundary loops."""
    # Simple grid-like mesh
    side = int(np.sqrt(n_tris) * 0.5) + 1
    x = np.linspace(0, 1, side)
    y = np.linspace(0, 1, side)
    xv, yv = np.meshgrid(x, y)
    
    points = np.column_stack([xv.flatten(), yv.flatten()])
    
    # Create triangles
    tris = []
    for i in range(side - 1):
        for j in range(side - 1):
            v0 = i * side + j
            v1 = i * side + (j + 1)
            v2 = (i + 1) * side + j
            v3 = (i + 1) * side + (j + 1)
            tris.append([v0, v1, v2])
            tris.append([v1, v3, v2])
    
    tris = np.array(tris[:n_tris], dtype=np.int32)
    return points, tris


def create_mesh_with_holes(n_tris=2000):
    """Create a mesh with internal holes (multiple boundary loops)."""
    points, tris = create_test_mesh(n_tris)
    
    # Remove some triangles to create internal holes
    center_x, center_y = 0.5, 0.5
    tri_centers = np.mean(points[tris], axis=1)
    
    # Create 2-3 circular holes
    holes = []
    for cx, cy in [(0.3, 0.3), (0.7, 0.7), (0.5, 0.5)]:
        dist = np.sqrt((tri_centers[:, 0] - cx)**2 + (tri_centers[:, 1] - cy)**2)
        holes.append(dist < 0.1)
    
    hole_mask = np.any(holes, axis=0)
    tris = tris[~hole_mask]
    
    return points, tris


def benchmark_function(func, *args, n_runs=100):
    """Benchmark a function with multiple runs."""
    # Warmup
    for _ in range(3):
        func(*args)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.median(times) * 1000, np.mean(times) * 1000, np.std(times) * 1000, result


def main():
    print("=" * 80)
    print("BOUNDARY LOOP EXTRACTION BENCHMARK")
    print("=" * 80)
    
    # Test different mesh sizes
    mesh_sizes = [100, 500, 1000, 2000, 5000]
    
    print("\n1. Vectorized boundary edge detection")
    print("-" * 80)
    for n_tris in mesh_sizes:
        points, tris = create_test_mesh(n_tris)
        median_ms, mean_ms, std_ms, edges = benchmark_function(_vectorized_boundary_edges, tris, n_runs=100)
        print(f"  {n_tris:5d} tris: {median_ms:8.3f} ms  (mean: {mean_ms:6.3f} ± {std_ms:5.3f}, {len(edges)} boundary edges)")
    
    print("\n2. Extract boundary loops (single loop)")
    print("-" * 80)
    for n_tris in mesh_sizes:
        points, tris = create_test_mesh(n_tris)
        median_ms, mean_ms, std_ms, loops = benchmark_function(extract_boundary_loops, points, tris, n_runs=50)
        n_loops = len(loops)
        total_verts = sum(len(loop) for loop in loops)
        print(f"  {n_tris:5d} tris: {median_ms:8.3f} ms  (mean: {mean_ms:6.3f} ± {std_ms:5.3f}, {n_loops} loops, {total_verts} verts)")
    
    print("\n3. Extract boundary loops (multiple holes)")
    print("-" * 80)
    for n_tris in mesh_sizes:
        points, tris = create_mesh_with_holes(n_tris)
        median_ms, mean_ms, std_ms, loops = benchmark_function(extract_boundary_loops, points, tris, n_runs=50)
        n_loops = len(loops)
        total_verts = sum(len(loop) for loop in loops)
        print(f"  {n_tris:5d} tris: {median_ms:8.3f} ms  (mean: {mean_ms:6.3f} ± {std_ms:5.3f}, {n_loops} loops, {total_verts} verts)")
    
    print("\n4. Count boundary loops")
    print("-" * 80)
    for n_tris in mesh_sizes:
        points, tris = create_mesh_with_holes(n_tris)
        median_ms, mean_ms, std_ms, count = benchmark_function(count_boundary_loops, points, tris, n_runs=50)
        print(f"  {n_tris:5d} tris: {median_ms:8.3f} ms  (mean: {mean_ms:6.3f} ± {std_ms:5.3f}, {count} loops)")
    
    print("\n5. Compact from arrays")
    print("-" * 80)
    for n_tris in mesh_sizes:
        points, tris = create_test_mesh(n_tris)
        # Add some tombstones
        mask = np.random.random(len(tris)) > 0.1
        tris_with_tombstones = tris.copy()
        tris_with_tombstones[~mask] = -1
        
        median_ms, mean_ms, std_ms, result = benchmark_function(compact_from_arrays, points, tris_with_tombstones, n_runs=100)
        new_pts, new_tris, mapping, active_idx = result
        print(f"  {n_tris:5d} tris: {median_ms:8.3f} ms  (mean: {mean_ms:6.3f} ± {std_ms:5.3f}, {len(new_tris)} active)")
    
    print("\n6. Find inverted triangles")
    print("-" * 80)
    for n_tris in mesh_sizes:
        points, tris = create_test_mesh(n_tris)
        # Invert a few triangles
        tris_copy = tris.copy()
        n_invert = max(1, n_tris // 100)
        indices = np.random.choice(len(tris_copy), n_invert, replace=False)
        # Swap columns 0 and 1 for each selected triangle
        for idx in indices:
            tris_copy[idx, [0, 1]] = tris_copy[idx, [1, 0]]
        
        median_ms, mean_ms, std_ms, inverted = benchmark_function(find_inverted_triangles, points, tris_copy, n_runs=100)
        print(f"  {n_tris:5d} tris: {median_ms:8.3f} ms  (mean: {mean_ms:6.3f} ± {std_ms:5.3f}, {len(inverted)} inverted)")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Potential optimization targets")
    print("=" * 80)
    
    # Test on a realistic-sized mesh
    n_test = 2000
    points, tris = create_mesh_with_holes(n_test)
    
    print(f"\nFor a {n_test}-triangle mesh with holes:")
    
    median_ms, _, _, edges = benchmark_function(_vectorized_boundary_edges, tris, n_runs=100)
    print(f"  _vectorized_boundary_edges:  {median_ms:6.3f} ms")
    
    median_ms, _, _, loops = benchmark_function(extract_boundary_loops, points, tris, n_runs=50)
    print(f"  extract_boundary_loops:       {median_ms:6.3f} ms  ← Main target")
    
    median_ms, _, _, count = benchmark_function(count_boundary_loops, points, tris, n_runs=50)
    print(f"  count_boundary_loops:         {median_ms:6.3f} ms")
    
    print("\nThe bottleneck is in the loop ordering algorithm (greedy neighbor walking).")
    print("This involves dict lookups, set operations, and Python loops.")
    print("Numba could help with the graph traversal and ordering logic.")


if __name__ == "__main__":
    main()
