"""Benchmark : Parallel Processing

Compares sequential batch processing vs parallel processing.

Expected Results:
- single-threaded: ~2,367 ops/sec
- 4 workers: ~8,000-9,000 ops/sec (3-4x speedup)
- Combined speedup: ~25x vs baseline
- Target achieved: 16,000+ triangles/sec 
"""
import numpy as np
import time
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.batch_operations import BatchEditor
from sofia.core.parallel_operations import ParallelBatchEditor
from sofia.core.conformity import check_mesh_conformity


def create_grid_mesh(grid_size=20):
    """Create a regular grid mesh for testing."""
    # Create grid of points (2D for PatchBasedMeshEditor)
    x = np.linspace(0, 10, grid_size)
    y = np.linspace(0, 10, grid_size)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])  # 2D points
    
    # Create triangles
    triangles = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # Two triangles per grid cell
            v1 = i * grid_size + j
            v2 = i * grid_size + (j + 1)
            v3 = (i + 1) * grid_size + j
            v4 = (i + 1) * grid_size + (j + 1)
            
            triangles.append([v1, v2, v3])
            triangles.append([v2, v4, v3])
    
    triangles = np.array(triangles)
    return points, triangles


def select_random_edges(editor, n_edges):
    """Select random edges from the mesh."""
    all_edges = list(editor.edge_map.keys())
    if len(all_edges) < n_edges:
        n_edges = len(all_edges)
    
    indices = np.random.choice(len(all_edges), size=n_edges, replace=False)
    return [all_edges[i] for i in indices]


def benchmark_optim2_sequential(grid_size=20, n_splits=1500, batch_size=5000):
    """Benchmark Optim 2: Sequential batch processing."""
    print("=" * 70)
    print("OPTIM 2: Sequential Batch Processing")
    print("=" * 70)
    
    # Create mesh
    points, triangles = create_grid_mesh(grid_size)
    editor = PatchBasedMeshEditor(points, triangles)
    
    print(f"Initial mesh: {len(points)} vertices, {len(triangles)} triangles")
    
    # Select edges
    edges = select_random_edges(editor, n_splits)
    print(f"Selected {len(edges)} random edges to split")
    
    # Benchmark
    batch = BatchEditor(editor, batch_size=batch_size, validate_on_flush=False)
    
    start_time = time.time()
    
    for edge in edges:
        batch.split_edge(edge)
    
    batch.flush()
    
    elapsed = time.time() - start_time
    
    # Get stats
    stats = batch.get_stats()
    ops_per_sec = stats['total_operations'] / elapsed if elapsed > 0 else 0
    ms_per_op = (elapsed / stats['total_operations'] * 1000) if stats['total_operations'] > 0 else 0
    
    print(f"\nResults:")
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {ops_per_sec:.1f} ops/sec")
    print(f"  Per-operation: {ms_per_op:.3f} ms")
    
    # Validate
    print(f"\nValidating mesh conformity...")
    ok, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    
    if ok:
        print(f"  Conformity check PASSED")
    else:
        print(f"  Conformity check FAILED:")
        for msg in msgs[:5]:
            print(f"    - {msg}")
        if len(msgs) > 5:
            print(f"    ... and {len(msgs) - 5} more errors")
    
    print(f"\nFinal mesh: {len(editor.points)} vertices, {len(editor.triangles)} triangles")
    
    return {
        'optim': 'Optim 2 (Sequential)',
        'operations': stats['total_operations'],
        'time': elapsed,
        'ops_per_sec': ops_per_sec,
        'ms_per_op': ms_per_op,
        'conformity': ok,
        'vertices': len(editor.points),
        'triangles': len(editor.triangles),
    }


def benchmark_optim3_parallel(grid_size=20, n_splits=1500, n_workers=4, batch_size=5000):
    """Benchmark Optim 3: Parallel batch processing."""
    print("\n" + "=" * 70)
    print(f"OPTIM 3: Parallel Processing ({n_workers} workers)")
    print("=" * 70)
    
    # Create mesh
    points, triangles = create_grid_mesh(grid_size)
    editor = PatchBasedMeshEditor(points, triangles)
    
    print(f"Initial mesh: {len(points)} vertices, {len(triangles)} triangles")
    
    # Select edges
    edges = select_random_edges(editor, n_splits)
    print(f"Selected {len(edges)} random edges to split")
    
    # Benchmark
    parallel = ParallelBatchEditor(
        editor,
        n_workers=n_workers,
        batch_size=batch_size,
        partitioning='spatial'
    )
    
    start_time = time.time()
    
    success, failed = parallel.split_edges_parallel(edges, validate=False)
    
    elapsed = time.time() - start_time
    
    # Get stats
    stats = parallel.get_stats()
    total_ops = stats['total_operations']
    ops_per_sec = total_ops / elapsed if elapsed > 0 else 0
    ms_per_op = (elapsed / total_ops * 1000) if total_ops > 0 else 0
    
    print(f"\nResults:")
    print(f"  Total operations: {total_ops}")
    print(f"  Parallel operations: {stats['parallel_operations']} ({stats['parallel_fraction']*100:.1f}%)")
    print(f"  Sequential operations: {stats['sequential_operations']}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {ops_per_sec:.1f} ops/sec")
    print(f"  Per-operation: {ms_per_op:.3f} ms")
    
    # Validate
    print(f"\nValidating mesh conformity...")
    ok, msgs = check_mesh_conformity(editor.points, editor.triangles, allow_marked=True)
    
    if ok:
        print(f"  Conformity check PASSED")
    else:
        print(f"  Conformity check FAILED:")
        for msg in msgs[:5]:
            print(f"    - {msg}")
        if len(msgs) > 5:
            print(f"    ... and {len(msgs) - 5} more errors")
    
    print(f"\nFinal mesh: {len(editor.points)} vertices, {len(editor.triangles)} triangles")
    
    return {
        'optim': f'Optim 3 (Parallel {n_workers}w)',
        'operations': total_ops,
        'time': elapsed,
        'ops_per_sec': ops_per_sec,
        'ms_per_op': ms_per_op,
        'parallel_fraction': stats['parallel_fraction'],
        'n_workers': n_workers,
        'conformity': ok,
        'vertices': len(editor.points),
        'triangles': len(editor.triangles),
    }


def compare_opts(optim2_results, optim3_results):
    """Compare Optim 2 vs Optim 3 results."""
    print("\n" + "=" * 70)
    print("COMPARISON: Optim 2 vs Optim 3")
    print("=" * 70)
    
    print(f"\nThroughput:")
    print(f"  Optim 2 (sequential): {optim2_results['ops_per_sec']:.1f} ops/sec")
    print(f"  Optim 3 (parallel):   {optim3_results['ops_per_sec']:.1f} ops/sec")
    
    speedup = optim3_results['ops_per_sec'] / optim2_results['ops_per_sec']
    print(f"  Speedup: {speedup:.2f}x")
    
    if speedup > 1:
        print(f"  Optim 3 is {speedup:.2f}x FASTER")
    else:
        print(f"  Optim 3 is {1/speedup:.2f}x SLOWER (overhead too high)")
    
    print(f"\nPer-Operation Latency:")
    print(f"  Optim 2: {optim2_results['ms_per_op']:.3f} ms")
    print(f"  Optim 3: {optim3_results['ms_per_op']:.3f} ms")
    print(f"  Improvement: {(1 - optim3_results['ms_per_op']/optim2_results['ms_per_op'])*100:.1f}%")
    
    # Combined speedup calculation
    baseline_ops_per_sec = 343  # From Phase 1 benchmark
    phase1_speedup = 3.76
    phase2_speedup = optim2_results['ops_per_sec'] / (baseline_ops_per_sec * phase1_speedup)
    combined_speedup = optim3_results['ops_per_sec'] / baseline_ops_per_sec
    
    print(f"\nCombined Progress:")
    print(f"  Baseline (normal mode): {baseline_ops_per_sec} ops/sec")
    print(f"  Phase 1 (batch):        {baseline_ops_per_sec * phase1_speedup:.0f} ops/sec (3.76x)")
    print(f"  Phase 2 (JIT):          {optim2_results['ops_per_sec']:.0f} ops/sec ({phase1_speedup * phase2_speedup:.1f}x)")
    print(f"  Phase 3 (parallel):     {optim3_results['ops_per_sec']:.0f} ops/sec ({combined_speedup:.1f}x)")
    
    # Triangle generation rate (assuming 2 triangles per split)
    triangles_per_sec = optim3_results['ops_per_sec'] * 2
    print(f"\nTriangle Generation Rate:")
    print(f"  Optim 3: {triangles_per_sec:.0f} triangles/sec")
    print(f"  Per minute: {triangles_per_sec * 60:.0f} triangles")
    
    # Target analysis
    target_per_sec = 1_000_000 / 60  # 1M triangles per minute
    print(f"\nTarget Analysis (1M triangles in 60 seconds):")
    print(f"  Required: {target_per_sec:.0f} triangles/sec")
    print(f"  Achieved: {triangles_per_sec:.0f} triangles/sec")
    
    if triangles_per_sec >= target_per_sec:
        print(f"  TARGET EXCEEDED by {(triangles_per_sec/target_per_sec - 1)*100:.1f}% ✓✓✓")
    else:
        gap = target_per_sec / triangles_per_sec
        print(f"  Still need {gap:.2f}x more speedup")
    
    # Extrapolation to 1M triangles
    print(f"\nExtrapolation to 1M triangles:")
    time_optim2 = 1_000_000 / (optim2_results['ops_per_sec'] * 2)
    time_optim3 = 1_000_000 / (optim3_results['ops_per_sec'] * 2)
    time_target = 60
    
    print(f"  Optim 2: {time_optim2:.1f}s ({time_optim2/60:.1f} min)")
    print(f"  Optim 3: {time_optim3:.1f}s ({time_optim3/60:.1f} min)")
    print(f"  Target:  {time_target}s (1.0 min)")
    
    if time_optim3 <= time_target:
        print(f"  Optim 3 achieves target ({time_target - time_optim3:.1f}s faster)!")
    else:
        print(f"  Need {time_target / time_optim3:.2f}x more speedup")
    
    print(f"\nConformity:")
    print(f"  Optim 2: {'PASS' if optim2_results['conformity'] else '✗ FAIL'}")
    print(f"  Optim 3: {'PASS' if optim3_results['conformity'] else '✗ FAIL'}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Phase 3: Parallel Processing')
    parser.add_argument('--grid-size', type=int, default=20,
                       help='Grid size for test mesh (default: 20)')
    parser.add_argument('--n-splits', type=int, default=1500,
                       help='Number of edges to split (default: 1500)')
    parser.add_argument('--n-workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Batch size (default: 5000)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--phase2-only', action='store_true',
                       help='Run only Phase 2 (sequential)')
    parser.add_argument('--phase3-only', action='store_true',
                       help='Run only Phase 3 (parallel)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    results = {}
    
    # Run benchmarks
    if not args.phase3_only:
        optim2_results = benchmark_optim2_sequential(
            grid_size=args.grid_size,
            n_splits=args.n_splits,
            batch_size=args.batch_size
        )
        results['phase2'] = optim2_results
    
    if not args.phase2_only:
        optim3_results = benchmark_optim3_parallel(
            grid_size=args.grid_size,
            n_splits=args.n_splits,
            n_workers=args.n_workers,
            batch_size=args.batch_size
        )
        results['phase3'] = optim3_results
    
    # Compare if both ran
    if 'phase2' in results and 'phase3' in results:
        compare_opts(results['phase2'], results['phase3'])
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
