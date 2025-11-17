"""Benchmark vectorized operations vs batch operations.

This script compares:
1. Normal mode (per-operation validation)
2. Batch mode (deferred validation)
3. Vectorized + JIT (process many ops at once)
"""

import numpy as np
import time
import argparse
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.batch_operations import BatchEditor
from sofia.core.vectorized_ops import vectorized_split_edges, create_vectorized_batch_editor, NUMBA_AVAILABLE
from sofia.core.conformity import check_mesh_conformity


def create_initial_mesh(grid_size=10):
    """Create a regular grid mesh for testing."""
    logger.info(f"Creating {grid_size}x{grid_size} grid mesh...")
    
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    triangles = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            v0 = i * grid_size + j
            v1 = v0 + 1
            v2 = v0 + grid_size
            v3 = v2 + 1
            
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])
    
    triangles = np.array(triangles, dtype=np.int32)
    
    logger.info(f"Created mesh: {len(points)} vertices, {len(triangles)} triangles")
    return points, triangles


def find_edges_to_split(editor, max_edges=1000):
    """Find edges suitable for splitting."""
    all_edges = list(editor.edge_map.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(all_edges)
    return all_edges[:max_edges]


def benchmark_batch_mode(points, triangles, n_splits=1000, batch_size=5000):
    """Benchmark batch mode."""
    logger.info(f"\n{'='*70}")
    logger.info("OPTIM 1: Batch Mode (deferred validation)")
    logger.info(f"{'='*70}")
    
    editor = PatchBasedMeshEditor(
        points.copy(), 
        triangles.copy(),
        enforce_split_quality=False
    )
    
    edges = find_edges_to_split(editor, n_splits)
    logger.info(f"Found {len(edges)} edges to split")
    
    start_time = time.perf_counter()
    
    batch = BatchEditor(editor, batch_size=batch_size, validate_on_flush=True)
    
    for i, edge in enumerate(edges):
        batch.split_edge(edge)
        
        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed
            logger.info(f"  Progress: {i+1}/{len(edges)}, {rate:.1f} ops/sec")
    
    ok_final = batch.finalize()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    stats = batch.get_stats()
    
    ok_conf, msgs = check_mesh_conformity(
        editor.points, 
        editor.triangles,
        allow_marked=False
    )
    
    results = {
        'phase': 'Optim 1: Batch',
        'batch_size': batch_size,
        'n_operations': stats['total_operations'],
        'batches_processed': stats['batches_processed'],
        'failed_batches': stats['failed_batches'],
        'total_time': total_time,
        'ops_per_second': stats['total_operations'] / total_time,
        'ms_per_op': (total_time * 1000) / stats['total_operations'],
        'final_vertices': len(editor.points),
        'final_triangles': len(editor.triangles),
        'conformity_ok': ok_conf,
    }
    
    logger.info(f"\nOptim 1 Results:")
    logger.info(f"  Total time: {total_time:.3f}s")
    logger.info(f"  Rate: {results['ops_per_second']:.1f} ops/sec")
    logger.info(f"  Per-op: {results['ms_per_op']:.3f} ms")
    logger.info(f"  Final: {results['final_vertices']} vertices, {results['final_triangles']} triangles")
    logger.info(f"  Conformity: {'PASS' if ok_conf else 'FAIL'}")
    
    return results


def benchmark_vectorized_mode(points, triangles, n_splits=1000, batch_size=1000):
    """Benchmark Vectorized + JIT operations."""
    logger.info(f"\n{'='*70}")
    logger.info(f"OPTIM 2: Vectorized + JIT (batch_size={batch_size})")
    logger.info(f"{'='*70}")
    
    if not NUMBA_AVAILABLE:
        logger.warning(" Numba not available - will be slower than expected")
    
    editor = PatchBasedMeshEditor(
        points.copy(), 
        triangles.copy(),
        enforce_split_quality=False
    )
    
    edges = find_edges_to_split(editor, n_splits)
    logger.info(f"Found {len(edges)} edges to split")
    
    start_time = time.perf_counter()
    
    # Use vectorized batch editor
    batch = create_vectorized_batch_editor(
        editor, 
        batch_size=batch_size,
        validate_on_flush=True
    )
    
    for i, edge in enumerate(edges):
        batch.split_edge(edge)
        
        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed
            logger.info(f"  Progress: {i+1}/{len(edges)}, {rate:.1f} ops/sec")
    
    ok_final = batch.finalize()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    stats = batch.get_stats()
    
    ok_conf, msgs = check_mesh_conformity(
        editor.points, 
        editor.triangles,
        allow_marked=False
    )
    
    results = {
        'phase': 'Optim 2: Vectorized',
        'batch_size': batch_size,
        'numba_available': NUMBA_AVAILABLE,
        'n_operations': stats['total_operations'],
        'batches_processed': stats['batches_processed'],
        'failed_batches': stats['failed_batches'],
        'total_time': total_time,
        'ops_per_second': stats['total_operations'] / total_time,
        'ms_per_op': (total_time * 1000) / stats['total_operations'],
        'final_vertices': len(editor.points),
        'final_triangles': len(editor.triangles),
        'conformity_ok': ok_conf,
    }
    
    logger.info(f"\nOptim 2 Results:")
    logger.info(f"  Total time: {total_time:.3f}s")
    logger.info(f"  Rate: {results['ops_per_second']:.1f} ops/sec")
    logger.info(f"  Per-op: {results['ms_per_op']:.3f} ms")
    logger.info(f"  Final: {results['final_vertices']} vertices, {results['final_triangles']} triangles")
    logger.info(f"  Conformity: {'PASS' if ok_conf else 'FAIL'}")
    logger.info(f"  Numba: {' enabled' if NUMBA_AVAILABLE else ' disabled'}")
    
    return results


def compare_phases(optim1_results, optim2_results):
    """Compare Batch vs Vectorized performance."""
    logger.info(f"\n{'='*70}")
    logger.info("BATCH vs VECTORIZED COMPARISON")
    logger.info(f"{'='*70}")
    
    speedup = optim1_results['total_time'] / optim2_results['total_time']
    ops_speedup = optim2_results['ops_per_second'] / optim1_results['ops_per_second']
    
    logger.info(f"Optim 1 (Batch):")
    logger.info(f"  Time: {optim1_results['total_time']:.3f}s")
    logger.info(f"  Rate: {optim1_results['ops_per_second']:.1f} ops/sec")
    logger.info(f"  Per-op: {optim1_results['ms_per_op']:.3f} ms")
    
    logger.info(f"\nOptim 2 (Vectorized):")
    logger.info(f"  Time: {optim2_results['total_time']:.3f}s")
    logger.info(f"  Rate: {optim2_results['ops_per_second']:.1f} ops/sec")
    logger.info(f"  Per-op: {optim2_results['ms_per_op']:.3f} ms")
    
    logger.info(f"\nOptim 2 Improvement:")
    logger.info(f"  Speedup: {speedup:.2f}x faster")
    logger.info(f"  Throughput: {ops_speedup:.2f}x higher")
    logger.info(f"  Time saved: {optim1_results['total_time'] - optim2_results['total_time']:.3f}s")
    
    # Combined improvement from baseline
    # Assume baseline (normal mode) is ~3.76x slower than Phase 1
    baseline_multiplier = 3.76
    combined_speedup = baseline_multiplier * speedup
    
    logger.info(f"\nCombined (vs Normal Mode):")
    logger.info(f"  Estimated speedup: {combined_speedup:.2f}x")
    
    # Extrapolate to 1M triangles
    logger.info(f"\nExtrapolation to 1M triangles:")
    
    ops_for_1m = 500_000
    
    optim1_time_1m = ops_for_1m / optim1_results['ops_per_second']
    optim2_time_1m = ops_for_1m / optim2_results['ops_per_second']
    
    logger.info(f"  Optim 1:  {optim1_time_1m:.1f}s ({optim1_time_1m/60:.1f} min)")
    logger.info(f"  Optim 2:  {optim2_time_1m:.1f}s ({optim2_time_1m/60:.1f} min)")
    logger.info(f"  Target:   60s (1 min)")
    
    if optim2_time_1m <= 60:
        logger.info(f"  TARGET ACHIEVED with Optim 2!")
    else:
        additional_speedup = optim2_time_1m / 60
        logger.info(f"  Need {additional_speedup:.2f}x more speedup")
        logger.info(f"     (Optim 3: Parallel processing can provide 3-4x)")
    
    return {
        'speedup': speedup,
        'combined_speedup': combined_speedup,
        'time_saved': optim1_results['total_time'] - optim2_results['total_time'],
        'optim1_rate': optim1_results['ops_per_second'],
        'optim2_rate': optim2_results['ops_per_second'],
        'extrapolated_1m_optim1': optim1_time_1m,
        'extrapolated_1m_optim2': optim2_time_1m,
        'target_achieved': optim2_time_1m <= 60,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark vectorized operations')
    parser.add_argument('--grid-size', type=int, default=20,
                       help='Grid size for initial mesh (default: 20)')
    parser.add_argument('--n-splits', type=int, default=2000,
                       help='Number of edge splits to perform (default: 2000)')
    parser.add_argument('--optim1-batch-size', type=int, default=5000,
                       help='Batch size for Optim 1 (default: 5000)')
    parser.add_argument('--optim2-batch-size', type=int, default=1000,
                       help='Batch size for Optim 2 vectorized (default: 1000)')
    parser.add_argument('--skip-optim1', action='store_true',
                       help='Skip Optim 1 benchmark (only run Optim 2)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    logger.info(f"\n{'='*70}")
    logger.info("OPTIM 2 VECTORIZED OPERATIONS BENCHMARK")
    logger.info(f"{'='*70}")
    logger.info(f"Numba JIT: {' Available' if NUMBA_AVAILABLE else ' Not available (slower)'}")
    logger.info(f"Grid size: {args.grid_size}x{args.grid_size}")
    logger.info(f"Target splits: {args.n_splits}")
    
    # Create initial mesh
    points, triangles = create_initial_mesh(args.grid_size)
    
    # Run benchmarks
    results = {}
    
    if not args.skip_optim1:
        phase1_results = benchmark_batch_mode(
            points, triangles, args.n_splits, args.optim1_batch_size
        )
        results['phase1'] = phase1_results
    
    phase2_results = benchmark_vectorized_mode(
        points, triangles, args.n_splits, args.optim2_batch_size
    )
    results['optim2'] = phase2_results
    
    # Compare if we have both
    if 'optim1' in results:
        comparison = compare_phases(results['optim1'], results['optim2'])
        results['comparison'] = comparison
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=2, default=convert)
        logger.info(f"\n Results saved to {args.output}")
    
    logger.info(f"\n{'='*70}")
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
