"""Benchmark batch operations vs normal operations for performance comparison.

This script tests the BatchEditor performance against normal operation execution
on a large-scale mesh refinement task.
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.batch_operations import BatchEditor
from sofia.core.conformity import check_mesh_conformity


def create_initial_mesh(grid_size=10):
    """Create a regular grid mesh for testing.
    
    Parameters
    ----------
    grid_size : int
        Number of vertices per side
        
    Returns
    -------
    points, triangles : ndarray
        Initial mesh
    """
    logger.info(f"Creating {grid_size}x{grid_size} grid mesh...")
    
    # Create grid of points
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Create triangles
    triangles = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # Vertices of the quad
            v0 = i * grid_size + j
            v1 = v0 + 1
            v2 = v0 + grid_size
            v3 = v2 + 1
            
            # Two triangles per quad
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])
    
    triangles = np.array(triangles, dtype=np.int32)
    
    logger.info(f"Created mesh: {len(points)} vertices, {len(triangles)} triangles")
    return points, triangles


def find_edges_to_split(editor, max_edges=1000):
    """Find edges suitable for splitting.
    
    Parameters
    ----------
    editor : PatchBasedMeshEditor
        The mesh editor
    max_edges : int
        Maximum number of edges to return
        
    Returns
    -------
    list of tuple
        Edges to split
    """
    # Get all edges from edge_map
    all_edges = list(editor.edge_map.keys())
    
    # Shuffle for variety
    rng = np.random.RandomState(42)
    rng.shuffle(all_edges)
    
    # Return subset
    return all_edges[:max_edges]


def benchmark_normal_mode(points, triangles, n_splits=1000):
    """Benchmark normal (per-operation validation) mode.
    
    Parameters
    ----------
    points, triangles : ndarray
        Initial mesh
    n_splits : int
        Number of edge splits to perform
        
    Returns
    -------
    dict
        Results including time, success count, final mesh size
    """
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK: Normal Mode (per-operation validation)")
    logger.info(f"{'='*60}")
    
    editor = PatchBasedMeshEditor(
        points.copy(), 
        triangles.copy(),
        enforce_split_quality=False  # Focus on performance, not quality
    )
    
    # Find edges to split
    edges = find_edges_to_split(editor, n_splits)
    logger.info(f"Found {len(edges)} edges to split")
    
    # Time the operations
    start_time = time.perf_counter()
    
    success_count = 0
    fail_count = 0
    
    for i, edge in enumerate(edges):
        ok, msg, info = editor.split_edge(edge)
        if ok:
            success_count += 1
        else:
            fail_count += 1
        
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed
            logger.info(f"  Progress: {i+1}/{len(edges)} splits, {rate:.1f} ops/sec")
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Final conformity check
    ok_conf, msgs = check_mesh_conformity(
        editor.points, 
        editor.triangles,
        allow_marked=True
    )
    
    results = {
        'mode': 'normal',
        'n_operations': len(edges),
        'success_count': success_count,
        'fail_count': fail_count,
        'total_time': total_time,
        'ops_per_second': len(edges) / total_time,
        'ms_per_op': (total_time * 1000) / len(edges),
        'final_vertices': len(editor.points),
        'final_triangles': len(editor.triangles),
        'conformity_ok': ok_conf,
    }
    
    logger.info(f"\nResults:")
    logger.info(f"  Total time: {total_time:.3f}s")
    logger.info(f"  Success: {success_count}, Failed: {fail_count}")
    logger.info(f"  Rate: {results['ops_per_second']:.1f} ops/sec")
    logger.info(f"  Per-op: {results['ms_per_op']:.2f} ms")
    logger.info(f"  Final mesh: {results['final_vertices']} vertices, {results['final_triangles']} triangles")
    logger.info(f"  Conformity: {'✓ PASS' if ok_conf else '✗ FAIL'}")
    
    return results


def benchmark_batch_mode(points, triangles, n_splits=1000, batch_size=5000):
    """Benchmark batch mode (deferred validation).
    
    Parameters
    ----------
    points, triangles : ndarray
        Initial mesh
    n_splits : int
        Number of edge splits to perform
    batch_size : int
        Batch size for validation
        
    Returns
    -------
    dict
        Results including time, success count, final mesh size
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARK: Batch Mode (batch_size={batch_size})")
    logger.info(f"{'='*60}")
    
    editor = PatchBasedMeshEditor(
        points.copy(), 
        triangles.copy(),
        enforce_split_quality=False
    )
    
    # Find edges to split
    edges = find_edges_to_split(editor, n_splits)
    logger.info(f"Found {len(edges)} edges to split")
    
    # Time the operations
    start_time = time.perf_counter()
    
    batch = BatchEditor(editor, batch_size=batch_size, validate_on_flush=True)
    
    for i, edge in enumerate(edges):
        batch.split_edge(edge)
        
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - start_time
            rate = (i + 1) / elapsed
            logger.info(f"  Progress: {i+1}/{len(edges)} splits queued, {rate:.1f} ops/sec")
    
    # Flush remaining operations
    ok_final = batch.finalize()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Get stats
    stats = batch.get_stats()
    
    # Final conformity check
    ok_conf, msgs = check_mesh_conformity(
        editor.points, 
        editor.triangles,
        allow_marked=False  # After compaction
    )
    
    results = {
        'mode': 'batch',
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
    
    logger.info(f"\nResults:")
    logger.info(f"  Total time: {total_time:.3f}s")
    logger.info(f"  Operations: {results['n_operations']}")
    logger.info(f"  Batches: {results['batches_processed']}, Failed: {results['failed_batches']}")
    logger.info(f"  Rate: {results['ops_per_second']:.1f} ops/sec")
    logger.info(f"  Per-op: {results['ms_per_op']:.2f} ms")
    logger.info(f"  Final mesh: {results['final_vertices']} vertices, {results['final_triangles']} triangles")
    logger.info(f"  Conformity: {'✓ PASS' if ok_conf else '✗ FAIL'}")
    
    return results


def compare_results(normal_results, batch_results):
    """Compare and print speedup analysis.
    
    Parameters
    ----------
    normal_results, batch_results : dict
        Results from both benchmarks
    """
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON")
    logger.info(f"{'='*60}")
    
    speedup = normal_results['total_time'] / batch_results['total_time']
    ops_speedup = batch_results['ops_per_second'] / normal_results['ops_per_second']
    
    logger.info(f"Normal mode:")
    logger.info(f"  Time: {normal_results['total_time']:.3f}s")
    logger.info(f"  Rate: {normal_results['ops_per_second']:.1f} ops/sec")
    logger.info(f"  Per-op: {normal_results['ms_per_op']:.2f} ms")
    
    logger.info(f"\nBatch mode:")
    logger.info(f"  Time: {batch_results['total_time']:.3f}s")
    logger.info(f"  Rate: {batch_results['ops_per_second']:.1f} ops/sec")
    logger.info(f"  Per-op: {batch_results['ms_per_op']:.2f} ms")
    
    logger.info(f"\nSpeedup:")
    logger.info(f"  Total time: {speedup:.2f}x faster")
    logger.info(f"  Throughput: {ops_speedup:.2f}x higher")
    logger.info(f"  Time saved: {normal_results['total_time'] - batch_results['total_time']:.3f}s")
    
    # Extrapolate to 1M triangles
    logger.info(f"\nExtrapolation to 1M triangles:")
    
    # Estimate operations needed (rough: each split adds ~2 triangles)
    ops_for_1m = 500_000
    
    normal_time_1m = ops_for_1m / normal_results['ops_per_second']
    batch_time_1m = ops_for_1m / batch_results['ops_per_second']
    
    logger.info(f"  Normal mode: {normal_time_1m:.1f}s ({normal_time_1m/60:.1f} min)")
    logger.info(f"  Batch mode:  {batch_time_1m:.1f}s ({batch_time_1m/60:.1f} min)")
    logger.info(f"  Target:      60s (1 min)")
    
    if batch_time_1m <= 60:
        logger.info(f"  ✓ TARGET ACHIEVED with batch mode!")
    else:
        additional_speedup = batch_time_1m / 60
        logger.info(f"  ✗ Need {additional_speedup:.1f}x more speedup")
    
    return {
        'speedup': speedup,
        'time_saved': normal_results['total_time'] - batch_results['total_time'],
        'normal_rate': normal_results['ops_per_second'],
        'batch_rate': batch_results['ops_per_second'],
        'extrapolated_1m_normal': normal_time_1m,
        'extrapolated_1m_batch': batch_time_1m,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark batch operations')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size for initial mesh (default: 10)')
    parser.add_argument('--n-splits', type=int, default=1000,
                       help='Number of edge splits to perform (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Batch size for batch mode (default: 5000)')
    parser.add_argument('--skip-normal', action='store_true',
                       help='Skip normal mode benchmark (only run batch)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Create initial mesh
    points, triangles = create_initial_mesh(args.grid_size)
    
    # Run benchmarks
    results = {}
    
    if not args.skip_normal:
        normal_results = benchmark_normal_mode(points, triangles, args.n_splits)
        results['normal'] = normal_results
    
    batch_results = benchmark_batch_mode(points, triangles, args.n_splits, args.batch_size)
    results['batch'] = batch_results
    
    # Compare if we have both
    if 'normal' in results:
        comparison = compare_results(results['normal'], results['batch'])
        results['comparison'] = comparison
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=2, default=convert)
        logger.info(f"\nResults saved to {args.output}")
    
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
