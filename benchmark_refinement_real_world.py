#!/usr/bin/env python3
"""
Benchmark refinement_scenario.py with the optimizations.

Compares performance with Numba optimizations enabled vs disabled.
"""
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sofia.core.logging_utils import configure_logging, get_logger
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.core import geometry, conformity, diagnostics

# Save original HAS_NUMBA values
ORIG_GEOM_NUMBA = geometry.HAS_NUMBA
ORIG_CONF_NUMBA = conformity.HAS_NUMBA
ORIG_DIAG_NUMBA = diagnostics.HAS_NUMBA

configure_logging(level='WARNING')
log = get_logger('benchmark_refinement')


def load_config(config_path):
    """Load JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def run_refinement_scenario(config_path, enable_numba=True):
    """Run refinement scenario and measure time."""
    # Set Numba flags
    geometry.HAS_NUMBA = enable_numba and ORIG_GEOM_NUMBA
    conformity.HAS_NUMBA = enable_numba and ORIG_CONF_NUMBA
    diagnostics.HAS_NUMBA = enable_numba and ORIG_DIAG_NUMBA
    
    # Load config
    cfg = load_config(config_path)
    mesh_cfg = cfg.get('mesh', {})
    auto_cfg = cfg.get('auto', {})
    
    # Build initial mesh
    npts = int(mesh_cfg.get('npts', 60))
    seed = int(mesh_cfg.get('seed', 7))
    pts, tris = build_random_delaunay(npts=npts, seed=seed)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    
    initial_stats = {
        'n_points': len(editor.points),
        'n_triangles': len(editor.triangles),
        'min_angle': editor.global_min_angle(),
    }
    
    # Time the auto-refinement
    start_time = time.perf_counter()
    
    # Run auto-refinement (simplified version focusing on core operations)
    if auto_cfg:
        target_h_factor = float(auto_cfg.get('target_h_factor', 0.5))
        max_h_iters = int(auto_cfg.get('max_h_iters', 10))
        max_splits = int(auto_cfg.get('max_h_splits_per_iter', 100))
        
        from sofia.core.quality import compute_h
        from sofia.core.diagnostics import compact_copy
        
        for iter_idx in range(max_h_iters):
            # Compute current mesh size
            pts_c, tris_c, _, _ = compact_copy(editor)
            if len(tris_c) == 0:
                break
                
            h_current = compute_h(editor, metric='avg_equilateral_h')
            h_target = h_current * target_h_factor
            
            # Find edges to split
            edges_to_split = []
            for tri_idx, tri in enumerate(editor.triangles):
                if np.all(tri == -1):
                    continue
                    
                t = [int(x) for x in tri]
                a, b, c = t[0], t[1], t[2]
                
                # Check edge lengths
                for v1, v2 in [(a, b), (b, c), (c, a)]:
                    edge_len = np.linalg.norm(editor.points[v1] - editor.points[v2])
                    if edge_len > h_target * 1.5:
                        edges_to_split.append((v1, v2, edge_len))
            
            if not edges_to_split or len(edges_to_split) == 0:
                break
            
            # Sort by length and split longest edges
            edges_to_split.sort(key=lambda x: x[2], reverse=True)
            n_splits = 0
            
            for v1, v2, _ in edges_to_split[:max_splits]:
                try:
                    editor.op_split_edge(v1, v2)
                    n_splits += 1
                except Exception:
                    pass
            
            if n_splits == 0:
                break
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    # Compute final stats
    pts_c, tris_c, _, _ = compact_copy(editor)
    final_stats = {
        'n_points': len(pts_c),
        'n_triangles': len(tris_c),
        'min_angle': editor.global_min_angle() if len(tris_c) > 0 else 0.0,
    }
    
    return elapsed, initial_stats, final_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark refinement scenario')
    parser.add_argument('config', help='Path to config JSON file')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per configuration')
    args = parser.parse_args()
    
    print("=" * 80)
    print("REFINEMENT SCENARIO BENCHMARK")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Runs per configuration: {args.runs}")
    print()
    
    # Check Numba availability
    print(f"Numba availability:")
    print(f"  geometry.HAS_NUMBA:    {ORIG_GEOM_NUMBA}")
    print(f"  conformity.HAS_NUMBA:  {ORIG_CONF_NUMBA}")
    print(f"  diagnostics.HAS_NUMBA: {ORIG_DIAG_NUMBA}")
    print()
    
    if not any([ORIG_GEOM_NUMBA, ORIG_CONF_NUMBA, ORIG_DIAG_NUMBA]):
        print("WARNING: Numba not available! Cannot compare optimizations.")
        print("Running baseline only...")
        print()
    
    # Warmup runs
    print("Warming up...")
    for _ in range(2):
        run_refinement_scenario(args.config, enable_numba=True)
    print()
    
    # Benchmark WITH Numba optimizations
    print("Running WITH Numba optimizations...")
    times_with = []
    for run in range(args.runs):
        elapsed, initial, final = run_refinement_scenario(args.config, enable_numba=True)
        times_with.append(elapsed)
        print(f"  Run {run+1}/{args.runs}: {elapsed:.3f}s "
              f"({initial['n_triangles']} → {final['n_triangles']} tris)")
    
    median_with = np.median(times_with)
    mean_with = np.mean(times_with)
    std_with = np.std(times_with)
    
    print(f"\nWith Numba: {median_with:.3f}s (median), {mean_with:.3f}s ± {std_with:.3f}s")
    print()
    
    # Benchmark WITHOUT Numba optimizations (if available)
    if any([ORIG_GEOM_NUMBA, ORIG_CONF_NUMBA, ORIG_DIAG_NUMBA]):
        print("Running WITHOUT Numba optimizations...")
        times_without = []
        for run in range(args.runs):
            elapsed, initial, final = run_refinement_scenario(args.config, enable_numba=False)
            times_without.append(elapsed)
            print(f"  Run {run+1}/{args.runs}: {elapsed:.3f}s "
                  f"({initial['n_triangles']} → {final['n_triangles']} tris)")
        
        median_without = np.median(times_without)
        mean_without = np.mean(times_without)
        std_without = np.std(times_without)
        
        print(f"\nWithout Numba: {median_without:.3f}s (median), {mean_without:.3f}s ± {std_without:.3f}s")
        print()
        
        # Compute speedup
        speedup = median_without / median_with
        improvement = (1 - median_with / median_without) * 100
        
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"  With Numba:     {median_with:.3f}s")
        print(f"  Without Numba:  {median_without:.3f}s")
        print(f"  Speedup:        {speedup:.2f}x")
        print(f"  Improvement:    {improvement:.1f}% faster")
        print("=" * 80)
    else:
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"  Execution time: {median_with:.3f}s (median)")
        print(f"  Note: Numba not available for comparison")
        print("=" * 80)
    
    # Restore original flags
    geometry.HAS_NUMBA = ORIG_GEOM_NUMBA
    conformity.HAS_NUMBA = ORIG_CONF_NUMBA
    diagnostics.HAS_NUMBA = ORIG_DIAG_NUMBA


if __name__ == '__main__':
    main()
