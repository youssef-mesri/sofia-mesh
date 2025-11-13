#!/usr/bin/env python3
"""
Benchmark real refinement scenario multiple times to measure performance impact.
"""
import subprocess
import time
import sys
import numpy as np
from pathlib import Path

def run_refinement(scenario_path, numba_enabled=True):
    """Run the refinement scenario and measure time."""
    env = {}
    if not numba_enabled:
        env['NUMBA_DISABLE_JIT'] = '1'
    
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, 'demos/refinement_scenario.py', 
         '--scenario', scenario_path, 
         '--no-plot'],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
        env={**subprocess.os.environ, **env}
    )
    end = time.perf_counter()
    
    if result.returncode != 0:
        print(f"Error running scenario: {result.stderr}")
        return None
    
    return end - start

def main():
    scenario = 'configs/refinement_scenario_h2_quad.json'
    n_runs = 5
    
    print("=" * 80)
    print("REAL-WORLD REFINEMENT BENCHMARK")
    print("=" * 80)
    print(f"Scenario: {scenario}")
    print(f"Runs: {n_runs}")
    print()
    
    # Warmup
    print("Warming up...")
    for _ in range(2):
        run_refinement(scenario, numba_enabled=True)
    print()
    
    # WITH Numba
    print(f"Running WITH Numba optimizations ({n_runs} runs)...")
    times_with = []
    for i in range(n_runs):
        t = run_refinement(scenario, numba_enabled=True)
        if t is not None:
            times_with.append(t)
            print(f"  Run {i+1}/{n_runs}: {t:.3f}s")
    
    if not times_with:
        print("ERROR: No successful runs with Numba")
        return
    
    median_with = np.median(times_with)
    mean_with = np.mean(times_with)
    std_with = np.std(times_with)
    print(f"\nWith Numba: {median_with:.3f}s (median), {mean_with:.3f}s ± {std_with:.3f}s")
    print()
    
    # WITHOUT Numba
    print(f"Running WITHOUT Numba optimizations ({n_runs} runs)...")
    times_without = []
    for i in range(n_runs):
        t = run_refinement(scenario, numba_enabled=False)
        if t is not None:
            times_without.append(t)
            print(f"  Run {i+1}/{n_runs}: {t:.3f}s")
    
    if not times_without:
        print("ERROR: No successful runs without Numba")
        return
    
    median_without = np.median(times_without)
    mean_without = np.mean(times_without)
    std_without = np.std(times_without)
    print(f"\nWithout Numba: {median_without:.3f}s (median), {mean_without:.3f}s ± {std_without:.3f}s")
    print()
    
    # Results
    speedup = median_without / median_with
    improvement = (1 - median_with / median_without) * 100
    time_saved = median_without - median_with
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  With Numba:     {median_with:.3f}s ± {std_with:.3f}s")
    print(f"  Without Numba:  {median_without:.3f}s ± {std_without:.3f}s")
    print(f"  Time saved:     {time_saved:.3f}s ({time_saved*1000:.0f}ms)")
    print(f"  Speedup:        {speedup:.2f}x")
    print(f"  Improvement:    {improvement:.1f}% faster")
    print("=" * 80)
    
    if speedup >= 1.2:
        print("Significant improvement!")
    elif speedup >= 1.1:
        print("Noticeable improvement")
    else:
        print("Tiny improvement")

if __name__ == '__main__':
    main()
