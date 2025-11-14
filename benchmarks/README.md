# SOFIA Benchmarks

This directory contains performance benchmarks for SOFIA's mesh operations.

## Benchmark Scripts

- **benchmark_boundary_loops.py** - Boundary loop performance
- **benchmark_comprehensive_validation.py** - Complete validation suite
- **benchmark_editor_incremental.py** - Incremental editor operations
- **benchmark_grid_optimization.py** - Grid-based optimizations
- **benchmark_incremental.py** - Incremental operations
- **benchmark_incremental_fair.py** - Fair comparison benchmarks
- **benchmark_numba.py** - Numba acceleration tests
- **benchmark_numba_comparison.py** - Python vs Numba comparison
- **benchmark_numba_direct.py** - Direct Numba integration
- **benchmark_real_world.py** - Real-world scenarios
- **benchmark_refinement_hotpaths.py** - Refinement hotpath analysis
- **benchmark_refinement_real_world.py** - Real-world refinement
- **benchmark_refinement_subprocess.py** - Subprocess-based refinement

## Results

Benchmark results are stored in `results/`:
- **batch_benchmark_results.json** - Batch operation results
- **phase2_results.json** - Phase 2 optimization results

## Running Benchmarks

```bash
# Run a specific benchmark
python benchmark_<name>.py

# Run all benchmarks (takes time)
for bench in benchmark_*.py; do python "$bench"; done
```

## Interpreting Results

Results include:
- Execution time (seconds)
- Operations per second
- Memory usage
- Comparison with baseline

See individual benchmark files for detailed metrics.
