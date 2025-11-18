# SOFIA Development Guide for AI Assistants

SOFIA (Scalable Operators for Field-driven Iso/Ani Adaptation) is a pure Python 2D triangular mesh modification library focused on quality-preserving local operations with strict conformity guarantees.

## Architecture Overview

### Core Component Hierarchy
- **`sofia/core/mesh_modifier2.py`** - Central `PatchBasedMeshEditor` class (1200+ lines), manages all mesh state and operations
- **`sofia/core/operations.py`** - Atomic mesh operations (split, flip, collapse, add/remove nodes, pocket fill)
- **`sofia/core/conformity.py`** - Mesh validity checking (edge-to-triangle maps, boundary detection, crossing edge detection)
- **`sofia/core/quality.py`** - Quality metrics (min angles, h-metric computation, area preservation checking)
- **`sofia/core/geometry.py`** - Low-level geometry primitives (triangle area/angles, orientation, point-in-polygon)

### Key Data Structures
The editor maintains **canonical storage** with amortized compaction:
- `points`: `(N,2)` float64 array, C-contiguous
- `triangles`: `(M,3)` int32 array, C-contiguous, with **tombstoning** (`[-1,-1,-1]` marks deleted triangles)
- `edge_map`: `Dict[(u,v)->Set[tri_idx]]` - edges to incident triangles (canonical: min vertex first)
- `v_map`: `Dict[v->Set[tri_idx]]` - vertices to incident triangles
- **Incremental structures** (`sofia/core/incremental.py`) avoid full rebuilds after local ops

**Why tombstoning?** SOFIA is based on patch/cavity editing that occurs iteratively in loops. Tombstoning aggregates expensive mesh map updates and compaction to the end of operations rather than paying the cost on every single triangle removal. This is critical for performance in iterative remeshing workflows.

### Operational Patterns

**Operation Lifecycle (all in `operations.py`):**
1. **Preflight quality check** - compute local min angles before modification
2. **Simulate compaction** (if `simulate_compaction_on_commit=True`) - validate operation wouldn't break topology
3. **Execute modification** - tombstone old triangles, append new ones, update maps
4. **Post-flight validation** - verify conformity preserved, quality non-worsening
5. **Stats tracking** - increment success/failure counters in `OpStats`

**Critical invariant preservation:**
- All triangles maintain **positive orientation** (CCW winding) via `ensure_positive_orientation()`
- **Conformity** must hold after every operation (edges shared by exactly 2 triangles, or 1 if boundary)
- **Area preservation** for node removal (see `AreaPreservationChecker` in `quality.py`)
- No **crossing edges** or **boundary loop increases** (when respective flags enabled)

## Developer Workflows

### Running Tests
```bash
pytest -q                          # Quick run (test output in test-logs/ on failure)
pytest sofia/tests/test_greedy_remesh.py::test_name -v  # Specific test
pytest -k "boundary" --tb=short    # Tests matching pattern
```

Tests use **headless Matplotlib (Agg backend)** via `conftest.py` fixture that captures logs only on failure. All assertions use tolerances from `sofia/core/constants.py` - **never hardcode tolerance literals** (enforced by `test_no_raw_tolerance_literals.py`).

### Building and Installation
```bash
pip install -e .[dev]              # Editable install with dev dependencies
make test                          # Run test suite
make lint                          # Run flake8
make type                          # Run mypy (Python 3.10+)
```

Package entry point: `sofia-remesh` CLI (defined in `pyproject.toml`, implemented in `remesh_driver.py:main()`)

### Benchmarking
Benchmarks live in `benchmarks/` with JSON results in `benchmarks/results/`. Key benchmarks:
- `benchmark_incremental.py` - Tests incremental map updates vs full rebuilds
- `benchmark_batch_ops.py` - BatchEditor performance
- `benchmark_refinement_*.py` - Real-world adaptive refinement scenarios

## Project-Specific Conventions

### Mesh File I/O
SOFIA currently uses **minimal dependencies** (numpy, matplotlib, scipy) and avoids heavy third-party mesh libraries:
- **Current approach**: Work directly with numpy arrays (`points`, `triangles`)
- **File formats**: No built-in I/O yet - meshes are typically constructed programmatically or from numpy saves
- **Future additions welcome**:
  - `.msh` reader (Gmsh format) - useful for importing generated meshes
  - `.vtk` exporter - standard format for visualization in ParaView/VisIt
  - Keep implementations lightweight (avoid `meshio`, `pyvista` dependencies)
  - Place I/O utilities in `sofia/core/io.py` or `utilities/` directory

When implementing I/O, preserve the canonical data format:
```python
# Expected format after reading
points = np.array([[x0, y0], [x1, y1], ...], dtype=np.float64)  # (N, 2)
triangles = np.array([[v0, v1, v2], ...], dtype=np.int32)       # (M, 3)
```

### Tolerance Management
**Always import from `constants.py`:**
```python
from .constants import EPS_AREA, EPS_MIN_ANGLE_DEG, EPS_IMPROVEMENT, EPS_TINY
```
Common uses:
- `EPS_AREA` (1e-12): Minimum valid triangle area
- `EPS_MIN_ANGLE_DEG` (1e-9): Min angle comparison tolerance (degrees)
- `EPS_IMPROVEMENT` (1e-12): Quality improvement significance threshold

### Editor Initialization Patterns
```python
# Standard initialization
editor = PatchBasedMeshEditor(points, triangles)

# Production mode with safeguards
editor = PatchBasedMeshEditor(
    points, triangles,
    simulate_compaction_on_commit=True,      # Preflight topology checks
    reject_boundary_loop_increase=True,       # Prevent holes
    reject_crossing_edges=True,               # Prevent non-planar graphs
    use_incremental_structures=True           # 100-1000x faster map updates
)

# Boundary-flexible mode
editor = PatchBasedMeshEditor(
    points, triangles,
    virtual_boundary_mode=True,               # Allow boundary vertex removal
    enable_remove_quad_fastpath=True          # Optimize quad->2tri removal
)
```

### Remeshing Driver Selection
- **`greedy_remesh(editor, config)`** - Iterative vertex/edge passes, good for general quality improvement
- **`run_patch_batch_driver(editor, config)`** - Patch-based batch operations, faster for targeted regions
- Use `GreedyConfig` / `PatchDriverConfig` dataclasses for configuration (defined in `config.py`)

### Plotting and Visualization
```python
from sofia.core.visualization import plot_mesh

plot_mesh(editor.points, editor.tris, title="My Mesh", 
          show_indices=True,        # Label vertices/triangles
          highlight_edges=[...],    # Emphasize specific edges
          save_path="output.png")   # Headless-friendly
```
- Always use `save_path` parameter for CI/headless environments
- Matplotlib backend auto-set to Agg if no `MPLBACKEND` environment variable

### Compaction Strategy
The editor **avoids eager compaction** for performance - this is by design, not a bug:
```python
# Check if compaction needed
if editor.has_tombstones():
    editor.compact()  # Removes [-1,-1,-1] triangles, renumbers arrays

# Or use utility for one-off compacted copy
from sofia.core.diagnostics import compact_copy
compact_pts, compact_tris = compact_copy(editor)
```

**Design rationale:** Since SOFIA performs iterative patch/cavity operations in loops, immediate compaction would trigger expensive map rebuilds and array reallocation on every operation. Tombstoning defers this cost until the end of a remeshing pass or when explicitly requested.

### Module Organization Quirks
- **Legacy shims exist** at repo root (`mesh_modifier2.py`, `debug_check.py`) but emit `DeprecationWarning` - always import from `sofia.core.*`
- **Public API facade** in `sofia/__init__.py` uses lazy imports to avoid circular dependencies
- **Pocket fill strategies** moved from `pocket_fill.py` to `triangulation.py` (old module is deprecated proxy)

## Common Pitfalls

1. **Hardcoded tolerances** - Tests will fail! Use `constants.py` imports
2. **Forgetting to check `has_tombstones()`** before array operations - tombstoned rows are `[-1,-1,-1]`
3. **Mutating `points`/`triangles` directly** - Use editor methods or property setters to maintain map consistency
4. **Assuming immediate compaction** - Operations may leave tombstones; call `compact()` explicitly if needed
5. **Missing conformity checks** - Always validate with `check_mesh_conformity()` after manual modifications
6. **Non-canonical edge keys** - Use `(min(u,v), max(u,v))` form for edge map lookups

## Testing Patterns

### Fixture Usage (from `conftest.py`)
- `capture_test_logs` - Auto-captures logs, writes to `test-logs/` only on failure
- Use `tmp_path` fixture for temporary files (pytest built-in)

### Assertion Style
```python
# Quality assertions
assert editor.mesh_min_angle() > 25.0 - EPS_MIN_ANGLE_DEG

# Area preservation
from sofia.core.quality import AreaPreservationChecker
checker = AreaPreservationChecker()
assert checker.check(removed_area, candidate_area)

# Conformity (raises on failure)
from sofia.core.conformity import check_mesh_conformity
check_mesh_conformity(points, triangles)  # Raises if invalid
```

## Performance Optimization Guidelines

1. **Use incremental structures** - Set `use_incremental_structures=True` on editor init
2. **Batch operations** - Use `BatchEditor` (from `batch_operations.py`) for 5-10x speedup on large-scale generation
3. **Vectorized quality checks** - Prefer `triangles_min_angles(points, triangles)` over loops
4. **Amortize expensive checks** - Use `strict_check_cooldown` in `GreedyConfig` to skip redundant validations
5. **Optional Numba JIT** - `vectorized_ops.py` has JIT-accelerated helpers if numba installed (graceful fallback)

## Examples Directory Structure

Examples in `examples/` follow difficulty progression:
- **Beginner**: `basic_remeshing.py`, `quality_improvement.py` - Single operation demos
- **Intermediate**: `boundary_operations.py`, `adaptive_refinement.py` - Multi-step workflows
- **Advanced**: `anisotropic_*.py` - Metric-based adaptation with tensors

All examples produce PNG visualizations and are tested via `scripts/test_examples.sh`.

## Configuration Files

- JSON configs in `configs/` define remeshing scenarios (see `configs/README.md`)
- Example: `refinement_scenario_h2.json` specifies target h-metric, boundary constraints
- Load via: `config = PatchDriverConfig(**json.load(open('configs/patch_cfg.json')))`

---

**Key Principle:** SOFIA prioritizes *correctness* (topology preservation, quality guarantees) over raw speed. When in doubt, add validation and document assumptions in docstrings.
