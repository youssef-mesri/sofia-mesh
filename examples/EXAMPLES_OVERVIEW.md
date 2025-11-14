# Sofia Examples Overview

This directory contains example scripts demonstrating various mesh operations and remeshing techniques.

## Quick Reference: Anisotropic Remeshing Examples

| Example | Difficulty | Time | Key Feature | When to Use |
|---------|-----------|------|-------------|-------------|
| `anisotropic_remeshing.py` | Advanced | 15 min | Basic metric-guided adaptation | Learn fundamentals of anisotropic remeshing |
| `anisotropic_levelset_adaptation.py` | Advanced | 20 min | Level-set based metric | Adapt to curved features and iso-contours |
| `anisotropic_remeshing_normalized.py` | Expert | 30 min | Metric normalization + budget control | Target specific vertex count, production use |
| `anisotropic_boundary_adaptation.py` | Advanced | 15 min | Boundary layer insertion | CFD preprocessing, boundary layer meshing |

**Recommended learning path:** 
9 → 11 → 12 (anisotropic_remeshing → levelset → normalized)

---

## Main Examples

### Anisotropic Remeshing

#### `anisotropic_levelset_adaptation.py` **Recommended**
A clean, well-documented implementation of anisotropic remeshing with level-set based metric and boundary preservation.

**Features:**
- Anisotropic metric field based on distance to a level-set function (sinusoidal curve)
- Automatic boundary preservation (no manual protection needed for collapse)
- Split/collapse/smooth operations
- Comprehensive visualization with statistics
- Quality checks disabled for anisotropic meshes (elongated triangles are desired)

**Usage:**
```bash
python anisotropic_levelset_adaptation.py --max-iter 10 --alpha 0.8 --beta 0.7
```

**Key innovations:**
- Level-set function φ(x,y) = y - (0.5 + 0.15·sin(4πx)) defines the feature
- Metric adapts based on distance to iso-contours
- Boundary vertices automatically detected and preserved during edge collapse
- Quality checks based on angles are disabled (not suitable for anisotropic meshes)
- C++ backend disabled to use Python implementation with boundary detection

**Results:**
- Error reduction: ~3x
- Metric edge lengths: well-aligned to ideal (L_M ≈ 1.0)
- Boundary perfectly preserved (deviation = 0.00)

#### `anisotropic_remeshing_normalized.py` **Expert-Level**
Advanced anisotropic remeshing with **metric normalization** for precise computational budget control.

**Features:**
- All features of `anisotropic_levelset_adaptation.py` plus:
- **Metric normalization**: Target specific vertex count via determinant normalization
- **Mesh quadrature integration**: 50× faster than grid sampling (0.01s vs 0.5s)
- **Independent control**: Decouple anisotropy ratio from vertex count
- Calibration factor for accurate vertex count targeting

**Usage:**
```bash
# Target 350 vertices
python anisotropic_remeshing_normalized.py --target-complexity 350 --alpha 1.3 --beta 0.5

# Target 500 vertices with more iterations
python anisotropic_remeshing_normalized.py --target-complexity 500 --max-iter 10

# Disable normalization (original behavior)
python anisotropic_remeshing_normalized.py --no-normalize
```

**Key innovation - Metric Normalization:**
- Formula: M_normalized = α² · M where α = sqrt(N_target / (C · ∫√det(M) dΩ))
- Calibration factor C ≈ 2.3 (for alpha=1.3, beta=0.5)
- Mesh quadrature evaluates integral on triangle centroids
- Target 350 → Actual 345 vertices (0.99 ratio, nearly perfect!)

**Validation Results:**
| Target | Actual | Ratio | Error |
|--------|--------|-------|-------|
| 150 | 200 | 1.33 | +33% |
| 250 | 291 | 1.16 | +16% |
| 350 | 345 | 0.99 | -1%  |
| 500 | 426 | 0.85 | -15% |

**Perfect for:**
- Computational budget control in large-scale simulations
- Comparing anisotropic strategies at fixed complexity
- Production meshes with resource constraints

**Documentation:**
- Theory: `README_metric_normalization.md`
- Performance: `MESH_QUADRATURE_RESULTS.md`

#### `anisotropic_remeshing.py`
Full-featured anisotropic remeshing with extensive diagnostics and boundary layer support.

**Features:**
- Complete remeshing pipeline with boundary layer insertion
- Extensive diagnostics and validation
- Multiple visualization panels
- Detailed logging and statistics

**Usage:**
```bash
python anisotropic_remeshing.py --max-iter 20 --npts 80
```

#### `anisotropic_boundary_adaptation.py`
Focused on boundary layer adaptation with normal vector alignment.

**Features:**
- Boundary layer mesh generation
- Normal vector computation and alignment
- Visualization of boundary layers

**Usage:**
```bash
python anisotropic_boundary_adaptation.py
```

---

### Basic Mesh Operations

#### `basic_remeshing.py`
Simple isotropic remeshing example showing fundamental operations.

**Features:**
- Edge split
- Edge collapse
- Edge flip
- Laplacian smoothing

#### `boundary_operations.py`
Demonstrates boundary-aware mesh operations.

**Features:**
- Boundary detection
- Boundary-constrained operations
- Topology preservation

#### `boundary_refinement.py`
Example of adaptive boundary refinement.

**Features:**
- Boundary edge refinement
- Distance-based adaptation
- Quality improvement near boundaries

---

### Advanced Examples

#### `adaptive_refinement.py`
Adaptive mesh refinement based on solution error.

**Features:**
- Error-driven refinement
- Hierarchical mesh adaptation
- Coarsening and refinement

#### `combined_refinement.py`
Combines multiple refinement strategies.

**Features:**
- Multi-criteria refinement
- Boundary and interior adaptation
- Quality-aware operations

#### `mesh_coarsening.py`
Demonstrates mesh simplification and coarsening.

**Features:**
- Edge collapse strategies
- Quality preservation during coarsening
- Feature preservation

#### `mesh_workflow.py`
Complete mesh processing workflow.

**Features:**
- Full pipeline example
- Multiple operations in sequence
- Statistics and validation

#### `quality_improvement.py`
Focused on improving mesh quality metrics.

**Features:**
- Quality-driven flipping
- Smoothing strategies
- Quality metrics computation

---

## Key Concepts

### Anisotropic Remeshing
- **Goal**: Adapt mesh to directional features (e.g., boundary layers, shocks, level-sets)
- **Metric**: 2×2 tensor defining desired edge lengths in different directions
- **Operations**: Split long edges, collapse short edges (in metric space)
- **Challenge**: Elongated triangles are desired, not defects!

### Metric Normalization (Advanced)
- **Goal**: Control computational budget by targeting specific vertex count
- **Method**: Scale metric uniformly by factor alpha^2 to achieve target complexity
- **Formula**: M_normalized = alpha^2 · M where alpha = sqrt(N_target / (C · \int \sqrt(det(M)) d\Omega))
- **Integration**: Mesh-based quadrature evaluates \int \sqrt(det(M)) d\Omega on triangle centroids
- **Calibration**: Empirical factor C relates integral to actual vertex count
- **Benefit**: Independent control of anisotropy ratio and vertex count
- **Accuracy**: Typically ±15-30%, excellent for 300-500 vertex range

### Boundary Preservation
- **Important**: Domain boundary must remain unchanged during remeshing
- **Implementation**: Automatic detection of boundary vertices in `op_edge_collapse`
- **Collapse strategy**: If vertex on boundary -> collapse to boundary vertex
- **Smoothing**: Exclude all boundary vertices (not just corners)

### Quality Checks
- **Isotropic meshes**: Angle-based quality checks are appropriate
- **Anisotropic meshes**: Angle-based checks MUST BE DISABLED
- **Why**: Elongated triangles have small angles by design
- **Configuration**: Set `enforce_collapse_quality = False` and `enforce_split_quality = False`

---

## Running Examples

### Prerequisites
```bash
cd /path/to/Sofia/publication_prep
pip install -e .
```

### Basic execution
```bash
cd examples
python anisotropic_levelset_adaptation.py
```

### With parameters
```bash
python anisotropic_levelset_adaptation.py --max-iter 20 --alpha 0.8 --beta 0.7 --h-perp 0.008 --h-tang 0.15
```

### Metric normalization example
```bash
python anisotropic_remeshing_normalized.py --target-complexity 350 --alpha 1.3 --beta 0.5 --max-iter 5
```

### View results
```bash
# Images are saved in the same directory
ls -lh *.png
```

---

## Output Files

- `simple_remesh_result.png` - Visualization from anisotropic_levelset_adaptation.py
- `simple_remesh_normalized_result.png` - Visualization from anisotropic_remeshing_normalized.py
- `anisotropic_remeshing_result.png` - Visualization from anisotropic_remeshing.py
- `anisotropic_boundary_adaptation_result.png` - Visualization from boundary adaptation

---

## Implementation Notes

### Edge Collapse with Boundary Preservation

The key innovation is in `sofia/core/operations.py::op_edge_collapse`:

```python
# Detect if vertices are on boundary
u_is_boundary = is_boundary_vertex_from_maps(u, editor.edge_map)
v_is_boundary = is_boundary_vertex_from_maps(v, editor.edge_map)

# Decide collapse position
if u_is_boundary and v_is_boundary:
    p_new = editor.points[u].copy()  # Both on boundary
elif u_is_boundary:
    p_new = editor.points[u].copy()  # Collapse to u
elif v_is_boundary:
    p_new = editor.points[v].copy()  # Collapse to v
else:
    p_new = 0.5 * (editor.points[u] + editor.points[v])  # Interior: midpoint
```

### Disabling C++ Backend

To use Python implementation with boundary detection:

```python
# In mesh_modifier2.py, line ~1415
if False and self.use_cpp_core and ...:  # Force Python fallback
    ...
```

The C++ backend always collapses to midpoint and doesn't support boundary preservation yet.

---

## Troubleshooting

### "Boundary is deformed"
- Check that C++ backend is disabled for edge_collapse
- Verify `is_boundary_vertex_from_maps` is called
- Ensure all boundary vertices are protected during smoothing

### "Too many rejections during collapse"
- Disable angle-based quality checks for anisotropic meshes
- Set `editor.enforce_collapse_quality = False`
- Set `editor.enforce_split_quality = False`

### "Mesh quality degraded"
- For isotropic: Enable quality checks
- For anisotropic: This is normal! Check metric edge lengths instead

---

## References

- Sofia documentation: `/path/to/Sofia/publication_prep/README.md`
- Anisotropic remeshing: `README_anisotropic.md`
- Metric normalization theory: `README_metric_normalization.md`
- Mesh quadrature performance: `MESH_QUADRATURE_RESULTS.md`
- Core operations: `sofia/core/operations.py`
- Mesh editor: `sofia/core/mesh_modifier2.py`
