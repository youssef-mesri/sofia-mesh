# Sofia Examples Overview

This directory contains example scripts demonstrating various mesh operations and remeshing techniques.

## Main Examples

### Anisotropic Remeshing

#### `simple_anisotropic_remeshing.py` ⭐ **Recommended**
A clean, well-documented implementation of anisotropic remeshing with boundary preservation.

**Features:**
- Anisotropic metric field based on distance to a sinusoidal curve
- Automatic boundary preservation (no manual protection needed for collapse)
- Split/collapse/smooth operations
- Comprehensive visualization with statistics
- Quality checks disabled for anisotropic meshes (elongated triangles are desired)

**Usage:**
```bash
python simple_anisotropic_remeshing.py --max-iter 10 --alpha 0.8 --beta 0.7
```

**Key innovations:**
- Boundary vertices automatically detected and preserved during edge collapse
- Quality checks based on angles are disabled (not suitable for anisotropic meshes)
- C++ backend disabled to use Python implementation with boundary detection

**Results:**
- Error reduction: ~3x
- Metric edge lengths: well-aligned to ideal (L_M ≈ 1.0)
- Boundary perfectly preserved (deviation = 0.00)

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
- **Goal**: Adapt mesh to directional features (e.g., boundary layers, shocks)
- **Metric**: 2×2 tensor defining desired edge lengths in different directions
- **Operations**: Split long edges, collapse short edges (in metric space)
- **Challenge**: Elongated triangles are desired, not defects!

### Boundary Preservation
- **Important**: Domain boundary must remain unchanged during remeshing
- **Implementation**: Automatic detection of boundary vertices in `op_edge_collapse`
- **Collapse strategy**: If vertex on boundary → collapse to boundary vertex
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
python simple_anisotropic_remeshing.py
```

### With parameters
```bash
python simple_anisotropic_remeshing.py --max-iter 20 --alpha 0.8 --beta 0.7 --h-perp 0.008 --h-tang 0.15
```

### View results
```bash
# Images are saved in the same directory
ls -lh *.png
```

---

## Output Files

- `simple_remesh_result.png` - Visualization from simple_anisotropic_remeshing.py
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
- Core operations: `sofia/core/operations.py`
- Mesh editor: `sofia/core/mesh_modifier2.py`
