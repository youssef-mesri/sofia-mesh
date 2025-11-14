# SOFIA Examples

This directory contains practical examples demonstrating SOFIA's mesh editing capabilities, from basic operations to advanced boundary refinement and complete workflows.

## Quick Start Examples

### 1. Basic Remeshing (`basic_remeshing.py`)
**Difficulty:** Beginner  
**Time:** ~5 minutes

Learn the fundamental mesh operations:
- Create a simple mesh from scratch
- Split edges to add refinement
- Flip edges to improve quality
- Collapse edges to simplify
- Check mesh quality metrics
- VisuaComplete Workflow (15 min)
     **Track 4 - Anisotropic Adaptation (Advanced to Expert):**
9. Anisotropic Remeshing → 11. Anisotropic Level-Set Adaptation → 
12. Anisotropic Remeshing with Normalization [Expert] → 10. Anisotropic Boundary Adaptation ↓
Anisotropic Remeshing (15 min)
         ↓
Anisotropic Level-Set Adaptation (20 min)
         ↓
Anisotropic Remeshing with Normalization (30 min) [Expert]
         ↓
Anisotropic Boundary Adaptation (15 min)-by-step results

```bash
python examples/basic_remeshing.py
```

**Output:** `basic_remeshing_result.png` (6-panel visualization showing each operation)

**Key Concepts:** PatchBasedMeshEditor, split_edge(), flip_edge(), collapse_edge()

---

### 2. Quality Improvement (`quality_improvement.py`)
**Difficulty:** Beginner  
**Time:** ~5 minutes

Mesh refinement through edge subdivision:
- Start with a coarse mesh
- Identify and split long edges
- Track mesh density evolution
- Compare triangle size distributions

```bash
python examples/quality_improvement.py
```

**Output:** `quality_improvement_result.png` (4-panel before/after comparison)

**Key Concepts:** Edge-based refinement, adaptive meshing, quality metrics

---

### 3. Boundary Operations (`boundary_operations.py`)
**Difficulty:** Intermediate  
**Time:** ~10 minutes

Safe manipulation of mesh boundaries:
- Work with non-convex L-shaped domain
- Split boundary edges
- Remove boundary vertices
- Maintain mesh conformity throughout

```bash
python examples/boundary_operations.py
```

**Output:** `boundary_operations_result.png` (before/after boundary visualization)

**Key Concepts:** Boundary detection, conformity preservation, edge/node operations

---

## Advanced Examples

### 4. Adaptive Refinement (`adaptive_refinement.py`)
**Difficulty:** Intermediate  
**Time:** ~10 minutes

Target-based adaptive mesh refinement:
- Build random Delaunay mesh
- Identify largest triangles
- Iteratively refine based on area criterion
- Track uniformity improvements
- Analyze area distribution evolution

```bash
python examples/adaptive_refinement.py
```

**Output:** `adaptive_refinement_result.png` (comprehensive 6-panel analysis)

**Stats Example:**
- Initial: 42 triangles -> Refined: 57 triangles (+36%)
- Area uniformity improved by 9%
- Targeted refinement preserves mesh structure

**Key Concepts:** Adaptive refinement, triangle area metrics, iterative subdivision

---

### 5. Mesh Coarsening (`mesh_coarsening.py`)
**Difficulty:** Advanced  
**Time:** ~15 minutes

Simplification through edge collapse:
- Start with dense mesh (160+ triangles)
- Identify shortest interior edges
- Safely collapse while maintaining quality
- Generate LOD (Level of Detail) variants
- Analyze edge length distribution changes

```bash
python examples/mesh_coarsening.py
```

**Output:** `mesh_coarsening_result.png` (6-panel coarsening analysis)

**Stats Example:**
- Initial: 162 triangles → Coarsened: 102 triangles (-37%)
- 30 successful collapses, 100% success rate
- Mean edge length: 0.16 -> 0.21 (+31%)
- Quality preserved throughout

**Key Concepts:** Edge collapse, mesh simplification, LOD generation, quality preservation

---

### 6. Complete Workflow (`mesh_workflow.py`)
**Difficulty:** Advanced  
**Time:** ~15 minutes

Full mesh editing pipeline combining multiple operations:
- Create initial mesh
- **Refine** specific regions (e.g., circular area)
- **Coarsen** other regions (remove low-degree vertices)
- Validate conformity at each step
- Track mesh evolution through workflow

```bash
python examples/mesh_workflow.py
```

**Output:** `mesh_workflow_result.png` (3-stage workflow visualization)

**Workflow Example:**
- Stage 1: Initial mesh (82 triangles)
- Stage 2: After refinement (+22 triangles in target region)
- Stage 3: After coarsening (-16 triangles outside region)
- Final: 88 triangles, all operations conformity-checked 

**Key Concepts:** Multi-stage editing, region-based operations, conformity validation

---

### 7. Boundary Refinement (`boundary_refinement.py`)
**Difficulty:** Intermediate  
**Time:** ~10 minutes

Selective refinement of mesh boundaries:
- Create a mesh with a circular boundary
- Identify boundary vs interior edges
- Refine boundary edges based on length threshold
- Maintain mesh conformity during boundary operations
- Analyze boundary resolution improvement

```bash
python examples/boundary_refinement.py
```

**Output:** `boundary_refinement_result.png` (6-panel analysis with zoom views)

**Results:** 50% reduction in max boundary edge length, boundary uniformity improved

**Perfect for:** Curved boundary approximation, boundary layer meshing

**Key Concepts:** Boundary edge identification, selective refinement, edge length thresholds

---

### 8. Combined Refinement (`combined_refinement.py`)
**Difficulty:** Advanced  
**Time:** ~15 minutes

Multi-region refinement with different strategies:
- Create an L-shaped domain with sharp corners
- Apply different refinement criteria to boundary vs interior
- Preserve sharp features during refinement
- Feature-aware refinement near corners
- Analyze refinement effectiveness by region

```bash
python examples/combined_refinement.py
```

**Output:** `combined_refinement_result.png` (8-panel comprehensive analysis)

**Refinement Strategy:**
- Boundary edges: refine if length > 0.3 (87.5% max length reduction)
- Interior edges: refine if length > 0.5 (65.2% max length reduction)
- Corner regions: extra refinement for feature preservation

**Perfect for:** Complex domains, feature-aware meshing, non-convex geometries

**Key Concepts:** Multi-criteria refinement, feature preservation, region-based strategies

---

### 9. Anisotropic Remeshing (`anisotropic_remeshing.py`)
**Difficulty:** Advanced  
**Time:** ~15 minutes

Metric-guided anisotropic mesh adaptation:
- Build random Delaunay mesh
- Define spatially-varying metric tensor field M(x)
- Compute metric edge lengths L_M(e) = sqrt((q-p)^T M (q-p))
- Iteratively refine/coarsen based on metric lengths
- Split edges where L_M > alpha, collapse where L_M < beta
- Maintain mesh conformity throughout
- Visualize metric ellipses showing directional scaling

```bash
python examples/anisotropic_remeshing.py
```

**Output:** `anisotropic_remeshing_result.png` (6-panel comprehensive visualization)

**Technical Highlights:**
- Metric tensor M(x): 2×2 symmetric positive-definite matrix at each point
- Pure Python implementation using split/collapse operations only
- Preserves mesh conformity (each edge shared by ≤2 triangles)
- Blue ellipses visualize the anisotropic field (axes = eigenvectors, scaling = eigenvalues)

**Results Example:**
- Initial: 84 vertices, 162 triangles
- After: 23 vertices, 37 triangles (-77% triangles, significant coarsening)
- Operations: 3 splits, 64 collapses, 0 flips
- Conformity: Maintained throughout
- Metric deviation: 0.996 -> 0.898 (9.8% improvement)

**Perfect for:** Understanding metric-guided adaptation, anisotropic features

**Key Concepts:** Metric tensors, Riemannian geometry, anisotropic adaptation, conformity preservation

**Note:** This example uses split/collapse only to maintain conformity in pure Python.
For full remeshing with flips and metric-space smoothing, see `demos/adapt_scenario.py` with C++ core.

---

### 11. Anisotropic Level-Set Adaptation (`anisotropic_levelset_adaptation.py`)
**Difficulty:** Advanced  
**Time:** ~20 minutes

Anisotropic remeshing around a curved feature (sine curve):
- Create initial Delaunay mesh from domain boundary
- Define anisotropic metric adapted to sine curve y = sin(2πx)
- **Metric properties:** Fine resolution perpendicular to curve, coarse along curve
- Iterative remeshing: split long edges (L_M > alpha), collapse short edges (L_M < beta)
- Laplacian smoothing in physical space for quality improvement
- Visualize mesh evolution and metric field with ellipses

```bash
python examples/anisotropic_levelset_adaptation.py
```

**Output:** `simple_remesh_result.png` (comprehensive 6-panel visualization)

**Metric Design:**
- **h_perp = 0.008**: Fine perpendicular resolution (captures curve curvature)
- **h_tang = 0.15**: Coarse tangential resolution (efficient along curve)
- **Anisotropy ratio:** 18.75:1 (highly elongated triangles)
- Smooth transition from near-curve to far-field (d0 = 0.06)

**Results Example:**
- Initial: 41 vertices, 64 triangles
- Final: 597 vertices, 1150 triangles (+1357% growth)
- Operations per iteration: ~100-150 splits, ~20-30 collapses
- Mesh aligns with curve: tangent vectors follow sin(2\pi*x)

**Perfect for:** 
- Curved feature alignment
- Understanding metric-guided adaptation mechanics
- Boundary layer anisotropic patterns

**Key Concepts:** 
- Metric tensor construction from curve geometry
- Gradient-based normal/tangent computation
- Smooth distance-based metric transitions
- Iterative split/collapse cycles
- Laplacian smoothing for triangle quality

---

### 12. Anisotropic Remeshing with Metric Normalization (`anisotropic_remeshing_normalized.py`)
**Difficulty:** Expert  
**Time:** ~30 minutes

**Advanced anisotropic remeshing with precise computational budget control via metric normalization.**

This example extends `anisotropic_levelset_adaptation.py` with determinant-based normalization to target a specific vertex count while maintaining anisotropy:

**Key Innovation:**
- **Metric Normalization:** Scale metric uniformly to achieve target vertex count
- **Formula:** M_normalized = alpha · M where alpha = sqrt(target / (C · \int \sqrt(det(M)) d\Omega))
- **Independent Control:** Decouple anisotropy ratio from vertex count
- **Mesh Quadrature:** faster and accurate integration than grid sampling (0.01s vs 0.5s)

```bash
# Target 350 vertices (default parameters)
python examples/anisotropic_remeshing_normalized.py --target-complexity 350

# Target 500 vertices with custom thresholds
python examples/anisotropic_remeshing_normalized.py --target-complexity 500 --alpha 1.3 --beta 0.5 --max-iter 3

# Disable normalization (original behavior)
python examples/anisotropic_remeshing_normalized.py --no-normalize
```

**Output:** `simple_remesh_normalized_result.png` (6-panel analysis)

**Command-Line Options:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target-complexity` | 1000 | Target number of vertices (not a factor!) |
| `--alpha` | 1.3 | Split threshold (L_M > alpha) |
| `--beta` | 0.5 | Collapse threshold (L_M < beta) |
| `--calibration-factor` | 2.3 | Empirical constant (for alpha=1.3, beta=0.5) |
| `--max-iter` | 15 | Maximum remeshing iterations |
| `--no-normalize` | False | Disable normalization |

**Validation Results (C=2.3, alpha=1.3, beta=0.5):**
| Target | Actual | Ratio | Error |
|--------|--------|-------|-------|
| 150 | 200 | 1.33 | +33% |
| 250 | 291 | 1.16 | +16% |
| **350** | **345** | **0.99** | **-1%** |
| 500 | 426 | 0.85 | -15% |

**Accuracy:** Typically ±15-30%, excellent for 300-500 vertex range

**Technical Details:**
- **Integration:** Mesh-based quadrature evaluates \int \sqrt(det(M)) d\Omega on triangle centroids
- **Calibration:** Factor C=2.3 relates integral to actual vertex count
- **Scaling:** Uniform alpha applied to all metrics (preserves anisotropy directions)
- **Performance:** Quadrature ~0.01s (vs 0.5s for 100×100 grid sampling)

**Perfect for:**
- Computational budget control in simulations
- Comparing anisotropic strategies at fixed complexity
- Large-scale applications requiring vertex count limits
- Production meshes with resource constraints

**Key Concepts:**
- Metric determinant as vertex density indicator
- \int \sqrt(det(M)) d\Omega approximates vertex count
- Uniform metric scaling preserves anisotropy
- Empirical calibration for remeshing algorithms
- Mesh quadrature for efficient integration

**Documentation:**
- See `README_metric_normalization.md` for theory and detailed usage
- See `MESH_QUADRATURE_RESULTS.md` for performance analysis

**Note:** The calibration factor C depends on split/collapse thresholds. 
Different (alpha, beta) values require recalibration for optimal accuracy.

---

### 10. Anisotropic Boundary Adaptation (`anisotropic_boundary_adaptation.py`)
**Difficulty:** Advanced  
**Time:** ~15 minutes

Anisotropic mesh adaptation with boundary layer vertex insertion and protection:
- Create unit square mesh with explicit boundary tracking
- **Insert boundary layer vertices** at geometric progression distances from boundaries
- **Protect BL vertices** from collapse operations to preserve layer structure
- Define boundary-layer metric (anisotropic near edges)
- Adapt boundary edges based on metric length
- Adapt interior edges independently
- Visualize metric ellipses showing directionality
- Compare initial --> BL insertion --> final adaptation

```bash
python examples/anisotropic_boundary_adaptation.py
```

**Output:** `anisotropic_boundary_adaptation_result.png` (9-panel analysis with 3×3 grid)

**Example Results:**
- Initial: 35 vertices (20 boundary), 48 triangles
- After BL insertion: 225 vertices (+190 BL vertices), 428 triangles
- Final: 275 vertices (20 boundary), 452 triangles (+841% growth)
- Operations: 190 BL vertex insertions, 0 boundary splits, 31 interior splits, **19 collapses only**
- BL vertices protected: 190 (all preserved throughout adaptation)
- Conformity: Maintained throughout
- BL structure: **5 layers × 38 vertices/layer** = 190 BL vertices
- Layer discretization: 20 boundary vertices + 20 edge midpoints per layer
- Boundary layer: **5 layers** at heights 0.020, 0.030, 0.045, 0.068, 0.101 (total ~0.10, growth ratio 1.5)

**Perfect for:** Boundary layer meshing, anisotropic problems with boundary features, structured near-wall refinement, CFD preprocessing

**Key Concepts:** 
- Boundary layer vertex insertion at geometric progression
- **Protected vertex sets** to preserve BL structure during adaptation
- Outward normal computation from boundary edges
- Delaunay retriangulation after vertex insertion
- Boundary edge identification (edges shared by 1 triangle)
- Metric tensors with lambda = 1/h^2
- Independent boundary/interior thresholds
- Selective collapse: only non-BL vertices
- Conformity preservation

**Visualization Features:**
- 9-panel layout showing full progression
- Top row: Initial -> After BL insertion -> Final mesh
- Middle row: Zoomed views of boundary layer detail
- Bottom row: Vertex progression, operations breakdown, statistics with BL protection info
- Red bold lines = boundary edges (shared by 1 triangle)
- Gray lines = interior edges (shared by 2 triangles)
- Blue ellipses show metric anisotropy and orientation near boundaries
- Legend explaining edge types

**Boundary Layer Technique:**
- **Dual insertion strategy per layer:**
  - Vertices inserted at each boundary vertex (along averaged normal)
  - Additional vertices inserted at edge midpoints (along edge normal)
  - Total: 38 vertices per layer (20 + 20 for this mesh)
- Geometric progression: h, h*r, h*r^2, h*r^3, h*r^4 (default: h=0.02, r=1.5, 5 layers)
- Layer heights: 0.020 -> 0.030 -> 0.045 -> 0.068 -> 0.101 (cumulative ~0.10)
- Automatically detects correct normal direction (points toward domain interior)
- Global Delaunay retriangulation to incorporate new vertices
- Preserves boundary vertex positions
- **All BL vertices tracked and protected from edge collapse operations**
- **190 BL vertices** create production-quality boundary layer with fine layer-by-layer discretization

**Critical Feature - BL Vertex Protection:**
Without protection, collapse operations would destroy the carefully placed BL vertices.
With protection, only 19 collapses occur (all in interior far from BL), preserving the
boundary layer structure while still allowing interior adaptation. The dual insertion
strategy (boundary vertices + edge midpoints) creates 38 vertices per layer, providing
excellent boundary layer discretization suitable for CFD simulations.

---

## Output Files

All examples generate figures showing:
- **Before/after comparisons**
- **Quality metric distributions**
- **Mesh statistics**
- **Step-by-step evolution**

Generated files:
- `*_result.png` - Visualization outputs
- Stored in current working directory

---

## Requirements

```bash
# Core dependencies (already in requirements.txt)
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.3
```

All examples work with the pure Python implementation on the `main` branch.

---

## Tips

**For Beginners:**
1. Start with `basic_remeshing.py` to understand core operations
2. Move to `quality_improvement.py` for refinement basics
3. Explore `boundary_operations.py` for geometric constraints

**For Advanced Users:**
1. Study `adaptive_refinement.py` for targeted refinement strategies
2. Use `mesh_coarsening.py` for simplification and LOD
3. Explore `boundary_refinement.py` for boundary-specific operations
4. Try `combined_refinement.py` for multi-criteria strategies
5. Combine techniques in `mesh_workflow.py` for complex scenarios
6. Master `anisotropic_remeshing.py` for metric-guided adaptation basics
7. Learn `anisotropic_levelset_adaptation.py` for level-set based metrics

**For Expert Users:**
1. Study `anisotropic_remeshing_normalized.py` for computational budget control
2. Understand metric normalization theory in `README_metric_normalization.md`
3. Analyze mesh quadrature performance in `MESH_QUADRATURE_RESULTS.md`
4. Experiment with different calibration factors for your use case
5. Apply to `anisotropic_boundary_adaptation.py` for production boundary layers

**Performance Notes:**
- Examples use small meshes (20-80 vertices) for fast execution
- Scale up point counts for production use
- Consider C++ backend for large-scale applications (>10K triangles)

---

## Learning Path

```
Basic Remeshing (5 min)
         
Quality Improvement (5 min)  
         
Boundary Operations (10 min)

Adaptive Refinement (10 min)   

Boundary Refinement (10 min)

Combined Refinement (15 min) 

Mesh Coarsening   (15 min)      

Complete Workflow (15 min)
         
Anisotropic Remeshing (15 min)
         
Simple Anisotropic Remeshing (20 min)
         
Anisotropic Remeshing with Normalization (30 min) [Expert]
         
Anisotropic Boundary Adaptation (15 min)
```

**Total Learning Time:** ~2.5 hours for all examples (3 hours with expert-level normalization)

**Recommended Tracks:**

**Track 1 - Basics (Beginner):**
1. Basic Remeshing -> 2. Quality Improvement -> 3. Boundary Operations

**Track 2 - Refinement & Adaptation (Intermediate):**
4. Adaptive Refinement -> 7. Boundary Refinement -> 8. Combined Refinement

**Track 3 - Advanced Operations (Advanced):**
5. Mesh Coarsening -> 6. Complete Workflow

**Track 4 - Anisotropic Adaptation (Advanced to Expert):**
9. Anisotropic Remeshing -> 11. Simple Anisotropic Remeshing → 
12. Anisotropic Remeshing with Normalization [Expert] -> 10. Anisotropic Boundary Adaptation

---

## Troubleshooting

**Import Errors:**
```bash
# Install SOFIA in development mode
cd /path/to/sofia
pip install -e .
```

**Visualization Not Showing:**
```python
# Add to end of example script
plt.show()  # Display interactively instead of just saving
```

**Performance Issues:**
- Reduce `npts` parameter in mesh generation
- Decrease iteration counts in refinement loops
- Use smaller refinement regions

---

## Further Reading

- [SOFIA Documentation](../docs/)
- [API Reference](../docs/API.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Demos Directory](../demos/) - Additional visualization tools

---

## Contributing

Have a new example idea? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Adding new examples
- Writing clear documentation
- Including test cases
- Maintaining code quality

---

**Questions?** Open an issue on GitHub or check the documentation in `docs/`.
- Maintain area preservation

```bash
python boundary_operations.py
```

**Output:** `boundary_operations_result.png`

---

## Running All Examples

```bash
# Run all examples
for script in *.py; do
    echo "Running $script..."
    python "$script"
done
```

---

## Requirements

All examples require:
- SOFIA installed (`pip install sofia-mesh`)
- Matplotlib for visualization
- NumPy and SciPy (automatically installed with SOFIA)

---

## Customization

Feel free to modify these examples:
- Change mesh sizes
- Adjust quality targets
- Experiment with parameters
- Add your own visualizations

---

## Need Help?

- **API Reference:** See `docs/API_REFERENCE.md`
- **Discussions:** https://github.com/youssef-mesri/sofia/discussions
- **Issues:** https://github.com/youssef-mesri/sofia/issues

---

## Contributing Examples

Have a cool example? Please contribute!

1. Create your example script
2. Add clear comments and docstrings
3. Test it works from scratch
4. Submit a pull request

See `CONTRIBUTING.md` for details.
