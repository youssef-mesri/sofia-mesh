# SOFIA Examples

This directory contains practical examples demonstrating SOFIA's mesh editing capabilities, from basic operations to advanced boundary refinement and complete workflows.

## 🎯 Quick Start Examples

### 1. Basic Remeshing (`basic_remeshing.py`)
**Difficulty:** ⭐ Beginner  
**Time:** ~5 minutes

Learn the fundamental mesh operations:
- Create a simple mesh from scratch
- Split edges to add refinement
- Flip edges to improve quality
- Collapse edges to simplify
- Check mesh quality metrics
- Visualize step-by-step results

```bash
python examples/basic_remeshing.py
```

**Output:** `basic_remeshing_result.png` (6-panel visualization showing each operation)

**Key Concepts:** PatchBasedMeshEditor, split_edge(), flip_edge(), collapse_edge()

---

### 2. Quality Improvement (`quality_improvement.py`)
**Difficulty:** ⭐ Beginner  
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
**Difficulty:** ⭐⭐ Intermediate  
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

## 🚀 Advanced Examples

### 4. Adaptive Refinement (`adaptive_refinement.py`)
**Difficulty:** ⭐⭐ Intermediate  
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
- Initial: 42 triangles → Refined: 57 triangles (+36%)
- Area uniformity improved by 9%
- Targeted refinement preserves mesh structure

**Key Concepts:** Adaptive refinement, triangle area metrics, iterative subdivision

---

### 5. Mesh Coarsening (`mesh_coarsening.py`)
**Difficulty:** ⭐⭐⭐ Advanced  
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
- Mean edge length: 0.16 → 0.21 (+31%)
- Quality preserved throughout

**Key Concepts:** Edge collapse, mesh simplification, LOD generation, quality preservation

---

### 6. Complete Workflow (`mesh_workflow.py`)
**Difficulty:** ⭐⭐⭐ Advanced  
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
- Final: 88 triangles, all operations conformity-checked ✓

**Key Concepts:** Multi-stage editing, region-based operations, conformity validation

---

### 7. Boundary Refinement (`boundary_refinement.py`)
**Difficulty:** ⭐⭐ Intermediate  
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
**Difficulty:** ⭐⭐⭐ Advanced  
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

## 📊 Output Files

All examples generate high-quality visualizations (150 DPI) showing:
- **Before/after comparisons**
- **Quality metric distributions**
- **Mesh statistics**
- **Step-by-step evolution**

Generated files:
- `*_result.png` - Visualization outputs
- Stored in current working directory

---

## 🔧 Requirements

```bash
# Core dependencies (already in requirements.txt)
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.3
```

All examples work with the pure Python implementation on the `main` branch.

---

## 💡 Tips

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

**Performance Notes:**
- Examples use small meshes (20-80 vertices) for fast execution
- Scale up point counts for production use
- Consider C++ backend for large-scale applications (>10K triangles)

---

## 📖 Learning Path

```
Basic Remeshing (5 min)
         ↓
Quality Improvement (5 min)  
         ↓
Boundary Operations (10 min)
         ↓
    ┌────┴────┐
    ↓         ↓
Adaptive    Boundary
Refinement  Refinement
(10 min)    (10 min)
    ↓         ↓
Mesh       Combined
Coarsening Refinement
(15 min)   (15 min)
    └────┬────┘
         ↓
Complete Workflow (15 min)
```

**Total Learning Time:** ~1.5 hours for all examples

---

## 🐛 Troubleshooting

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

## 📚 Further Reading

- [SOFIA Documentation](../docs/)
- [API Reference](../docs/API.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Demos Directory](../demos/) - Additional visualization tools

---

## 🤝 Contributing

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

- 📖 **API Reference:** See `docs/API_REFERENCE.md`
- 💬 **Discussions:** https://github.com/youssef-mesri/sofia/discussions
- 🐛 **Issues:** https://github.com/youssef-mesri/sofia/issues

---

## Contributing Examples

Have a cool example? Please contribute!

1. Create your example script
2. Add clear comments and docstrings
3. Test it works from scratch
4. Submit a pull request

See `CONTRIBUTING.md` for details.
