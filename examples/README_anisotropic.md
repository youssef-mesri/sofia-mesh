# Anisotropic Remeshing Example

## Description

This example demonstrates **metric-based anisotropic mesh adaptation** to align a mesh with a sinusoidal curve:

```
y = 0.5 + 0.15 * sin(4πx)
```

The algorithm uses a **Hessian-based metric** derived from the level-set function φ(x,y) = y - f(x), which naturally captures:
- Curvature of the curve (second derivatives)
- Anisotropic directions (eigenvectors of Hessian)
- Proper alignment with geometric features

## Features

- ✅ **Metric normalization** to control target vertex count
- ✅ **Edge flipping** for better metric alignment
- ✅ **Laplacian smoothing** to improve element quality
- ✅ **Extreme anisotropy** (ratio up to 70:1)
- ✅ **Pure Python implementation** (no C++ core required)
- ✅ **Conforming meshes** (each edge shared by ≤ 2 triangles)

## Usage

### Basic usage (default: extreme anisotropy 200:1)
```bash
python anisotropic_remeshing.py
```

**Recommended settings for strong anisotropy:**
```bash
python anisotropic_remeshing.py --max-iter 30 --alpha 1.5 --beta 0.5
```

This produces:
- ~600 vertices, ~1100 triangles
- Min angle: ~0.1° (extreme anisotropy!)
- Max L_M: ~1.5 (excellent metric satisfaction)
- Conforming mesh ✓

### Control mesh density

**Coarse mesh (~400 vertices):**
```bash
python anisotropic_remeshing.py --target-vertices 200 --alpha 2.0
```

**Medium mesh (~1000 vertices):**
```bash
python anisotropic_remeshing.py --target-vertices 400 --alpha 2.0
```

**Fine mesh (~1600 vertices):**
```bash
python anisotropic_remeshing.py --target-vertices 800 --alpha 1.5
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target-vertices` | 300 | Target number of vertices (actual ≈ 2-3×) |
| `--alpha` | 1.3 | Split threshold (higher = less aggressive) |
| `--beta` | 0.7 | Collapse threshold |
| `--npts` | 150 | Initial Delaunay mesh size |
| `--max-iter` | 40 | Maximum remeshing iterations |
| `--seed` | 42 | Random seed for reproducibility |

## Algorithm Overview

The remeshing process consists of the following steps:

1. **Metric Construction (Hessian-based)**
   - Compute Hessian H(φ) of level-set function
   - Extract eigenvalues/eigenvectors for anisotropic directions
   - Normalize metric to target vertex count

2. **Iterative Remeshing Loop**
   - **Split**: edges where L_M > α (metric edge length too long)
   - **Collapse**: edges where L_M < β (metric edge length too short)
   - **Flip**: edges to improve metric alignment (area-preserving)
   - **Smooth**: interior vertices (Laplacian, every 2 iterations)

3. **Convergence**
   - Stop when max|L_M - 1| < tolerance
   - Or when no operations possible
   - Typical: 15-30 iterations

## Metric Details

### Hessian Computation

For the level-set φ(x,y) = y - f(x):

```
H(φ) = [[-f''(x),  0    ],
        [0,        0    ]]
```

For our sinusoidal curve:
```
f(x) = 0.5 + 0.15·sin(4πx)
f'(x) = 0.15·4π·cos(4πx)
f''(x) = -0.15·(4π)²·sin(4πx)
```

### Metric Tensor

The metric tensor M controls mesh sizing:

```
M = R @ Λ @ R^T
```

where:
- **R** = [tangent | normal] (eigenvector matrix)
- **Λ** = diag(1/h_tangent², 1/h_normal²) (eigenvalue matrix)

### Anisotropic Sizing

| Location | h_⊥ (normal) | h_∥ (tangent) | Ratio |
|----------|-------------|---------------|-------|
| Near curve | 0.005 | 1.0 | **200:1** |
| Transition (d < 0.25) | 0.005→0.30 | 1.0 | 200:1→1:1 |
| Far field | 0.30 | 0.30 | 1:1 |

**Note:** Extreme anisotropy (200:1) produces triangles with aspect ratios up to 500:1 and minimum angles as low as 0.1°.

### Normalization

The metric is normalized using the complexity formula:

```
C = ∫_Ω √det(M(x)) dx
```

Normalization factor:
```
scale = (N_target / C)^(2/d)  where d=2 (dimensions)
M_normalized = scale · M
```

## Results

### Performance Metrics

With default settings (target=300, alpha=1.3):

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Max L_M | 46.88 | 1.46 | **96.9%** |
| Min angle | 0.47° | 2.42° | **5.2×** |
| Vertices | 154 | ~500 | |
| Triangles | 302 | ~1000 | |

### Vertex Count Control

| Target | Alpha | Actual Vertices | Ratio |
|--------|-------|-----------------|-------|
| 200 | 2.0 | ~400 | 2.0× |
| 300 | 1.3 | ~500 | 1.7× |
| 400 | 2.0 | ~1000 | 2.5× |
| 800 | 1.5 | ~1600 | 2.0× |

**Note:** Actual vertex count is typically 2-3× the target due to anisotropic refinement near the curve.

## Visualization

The output shows 6 panels:
1. **Initial mesh** (random Delaunay)
2. **Final adapted mesh** with sine curve overlay
3. **Metric ellipses** (blue) showing anisotropic field
4. **Zoom on curve** to visualize stretched triangles
5. **Edge length distribution** (L_M histogram)
6. **Statistics** panel

### Interpreting Ellipses

- **Highly elongated ellipses** → extreme anisotropy (200:1 stretched elements)
- **Circular ellipses** → isotropic (equilateral triangles)  
- **Ellipse orientation** → principal directions of metric tensor
- **Ellipse size** → mesh density (smaller = finer mesh)

### Observing Anisotropy

To see **strong anisotropy** in the mesh:
- Look for **very thin, elongated triangles** along the sine curve
- Check **minimum angle** (should be < 1° for extreme anisotropy)
- Zoom panel shows stretched elements most clearly
- Ellipses should be very elongated (major/minor axis ratio ~200)

## Implementation Notes

### Quality Enforcement

Quality checks are **disabled** (`enforce_split_quality=False`) to allow extreme anisotropy:
- Minimum angles can be < 5° (normal for 200:1 anisotropy)
- Area preservation checked during edge flipping
- Conformity preserved throughout

### Node Removal (TODO - Requires Core Fix)

**Status**: Implemented but non-functional due to `remove_node_with_patch` failures.

**Implemented approach** (`remove_nodes_by_metric_quality`):
- ✅ Direct metric-quality optimization (not degree-based heuristic)
- ✅ Evaluates cavity quality: `Q = mean(|L_M - 1|)` for all edges in star
- ✅ Simulates removal and computes new cavity quality
- ✅ Accepts only if `Q_after < Q_before - threshold`
- ✅ Ranks candidates by worst quality first
- ✅ Verifies mesh conformity before attempting removal

**Why it doesn't work**:
`remove_node_with_patch` systematically fails even with:
- Conforming meshes
- Interior nodes
- Good-quality cavities

**Root cause analysis**:
The core `remove_node_with_patch` function likely has:
1. **Euclidean quality checks** instead of metric-aware checks
2. **Too strict area preservation** constraints
3. **Orientation issues** in retriangulated cavity
4. **Missing support** for highly anisotropic elements

**Required fix** (in `sofia/core/operations.py`):
```python
def op_remove_node_with_patch(editor, v_idx, metric_fn=None):
    # ... existing code ...
    
    # Replace Euclidean quality check with metric-aware check:
    if metric_fn is not None:
        quality_before = compute_cavity_metric_quality(old_tris, metric_fn)
        quality_after = compute_cavity_metric_quality(new_tris, metric_fn)
        
        if quality_after < quality_before:  # Metric improvement
            accept_retriangulation = True
    else:
        # Fallback to Euclidean quality (current behavior)
        # ...
```

**Workaround**: Currently disabled. Use edge flipping + collapse for topology optimization.

### Smoothing Strategy

Laplacian smoothing applied every 2 iterations:
- **ω = 0.3** (relaxation factor)
- **3 passes** per smoothing step
- **Interior vertices only** (boundary fixed)
- Improves min angle by ~5×

### Edge Flipping

Metric-based flipping with area preservation:
- Check variance of metric edge lengths in diamond
- Accept flip if variance decreases
- Rollback if area changes > 1%
- Typically 20-30 flips per run

## References

1. **Alauzet, F., & Loseille, A.** (2016). "A decade of progress on anisotropic mesh adaptation for computational fluid dynamics." *CAD Computer Aided Design*, 72, 13-39.

2. **Loseille, A., & Alauzet, F.** (2011). "Continuous mesh framework part I: well-posed continuous interpolation error." *SIAM Journal on Numerical Analysis*, 49(1), 38-60.

3. **Frey, P. J., & Alauzet, F.** (2005). "Anisotropic mesh adaptation for CFD computations." *Computer Methods in Applied Mechanics and Engineering*, 194(48-49), 5068-5082.

## Troubleshooting

### Too many vertices

**Problem:** Final mesh has many more vertices than target.

**Solution:** Increase `--alpha` (e.g., from 1.3 to 2.0) to reduce splitting aggressiveness.

### Poor convergence

**Problem:** Max L_M doesn't decrease below 2.0.

**Solution:** 
- Increase `--max-iter` (e.g., to 60)
- Decrease `--alpha` slightly (e.g., to 1.2)
- Check that smoothing is enabled

### Mesh quality issues

**Problem:** Very small angles or degenerate elements.

**Solution:**
- Reduce anisotropy ratio in `simple_metric()` (h_max/h_min)
- Increase smoothing frequency
- Enable quality enforcement (may limit anisotropy)

### Non-conforming mesh

**Problem:** Warning about non-conforming edges.

**Solution:**
- Usually resolved by compaction
- If persistent, reduce `--alpha` and `--beta` gap
- Check boundary adaptation

## License

Part of the SOFIA mesh adaptation framework.
