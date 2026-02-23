# SOFIA API Documentation

Complete reference for SOFIA's mesh modification and analysis tools.

## Table of Contents

- [Core Classes](#core-classes)
  - [PatchBasedMeshEditor](#patchbasedmesheditor)
- [Mesh Operations](#mesh-operations)
  - [Edge Operations](#edge-operations)
  - [Vertex Operations](#vertex-operations)
  - [Boundary Operations](#boundary-operations)
- [Quality and Analysis](#quality-and-analysis)
- [Remeshing](#remeshing)
- [I/O Operations](#io-operations)
- [Utilities](#utilities)

---

## Core Classes

### PatchBasedMeshEditor

The main class for mesh editing operations.

#### Constructor

```python
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor

editor = PatchBasedMeshEditor(points, triangles)
```

**Parameters:**
- `points` (np.ndarray): Nx2 array of vertex coordinates
- `triangles` (np.ndarray): Mx3 array of triangle vertex indices

**Attributes:**
- `points` (np.ndarray): Current mesh vertices
- `triangles` (np.ndarray): Current mesh triangles (inactive triangles marked with -1)
- `edge_map` (dict): Mapping from edges (sorted tuples) to triangle indices
- `stats` (dict): Statistics about operations performed

**Example:**
```python
import numpy as np
from scipy.spatial import Delaunay

# Create initial points
points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
tri = Delaunay(points)

# Create editor
editor = PatchBasedMeshEditor(points, tri.simplices)
```

---

## Mesh Operations

### Edge Operations

#### `split_edge(edge, split_point=None, config=None)`

Split an edge, optionally at a specified point.

**Parameters:**
- `edge` (tuple): Edge as (v1, v2) where v1 < v2
- `split_point` (np.ndarray, optional): Coordinates for new vertex (default: midpoint)
- `config` (SplitConfig, optional): Configuration options

**Returns:**
- `success` (bool): True if operation succeeded
- `new_vertex` (int or None): Index of newly created vertex
- `info` (dict): Additional information about the operation

**Example:**
```python
# Split edge at midpoint
success, new_v, info = editor.split_edge((0, 1))

# Split edge at specific point
split_point = np.array([0.3, 0.5])
success, new_v, info = editor.split_edge((0, 1), split_point=split_point)
```

---

#### `edge_collapse(edge, target_point=None, config=None)`

Collapse an edge, merging two vertices into one.

**Parameters:**
- `edge` (tuple): Edge as (v1, v2) to collapse
- `target_point` (np.ndarray, optional): Position for merged vertex (default: midpoint)
- `config` (CollapseConfig, optional): Configuration options

**Returns:**
- `success` (bool): True if operation succeeded
- `kept_vertex` (int or None): Index of the vertex that remains
- `info` (dict): Additional information about the operation

**Example:**
```python
# Collapse edge to midpoint
success, kept_v, info = editor.edge_collapse((0, 1))

# Collapse edge to specific point
target = np.array([0.2, 0.3])
success, kept_v, info = editor.edge_collapse((0, 1), target_point=target)
```

---

#### `flip_edge(edge, config=None)`

Flip an edge between two triangles.

**Parameters:**
- `edge` (tuple): Edge as (v1, v2) to flip
- `config` (FlipConfig, optional): Configuration options

**Returns:**
- `success` (bool): True if operation succeeded
- `new_edge` (tuple or None): The new edge after flipping
- `info` (dict): Additional information about the operation

**Example:**
```python
# Flip an edge
success, new_edge, info = editor.flip_edge((0, 1))
```

---

### Vertex Operations

#### `insert_vertex(point, config=None)`

Insert a new vertex into the mesh.

**Parameters:**
- `point` (np.ndarray): Coordinates [x, y] for new vertex
- `config` (InsertConfig, optional): Configuration options

**Returns:**
- `success` (bool): True if operation succeeded
- `new_vertex` (int or None): Index of newly inserted vertex
- `info` (dict): Additional information about the operation

**Example:**
```python
# Insert vertex at specific location
new_point = np.array([0.5, 0.5])
success, new_v, info = editor.insert_vertex(new_point)
```

---

#### `remove_vertex(vertex, config=None)`

Remove a vertex and retriangulate the resulting hole.

**Parameters:**
- `vertex` (int): Index of vertex to remove
- `config` (RemoveConfig, optional): Configuration options

**Returns:**
- `success` (bool): True if operation succeeded
- `removed_vertex` (int or None): Index of removed vertex
- `info` (dict): Additional information about the operation

**Example:**
```python
# Remove a vertex
success, removed_v, info = editor.remove_vertex(5)
```

---

### Boundary Operations

#### `split_boundary_edge(edge, split_point=None)`

Split an edge on the mesh boundary.

**Parameters:**
- `edge` (tuple): Boundary edge as (v1, v2)
- `split_point` (np.ndarray, optional): Position for split (default: midpoint)

**Returns:**
- `success` (bool): True if operation succeeded
- `new_vertex` (int or None): Index of newly created vertex
- `info` (dict): Additional information

**Example:**
```python
# Split a boundary edge
success, new_v, info = editor.split_boundary_edge((0, 1))
```

---

#### `remove_boundary_vertex(vertex, config=None)`

Remove a vertex on the mesh boundary.

**Parameters:**
- `vertex` (int): Index of boundary vertex to remove
- `config` (BoundaryRemoveConfig, optional): Configuration options

**Returns:**
- `success` (bool): True if operation succeeded
- `removed_vertex` (int or None): Index of removed vertex
- `info` (dict): Additional information

**Example:**
```python
from sofia.core.config import BoundaryRemoveConfig

# Remove boundary vertex with configuration
config = BoundaryRemoveConfig(require_area_preservation=True)
success, removed_v, info = editor.remove_boundary_vertex(3, config=config)
```

---

## Quality and Analysis

### Quality Metrics

```python
from sofia.core.quality import mesh_min_angle, triangle_quality

# Compute minimum angle in entire mesh
min_angle = mesh_min_angle(points, triangles)

# Compute quality metrics for specific triangle
quality = triangle_quality(points, triangle)
```

#### `mesh_min_angle(points, triangles)`

Compute the minimum angle across all active triangles.

**Parameters:**
- `points` (np.ndarray): Mesh vertices
- `triangles` (np.ndarray): Mesh triangles

**Returns:**
- `float`: Minimum angle in degrees

---

#### `triangle_quality(points, triangle)`

Compute quality metrics for a single triangle.

**Parameters:**
- `points` (np.ndarray): Mesh vertices
- `triangle` (np.ndarray): Triangle vertex indices [v1, v2, v3]

**Returns:**
- `dict`: Quality metrics including:
  - `min_angle`: Minimum angle in degrees
  - `area`: Triangle area
  - `aspect_ratio`: Ratio of circumradius to inradius

---

### Conformity Checking

```python
from sofia.core.conformity import check_mesh_conformity

# Check if mesh is conforming
is_valid, message = check_mesh_conformity(points, triangles)
```

#### `check_mesh_conformity(points, triangles)`

Verify mesh conformity (each edge shared by at most 2 triangles).

**Parameters:**
- `points` (np.ndarray): Mesh vertices
- `triangles` (np.ndarray): Mesh triangles

**Returns:**
- `is_valid` (bool): True if mesh is conforming
- `message` (str): Description of conformity status

---

## Remeshing

### Anisotropic Remeshing

```python
from sofia.core.anisotropic_remesh import anisotropic_local_remesh

# Define metric field
def metric_function(x):
    """Return 2x2 metric tensor at point x"""
    # Example: anisotropic metric
    return np.array([[1.0, 0.0], [0.0, 100.0]])

# Perform anisotropic remeshing
editor = anisotropic_local_remesh(
    editor,
    metric_fn=metric_function,
    alpha=0.8,    # Split threshold
    beta=0.7,     # Collapse threshold
    max_iter=10
)
```

#### `anisotropic_local_remesh(editor, metric_fn, alpha, beta, max_iter)`

Perform metric-based anisotropic mesh adaptation.

**Parameters:**
- `editor` (PatchBasedMeshEditor): Mesh editor
- `metric_fn` (callable): Function returning 2x2 metric tensor at each point
- `alpha` (float): Threshold for edge splitting (L_M > alpha)
- `beta` (float): Threshold for edge collapse (L_M < beta)
- `max_iter` (int): Maximum number of iterations

**Returns:**
- `PatchBasedMeshEditor`: Modified mesh editor

---

### Greedy Remeshing

```python
from sofia.core.remesh_driver import greedy_remesh

# Perform greedy remeshing with target edge length
editor = greedy_remesh(
    editor,
    target_length=0.1,
    max_iter=10,
    strict_mode=False
)
```

#### `greedy_remesh(editor, target_length, max_iter, strict_mode)`

Iterative mesh refinement/coarsening to achieve target edge length.

**Parameters:**
- `editor` (PatchBasedMeshEditor): Mesh editor
- `target_length` (float): Target edge length
- `max_iter` (int): Maximum iterations
- `strict_mode` (bool): Enable strict quality checking

**Returns:**
- `PatchBasedMeshEditor`: Modified mesh editor

---

## I/O Operations

### Reading and Writing Meshes

```python
from sofia.core.io import read_msh, write_vtk

# Read mesh from Gmsh .msh file
points, triangles = read_msh('mesh.msh')

# Write mesh to VTK format for visualization
write_vtk('output.vtk', points, triangles)
```

#### Supported Formats

- **MSH**: Gmsh format (`.msh` files, versions 2.2 and 4.1, ASCII mode)
- **VTK**: VTK legacy format (`.vtk` files) for ParaView/VisIt visualization

**Note:** OBJ and MEDIT formats are not currently implemented.

---

## Utilities

### Building Random Delaunay Meshes

```python
from sofia.core.mesh_modifier2 import build_random_delaunay

# Create random Delaunay mesh in unit square
editor = build_random_delaunay(n_points=100, seed=42)
```

#### `build_random_delaunay(n_points, seed, bounds)`

Create a random Delaunay triangulation.

**Parameters:**
- `n_points` (int): Number of random points
- `seed` (int, optional): Random seed for reproducibility
- `bounds` (tuple, optional): Domain bounds as ((xmin, xmax), (ymin, ymax))

**Returns:**
- `PatchBasedMeshEditor`: Mesh editor with random Delaunay mesh

---

### Visualization

```python
from sofia.core.visualization import plot_mesh

# Visualize mesh
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plot_mesh(editor.points, editor.triangles, ax=ax)
plt.show()
```

#### `plot_mesh(points, triangles, ax, show_vertices, show_edges)`

Plot a triangular mesh.

**Parameters:**
- `points` (np.ndarray): Mesh vertices
- `triangles` (np.ndarray): Mesh triangles
- `ax` (matplotlib.axes.Axes): Matplotlib axis
- `show_vertices` (bool): Display vertex markers
- `show_edges` (bool): Display mesh edges

**Returns:**
- `matplotlib.axes.Axes`: Modified axis

---

## Configuration Objects

### SplitConfig

```python
from sofia.core.config import SplitConfig

config = SplitConfig(
    enforce_quality=True,
    min_quality_threshold=0.5,
    prevent_boundary_loops=True
)
```

### CollapseConfig

```python
from sofia.core.config import CollapseConfig

config = CollapseConfig(
    enforce_quality=True,
    min_quality_threshold=0.5,
    preserve_boundaries=True
)
```

### RemoveConfig

```python
from sofia.core.config import RemoveConfig

config = RemoveConfig(
    strategy='ear_clip',  # or 'delaunay', 'greedy'
    enforce_quality=True,
    fallback_strategies=['delaunay', 'greedy']
)
```

---

## Advanced Topics

### Custom Removal Strategies

Implement custom strategies for vertex removal by extending the base removal strategy class.

### Batch Operations

Process multiple operations efficiently:

```python
from sofia.core.batch_operations import batch_split_edges

# Split multiple edges at once
edges = [(0, 1), (2, 3), (4, 5)]
results = batch_split_edges(editor, edges)
```

### Statistics Tracking

Access operation statistics:

```python
# Get statistics
stats = editor.stats

print(f"Splits: {stats['splits']}")
print(f"Collapses: {stats['collapses']}")
print(f"Flips: {stats['flips']}")
```

---

## Best Practices

1. **Always check return values**: Operations may fail due to quality or topological constraints
2. **Use configuration objects**: Customize behavior for specific use cases
3. **Update internal maps**: Call `editor._update_maps()` after manual modifications
4. **Validate mesh conformity**: Use `check_mesh_conformity()` to verify mesh integrity
5. **Monitor quality**: Track `mesh_min_angle()` during iterative operations

---

## See Also

- [Examples](../examples/README.md) - Practical usage examples
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Paper](../paper.md) - Theoretical background and algorithms

---

**Last Updated:** February 23, 2026  
**Version:** 0.1.0
