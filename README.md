# SOFIA - Scalable Operators for Field-driven Iso/Ani Adaptation

<div align="center">

**A Modern 2D Triangular Mesh Modification Library**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18492172.svg)](https://doi.org/10.5281/zenodo.18492172)

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •
[Examples](#-examples) •
[Roadmap](#-roadmap)

</div>

---

## What is SOFIA?

SOFIA is a **pure Python** library for **2D triangular mesh modification** and **remeshing**. It provides robust, high-quality local operations to edit, refine, and optimize meshes while maintaining mesh conformity and quality.

Perfect for:
- **Computational scientists** working with 2D simulations
- **Computer graphics** professionals needing mesh processing
- **Numerical analysis** requiring adaptive mesh refinement
- **Researchers** in mesh generation and optimization

---

## Features

### Core Operations
- **Edge Split** - Midpoint and Delaunay-based splitting
- **Edge Collapse** - Quality-preserving mesh coarsening
- **Edge Flip** - Improve triangle quality through flipping
- **Vertex Insertion/Removal** - Add or remove mesh vertices
- **Pocket Filling** - Fill holes in meshes intelligently
- **Boundary Operations** - Safe manipulation of mesh boundaries

### Quality & Robustness
- **Quality Metrics** - Min angle, area, shape measures
- **Conformity Checks** - Ensure valid mesh topology
- **Amortized Validation** - Fast incremental checking
- **Strict Mode** - Optional rigorous validation
- **Area Preservation** - Maintain geometric properties

### Workflows
- **Greedy Remeshing** - Iterative quality improvement
-  **Patch-based Processing** - Batch operations on mesh regions
- **Visualization** - Built-in plotting with Matplotlib
- **Extensive Testing** - 50+ unit tests for reliability

---

## Installation

### Using pip (Recommended)

```bash
pip install sofia-mesh
```

### From Source

```bash
git clone https://github.com/youssef-mesri/sofia.git
cd sofia
pip install -e .
```

### Requirements
- Python 3.8 or higher
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Matplotlib ≥ 3.3

---

## Quick Start

### Basic Example - Improve Mesh Quality

```python
import numpy as np
from sofia import PatchBasedMeshEditor

# Create a simple mesh
points = np.array([
    [0.0, 0.0], [1.0, 0.0], [0.5, 0.8],
    [0.0, 1.0], [1.0, 1.0]
])
triangles = np.array([
    [0, 1, 2], [0, 2, 3], [1, 4, 2], [3, 2, 4]
])

# Initialize editor
editor = PatchBasedMeshEditor(points, triangles)

# Split an edge to refine the mesh
success = editor.split_edge(edge_id=0, method='midpoint')
print(f"Edge split: {success}")

# Get the updated mesh
new_points = editor.pts
new_triangles = editor.tris

# Check mesh quality
min_angle = editor.mesh_min_angle()
print(f"Minimum angle: {min_angle:.2f}°")
```

### Remeshing Example

```python
from sofia.core.remesh_driver import greedy_remesh

# Iteratively improve mesh quality
improved_editor = greedy_remesh(
    editor,
    target_min_angle=25.0,  # Target minimum angle in degrees
    max_iterations=100
)

print(f"Final quality: {improved_editor.mesh_min_angle():.2f}°")
```

---

## Documentation

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Examples](examples/)** - Practical usage examples
- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[Publication Strategy](docs/PUBLICATION_STRATEGY.md)** - Development roadmap

---

## Examples

Check out the `examples/` directory for more:

- `basic_remeshing.py` - Simple mesh refinement
- `quality_improvement.py` - Iterative quality optimization
- `boundary_operations.py` - Boundary manipulation

---

## Roadmap

### Current (v0.1.0) - Python Implementation
- Pure Python implementation
- All core features working
- Suitable for meshes up to ~10K triangles
- Easy installation, no compilation needed

### Next (v2.0) - High-Performance C++ Backend
- **20-50x performance improvement** 🚀
- Transparent integration (same API)
- Suitable for meshes up to 1M+ triangles
- Optional: falls back to Python if C++ unavailable
- **Status:** In development (private branch)

### Future (v3.0+)
- GPU acceleration
- 3D tetrahedral mesh support
- Parallel batch processing
- Advanced anisotropic remeshing

---

## Citation

If you use SOFIA in your research, please cite:

```bibtex
@software{sofia2025,
  author = {Mesri, Youssef},
    title = {SOFIA: Scalable Operators for Field-driven Iso/Ani Adaptation},
  year = {2025},
  url = {https://github.com/youssef-mesri/sofia},
  version = {1.0.0}
}
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- **Found a bug?** [Open an issue](https://github.com/youssef-mesri/sofia/issues)
- **Have an idea?** [Start a discussion](https://github.com/youssef-mesri/sofia/discussions)
- **Want to contribute?** [Submit a pull request](https://github.com/youssef-mesri/sofia/pulls)

---

## License

SOFIA is released under the [MIT License](LICENSE).

---

## Acknowledgments

- Inspired by classical mesh modification algorithms
- Built with modern Python best practices
- Developed for research in computational geometry

---

## Contact

**Author:** Youssef Mesri

- **Issues:** [GitHub Issues](https://github.com/youssef-mesri/sofia/issues)
- **Discussions:** [GitHub Discussions](https://github.com/youssef-mesri/sofia/discussions)
- **Email:** [youssef.mesri@minesparis.psl.eu](mailto:youssef.mesri@minesparis.psl.eu)
- **ORCID:** [0000-0002-5136-5435](https://orcid.org/0000-0002-5136-5435) 

---

<div align="center">

**Made with love for the computational geometry community**

Star us on GitHub if you find SOFIA useful!

</div>
