---
title: 'SOFIA: A Python Library for High-Quality 2D Triangular Mesh Adaptation'
tags:
  - Python
  - mesh generation
  - triangular mesh
  - computational geometry
  - mesh adaptation
  - finite element method
authors:
  - name: Youssef Mesri
    orcid: https://orcid.org/0000-0002-5136-5435
    affiliation: 1
affiliations:
 - name: MINES Paris - PSL, France
   index: 1
date: 4 November 2025
bibliography: paper.bib
---

# Summary

Triangular meshes are fundamental data structures in computational science and engineering, serving as the backbone for numerical simulations in fluid dynamics, structural analysis, heat transfer, and many other fields. The quality of a mesh directly impacts the accuracy, stability, and convergence of numerical solutions [@mesri2008dynamic;@manzinali2018adaptive;@Persson2004]. `SOFIA` (Scalable Operators for Field-driven Iso/Ani Adaptation) is a Python library that provides robust and efficient tools for 2D triangular mesh modification, refinement, and quality improvement through local topological operations, with particular emphasis on **anisotropic mesh adaptation** and **automatic boundary preservation**.

# Statement of Need

While several mesh generation tools exist [@Shewchuk1996;@Geuzaine2009;@MeshPy;@PyMesh], there is a gap in the Python ecosystem for a lightweight, well-documented library focused specifically on **mesh adaptation** and **local modification operations**. Existing solutions either require complex dependencies (C/C++ bindings), lack comprehensive documentation, or focus primarily on initial mesh generation rather than adaptive remeshing during simulation workflows.

`SOFIA` addresses these needs by providing:

1. **Pure Python implementation with optional C++ acceleration**: Easy to install and modify, with optional C++ backend for performance-critical operations
2. **Local topological operations**: Fine-grained control over mesh modification through fundamental operations (split, collapse, flip, add, remove)
3. **Anisotropic mesh adaptation**: Metric-based adaptation with automatic handling of highly stretched elements near features (boundary layers, shocks, fronts)
4. **Automatic boundary preservation**: Edge collapse that automatically detects and preserves domain boundaries
5. **Quality-driven adaptation**: Built-in quality metrics and optimization strategies handling both isotropic/anisotropic meshes
6. **Boundary-aware operations**: Safe manipulation of mesh boundaries with conformity preservation
7. **Production-ready reliability**: Extensive test suite (100+ unit tests) ensuring robustness

The library is designed for computational scientists, researchers, and engineers who need to adaptively refine/coarsen meshes during simulations, optimize existing meshes, implement custom mesh adaptation strategies, or generate boundary layer meshes for high-Reynolds number flows.

# Functionality

## Core Operations

`SOFIA` implements the fundamental local operations for triangular mesh modification:

- **Edge operations**: Split edges at midpoint or custom locations, collapse edges, and flip edges
- **Vertex operations**: Insert and remove vertices using cavity re-triangulation
- **Pocket filling**: Fill holes in meshes after vertex removal or other operations
- **Boundary operations**: Safely manipulate boundary edges and vertices while maintaining domain conformity
- **Automatic boundary detection**: Edge collapse automatically detect boundary vertices and preserve domain geometry.

## Anisotropic Mesh Adaptation

A key feature of `SOFIA` is its support for anisotropic mesh adaptation, crucial for capturing directional features [@mesri2006continuous;@mesri2008dynamic;@Alauzet2010;@Loseille2011;@mesri2016optimal]:

- **Metric tensor field**: User-defined symmetric positive-definite tensor field specifying desired mesh resolution and anisotropy
- **Metric-based edge lengths**: Operations use metric edge length $L_M(e) = \sqrt{(p_2-p_1)^T M (p_2-p_1)}$ instead of Euclidean distance
- **Boundary layer support**: Natural support for highly anisotropic elements near boundaries (aspect ratios up to 10:1 or higher)
- **Smooth transitions**: Metric fields can specify smooth transitions from anisotropic to isotropic regions

### Key Innovation: Boundary-aware Edge Collapse

Traditional edge collapse operations use the midpoint of the collapsed edge, which can deform domain boundaries. `SOFIA` implements an edge collapse that:

1. **Automatically detects boundary vertices** using topological information (edges with only one incident triangle)
2. **Preserves boundaries** by collapsing to the boundary vertex instead of the midpoint when one or both endpoints lie on the boundary
3. **Requires no manual vertex protection** the boundary preservation is built into the operation itself
4. **Maintains geometric accuracy** with zero deviation from straight boundaries

This innovation is particularly important for anisotropic remeshing, where many edges near boundaries need to be collapsed while maintaining domain geometry [@mesri2012automatic].

### Boundary Layer Insertion and Adaptation

For high-Reynolds number flow simulations and other applications requiring fine resolution near boundaries, `SOFIA` provides boundary layer mesh generation capabilities:

- **Structured layer insertion**: Insert vertices at geometric progression distances from boundaries (e.g., $y_i = y_0 \cdot r^i$)
- **Direction-aware metrics**: Compute normal and tangential directions to boundaries for proper metric alignment 
- **Progressive refinement**: Smooth transition from very fine resolution at walls to coarse resolution elsewhere

The workflow combines explicit boundary layer construction with metric-driven adaptation for High fedility simulations.

## Quality Management

The library provides quality assessment and improvement tools with separate handling for isotropic/anisotropic meshes:

- **Quality metrics**: Minimum angle, area ratios, aspect ratios, and shape measures for triangles
- **Metric-based validation**: For anisotropic meshes, quality assessed using metric edge length distribution rather than geometric angles
- **Quality-driven refinement**: Adaptive refinement based on element size, quality thresholds, or user-defined criteria
- **Optimization strategies**: Vertex smoothing with boundary protection, edge flipping cascades, and iterative quality improvement
- **Validation framework**: Incremental topology checking and optional strict validation mode

## Adaptive Refinement Workflows

`SOFIA` supports multiple adaptation strategies for both isotropic/anisotropic meshes:

- **Isotropic refinement**: Uniform or area-based refinement maintaining equilateral triangles
- **Anisotropic refinement**: Metric-driven adaptation with split/collapse/smooth cycles
- **Edge-based refinement**: Selectively refine specific edges (e.g., boundary refinement)
- **Quality-based refinement**: Target low-quality elements for improvement
- **Combined workflows**: Multi-criteria refinement for complex adaptation scenarios
- **Mesh coarsening**: Reduce element count while preserving geometric features and quality
- **Boundary layer generation**: Create highly stretched elements near boundaries for viscous flow simulations

## Integration and Extensibility

The modular architecture allows easy integration into existing simulation pipelines:

- **Simple API**: Intuitive mesh editor interface with clear operation semantics
- **Batch operations**: Efficient processing of multiple mesh regions
- **Visualization tools**: Built-in plotting with Matplotlib for analysis and debugging
- **Extensible framework**: Easy to add custom quality metrics or adaptation strategies

# Examples

The repository includes comprehensive examples demonstrating various use cases. Each example includes visualization and quantitative metrics. For instance, the simple anisotropic remeshing achieves perfect boundary preservation (max deviation = 0.00e+00) while reducing approximation error by 3×, demonstrating the effectiveness of automatic boundary detection in edge collapse operations [@manzinali2018adaptive;@mesri2012automatic].

# Performance and Testing

`SOFIA` prioritizes reliability and correctness:

- **Comprehensive test suite**: Over 100 unit tests covering all operations and edge cases
- **Continuous validation**: Tests run on every commit to ensure stability
- **Topology verification**: Built-in checks for mesh conformity and validity
- **Benchmarking**: Performance tests for batch operations and large-scale refinement

The library handles meshes ranging from tens to thousands of elements efficiently. For anisotropic remeshing, typical performance on a 1000-element mesh:
- 10-20 iterations to convergence
- ~300-400 split operations (long edges in metric space)
- ~200-300 collapse operations (short edges, with automatic boundary preservation)
- Total execution time: <5 seconds on modern hardware

# Comparison with Existing Tools

`SOFIA` complements and extends existing tools in the Python ecosystem [@Shewchuk1996;@MeshPy;@PyMesh;@Logg2012]:

- **Triangle/MeshPy**: Focused on initial mesh generation; `SOFIA` specializes in adaptation and modification, with anisotropic support
- **PyMesh**: Requires C++ dependencies; `SOFIA` offers pure Python with optional acceleration
- **CGAL bindings**: Complex API; `SOFIA` provides simpler, Pythonic interface
- **FEniCS/Firedrake mesh tools**: Tightly coupled to FEM frameworks; `SOFIA` is framework-agnostic
- **MMG/BAMG**: C/Fortran anisotropic remeshers; `SOFIA` offers Python integration and automatic boundary preservation

Key differentiators:
- **Automatic boundary preservation** in edge collapse (unique feature)
- **Adaptive quality checking** (different criteria for isotropic vs anisotropic)
- **Pure Python flexibility** with optional C++ performance
- **Educational value** with clear, readable implementations

# Future Development

Ongoing and planned enhancements include:
- **3D tetrahedral mesh support**: Extend core operations to 3D meshes

# Acknowledgements

We acknowledge contributions from the computational geometry and mesh generation communities, and thank the reviewers for their valuable feedback.

# References
