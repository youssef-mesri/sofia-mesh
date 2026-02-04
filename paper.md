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
    orcid: 0000-0002-5136-5435
    affiliation: 1
affiliations:
 - name: MINES Paris - PSL, France
   index: 1
date: 4 November 2025
bibliography: paper.bib
---

# Summary

Triangular meshes are fundamental data structures in computational science and engineering, serving as the backbone for numerical simulations in fluid dynamics, structural analysis, heat transfer, and many other fields. The quality of a mesh directly impacts the accuracy, stability, and convergence of numerical solutions [@mesri2008dynamic;@manzinali2018adaptive;@Persson2004]. `SOFIA` (Scalable Operators for Field-driven Iso/Ani Adaptation) is a Python library that provides robust and efficient tools for 2D triangular mesh modification, refinement, and quality improvement through local topological operations, with particular emphasis on **anisotropic mesh adaptation** and **automatic boundary preservation**.

In its current version, `SOFIA` focuses on **mesh topology modification (h-adaptation)**. That is, it changes the mesh by locally splitting/collapsing/flipping edges and inserting/removing vertices to refine or coarsen the discretisation. Limited vertex relocation is available through smoothing to improve element quality, but `SOFIA` does not currently implement full **mesh movement (r-adaptation)** driven by a PDE-based moving mesh method; such r-adaptation workflows are left to external solvers or future work.

# Statement of Need

While several mesh generation tools exist [@Shewchuk1996;@Geuzaine2009;@MeshPy;@PyMesh], there is a gap in the Python ecosystem for a lightweight, well-documented library focused specifically on **mesh adaptation** and **local modification operations**. Existing solutions either require complex dependencies (C/C++ bindings), lack comprehensive documentation, or focus primarily on initial mesh generation rather than adaptive remeshing during simulation workflows.

`SOFIA` addresses these needs by providing a **pure Python** implementation of:

1. **Local topological operations** (split, collapse, flip, insert, remove) enabling fine-grained mesh modification
2. **Metric-based anisotropic h-adaptation** driven by a user-supplied tensor field (see below)
3. **Automatic boundary preservation** built into edge collapse, so boundary conformity can be maintained without manual vertex “locking”
4. **Quality management** with metrics and optimisation strategies for isotropic and anisotropic meshes
5. **Reliability** supported by an extensive unit test suite

The repository README mentions a future C++ backend; however, the current release is a pure Python package and this paper describes functionality available in the present version.

The library is designed for computational scientists, researchers, and engineers who need to adaptively refine/coarsen meshes during simulations, optimize existing meshes, implement custom mesh adaptation strategies, or generate boundary layer meshes for high-Reynolds number flows.

# Functionality

## Core Operations

`SOFIA` implements the fundamental local operations for triangular mesh modification:

- **Edge operations**: Split edges at midpoint or custom locations, collapse edges, and flip edges
- **Vertex operations**: Insert and remove vertices using cavity re-triangulation
- **Pocket filling**: Fill holes in meshes after vertex removal or other operations
- **Boundary operations**: Safely manipulate boundary edges and vertices while maintaining domain conformity
- **Automatic boundary detection**: Edge collapse operations automatically detect boundary vertices and preserve domain geometry.

## Anisotropic Mesh Adaptation

A key feature of `SOFIA` is its support for anisotropic mesh adaptation, crucial for capturing directional features [@mesri2006continuous;@mesri2008dynamic;@Alauzet2010;@Loseille2011;@mesri2016optimal]:

![Example of an anisotropic adapted triangular mesh produced with `SOFIA`. The adaptation is driven by a user-supplied metric tensor field, resulting in strongly stretched elements aligned with the target features while preserving the domain boundary.](docs/images/anisotropic_remeshing_levelset.png)

- **Metric tensor field**: User-defined symmetric positive-definite tensor field specifying desired mesh resolution and anisotropy
- **Metric-based edge lengths**: Operations use metric edge length $L_M(e) = \sqrt{(p_2-p_1)^T M (p_2-p_1)}$ instead of Euclidean distance
- **Boundary layer support**: Natural support for highly anisotropic elements near boundaries (high aspect ratios)
- **Smooth transitions**: Metric fields can specify smooth transitions from anisotropic to isotropic regions

Here, $p_1$ and $p_2$ are the endpoints of an edge $e$ in physical coordinates, $M$ is the symmetric positive-definite metric tensor (or a suitable edge-averaged metric), and $L_M(e)$ is the target length measure used to decide whether an edge should be split or collapsed. This is a **metric-based h-adaptation** approach (in the spirit of local mesh modification governed by a metric) rather than a hierarchical refinement strategy.

### Key Innovation: Boundary-aware Edge Collapse

Traditional edge collapse operations often collapse to the midpoint of an edge, which can deform domain boundaries. In `SOFIA`, boundary preservation is built into the collapse decision: boundary vertices are detected from topology (boundary edges have only one incident triangle), and when collapsing an edge touching the boundary the operation collapses to the boundary vertex instead of the midpoint. This avoids manual boundary “protection” and helps maintain geometric fidelity during aggressive coarsening near boundaries [@mesri2012automatic].

This innovation is particularly important for anisotropic remeshing, where many edges near boundaries need to be collapsed while maintaining domain geometry [@mesri2012automatic].

### Boundary Layer Insertion and Adaptation

For high-Reynolds number flow simulations and other applications requiring fine resolution near boundaries, `SOFIA` provides boundary layer mesh generation capabilities:

- **Structured layer insertion**: Insert vertices at geometric progression distances from boundaries (e.g., $y_i = y_0 \cdot r^i$)
- **Direction-aware metrics**: Compute normal and tangential directions to boundaries for proper metric alignment 
- **Progressive refinement**: Smooth transition from very fine resolution at walls to coarse resolution elsewhere

The workflow combines explicit boundary layer construction with metric-driven adaptation for high-fidelity simulations.

![Anisotropic mesh adaptation with boundary layer insertion. Panel (a) shows the initial mesh with 106 triangles, while panel (b) demonstrates the adapted mesh with 522 triangles after inserting 4 boundary layers and performing metric-based adaptation. The boundary layers use a geometric progression with first layer height of 0.015 and growth ratio of 1.4, creating highly anisotropic elements near the boundary while maintaining smooth transitions to isotropic regions in the interior.](docs/images/paper_figure_anisotropic_adaptation_v2.png)

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

The library handles meshes ranging from tens to thousands of elements. The repository also includes benchmarks focused on the split/collapse/flip kernels used during anisotropic remeshing.

# State of the Field and Comparison

`SOFIA` complements and extends existing tools in the Python ecosystem:

- **Triangle/MeshPy** [@Shewchuk1996;@MeshPy]: Focused on initial mesh generation; `SOFIA` specialises in adaptation and modification, including anisotropy.
- **PyMesh** [@PyMesh]: Requires substantial C++ dependencies; `SOFIA` aims to remain lightweight and hackable in pure Python.
- **FEniCS/Firedrake mesh tools** [@Logg2012]: Useful within FEM frameworks; for metric-based adaptation workflows connected to Firedrake via PETSc/ParMmg see e.g. [@wallwork2022parmmsg]. `SOFIA` remains framework-agnostic and can be used upstream of different PDE codes.
- **MMG/BAMG and related remeshers** [@MMG;@BAMG]: High-performance anisotropic remeshers; `SOFIA` provides a Python-native workflow and emphasises boundary-aware collapse.

`SOFIA`’s main differentiators are automatic boundary preservation during edge collapse, quality-driven operations with anisotropic-aware checks, and a readable pure-Python codebase that is easy to extend for research and teaching.

# Acknowledgements

We acknowledge contributions from the computational geometry and mesh generation communities, and thank the reviewers for their valuable feedback.

# References
