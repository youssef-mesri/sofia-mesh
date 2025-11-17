"""Lightweight mesh file I/O for SOFIA.

Provides readers/writers for common mesh formats without heavy dependencies:
- read_msh: Import Gmsh .msh format (ASCII, version 2.2 and 4.1)
- write_vtk: Export legacy VTK format for ParaView/VisIt visualization

All functions maintain SOFIA's canonical data format:
    points: (N, 2) float64 array
    triangles: (M, 3) int32 array
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings


def read_msh(filepath: str, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Read a 2D triangular mesh from Gmsh .msh file (ASCII format).
    
    Supports Gmsh format versions 2.2 and 4.1 (ASCII mode only).
    Extracts only 2D triangular elements (type 2 in v2.2, type 2 in v4.1).
    
    Parameters
    ----------
    filepath : str
        Path to .msh file
    verbose : bool, default=False
        Print parsing information
        
    Returns
    -------
    points : (N, 2) ndarray of float64
        Vertex coordinates (x, y)
    triangles : (M, 3) ndarray of int32
        Triangle connectivity (0-indexed)
        
    Raises
    ------
    ValueError
        If file format is unsupported or contains no triangles
    FileNotFoundError
        If file doesn't exist
        
    Notes
    -----
    - Only extracts 2D coordinates (x, y); ignores z if present
    - Converts from Gmsh's 1-indexed to 0-indexed vertices
    - Skips non-triangle elements (lines, quads, tets, etc.)
    
    Examples
    --------
    >>> points, triangles = read_msh('mesh.msh')
    >>> from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
    >>> editor = PatchBasedMeshEditor(points, triangles)
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f]
    
    if not lines:
        raise ValueError(f"Empty file: {filepath}")
    
    # Detect version
    version = None
    for i, line in enumerate(lines):
        if line.startswith('$MeshFormat'):
            version_line = lines[i + 1].split()
            version = float(version_line[0])
            if verbose:
                print(f"Detected Gmsh format version {version}")
            break
    
    if version is None:
        raise ValueError("Could not detect Gmsh format version (no $MeshFormat section)")
    
    # Parse based on version
    if 2.0 <= version < 3.0:
        return _read_msh_v2(lines, verbose)
    elif 4.0 <= version < 5.0:
        return _read_msh_v4(lines, verbose)
    else:
        raise ValueError(f"Unsupported Gmsh format version: {version}")


def _read_msh_v2(lines, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Parse Gmsh format 2.2 (legacy ASCII format)."""
    # Find $Nodes section
    node_start = None
    node_end = None
    for i, line in enumerate(lines):
        if line == '$Nodes':
            node_start = i
        elif line == '$EndNodes':
            node_end = i
            break
    
    if node_start is None or node_end is None:
        raise ValueError("Missing $Nodes section in .msh file")
    
    num_nodes = int(lines[node_start + 1])
    node_coords = {}
    
    for i in range(node_start + 2, node_end):
        parts = lines[i].split()
        node_id = int(parts[0])  # 1-indexed
        x, y = float(parts[1]), float(parts[2])
        node_coords[node_id] = (x, y)
    
    if verbose:
        print(f"Read {len(node_coords)} nodes")
    
    # Find $Elements section
    elem_start = None
    elem_end = None
    for i, line in enumerate(lines):
        if line == '$Elements':
            elem_start = i
        elif line == '$EndElements':
            elem_end = i
            break
    
    if elem_start is None or elem_end is None:
        raise ValueError("Missing $Elements section in .msh file")
    
    num_elements = int(lines[elem_start + 1])
    triangles_list = []
    
    for i in range(elem_start + 2, elem_end):
        parts = lines[i].split()
        elem_type = int(parts[1])
        
        # Element type 2 = 3-node triangle in Gmsh v2
        if elem_type == 2:
            num_tags = int(parts[2])
            # Triangle node IDs start after: elem_id, type, num_tags, tags...
            start_idx = 3 + num_tags
            v0 = int(parts[start_idx])
            v1 = int(parts[start_idx + 1])
            v2 = int(parts[start_idx + 2])
            triangles_list.append([v0, v1, v2])
    
    if not triangles_list:
        raise ValueError("No triangular elements found in .msh file")
    
    if verbose:
        print(f"Read {len(triangles_list)} triangles (out of {num_elements} total elements)")
    
    # Convert to 0-indexed arrays
    node_ids = sorted(node_coords.keys())
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    
    points = np.array([node_coords[nid] for nid in node_ids], dtype=np.float64)
    triangles = np.array([[id_to_idx[v0], id_to_idx[v1], id_to_idx[v2]] 
                          for v0, v1, v2 in triangles_list], dtype=np.int32)
    
    return points, triangles


def _read_msh_v4(lines, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Parse Gmsh format 4.1 (modern ASCII format)."""
    # Find $Nodes section
    node_start = None
    node_end = None
    for i, line in enumerate(lines):
        if line == '$Nodes':
            node_start = i
        elif line == '$EndNodes':
            node_end = i
            break
    
    if node_start is None or node_end is None:
        raise ValueError("Missing $Nodes section in .msh file")
    
    # v4 format: numEntityBlocks numNodes minNodeTag maxNodeTag
    header = lines[node_start + 1].split()
    num_nodes = int(header[1])
    
    node_coords = {}
    i = node_start + 2
    
    while i < node_end:
        # Entity block header: entityDim entityTag parametric numNodesInBlock
        block_header = lines[i].split()
        num_nodes_in_block = int(block_header[3])
        i += 1
        
        # Read node tags
        node_tags = []
        for j in range(num_nodes_in_block):
            node_tags.append(int(lines[i]))
            i += 1
        
        # Read node coordinates
        for j in range(num_nodes_in_block):
            coords = lines[i].split()
            x, y = float(coords[0]), float(coords[1])
            node_coords[node_tags[j]] = (x, y)
            i += 1
    
    if verbose:
        print(f"Read {len(node_coords)} nodes")
    
    # Find $Elements section
    elem_start = None
    elem_end = None
    for i, line in enumerate(lines):
        if line == '$Elements':
            elem_start = i
        elif line == '$EndElements':
            elem_end = i
            break
    
    if elem_start is None or elem_end is None:
        raise ValueError("Missing $Elements section in .msh file")
    
    # v4 format: numEntityBlocks numElements minElementTag maxElementTag
    header = lines[elem_start + 1].split()
    num_elements = int(header[1])
    
    triangles_list = []
    i = elem_start + 2
    
    while i < elem_end:
        # Entity block header: entityDim entityTag elementType numElementsInBlock
        block_header = lines[i].split()
        elem_type = int(block_header[2])
        num_elems_in_block = int(block_header[3])
        i += 1
        
        # Element type 2 = 3-node triangle in Gmsh v4
        if elem_type == 2:
            for j in range(num_elems_in_block):
                parts = lines[i].split()
                # Format: elemTag node1 node2 node3
                v0, v1, v2 = int(parts[1]), int(parts[2]), int(parts[3])
                triangles_list.append([v0, v1, v2])
                i += 1
        else:
            # Skip non-triangle elements
            i += num_elems_in_block
    
    if not triangles_list:
        raise ValueError("No triangular elements found in .msh file")
    
    if verbose:
        print(f"Read {len(triangles_list)} triangles")
    
    # Convert to 0-indexed arrays
    node_ids = sorted(node_coords.keys())
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    
    points = np.array([node_coords[nid] for nid in node_ids], dtype=np.float64)
    triangles = np.array([[id_to_idx[v0], id_to_idx[v1], id_to_idx[v2]] 
                          for v0, v1, v2 in triangles_list], dtype=np.int32)
    
    return points, triangles


def write_vtk(filepath: str, 
              points: np.ndarray, 
              triangles: np.ndarray,
              point_data: Optional[Dict[str, np.ndarray]] = None,
              cell_data: Optional[Dict[str, np.ndarray]] = None,
              title: str = "SOFIA mesh") -> None:
    """Write 2D triangular mesh to legacy VTK format (ASCII).
    
    Exports mesh for visualization in ParaView, VisIt, or other VTK-compatible tools.
    Uses legacy VTK format (simple ASCII, widely supported).
    
    Parameters
    ----------
    filepath : str
        Output .vtk file path
    points : (N, 2) or (N, 3) ndarray
        Vertex coordinates. If 2D, z=0 is added.
    triangles : (M, 3) ndarray
        Triangle connectivity (0-indexed)
    point_data : dict, optional
        Scalar/vector data at vertices. Keys are field names, values are arrays:
        - Scalars: (N,) array
        - Vectors: (N, 2) or (N, 3) array
    cell_data : dict, optional
        Scalar/vector data at triangle centers. Keys are field names, values are arrays:
        - Scalars: (M,) array  
        - Vectors: (M, 2) or (M, 3) array
    title : str, default="SOFIA mesh"
        Dataset title/description
        
    Examples
    --------
    >>> # Basic export
    >>> write_vtk('output.vtk', points, triangles)
    
    >>> # Export with quality metric
    >>> from sofia.core.geometry import triangles_min_angles
    >>> min_angles = triangles_min_angles(points, triangles)
    >>> write_vtk('quality.vtk', points, triangles, 
    ...           cell_data={'min_angle': min_angles})
    
    >>> # Export with vertex displacement
    >>> displacement = new_points - points
    >>> write_vtk('displaced.vtk', new_points, triangles,
    ...           point_data={'displacement': displacement})
    """
    points = np.asarray(points)
    triangles = np.asarray(triangles)
    
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"points must be (N, 2) or (N, 3), got shape {points.shape}")
    
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"triangles must be (M, 3), got shape {triangles.shape}")
    
    # Ensure 3D coordinates (add z=0 if 2D)
    if points.shape[1] == 2:
        points_3d = np.column_stack([points, np.zeros(len(points))])
    else:
        points_3d = points
    
    num_points = len(points_3d)
    num_triangles = len(triangles)
    
    with open(filepath, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 2.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Points
        f.write(f"POINTS {num_points} double\n")
        for pt in points_3d:
            f.write(f"{pt[0]:.16e} {pt[1]:.16e} {pt[2]:.16e}\n")
        
        # Cells (triangles)
        # Format: numIndices v0 v1 v2
        f.write(f"\nCELLS {num_triangles} {num_triangles * 4}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
        
        # Cell types (5 = triangle in VTK)
        f.write(f"\nCELL_TYPES {num_triangles}\n")
        for _ in range(num_triangles):
            f.write("5\n")
        
        # Point data
        if point_data:
            f.write(f"\nPOINT_DATA {num_points}\n")
            for name, data in point_data.items():
                data = np.asarray(data)
                if data.ndim == 1:
                    # Scalar field
                    f.write(f"SCALARS {name} double 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for val in data:
                        f.write(f"{val:.16e}\n")
                elif data.ndim == 2 and data.shape[1] in (2, 3):
                    # Vector field
                    if data.shape[1] == 2:
                        data_3d = np.column_stack([data, np.zeros(len(data))])
                    else:
                        data_3d = data
                    f.write(f"VECTORS {name} double\n")
                    for vec in data_3d:
                        f.write(f"{vec[0]:.16e} {vec[1]:.16e} {vec[2]:.16e}\n")
                else:
                    warnings.warn(f"Skipping point_data['{name}'] with unsupported shape {data.shape}")
        
        # Cell data
        if cell_data:
            f.write(f"\nCELL_DATA {num_triangles}\n")
            for name, data in cell_data.items():
                data = np.asarray(data)
                if data.ndim == 1:
                    # Scalar field
                    f.write(f"SCALARS {name} double 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for val in data:
                        f.write(f"{val:.16e}\n")
                elif data.ndim == 2 and data.shape[1] in (2, 3):
                    # Vector field
                    if data.shape[1] == 2:
                        data_3d = np.column_stack([data, np.zeros(len(data))])
                    else:
                        data_3d = data
                    f.write(f"VECTORS {name} double\n")
                    for vec in data_3d:
                        f.write(f"{vec[0]:.16e} {vec[1]:.16e} {vec[2]:.16e}\n")
                else:
                    warnings.warn(f"Skipping cell_data['{name}'] with unsupported shape {data.shape}")


__all__ = ['read_msh', 'write_vtk']
