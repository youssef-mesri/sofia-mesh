#!/usr/bin/env python3
"""
Convert SU2 mesh format to Gmsh v2.2 MSH format.
Supports 2D triangular and quadrilateral elements.
"""

import numpy as np
import sys


def read_su2_mesh(filepath):
    """Read SU2 mesh file and extract triangular elements.
    
    SU2 element types:
    - 3: Line
    - 5: Triangle  
    - 9: Quadrilateral
    
    Args:
        filepath: Path to .su2 file
        
    Returns:
        points: (N, 2) array of coordinates
        triangles: (M, 3) array of triangle connectivity (0-indexed)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    ndim = None
    nelem = None
    npoin = None
    
    elements = []
    points = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('NDIME='):
            ndim = int(line.split('=')[1])
            print(f"Dimension: {ndim}")
            
        elif line.startswith('NELEM='):
            nelem = int(line.split('=')[1])
            print(f"Elements: {nelem}")
            i += 1
            
            # Read elements
            for j in range(nelem):
                parts = lines[i + j].strip().split()
                elem_type = int(parts[0])
                
                if elem_type == 5:  # Triangle
                    v0, v1, v2 = int(parts[1]), int(parts[2]), int(parts[3])
                    elements.append([v0, v1, v2])
                    
                elif elem_type == 9:  # Quadrilateral - split into 2 triangles
                    v0, v1, v2, v3 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    # Split quad into triangles: (v0, v1, v2) and (v0, v2, v3)
                    elements.append([v0, v1, v2])
                    elements.append([v0, v2, v3])
                    
            i += nelem
            continue
            
        elif line.startswith('NPOIN='):
            npoin = int(line.split('=')[1])
            print(f"Points: {npoin}")
            i += 1
            
            # Read points
            for j in range(npoin):
                parts = lines[i + j].strip().split()
                x = float(parts[0])
                y = float(parts[1])
                # parts[2] is the point index (ignored, we use line order)
                points.append([x, y])
                
            i += npoin
            continue
            
        i += 1
    
    points = np.array(points, dtype=np.float64)
    triangles = np.array(elements, dtype=np.int32)
    
    print(f"\nConverted mesh:")
    print(f"  Vertices: {len(points)}")
    print(f"  Triangles: {len(triangles)}")
    
    return points, triangles


def write_msh_v2(filepath, points, triangles):
    """Write Gmsh v2.2 ASCII format.
    
    Args:
        filepath: Output .msh file path
        points: (N, 2) array of coordinates
        triangles: (M, 3) array of triangle connectivity (0-indexed)
    """
    with open(filepath, 'w') as f:
        # Header
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # Nodes (1-indexed in Gmsh)
        f.write("$Nodes\n")
        f.write(f"{len(points)}\n")
        for i, (x, y) in enumerate(points):
            f.write(f"{i+1} {x:.16e} {y:.16e} 0.0\n")
        f.write("$EndNodes\n")
        
        # Elements (triangles only, type 2)
        f.write("$Elements\n")
        f.write(f"{len(triangles)}\n")
        for i, (v0, v1, v2) in enumerate(triangles):
            # elem_num, type=2 (triangle), ntags=0, v0+1, v1+1, v2+1 (1-indexed)
            f.write(f"{i+1} 2 0 {v0+1} {v1+1} {v2+1}\n")
        f.write("$EndElements\n")
    
    print(f"Saved to: {filepath}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_su2_to_msh.py <input.su2> [output.msh]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = 'meshes/' + os.path.basename(input_file.replace('.su2', '.msh'))
    
    print(f"Converting: {input_file} -> {output_file}")
    print("-" * 60)
    
    points, triangles = read_su2_mesh(input_file)
    write_msh_v2(output_file, points, triangles)
    
    print("-" * 60)
    print("Conversion complete!")


if __name__ == '__main__':
    main()
