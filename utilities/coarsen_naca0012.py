#!/usr/bin/env python3
"""
Coarsen NACA0012 mesh using edge collapse operations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sofia.core.io import read_msh
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.quality import mesh_min_angle


def identify_boundary_vertices(editor):
    """Identify boundary vertices."""
    editor._update_maps()
    boundary_vertices = set()
    for (u, v), tris in editor.edge_map.items():
        if len(tris) == 1:  # Boundary edge
            boundary_vertices.add(u)
            boundary_vertices.add(v)
    return boundary_vertices


def coarsen_mesh(editor, target_triangles=5000, protect_boundary=True):
    """Coarsen mesh by collapsing short interior edges.
    
    Args:
        editor: Mesh editor
        target_triangles: Target number of triangles
        protect_boundary: If True, avoid collapsing boundary edges
        
    Returns:
        Number of successful collapses
    """
    print(f"\nCoarsening mesh from {len(editor.triangles)} to ~{target_triangles} triangles...")
    
    if protect_boundary:
        boundary_vertices = identify_boundary_vertices(editor)
        print(f"  Protecting {len(boundary_vertices)} boundary vertices")
    else:
        boundary_vertices = set()
    
    n_collapses = 0
    max_iterations = 20
    
    for iteration in range(max_iterations):
        current_tris = len([t for t in editor.triangles if not np.all(t == -1)])
        
        if current_tris <= target_triangles:
            print(f"  Reached target: {current_tris} triangles")
            break
        
        print(f"\n  Iteration {iteration + 1}/{max_iterations}:")
        print(f"    Current: {current_tris} triangles")
        
        editor._update_maps()
        
        # Collect all edges with their lengths
        edge_lengths = []
        for (u, v), tris in editor.edge_map.items():
            # Skip boundary edges if protecting boundary
            if protect_boundary and (u in boundary_vertices or v in boundary_vertices):
                # Allow collapse of boundary edges only if both vertices are on boundary
                if not (u in boundary_vertices and v in boundary_vertices):
                    continue
            
            p1, p2 = editor.points[u], editor.points[v]
            length = np.linalg.norm(p2 - p1)
            edge_lengths.append((length, (u, v)))
        
        # Sort by length (shortest first)
        edge_lengths.sort()
        
        # Try to collapse shortest edges
        collapses_this_iter = 0
        max_attempts = min(1000, len(edge_lengths))
        
        for i, (length, edge) in enumerate(edge_lengths[:max_attempts]):
            if collapses_this_iter >= 500:  # Limit per iteration
                break
            
            try:
                ok, msg, _ = editor.edge_collapse(edge=edge)
                if ok:
                    collapses_this_iter += 1
                    n_collapses += 1
                    
                    # Update maps periodically
                    if collapses_this_iter % 100 == 0:
                        editor._update_maps()
            except Exception as e:
                pass
        
        print(f"    Collapsed: {collapses_this_iter} edges")
        
        if collapses_this_iter == 0:
            print("    No more edges can be collapsed")
            break
    
    return n_collapses


def write_msh_v2(filepath, points, triangles):
    """Write Gmsh v2.2 ASCII format."""
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


def main():
    print("=" * 70)
    print("NACA0012 Mesh Coarsening")
    print("=" * 70)
    
    # Load mesh
    print("\n1. Loading mesh from meshes/naca0012.msh...")
    points, triangles = read_msh('meshes/naca0012.msh', verbose=False)
    print(f"   Initial: {len(points)} vertices, {len(triangles)} triangles")
    
    editor = PatchBasedMeshEditor(points, triangles)
    
    initial_min_angle = mesh_min_angle(editor.points, editor.triangles)
    print(f"   Initial min angle: {initial_min_angle:.2f}°")
    
    # Coarsen
    print("\n2. Coarsening mesh...")
    target_tris = 5000  # Target number of triangles
    n_collapses = coarsen_mesh(editor, target_triangles=target_tris, protect_boundary=True)
    
    print(f"\n   Total collapses: {n_collapses}")
    
    # Compact
    print("\n3. Compacting mesh...")
    if editor.has_tombstones():
        editor.compact_triangle_indices()
    
    final_min_angle = mesh_min_angle(editor.points, editor.triangles)
    print(f"   Final: {len(editor.points)} vertices, {len(editor.triangles)} triangles")
    print(f"   Final min angle: {final_min_angle:.2f}°")
    print(f"   Reduction: {len(triangles)} -> {len(editor.triangles)} triangles")
    print(f"   Ratio: {100.0 * len(editor.triangles) / len(triangles):.1f}%")
    
    # Save
    output_file = 'meshes/naca0012_coarse.msh'
    print(f"\n4. Saving to {output_file}...")
    write_msh_v2(output_file, editor.points, editor.triangles)
    print(f"   Saved: {output_file}")
    
    print("\n" + "=" * 70)
    print("Coarsening complete!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
