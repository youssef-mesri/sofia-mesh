#!/usr/bin/env python3
"""
Aggressively coarsen NACA0012 mesh using both edge_collapse and remove_node.
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


def aggressive_coarsen(editor, target_triangles=3000, protect_boundary=True):
    """Aggressively coarsen mesh using edge_collapse and remove_node.
    
    Args:
        editor: Mesh editor
        target_triangles: Target number of triangles
        protect_boundary: If True, avoid removing boundary vertices
        
    Returns:
        (n_collapses, n_removals)
    """
    print(f"\nAggressive coarsening from {len(editor.triangles)} to ~{target_triangles} triangles...")
    
    if protect_boundary:
        boundary_vertices = identify_boundary_vertices(editor)
        print(f"  Protecting {len(boundary_vertices)} boundary vertices")
    else:
        boundary_vertices = set()
    
    n_collapses = 0
    n_removals = 0
    max_iterations = 15
    
    for iteration in range(max_iterations):
        current_tris = len([t for t in editor.triangles if not np.all(t == -1)])
        
        if current_tris <= target_triangles:
            print(f"  Reached target: {current_tris} triangles")
            break
        
        print(f"\n  Iteration {iteration + 1}/{max_iterations}:")
        print(f"    Current: {current_tris} triangles")
        
        editor._update_maps()
        
        # Phase 1: Edge collapse (short interior edges)
        edge_lengths = []
        for (u, v), tris in editor.edge_map.items():
            # Skip boundary edges
            if protect_boundary and (u in boundary_vertices or v in boundary_vertices):
                continue
            
            p1, p2 = editor.points[u], editor.points[v]
            length = np.linalg.norm(p2 - p1)
            edge_lengths.append((length, (u, v)))
        
        edge_lengths.sort()
        
        collapses_this_iter = 0
        max_collapse_attempts = min(500, len(edge_lengths))
        
        for i, (length, edge) in enumerate(edge_lengths[:max_collapse_attempts]):
            if collapses_this_iter >= 300:
                break
            
            try:
                ok, msg, _ = editor.edge_collapse(edge=edge)
                if ok:
                    collapses_this_iter += 1
                    n_collapses += 1
                    
                    if collapses_this_iter % 100 == 0:
                        editor._update_maps()
            except:
                pass
        
        print(f"    Edge collapses: {collapses_this_iter}")
        
        # Phase 2: Node removal (interior nodes with low degree)
        editor._update_maps()
        
        # Find interior vertices (not on boundary)
        all_vertices = set(range(len(editor.points)))
        interior_vertices = all_vertices - boundary_vertices
        
        # Compute vertex degrees (number of incident triangles)
        vertex_degrees = {}
        for v in interior_vertices:
            incident_tris = editor.v_map.get(v, set())
            # Filter out tombstones
            active_tris = [t for t in incident_tris if not np.all(editor.triangles[t] == -1)]
            vertex_degrees[v] = len(active_tris)
        
        # Sort by degree (prefer low-degree vertices for removal)
        sorted_vertices = sorted(vertex_degrees.items(), key=lambda x: x[1])
        
        removals_this_iter = 0
        max_removal_attempts = min(1000, len(sorted_vertices))  # Increased from 300
        
        for v, degree in sorted_vertices[:max_removal_attempts]:
            if removals_this_iter >= 500:  # Increased from 200
                break
            
            # Try to remove vertices with degree 3-12 (much more permissive)
            # Degree 3 = minimum for triangulation
            # Higher degrees are fine, remove_node will handle retriangulation
            if degree < 3 or degree > 12:
                continue
            
            try:
                ok = editor.remove_node(v)
                if ok:
                    removals_this_iter += 1
                    n_removals += 1
                    
                    if removals_this_iter % 50 == 0:
                        editor._update_maps()
            except:
                pass
        
        print(f"    Node removals: {removals_this_iter}")
        
        if collapses_this_iter == 0 and removals_this_iter == 0:
            print("    No more operations possible")
            break
    
    return n_collapses, n_removals


def write_msh_v2(filepath, points, triangles):
    """Write Gmsh v2.2 ASCII format."""
    with open(filepath, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        f.write("$Nodes\n")
        f.write(f"{len(points)}\n")
        for i, (x, y) in enumerate(points):
            f.write(f"{i+1} {x:.16e} {y:.16e} 0.0\n")
        f.write("$EndNodes\n")
        
        f.write("$Elements\n")
        f.write(f"{len(triangles)}\n")
        for i, (v0, v1, v2) in enumerate(triangles):
            f.write(f"{i+1} 2 0 {v0+1} {v1+1} {v2+1}\n")
        f.write("$EndElements\n")


def main():
    print("=" * 70)
    print("NACA0012 Aggressive Coarsening (edge_collapse + remove_node)")
    print("=" * 70)
    
    # Load coarse mesh
    input_file = 'meshes/naca0012_coarse.msh'
    print(f"\n1. Loading mesh from {input_file}...")
    points, triangles = read_msh(input_file, verbose=False)
    print(f"   Initial: {len(points)} vertices, {len(triangles)} triangles")
    
    editor = PatchBasedMeshEditor(points, triangles)
    
    initial_min_angle = mesh_min_angle(editor.points, editor.triangles)
    print(f"   Initial min angle: {initial_min_angle:.2f}°")
    
    # Aggressive coarsening
    print("\n2. Aggressive coarsening...")
    target_tris = 3000
    n_collapses, n_removals = aggressive_coarsen(editor, target_triangles=target_tris, 
                                                 protect_boundary=True)
    
    print(f"\n   Total operations:")
    print(f"     Edge collapses: {n_collapses}")
    print(f"     Node removals:  {n_removals}")
    print(f"     Total:          {n_collapses + n_removals}")
    
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
    output_file = 'meshes/naca0012_ultra_coarse.msh'
    print(f"\n4. Saving to {output_file}...")
    write_msh_v2(output_file, editor.points, editor.triangles)
    print(f"   Saved: {output_file}")
    
    print("\n" + "=" * 70)
    print("Aggressive coarsening complete!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
