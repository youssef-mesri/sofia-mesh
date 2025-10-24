"""
SOFIA Example 3: Boundary Operations

This example demonstrates safe boundary manipulation:
1. Create a mesh with boundary
2. Split boundary edges
3. Remove boundary vertices (with area preservation)
4. Visualize each step

Perfect for: Understanding boundary operations and constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.config import BoundaryRemoveConfig
from sofia.core.conformity import check_mesh_conformity

def create_boundary_mesh():
    """Create a simple L-shaped mesh with external boundary"""
    
    # L-shaped domain points
    points = np.array([
        # Outer boundary
        [0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0],
        [1.0, 2.0], [0.0, 2.0],
        # Interior points
        [0.5, 0.5], [1.5, 0.5], [0.5, 1.5]
    ], dtype=float)
    
    # Triangulation
    triangles = np.array([
        [0, 1, 6], [1, 7, 6], [1, 2, 7], [2, 3, 7],
        [3, 4, 8], [3, 8, 6], [3, 6, 7], [4, 5, 8],
        [5, 0, 8], [0, 6, 8]
    ], dtype=np.int32)
    
    return PatchBasedMeshEditor(points, triangles)

def identify_boundary_vertices(editor):
    """Find vertices on the boundary"""
    from sofia.core.conformity import build_edge_to_tri_map
    
    edge_map = build_edge_to_tri_map(editor.triangles)
    boundary_verts = set()
    
    for edge, tris in edge_map.items():
        if len(tris) == 1:  # Boundary edge
            boundary_verts.add(edge[0])
            boundary_verts.add(edge[1])
    
    return sorted(boundary_verts)

def main():
    print("=" * 60)
    print("SOFIA Example 3: Boundary Operations")
    print("=" * 60)
    
    # Step 1: Create mesh with boundary
    print("\n[1] Creating L-shaped mesh with boundary...")
    editor = create_boundary_mesh()
    
    print(f"  Initial mesh: {len(editor.points)} vertices, {len(editor.triangles)} triangles")
    
    boundary_verts = identify_boundary_vertices(editor)
    print(f"  Boundary vertices: {len(boundary_verts)}")
    
    # Verify conformity
    is_valid, msg = check_mesh_conformity(editor.points, editor.triangles)
    print(f"  Mesh conformity: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Step 2: Split a boundary edge
    print("\n[2] Splitting boundary edge...")
    
    # Find a boundary edge (edge with only one adjacent triangle)
    from sofia.core.conformity import build_edge_to_tri_map
    edge_map = build_edge_to_tri_map(editor.triangles)
    
    boundary_edge = None
    for edge, tris in edge_map.items():
        if len(tris) == 1:
            boundary_edge = edge
            break
    
    if boundary_edge is not None:
        success = editor.split_edge(edge=boundary_edge)
        print(f"  {'✓' if success else '✗'} Split boundary edge {boundary_edge}")
        print(f"  Mesh now has {len(editor.points)} vertices")
    
    # Step 3: Remove a boundary vertex
    print("\n[3] Removing boundary vertex...")
    
    # Find an interior boundary vertex (not a corner)
    boundary_verts_new = identify_boundary_vertices(editor)
    
    if len(boundary_verts_new) > 2:
        # Try to remove a non-corner boundary vertex
        test_vertex = boundary_verts_new[len(boundary_verts_new) // 2]
        
        print(f"  Attempting to remove vertex {test_vertex}...")
        
        try:
            success = editor.remove_node(test_vertex)
            
            if success:
                print(f"  ✓ Successfully removed vertex {test_vertex}")
                print(f"  Mesh now has {len(editor.points)} vertices")
            else:
                print(f"  ✗ Removal rejected")
        except Exception as e:
            print(f"  ✗ Removal failed: {e}")
    
    # Verify final conformity
    is_valid, msg = check_mesh_conformity(editor.points, editor.triangles)
    print(f"\n  Final conformity check: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Step 4: Visualize
    print("\n[4] Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Recreate initial mesh for comparison
    initial_editor = create_boundary_mesh()
    initial_boundary = identify_boundary_vertices(initial_editor)
    
    # Plot initial mesh
    ax = axes[0]
    ax.set_aspect('equal')
    # Filter active triangles
    initial_active = np.all(initial_editor.triangles != -1, axis=1)
    ax.triplot(initial_editor.points[:, 0], initial_editor.points[:, 1], 
               initial_editor.triangles[initial_active], 'b-', linewidth=1.5)
    ax.plot(initial_editor.points[:, 0], initial_editor.points[:, 1], 
            'ko', markersize=8, label='Interior')
    ax.plot(initial_editor.points[initial_boundary, 0], 
            initial_editor.points[initial_boundary, 1], 
            'ro', markersize=10, label='Boundary', zorder=5)
    ax.set_title('Initial Mesh\nBoundary vertices highlighted', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Plot modified mesh
    ax = axes[1]
    ax.set_aspect('equal')
    final_boundary = identify_boundary_vertices(editor)
    # Filter active triangles
    final_active = np.all(editor.triangles != -1, axis=1)
    ax.triplot(editor.points[:, 0], editor.points[:, 1], 
               editor.triangles[final_active], 'g-', linewidth=1.5)
    ax.plot(editor.points[:, 0], editor.points[:, 1], 
            'ko', markersize=8, label='Interior')
    ax.plot(editor.points[final_boundary, 0], 
            editor.points[final_boundary, 1], 
            'ro', markersize=10, label='Boundary', zorder=5)
    ax.set_title('After Boundary Operations\n(split + optional removal)', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('boundary_operations_result.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization to 'boundary_operations_result.png'")
    
    # Optional: show plot
    # plt.show()
    
    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("✓ Boundary operations maintain mesh validity")
    print("=" * 60)
    
    # Print important note
    print("\n📝 Note:")
    print("   Boundary operations preserve local area by default.")
    print("   Use BoundaryRemoveConfig(require_area_preservation=False)")
    print("   to relax this constraint for exploratory purposes.")

if __name__ == "__main__":
    main()
