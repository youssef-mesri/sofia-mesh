"""
SOFIA Example 1: Basic Remeshing

This example demonstrates the most basic usage of SOFIA:
1. Create a simple triangular mesh
2. Perform basic operations (edge split)
3. Check mesh quality
4. Visualize the result

Perfect for: First-time users, quick start guide
"""

import numpy as np
import matplotlib.pyplot as plt
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.quality import mesh_min_angle

def main():
    print("=" * 60)
    print("SOFIA Example 1: Basic Remeshing")
    print("=" * 60)
    
    # Step 1: Create a simple square mesh (4 triangles)
    print("\n[1] Creating initial mesh...")
    
    points = np.array([
        [0.0, 0.0],  # Bottom-left
        [1.0, 0.0],  # Bottom-right
        [1.0, 1.0],  # Top-right
        [0.0, 1.0],  # Top-left
        [0.5, 0.5],  # Center
    ], dtype=float)
    
    triangles = np.array([
        [0, 1, 4],  # Bottom triangle
        [1, 2, 4],  # Right triangle
        [2, 3, 4],  # Top triangle
        [3, 0, 4],  # Left triangle
    ], dtype=np.int32)
    
    print(f"  Initial mesh: {len(points)} vertices, {len(triangles)} triangles")
    
    # Step 2: Initialize the mesh editor
    print("\n[2] Initializing mesh editor...")
    editor = PatchBasedMeshEditor(points, triangles)
    
    # Step 3: Check initial quality
    print("\n[3] Checking initial mesh quality...")
    min_angle_before = mesh_min_angle(editor.points, editor.triangles)
    print(f"  Minimum angle before: {min_angle_before:.2f}°")
    
    # Step 4: Refine the mesh by splitting edges
    print("\n[4] Refining mesh by splitting edges...")
    # Define edges as (vertex1, vertex2) pairs
    edges_to_split = [(0, 1), (1, 2), (2, 3)]  # Split first 3 edges
    
    for edge in edges_to_split:
        success = editor.split_edge(edge=edge)
        if success:
            print(f"  Split edge {edge}")
        else:
            print(f"  Failed to split edge {edge}")
    
    # Step 5: Check final quality
    print("\n[5] Checking final mesh quality...")
    min_angle_after = mesh_min_angle(editor.points, editor.triangles)
    print(f"  Minimum angle after: {min_angle_after:.2f}°")
    print(f"  Final mesh: {len(editor.points)} vertices, {len(editor.triangles)} triangles")
    
    # Step 6: Visualize
    print("\n[6] Creating visualization...")
    
    # Filter active triangles (remove tombstones with -1)
    active_tris_mask = np.all(editor.triangles != -1, axis=1)
    active_tris = editor.triangles[active_tris_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot initial mesh
    ax1.set_aspect('equal')
    ax1.triplot(points[:, 0], points[:, 1], triangles, 'b-', linewidth=1.5)
    ax1.plot(points[:, 0], points[:, 1], 'ro', markersize=8)
    ax1.set_title(f'Initial Mesh\nMin angle: {min_angle_before:.2f}°')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    
    # Plot refined mesh
    ax2.set_aspect('equal')
    ax2.triplot(editor.points[:, 0], editor.points[:, 1], active_tris, 'g-', linewidth=1.5)
    ax2.plot(editor.points[:, 0], editor.points[:, 1], 'ro', markersize=8)
    ax2.set_title(f'Refined Mesh\nMin angle: {min_angle_after:.2f}°')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basic_remeshing_result.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'basic_remeshing_result.png'")
    
    # Optional: show plot
    # plt.show()
    
    print("\n" + "=" * 60)
    print(" Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
