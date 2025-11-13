"""
SOFIA Example 2: Quality Improvement

This example demonstrates mesh quality improvement through edge refinement.

What this example shows:
1. Start with a coarse mesh
2. Refine edges that form poor-quality triangles
3. Track quality metrics throughout the process
4. Visualize before and after

Perfect for: Understanding mesh refinement and quality improvement
"""

import numpy as np
import matplotlib.pyplot as plt
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from scipy.spatial import Delaunay

def create_initial_mesh():
    """Create a simple coarse mesh.
    
    We create a square domain with a coarse triangulation.
    """
    # Create a simple square with just 5 points (very coarse)
    points = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
        [1.0, 1.0],   # Center point
    ], dtype=float)
    
    # Create a Delaunay triangulation
    tri = Delaunay(points)
    
    # Create the mesh editor
    editor = PatchBasedMeshEditor(points, tri.simplices)
    
    return editor

def compute_quality_stats(editor):
    """Compute various quality metrics for the mesh."""
    from sofia.core.geometry import triangles_min_angles
    
    # Filter active triangles
    active_mask = np.all(editor.triangles != -1, axis=1)
    active_tris = editor.triangles[active_mask]
    
    if len(active_tris) == 0:
        return {
            'min_angle': 0,
            'max_angle': 0,
            'mean_angle': 0,
            'n_vertices': len(editor.points),
            'n_triangles': 0
        }
    
    # Compute minimum angles for all triangles
    min_angles = triangles_min_angles(editor.points, active_tris)
    
    return {
        'min_angle': float(np.min(min_angles)),
        'max_angle': float(np.max(min_angles)),
        'mean_angle': float(np.mean(min_angles)),
        'n_vertices': len(editor.points),
        'n_triangles': len(active_tris)
    }

def refine_long_edges(editor, max_edge_length=0.8):
    """Refine edges longer than max_edge_length.
    
    Returns the number of edges split.
    """
    splits_made = 0
    
    # Iterate multiple times as splitting creates new edges
    for iteration in range(5):
        edges_to_split = []
        
        # Find long edges
        for edge in editor.edge_map.keys():
            v1, v2 = edge
            length = np.linalg.norm(editor.points[v1] - editor.points[v2])
            
            if length > max_edge_length:
                edges_to_split.append(edge)
        
        if not edges_to_split:
            break  # No more long edges
        
        # Split the long edges
        for edge in edges_to_split:
            if edge in editor.edge_map:  # Edge might have been modified
                success = editor.split_edge(edge=edge)
                if success:
                    splits_made += 1
    
    return splits_made

def main():
    """Run the quality improvement demonstration."""
    print("=" * 60)
    print("Sofia Mesh Quality Improvement Example")
    print("=" * 60)
    
    # Step 1: Create initial coarse mesh
    print("\n[1] Creating initial coarse mesh...")
    mesh_editor = create_initial_mesh()
    
    # Save initial state for comparison
    initial_points = mesh_editor.points.copy()
    initial_tris_mask = np.all(mesh_editor.triangles != -1, axis=1)
    initial_triangles = mesh_editor.triangles[initial_tris_mask].copy()
    
    initial_stats = compute_quality_stats(mesh_editor)
    print(f"  Initial mesh:")
    print(f"    Vertices: {initial_stats['n_vertices']}")
    print(f"    Triangles: {initial_stats['n_triangles']}")
    print(f"    Min angle: {initial_stats['min_angle']:.2f}°")
    print(f"    Mean angle: {initial_stats['mean_angle']:.2f}°")
    
    # Step 2: Refine by splitting long edges
    print("\n[2] Refining mesh by splitting long edges...")
    print("  Target max edge length: 0.8")
    
    splits_made = refine_long_edges(mesh_editor, max_edge_length=0.8)
    print(f"  Split {splits_made} edges")
    
    # Step 3: Compute and display final statistics
    print("\n[3] Computing final statistics...")
    final_stats = compute_quality_stats(mesh_editor)
    print(f"  Refined mesh:")
    print(f"    Vertices: {final_stats['n_vertices']}")
    print(f"    Triangles: {final_stats['n_triangles']}")
    print(f"    Min angle: {final_stats['min_angle']:.2f}°")
    print(f"    Mean angle: {final_stats['mean_angle']:.2f}°")
    
    # Step 4: Display improvement metrics
    print("\n[4] Mesh refinement:")
    print(f"  Vertices: {initial_stats['n_vertices']} → {final_stats['n_vertices']} "
          f"(+{final_stats['n_vertices'] - initial_stats['n_vertices']})")
    print(f"  Triangles: {initial_stats['n_triangles']} → {final_stats['n_triangles']} "
          f"(+{final_stats['n_triangles'] - initial_stats['n_triangles']})")
    print(f"  Mean angle: {initial_stats['mean_angle']:.2f}° → {final_stats['mean_angle']:.2f}°")
    
    # Step 5: Create visualization
    print("\n[5] Creating visualization...")
    from sofia.core.geometry import triangles_min_angles
    
    fig = plt.figure(figsize=(15, 10))
    
    # Filter active triangles for plotting
    initial_active = np.all(initial_triangles != -1, axis=1)
    final_active = np.all(mesh_editor.triangles != -1, axis=1)
    
    # Before mesh
    ax = plt.subplot(2, 2, 1)
    ax.set_aspect('equal')
    ax.set_title(f'Initial Mesh ({initial_stats["n_triangles"]} triangles)', 
                 fontsize=12, fontweight='bold')
    ax.triplot(initial_points[:, 0], initial_points[:, 1],
               initial_triangles[initial_active], 'b-', linewidth=1.5)
    ax.plot(initial_points[:, 0], initial_points[:, 1],
            'ro', markersize=8, label='Vertices')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.2, 2.2)
    
    # After mesh
    ax = plt.subplot(2, 2, 2)
    ax.set_aspect('equal')
    ax.set_title(f'Refined Mesh ({final_stats["n_triangles"]} triangles)', 
                 fontsize=12, fontweight='bold')
    ax.triplot(mesh_editor.points[:, 0], mesh_editor.points[:, 1],
               mesh_editor.triangles[final_active], 'g-', linewidth=1)
    ax.plot(mesh_editor.points[:, 0], mesh_editor.points[:, 1],
            'ro', markersize=6, label='Vertices')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.2, 2.2)
    
    # Before angle distribution
    ax = plt.subplot(2, 2, 3)
    initial_angles = triangles_min_angles(initial_points, initial_triangles[initial_active])
    ax.hist(initial_angles, bins=15, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(initial_stats['mean_angle'], color='green', linestyle='--',
               linewidth=2, label=f"Mean: {initial_stats['mean_angle']:.1f}°")
    ax.set_xlabel('Minimum Angle (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Initial Angle Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    
    # After angle distribution
    ax = plt.subplot(2, 2, 4)
    final_angles = triangles_min_angles(mesh_editor.points, mesh_editor.triangles[final_active])
    ax.hist(final_angles, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(final_stats['mean_angle'], color='green', linestyle='--',
               linewidth=2, label=f"Mean: {final_stats['mean_angle']:.1f}°")
    ax.set_xlabel('Minimum Angle (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Refined Angle Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    
    plt.tight_layout()
    plt.savefig('quality_improvement_result.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'quality_improvement_result.png'")
    
    print("\n" + "=" * 60)
    print(" Example completed successfully!")
    print(f" Mesh refined from {initial_stats['n_triangles']} to {final_stats['n_triangles']} triangles")
    print("=" * 60)

if __name__ == "__main__":
    main()
