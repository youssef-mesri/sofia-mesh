#!/usr/bin/env python3
"""
SOFIA Example 4: Adaptive Mesh Refinement

This example demonstrates adaptive refinement based on triangle size.
We start with a coarse mesh and iteratively refine the largest triangles
until reaching a target refinement level.

What this example shows:
1. Building a random Delaunay mesh
2. Computing triangle areas to identify refinement targets
3. Splitting edges of large triangles
4. Tracking mesh statistics during refinement
5. Visualizing the refinement process

Perfect for: Understanding adaptive mesh refinement strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.core.geometry import triangle_area

def compute_triangle_areas(editor):
    """Compute areas of all active triangles."""
    areas = []
    tri_indices = []
    
    for idx, tri in enumerate(editor.triangles):
        if np.all(tri != -1):
            coords = editor.points[tri]
            area = triangle_area(coords[0], coords[1], coords[2])
            areas.append(area)
            tri_indices.append(idx)
    
    return np.array(areas), tri_indices

def refine_largest_triangles(editor, fraction=0.2, max_splits=50):
    """Refine the largest triangles by splitting their longest edges.
    
    Args:
        editor: Mesh editor
        fraction: Fraction of largest triangles to refine (0-1)
        max_splits: Maximum number of edge splits per iteration
        
    Returns:
        Number of edges split
    """
    areas, tri_indices = compute_triangle_areas(editor)
    
    if len(areas) == 0:
        return 0
    
    # Sort triangles by area (largest first)
    sorted_idx = np.argsort(areas)[::-1]
    n_to_refine = max(1, int(len(areas) * fraction))
    
    splits_made = 0
    refined_triangles = set()
    
    for i in sorted_idx[:n_to_refine]:
        if splits_made >= max_splits:
            break
            
        tri_idx = tri_indices[i]
        tri = editor.triangles[tri_idx]
        
        if np.any(tri == -1) or tri_idx in refined_triangles:
            continue
        
        # Find longest edge of this triangle
        edges = [
            (int(tri[0]), int(tri[1])),
            (int(tri[1]), int(tri[2])),
            (int(tri[2]), int(tri[0]))
        ]
        
        longest_edge = None
        max_length = 0
        
        for edge in edges:
            v1, v2 = edge
            length = np.linalg.norm(editor.points[v1] - editor.points[v2])
            if length > max_length:
                max_length = length
                longest_edge = edge
        
        # Split the longest edge
        if longest_edge and longest_edge in editor.edge_map:
            success = editor.split_edge(edge=longest_edge)
            if success:
                splits_made += 1
                refined_triangles.add(tri_idx)
    
    return splits_made

def main():
    """Run the adaptive refinement demonstration."""
    print("=" * 70)
    print("SOFIA Example 4: Adaptive Mesh Refinement")
    print("=" * 70)
    
    # Step 1: Create initial coarse mesh
    print("\n[1] Creating initial random Delaunay mesh...")
    np.random.seed(42)
    initial_pts, initial_tris = build_random_delaunay(npts=20, seed=42)
    editor = PatchBasedMeshEditor(initial_pts.copy(), initial_tris.copy())
    
    # Save initial state
    initial_points = editor.points.copy()
    initial_active = np.all(editor.triangles != -1, axis=1)
    initial_triangles = editor.triangles[initial_active].copy()
    
    areas, _ = compute_triangle_areas(editor)
    print(f"  Initial mesh:")
    print(f"    Vertices: {len(editor.points)}")
    print(f"    Triangles: {len(areas)}")
    print(f"    Mean triangle area: {np.mean(areas):.4f}")
    print(f"    Max triangle area: {np.max(areas):.4f}")
    
    # Step 2: Adaptive refinement iterations
    print("\n[2] Applying adaptive refinement...")
    print("  Strategy: Refine 20% largest triangles per iteration")
    
    refinement_iterations = 3
    total_splits = 0
    
    for iteration in range(refinement_iterations):
        splits = refine_largest_triangles(editor, fraction=0.2, max_splits=30)
        total_splits += splits
        
        areas, _ = compute_triangle_areas(editor)
        print(f"  Iteration {iteration + 1}: Split {splits} edges, "
              f"{len(areas)} triangles, max area = {np.max(areas):.4f}")
    
    print(f"\n  Total edge splits: {total_splits}")
    
    # Step 3: Final statistics
    print("\n[3] Final mesh statistics...")
    final_areas, _ = compute_triangle_areas(editor)
    final_active = np.all(editor.triangles != -1, axis=1)
    
    print(f"  Final mesh:")
    print(f"    Vertices: {len(editor.points)} (+{len(editor.points) - len(initial_points)})")
    print(f"    Triangles: {len(final_areas)} (+{len(final_areas) - len(areas)})")
    print(f"    Mean triangle area: {np.mean(final_areas):.4f}")
    print(f"    Max triangle area: {np.max(final_areas):.4f}")
    print(f"    Area std dev: {np.std(final_areas):.4f}")
    
    # Step 4: Visualization
    print("\n[4] Creating visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Initial mesh
    ax = plt.subplot(2, 3, 1)
    ax.set_aspect('equal')
    ax.triplot(initial_points[:, 0], initial_points[:, 1],
               initial_triangles, 'b-', linewidth=1)
    ax.plot(initial_points[:, 0], initial_points[:, 1],
            'ro', markersize=6)
    ax.set_title(f'Initial Mesh\n{len(initial_triangles)} triangles',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Final mesh
    ax = plt.subplot(2, 3, 2)
    ax.set_aspect('equal')
    ax.triplot(editor.points[:, 0], editor.points[:, 1],
               editor.triangles[final_active], 'g-', linewidth=0.8)
    ax.plot(editor.points[:, 0], editor.points[:, 1],
            'ro', markersize=4)
    ax.set_title(f'Refined Mesh\n{len(final_areas)} triangles',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Triangle size distribution - before
    ax = plt.subplot(2, 3, 4)
    initial_areas_data, _ = compute_triangle_areas(
        PatchBasedMeshEditor(initial_points, initial_triangles)
    )
    ax.hist(initial_areas_data, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(initial_areas_data), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(initial_areas_data):.4f}')
    ax.set_xlabel('Triangle Area')
    ax.set_ylabel('Frequency')
    ax.set_title('Initial Area Distribution', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Triangle size distribution - after
    ax = plt.subplot(2, 3, 5)
    ax.hist(final_areas, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(final_areas), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(final_areas):.4f}')
    ax.set_xlabel('Triangle Area')
    ax.set_ylabel('Frequency')
    ax.set_title('Refined Area Distribution', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mesh density comparison
    ax = plt.subplot(2, 3, 3)
    metrics = [
        ('Initial', len(initial_triangles), len(initial_points)),
        ('Refined', len(final_areas), len(editor.points))
    ]
    x = np.arange(len(metrics))
    width = 0.35
    
    triangles = [m[1] for m in metrics]
    vertices = [m[2] for m in metrics]
    
    ax.bar(x - width/2, triangles, width, label='Triangles', color='steelblue')
    ax.bar(x + width/2, vertices, width, label='Vertices', color='coral')
    ax.set_ylabel('Count')
    ax.set_title('Mesh Complexity', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Statistics summary
    ax = plt.subplot(2, 3, 6)
    ax.axis('off')
    stats_text = f"""
Adaptive Refinement Summary
{'=' * 35}

Initial Mesh:
  • Vertices: {len(initial_points)}
  • Triangles: {len(initial_triangles)}
  • Mean area: {np.mean(initial_areas_data):.4f}

Refined Mesh:
  • Vertices: {len(editor.points)}
  • Triangles: {len(final_areas)}
  • Mean area: {np.mean(final_areas):.4f}

Refinement Process:
  • Iterations: {refinement_iterations}
  • Total splits: {total_splits}
  • Area reduction: {(1 - np.mean(final_areas)/np.mean(initial_areas_data)) * 100:.1f}%
  • Uniformity (1/std): {1/np.std(final_areas):.2f}
"""
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('adaptive_refinement_result.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'adaptive_refinement_result.png'")
    
    print("\n" + "=" * 70)
    print(" Example completed successfully!")
    print(f" Mesh refined from {len(initial_triangles)} to {len(final_areas)} triangles")
    print(f" Area uniformity improved by {(1 - np.std(final_areas)/np.std(initial_areas_data)) * 100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
