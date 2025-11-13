#!/usr/bin/env python3
"""
SOFIA Example 6: Mesh Editing Workflow

This example demonstrates a complete mesh editing workflow combining
multiple operations: refinement, node removal, and quality checks.

What this example shows:
1. Building an initial mesh
2. Refining specific regions by splitting edges
3. Removing unnecessary vertices
4. Validating mesh conformity throughout
5. Tracking mesh evolution step by step

Perfect for: Understanding complete mesh editing pipelines
"""

import numpy as np
import matplotlib.pyplot as plt
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay
from sofia.core.conformity import check_mesh_conformity
from sofia.core.quality import mesh_min_angle

def compute_mesh_stats(editor):
    """Compute basic mesh statistics."""
    active_mask = np.all(editor.triangles != -1, axis=1)
    active_tris = editor.triangles[active_mask]
    
    if len(active_tris) > 0:
        min_angle = mesh_min_angle(editor.points, active_tris)
    else:
        min_angle = 0
    
    return {
        'n_vertices': len(editor.points),
        'n_triangles': np.sum(active_mask),
        'n_edges': len(editor.edge_map),
        'min_angle': min_angle
    }

def refine_region(editor, center, radius, max_splits=10):
    """Refine triangles within a circular region.
    
    Args:
        editor: Mesh editor
        center: Center of refinement region (x, y)
        radius: Radius of refinement region
        max_splits: Maximum number of edge splits
        
    Returns:
        Number of edges split
    """
    center = np.array(center)
    splits = 0
    
    # Find triangles in the region
    for iteration in range(3):  # Multiple passes
        edges_to_split = []
        
        for edge in editor.edge_map.keys():
            v1, v2 = edge
            midpoint = (editor.points[v1] + editor.points[v2]) / 2
            
            # Check if edge midpoint is in region
            if np.linalg.norm(midpoint - center) < radius:
                length = np.linalg.norm(editor.points[v1] - editor.points[v2])
                edges_to_split.append((length, edge))
        
        if not edges_to_split or splits >= max_splits:
            break
        
        # Sort by length (longest first) and split
        edges_to_split.sort(reverse=True)
        
        for _, edge in edges_to_split[:5]:  # Split up to 5 per iteration
            if edge in editor.edge_map and splits < max_splits:
                success = editor.split_edge(edge=edge)
                if success:
                    splits += 1
    
    return splits

def remove_low_degree_vertices(editor, max_removals=5):
    """Remove interior vertices with low degree (ideally degree 4-5).
    
    Args:
        editor: Mesh editor
        max_removals: Maximum number of vertices to remove
        
    Returns:
        Number of vertices successfully removed
    """
    removals = 0
    
    for attempt in range(max_removals * 3):  # Try more times than target
        if removals >= max_removals:
            break
        
        # Find interior vertices with low degree
        candidates = []
        
        for v_idx, tri_set in editor.v_map.items():
            if len(tri_set) in [4, 5, 6]:  # Low to medium degree
                # Check if all incident edges are interior
                is_interior = True
                incident_edges = []
                
                for tri_idx in tri_set:
                    tri = editor.triangles[tri_idx]
                    if np.all(tri != -1):
                        for i in range(3):
                            if int(tri[i]) == v_idx or int(tri[(i+1)%3]) == v_idx:
                                edge = tuple(sorted([int(tri[i]), int(tri[(i+1)%3])]))
                                if v_idx in edge:
                                    incident_edges.append(edge)
                
                # Verify all edges are interior
                for edge in set(incident_edges):
                    if len(editor.edge_map.get(edge, [])) != 2:
                        is_interior = False
                        break
                
                if is_interior:
                    candidates.append((len(tri_set), v_idx))
        
        if not candidates:
            break
        
        # Sort by degree (lowest first)
        candidates.sort()
        
        # Try to remove the lowest degree vertex
        for _, v_idx in candidates[:3]:
            try:
                success = editor.remove_node(v_idx)
                if success:
                    removals += 1
                    break
            except:
                continue
    
    return removals

def main():
    """Run the mesh editing workflow demonstration."""
    print("=" * 70)
    print("SOFIA Example 6: Complete Mesh Editing Workflow")
    print("=" * 70)
    
    # Step 1: Create initial mesh
    print("\n[1] Creating initial mesh...")
    np.random.seed(789)
    pts, tris = build_random_delaunay(npts=40, seed=789)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    
    initial_stats = compute_mesh_stats(editor)
    valid, msg = check_mesh_conformity(editor.points, editor.triangles)
    
    print(f"  Initial mesh:")
    print(f"    Vertices: {initial_stats['n_vertices']}")
    print(f"    Triangles: {initial_stats['n_triangles']}")
    print(f"    Min angle: {initial_stats['min_angle']:.2f}°")
    print(f"    Conformity: {'✓ Valid' if valid else '✗ Invalid'}")
    
    # Save initial state
    initial_points = editor.points.copy()
    initial_active = np.all(editor.triangles != -1, axis=1)
    initial_triangles = editor.triangles[initial_active].copy()
    
    # Step 2: Refine a specific region
    print("\n[2] Refining central region...")
    center = np.mean(editor.points, axis=0)
    bbox_size = np.max(editor.points, axis=0) - np.min(editor.points, axis=0)
    radius = np.max(bbox_size) * 0.25
    
    print(f"  Refinement region: center={center}, radius={radius:.3f}")
    
    splits = refine_region(editor, center, radius, max_splits=15)
    
    refined_stats = compute_mesh_stats(editor)
    valid, msg = check_mesh_conformity(editor.points, editor.triangles)
    
    print(f"  After refinement:")
    print(f"    Edge splits: {splits}")
    print(f"    Vertices: {refined_stats['n_vertices']} (+{refined_stats['n_vertices'] - initial_stats['n_vertices']})")
    print(f"    Triangles: {refined_stats['n_triangles']} (+{refined_stats['n_triangles'] - initial_stats['n_triangles']})")
    print(f"    Conformity: {' Valid' if valid else ' Invalid'}")
    
    # Save refined state
    refined_points = editor.points.copy()
    refined_active = np.all(editor.triangles != -1, axis=1)
    refined_triangles = editor.triangles[refined_active].copy()
    
    # Step 3: Remove some vertices outside the refined region
    print("\n[3] Removing low-degree vertices outside refined region...")
    
    removals = remove_low_degree_vertices(editor, max_removals=8)
    
    final_stats = compute_mesh_stats(editor)
    valid, msg = check_mesh_conformity(editor.points, editor.triangles)
    
    print(f"  After removal:")
    print(f"    Vertices removed: {removals}")
    print(f"    Vertices: {final_stats['n_vertices']} ({final_stats['n_vertices'] - refined_stats['n_vertices']:+d})")
    print(f"    Triangles: {final_stats['n_triangles']} ({final_stats['n_triangles'] - refined_stats['n_triangles']:+d})")
    print(f"    Conformity: {'✓ Valid' if valid else '✗ Invalid'}")
    
    # Compact
    editor.compact_triangle_indices()
    
    # Step 4: Final validation
    print("\n[4] Final validation...")
    final_valid, final_msg = check_mesh_conformity(editor.points, editor.triangles)
    final_active = np.all(editor.triangles != -1, axis=1)
    
    print(f"  Final mesh:")
    print(f"    Vertices: {final_stats['n_vertices']}")
    print(f"    Triangles: {final_stats['n_triangles']}")
    print(f"    Edges: {final_stats['n_edges']}")
    print(f"    Min angle: {final_stats['min_angle']:.2f}°")
    print(f"    Conformity: {' Valid' if final_valid else ' Invalid - ' + final_msg}")
    
    # Step 5: Visualization
    print("\n[5] Creating visualization...")
    
    fig = plt.figure(figsize=(18, 6))
    
    # Initial mesh
    ax = plt.subplot(1, 3, 1)
    ax.set_aspect('equal')
    ax.triplot(initial_points[:, 0], initial_points[:, 1],
               initial_triangles, 'b-', linewidth=1)
    ax.plot(initial_points[:, 0], initial_points[:, 1],
            'ro', markersize=5)
    # Mark refinement region
    circle = plt.Circle(center, radius, fill=False, edgecolor='orange', 
                       linewidth=2, linestyle='--', label='Target region')
    ax.add_patch(circle)
    ax.set_title(f'Step 1: Initial Mesh\n{initial_stats["n_vertices"]} vertices, '
                 f'{initial_stats["n_triangles"]} triangles',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # After refinement
    ax = plt.subplot(1, 3, 2)
    ax.set_aspect('equal')
    ax.triplot(refined_points[:, 0], refined_points[:, 1],
               refined_triangles, 'g-', linewidth=0.8)
    ax.plot(refined_points[:, 0], refined_points[:, 1],
            'ro', markersize=4)
    circle = plt.Circle(center, radius, fill=False, edgecolor='orange',
                       linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.set_title(f'Step 2: After Refinement\n{refined_stats["n_vertices"]} vertices, '
                 f'{refined_stats["n_triangles"]} triangles\n(+{splits} splits)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Final mesh
    ax = plt.subplot(1, 3, 3)
    ax.set_aspect('equal')
    ax.triplot(editor.points[:, 0], editor.points[:, 1],
               editor.triangles[final_active], 'm-', linewidth=0.8)
    ax.plot(editor.points[:, 0], editor.points[:, 1],
            'ro', markersize=4)
    circle = plt.Circle(center, radius, fill=False, edgecolor='orange',
                       linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.set_title(f'Step 3: After Coarsening\n{final_stats["n_vertices"]} vertices, '
                 f'{final_stats["n_triangles"]} triangles\n(-{removals} vertices)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mesh_workflow_result.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'mesh_workflow_result.png'")
    
    print("\n" + "=" * 70)
    print(" Example completed successfully!")
    print(f" Workflow: {initial_stats['n_triangles']} → "
          f"{refined_stats['n_triangles']} → {final_stats['n_triangles']} triangles")
    print(f" All operations preserved mesh conformity: {final_valid}")
    print("=" * 70)

if __name__ == "__main__":
    main()
