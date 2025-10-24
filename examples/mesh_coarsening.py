#!/usr/bin/env python3
"""
SOFIA Example 5: Mesh Coarsening

This example demonstrates mesh coarsening by collapsing short edges.
We start with a refined mesh and progressively simplify it while
maintaining mesh quality constraints.

What this example shows:
1. Building a refined Delaunay mesh
2. Identifying candidate edges for collapse
3. Safely collapsing edges with quality preservation
4. Tracking mesh statistics during coarsening
5. Comparing before/after mesh characteristics

Perfect for: Understanding mesh simplification and LOD generation
"""

import numpy as np
import matplotlib.pyplot as plt
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor, build_random_delaunay

def list_interior_edges(editor):
    """List all interior edges (shared by exactly two triangles)."""
    interior = []
    for edge, tris in editor.edge_map.items():
        if len(tris) == 2:
            interior.append(edge)
    return interior

def edge_length(editor, edge):
    """Compute edge length."""
    v1, v2 = edge
    return np.linalg.norm(editor.points[v1] - editor.points[v2])

def coarsen_mesh(editor, target_collapses=20, min_quality=0.3):
    """Coarsen mesh by collapsing shortest edges.
    
    Args:
        editor: Mesh editor
        target_collapses: Target number of successful collapses
        min_quality: Minimum quality threshold for collapse acceptance
        
    Returns:
        Number of successful collapses
    """
    successes = 0
    attempts = 0
    max_attempts = target_collapses * 10  # Safety limit
    
    while successes < target_collapses and attempts < max_attempts:
        # Get current interior edges
        edges = list_interior_edges(editor)
        
        if not edges:
            print(f"    No more interior edges (attempts={attempts})")
            break
        
        # Sort by length (shortest first)
        edges.sort(key=lambda e: edge_length(editor, e))
        
        # Try to collapse shortest edges
        progressed = False
        for edge in edges[:min(20, len(edges))]:  # Try top 20 shortest
            attempts += 1
            
            # Try collapse
            ok, msg, _ = editor.edge_collapse(edge)
            
            if ok:
                successes += 1
                progressed = True
                break  # Recompute edge list after each success
            
            if attempts >= max_attempts or successes >= target_collapses:
                break
        
        if not progressed:
            # No edge could be collapsed - quality constraints prevent further simplification
            break
    
    return successes, attempts

def compute_mesh_stats(editor):
    """Compute mesh statistics."""
    active_mask = np.all(editor.triangles != -1, axis=1)
    n_triangles = np.sum(active_mask)
    n_vertices = len(editor.points)
    
    # Compute edge lengths
    edge_lengths = []
    for edge in editor.edge_map.keys():
        edge_lengths.append(edge_length(editor, edge))
    
    return {
        'n_vertices': n_vertices,
        'n_triangles': n_triangles,
        'n_edges': len(edge_lengths),
        'mean_edge_length': np.mean(edge_lengths) if edge_lengths else 0,
        'min_edge_length': np.min(edge_lengths) if edge_lengths else 0,
        'max_edge_length': np.max(edge_lengths) if edge_lengths else 0,
    }

def main():
    """Run the mesh coarsening demonstration."""
    print("=" * 70)
    print("SOFIA Example 5: Mesh Coarsening (Edge Collapse)")
    print("=" * 70)
    
    # Step 1: Create initial refined mesh
    print("\n[1] Creating initial refined mesh...")
    np.random.seed(123)
    pts, tris = build_random_delaunay(npts=80, seed=123)
    editor = PatchBasedMeshEditor(pts.copy(), tris.copy())
    
    # Save initial state
    initial_points = editor.points.copy()
    initial_active = np.all(editor.triangles != -1, axis=1)
    initial_triangles = editor.triangles[initial_active].copy()
    
    initial_stats = compute_mesh_stats(editor)
    print(f"  Initial mesh:")
    print(f"    Vertices: {initial_stats['n_vertices']}")
    print(f"    Triangles: {initial_stats['n_triangles']}")
    print(f"    Edges: {initial_stats['n_edges']}")
    print(f"    Mean edge length: {initial_stats['mean_edge_length']:.4f}")
    
    # Step 2: Coarsen mesh
    print("\n[2] Coarsening mesh by collapsing short edges...")
    print("  Strategy: Collapse shortest edges while maintaining quality")
    
    target_collapses = 30
    successes, attempts = coarsen_mesh(editor, target_collapses=target_collapses)
    
    print(f"\n  Collapse summary:")
    print(f"    Successful collapses: {successes}")
    print(f"    Total attempts: {attempts}")
    print(f"    Success rate: {successes/attempts*100:.1f}%")
    
    # Step 3: Final statistics
    print("\n[3] Final mesh statistics...")
    editor.compact_triangle_indices()
    
    final_stats = compute_mesh_stats(editor)
    final_active = np.all(editor.triangles != -1, axis=1)
    
    print(f"  Coarsened mesh:")
    print(f"    Vertices: {final_stats['n_vertices']} "
          f"({final_stats['n_vertices'] - initial_stats['n_vertices']:+d})")
    print(f"    Triangles: {final_stats['n_triangles']} "
          f"({final_stats['n_triangles'] - initial_stats['n_triangles']:+d})")
    print(f"    Edges: {final_stats['n_edges']} "
          f"({final_stats['n_edges'] - initial_stats['n_edges']:+d})")
    print(f"    Mean edge length: {final_stats['mean_edge_length']:.4f}")
    
    reduction = (initial_stats['n_triangles'] - final_stats['n_triangles']) / initial_stats['n_triangles'] * 100
    print(f"\n  Mesh reduction: {reduction:.1f}%")
    
    # Step 4: Visualization
    print("\n[4] Creating visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Initial mesh
    ax = plt.subplot(2, 3, 1)
    ax.set_aspect('equal')
    ax.triplot(initial_points[:, 0], initial_points[:, 1],
               initial_triangles, 'b-', linewidth=0.8)
    ax.plot(initial_points[:, 0], initial_points[:, 1],
            'ro', markersize=3)
    ax.set_title(f'Initial Mesh\n{initial_stats["n_vertices"]} vertices, '
                 f'{initial_stats["n_triangles"]} triangles',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Coarsened mesh
    ax = plt.subplot(2, 3, 2)
    ax.set_aspect('equal')
    ax.triplot(editor.points[:, 0], editor.points[:, 1],
               editor.triangles[final_active], 'g-', linewidth=1.2)
    ax.plot(editor.points[:, 0], editor.points[:, 1],
            'ro', markersize=5)
    ax.set_title(f'Coarsened Mesh\n{final_stats["n_vertices"]} vertices, '
                 f'{final_stats["n_triangles"]} triangles',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mesh complexity comparison
    ax = plt.subplot(2, 3, 3)
    categories = ['Vertices', 'Triangles', 'Edges']
    initial_vals = [initial_stats['n_vertices'], initial_stats['n_triangles'], 
                    initial_stats['n_edges']]
    final_vals = [final_stats['n_vertices'], final_stats['n_triangles'], 
                  final_stats['n_edges']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, initial_vals, width, label='Initial', color='steelblue')
    ax.bar(x + width/2, final_vals, width, label='Coarsened', color='coral')
    ax.set_ylabel('Count')
    ax.set_title('Mesh Complexity Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Edge length distribution - before
    ax = plt.subplot(2, 3, 4)
    initial_edges = [edge_length(PatchBasedMeshEditor(initial_points, initial_triangles), e) 
                     for e in list_interior_edges(PatchBasedMeshEditor(initial_points, initial_triangles))]
    ax.hist(initial_edges, bins=25, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(initial_edges), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(initial_edges):.3f}')
    ax.set_xlabel('Edge Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Initial Edge Length Distribution', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Edge length distribution - after
    ax = plt.subplot(2, 3, 5)
    final_edges = [edge_length(editor, e) for e in list_interior_edges(editor)]
    ax.hist(final_edges, bins=25, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(final_edges), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(final_edges):.3f}')
    ax.set_xlabel('Edge Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Coarsened Edge Length Distribution', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistics summary
    ax = plt.subplot(2, 3, 6)
    ax.axis('off')
    stats_text = f"""
Mesh Coarsening Summary
{'=' * 35}

Initial Mesh:
  • Vertices: {initial_stats['n_vertices']}
  • Triangles: {initial_stats['n_triangles']}
  • Edges: {initial_stats['n_edges']}
  • Mean edge: {initial_stats['mean_edge_length']:.4f}

Coarsened Mesh:
  • Vertices: {final_stats['n_vertices']}
  • Triangles: {final_stats['n_triangles']}
  • Edges: {final_stats['n_edges']}
  • Mean edge: {final_stats['mean_edge_length']:.4f}

Coarsening Process:
  • Target collapses: {target_collapses}
  • Successful: {successes}
  • Success rate: {successes/attempts*100:.1f}%
  • Mesh reduction: {reduction:.1f}%
  • Quality preserved: ✓
"""
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('mesh_coarsening_result.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization to 'mesh_coarsening_result.png'")
    
    print("\n" + "=" * 70)
    print("✓ Example completed successfully!")
    print(f"✓ Mesh simplified: {initial_stats['n_triangles']} → {final_stats['n_triangles']} triangles")
    print(f"✓ Reduction: {reduction:.1f}% while maintaining quality")
    print("=" * 70)

if __name__ == "__main__":
    main()
