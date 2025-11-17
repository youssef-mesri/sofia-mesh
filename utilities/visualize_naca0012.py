#!/usr/bin/env python3
"""
Regenerate NACA0012 mesh visualizations with proper zoom levels.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sofia.core.io import read_msh

def plot_mesh(points, triangles, title, filename, zoom_airfoil=True):
    """Plot mesh with proper zoom."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Identify boundary edges
    from collections import defaultdict
    edge_count = defaultdict(int)
    for tri in triangles:
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
            edge_count[edge] += 1
    
    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    interior_edges = [e for e, count in edge_count.items() if count == 2]
    
    # Plot interior edges
    interior_lines = [[points[e[0]], points[e[1]]] for e in interior_edges]
    if interior_lines:
        lc = LineCollection(interior_lines, colors='lightgray', linewidths=0.3, alpha=0.5)
        ax.add_collection(lc)
    
    # Plot boundary edges (red, bold)
    boundary_lines = [[points[e[0]], points[e[1]]] for e in boundary_edges]
    if boundary_lines:
        lc = LineCollection(boundary_lines, colors='red', linewidths=2.0, alpha=0.9)
        ax.add_collection(lc)
    
    # Set view limits
    if zoom_airfoil:
        # Zoom on airfoil region
        airfoil_mask = np.linalg.norm(points, axis=1) < 5.0
        airfoil_pts = points[airfoil_mask]
        if len(airfoil_pts) > 0:
            margin = 1.0
            ax.set_xlim(np.min(airfoil_pts[:, 0]) - margin, np.max(airfoil_pts[:, 0]) + margin)
            ax.set_ylim(np.min(airfoil_pts[:, 1]) - margin, np.max(airfoil_pts[:, 1]) + margin)
    else:
        # Show full domain
        margin = 10.0
        ax.set_xlim(np.min(points[:, 0]) - margin, np.max(points[:, 0]) + margin)
        ax.set_ylim(np.min(points[:, 1]) - margin, np.max(points[:, 1]) + margin)
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)


def main():
    print("=" * 70)
    print("Regenerating NACA0012 Visualizations")
    print("=" * 70)
    
    # Original mesh
    if os.path.exists('meshes/naca0012.msh'):
        print("\n1. Visualizing original mesh (meshes/naca0012.msh)...")
        points, triangles = read_msh('meshes/naca0012.msh', verbose=False)
        print(f"   {len(points)} vertices, {len(triangles)} triangles")
        
        plot_mesh(points, triangles, 
                 f"NACA0012 Original Mesh (Zoomed) - {len(triangles)} triangles",
                 "naca0012_original_zoom.png", zoom_airfoil=True)
        plot_mesh(points, triangles,
                 f"NACA0012 Original Mesh (Full) - {len(triangles)} triangles", 
                 "naca0012_original_full.png", zoom_airfoil=False)
    
    # Coarse mesh
    if os.path.exists('meshes/naca0012_coarse.msh'):
        print("\n2. Visualizing coarse mesh (meshes/naca0012_coarse.msh)...")
        points, triangles = read_msh('meshes/naca0012_coarse.msh', verbose=False)
        print(f"   {len(points)} vertices, {len(triangles)} triangles")
        
        plot_mesh(points, triangles,
                 f"NACA0012 Coarse Mesh (Zoomed) - {len(triangles)} triangles",
                 "naca0012_coarse_zoom.png", zoom_airfoil=True)
        plot_mesh(points, triangles,
                 f"NACA0012 Coarse Mesh (Full) - {len(triangles)} triangles",
                 "naca0012_coarse_full.png", zoom_airfoil=False)
    
    print("\n" + "=" * 70)
    print("Visualizations complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
