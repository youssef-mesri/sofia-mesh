#!/usr/bin/env python3
"""
SOFIA Example: Anisotropic Adaptation on NACA0012 Airfoil

This example demonstrates anisotropic mesh adaptation on a NACA0012 airfoil mesh:
1. Loading mesh from MSH file
2. Identifying airfoil boundary
3. Computing metric edge lengths for boundary and interior edges
4. Splitting/collapsing edges based on metric criteria
5. Protecting boundary vertices during adaptation
6. Visualizing metric field with ellipses

Perfect for: CFD preprocessing, airfoil meshing, anisotropic boundary features
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import math
from sofia.core.io import read_msh, write_vtk
from sofia.core.mesh_modifier2 import PatchBasedMeshEditor
from sofia.core.quality import mesh_min_angle


def naca0012_metric(x, boundary_thickness=0.5):
    """Construct anisotropic metric with refinement near airfoil.
    
    Args:
        x: 2D position array [x, y]
        boundary_thickness: Thickness of boundary layer region
        
    Returns:
        2x2 symmetric positive-definite metric tensor M(x)
    """
    # Distance from airfoil (approximated as distance from origin)
    dist = np.linalg.norm(x)
    
    # In boundary layer: want SMALL edges perpendicular, LARGE edges parallel
    if dist < boundary_thickness:
        h_perp = 0.05 + 0.2 * (dist / boundary_thickness)  # 0.05 to 0.25
        h_parallel = 0.5  # coarse parallel
    else:
        # In farfield: uniform coarse sizing
        h_perp = 0.5
        h_parallel = 0.5
    
    # Convert to metric eigenvalues: lambda = 1/h^2
    lam_perp = 1.0 / (h_perp ** 2)
    lam_parallel = 1.0 / (h_parallel ** 2)
    
    # Tangent direction (approximate as perpendicular to radial)
    if np.linalg.norm(x) > 1e-10:
        radial = x / np.linalg.norm(x)
        tangent = np.array([-radial[1], radial[0]])
    else:
        tangent = np.array([1.0, 0.0])
    
    # Build metric: M = lambda_parallel * (t⊗t) + lambda_perp * (n⊗n)
    M = lam_parallel * np.outer(tangent, tangent) + lam_perp * np.outer(radial, radial)
    
    return M


def metric_edge_length(editor, edge, metric_fn):
    """Compute metric length of edge using midpoint quadrature."""
    v1, v2 = edge
    p1, p2 = editor.points[v1], editor.points[v2]
    midpoint = 0.5 * (p1 + p2)
    
    M = metric_fn(midpoint)
    delta = p2 - p1
    
    # L_M = sqrt(delta^T M delta)
    lm = math.sqrt(max(0.0, delta @ M @ delta))
    return lm


def identify_boundary_edges(editor, boundary_vertices):
    """Identify boundary and interior edges."""
    editor._update_maps()
    
    boundary_edges = set()
    interior_edges = set()
    
    for (u, v), tris in editor.edge_map.items():
        edge = tuple(sorted([u, v]))
        if len(tris) == 1:  # Boundary edge
            boundary_edges.add(edge)
        elif len(tris) == 2:  # Interior edge
            interior_edges.add(edge)
    
    return boundary_edges, interior_edges


def refine_by_metric(editor, metric_fn, boundary_vertices,
                     alpha_split=1.5, max_splits=100):
    """Refine edges based on metric length.
    
    Args:
        editor: Mesh editor
        metric_fn: Metric function
        boundary_vertices: Set of boundary vertex indices
        alpha_split: Split edges where L_M > alpha_split
        max_splits: Max splits per iteration
        
    Returns:
        (boundary_splits, interior_splits)
    """
    editor._update_maps()
    
    boundary_edges, interior_edges = identify_boundary_edges(editor, boundary_vertices)
    
    # Compute metric lengths
    bnd_edge_metrics = [(metric_edge_length(editor, e, metric_fn), e) for e in boundary_edges]
    int_edge_metrics = [(metric_edge_length(editor, e, metric_fn), e) for e in interior_edges]
    
    bnd_edge_metrics.sort(reverse=True)
    int_edge_metrics.sort(reverse=True)
    
    bnd_splits = 0
    int_splits = 0
    
    # Split boundary edges
    for lm, edge in bnd_edge_metrics:
        if lm <= alpha_split or bnd_splits >= max_splits:
            break
        
        try:
            ok = editor.split_edge_delaunay(edge=edge)
            if ok:
                bnd_splits += 1
                editor._update_maps()
        except:
            pass
    
    # Split interior edges
    for lm, edge in int_edge_metrics:
        if lm <= alpha_split or int_splits >= max_splits:
            break
        
        try:
            ok = editor.split_edge_delaunay(edge=edge)
            if ok:
                int_splits += 1
                editor._update_maps()
        except:
            pass
    
    return bnd_splits, int_splits


def coarsen_interior_by_metric(editor, metric_fn, boundary_vertices,
                                beta_collapse=0.4, max_collapses=50):
    """Coarsen interior edges based on metric length."""
    editor._update_maps()
    
    _, interior_edges = identify_boundary_edges(editor, boundary_vertices)
    
    edge_metrics = [(metric_edge_length(editor, e, metric_fn), e) 
                    for e in interior_edges]
    edge_metrics.sort()
    
    collapses = 0
    for lm, edge in edge_metrics:
        if lm >= beta_collapse or collapses >= max_collapses:
            break
        
        try:
            ok, _, _ = editor.edge_collapse(edge=edge)
            if ok:
                collapses += 1
                editor._update_maps()
        except:
            pass
    
    return collapses


def plot_mesh_with_metric(editor, metric_fn, boundary_vertices, title="", filename=None, zoom_airfoil=True):
    """Plot mesh with metric ellipses.
    
    Args:
        zoom_airfoil: If True, zoom on airfoil region; if False, show full domain
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    active_mask = np.all(editor.triangles != -1, axis=1)
    active_tris = editor.triangles[active_mask]
    pts = editor.points
    
    boundary_edges, interior_edges = identify_boundary_edges(editor, boundary_vertices)
    
    # Plot interior edges
    interior_lines = [[pts[e[0]], pts[e[1]]] for e in interior_edges]
    if interior_lines:
        lc = LineCollection(interior_lines, colors='lightgray', linewidths=0.5, alpha=0.7)
        ax.add_collection(lc)
    
    # Plot boundary edges (red, bold)
    boundary_lines = [[pts[e[0]], pts[e[1]]] for e in boundary_edges]
    if boundary_lines:
        lc = LineCollection(boundary_lines, colors='red', linewidths=2.0, alpha=0.9)
        ax.add_collection(lc)
    
    # Set view limits
    if zoom_airfoil:
        # Zoom on airfoil region (distance < 5 from origin)
        airfoil_mask = np.linalg.norm(pts, axis=1) < 5.0
        airfoil_pts = pts[airfoil_mask]
        if len(airfoil_pts) > 0:
            margin = 1.0
            ax.set_xlim(np.min(airfoil_pts[:, 0]) - margin, np.max(airfoil_pts[:, 0]) + margin)
            ax.set_ylim(np.min(airfoil_pts[:, 1]) - margin, np.max(airfoil_pts[:, 1]) + margin)
    else:
        # Show full domain
        margin = 10.0
        ax.set_xlim(np.min(pts[:, 0]) - margin, np.max(pts[:, 0]) + margin)
        ax.set_ylim(np.min(pts[:, 1]) - margin, np.max(pts[:, 1]) + margin)
    
    # Plot metric ellipses (only in visible region)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    visible_mask = ((pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1]) &
                   (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1]))
    visible_indices = np.where(visible_mask)[0]
    
    n_ellipses = min(30, len(visible_indices) // 50)
    if n_ellipses > 0 and len(visible_indices) > 0:
        sample_indices = np.random.choice(visible_indices, 
                                         min(n_ellipses, len(visible_indices)), 
                                         replace=False)
        
        for idx in sample_indices:
            pos = pts[idx]
            M = metric_fn(pos)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            
            # Semi-axis lengths: h = 1/sqrt(lambda)
            h1 = 1.0 / math.sqrt(max(eigenvalues[0], 1e-10))
            h2 = 1.0 / math.sqrt(max(eigenvalues[1], 1e-10))
            
            # Angle
            angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
            angle_deg = math.degrees(angle)
            
            ellipse = mpatches.Ellipse(pos, width=2*h1, height=2*h2, 
                                       angle=angle_deg, 
                                       facecolor='blue', alpha=0.15, 
                                       edgecolor='blue', linewidth=1.0)
            ax.add_patch(ellipse)
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.close(fig)


def main():
    print("=" * 70)
    print("SOFIA: Anisotropic Adaptation on NACA0012 Airfoil")
    print("=" * 70)
    
    # 1. Load mesh
    print("\n1. Loading mesh from naca0012_ultra_coarse.msh...")
    points, triangles = read_msh('naca0012_ultra_coarse.msh', verbose=True)
    print(f"   Loaded: {len(points)} vertices, {len(triangles)} triangles")
    
    editor = PatchBasedMeshEditor(points, triangles)
    
    # 2. Identify boundary
    print("\n2. Identifying airfoil boundary...")
    editor._update_maps()
    boundary_vertices = set()
    for (u, v), tris in editor.edge_map.items():
        if len(tris) == 1:
            boundary_vertices.add(u)
            boundary_vertices.add(v)
    print(f"   Found {len(boundary_vertices)} boundary vertices")
    
    # 3. Initial quality
    print("\n3. Initial mesh quality:")
    initial_min_angle = mesh_min_angle(editor.points, editor.triangles)
    print(f"   Min angle: {initial_min_angle:.2f}°")
    print(f"   Triangles: {len(editor.triangles)}")
    
    # 4. Plot initial
    print("\n4. Plotting initial mesh...")
    plot_mesh_with_metric(editor, naca0012_metric, boundary_vertices,
                         title=f"Initial NACA0012 Mesh (Zoomed) - {len(editor.triangles)} triangles",
                         filename="naca0012_initial_zoom.png",
                         zoom_airfoil=True)
    plot_mesh_with_metric(editor, naca0012_metric, boundary_vertices,
                         title=f"Initial NACA0012 Mesh (Full Domain) - {len(editor.triangles)} triangles",
                         filename="naca0012_initial_full.png",
                         zoom_airfoil=False)
    
    # 5. Adapt
    print("\n5. Adapting mesh with anisotropic metric...")
    n_iterations = 3
    total_bnd_splits = 0
    total_int_splits = 0
    total_collapses = 0
    
    for iteration in range(n_iterations):
        print(f"\n   Iteration {iteration + 1}/{n_iterations}:")
        
        # Refine
        bnd_s, int_s = refine_by_metric(editor, naca0012_metric, boundary_vertices,
                                       alpha_split=1.5, max_splits=100)
        print(f"     Splits: {bnd_s} boundary, {int_s} interior")
        total_bnd_splits += bnd_s
        total_int_splits += int_s
        
        # Coarsen
        c = coarsen_interior_by_metric(editor, naca0012_metric, boundary_vertices,
                                      beta_collapse=0.4, max_collapses=50)
        print(f"     Collapses: {c}")
        total_collapses += c
    
    # 6. Compact
    if editor.has_tombstones():
        editor.compact_triangle_indices()
    
    # 7. Final quality
    print("\n6. Final mesh quality:")
    final_min_angle = mesh_min_angle(editor.points, editor.triangles)
    print(f"   Min angle: {final_min_angle:.2f}°")
    print(f"   Triangles: {len(editor.triangles)}")
    print(f"   Vertices: {len(editor.points)}")
    
    # 8. Plot final
    print("\n7. Plotting adapted mesh...")
    plot_mesh_with_metric(editor, naca0012_metric, boundary_vertices,
                         title=f"Adapted NACA0012 Mesh (Zoomed) - {len(editor.triangles)} triangles",
                         filename="naca0012_adapted_zoom.png",
                         zoom_airfoil=True)
    plot_mesh_with_metric(editor, naca0012_metric, boundary_vertices,
                         title=f"Adapted NACA0012 Mesh (Full Domain) - {len(editor.triangles)} triangles",
                         filename="naca0012_adapted_full.png",
                         zoom_airfoil=False)
    
    # 9. Export VTK
    print("\n8. Exporting to VTK...")
    min_angles = np.array([mesh_min_angle(editor.points, tri.reshape(1, -1)) 
                          for tri in editor.triangles])
    write_vtk('naca0012_adapted.vtk', editor.points, editor.triangles,
             cell_data={'min_angle': min_angles})
    print("   Saved: naca0012_adapted.vtk")
    
    # 10. Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Boundary splits:  {total_bnd_splits}")
    print(f"  Interior splits:  {total_int_splits}")
    print(f"  Collapses:        {total_collapses}")
    print(f"  Min angle: {initial_min_angle:.2f}° -> {final_min_angle:.2f}°")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

"""
SOFIA Example: Anisotropic Boundary Adaptation on NACA0012 Airfoil

Demonstrates anisotropic mesh adaptation on a realistic NACA0012 airfoil geometry.
Loads mesh from Gmsh .msh file and applies metric-based boundary layer refinement.

What this example shows:
1. Loading a mesh from Gmsh .msh format
2. Detecting boundary edges automatically
3. Computing anisotropic metric based on distance from airfoil
4. Adapting mesh with boundary protection
5. Visualizing metric field with ellipses
6. Exporting result to VTK for ParaView visualization

Perfect for: CFD preprocessing, airfoil meshing, anisotropic boundary layers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sofia import read_msh, write_vtk, PatchBasedMeshEditor
from sofia.core.quality import mesh_min_angle
from sofia.core.conformity import boundary_edges_from_map


def naca0012_metric(x, airfoil_points, boundary_thickness=0.1):
    """Construct anisotropic metric with strong refinement near airfoil."""
    dists = np.linalg.norm(airfoil_points - x, axis=1)
    min_idx = np.argmin(dists)
    min_dist = dists[min_idx]
    nearest_point = airfoil_points[min_idx]
    
    if min_dist > 1e-10:
        perp_dir = (x - nearest_point) / min_dist
    else:
        perp_dir = np.array([0.0, 1.0])
    
    theta = math.atan2(perp_dir[1], perp_dir[0])
    
    if min_dist < boundary_thickness:
        h_perp = 0.002 + 0.01 * (min_dist / boundary_thickness)
        h_parallel = 0.05
    else:
        decay = math.exp(-2.0 * (min_dist - boundary_thickness) / boundary_thickness)
        h_perp = 0.012 + 0.04 * (1.0 - decay)
        h_parallel = 0.05 + 0.1 * (1.0 - decay)
    
    lam_perp = 1.0 / (h_perp ** 2)
    lam_parallel = 1.0 / (h_parallel ** 2)
    
    c, sn = math.cos(theta), math.sin(theta)
    R = np.array([[c, -sn], [sn, c]])
    L = np.diag([lam_perp, lam_parallel])
    
    return R @ L @ R.T


def compute_metric_edge_length(editor, edge, metric_fn):
    """Compute edge length in metric space."""
    u, v = edge
    p0, p1 = editor.points[u], editor.points[v]
    mid = 0.5 * (p0 + p1)
    M = metric_fn(mid)
    delta = p1 - p0
    return math.sqrt(float(delta @ M @ delta))


def identify_boundary_vertices(editor):
    """Identify vertices on the mesh boundary."""
    boundary_edges = boundary_edges_from_map(editor.edge_map)
    boundary_verts = set()
    for u, v in boundary_edges:
        boundary_verts.add(u)
        boundary_verts.add(v)
    return boundary_verts


def extract_airfoil_boundary_points(editor):
    """Extract airfoil surface points."""
    boundary_verts = identify_boundary_vertices(editor)
    return editor.points[sorted(boundary_verts)]


def adapt_mesh_with_metric(editor, metric_fn, h_target=1.0, 
                           max_iterations=50, protect_boundary=True):
    """Adapt mesh based on metric field."""
    boundary_verts = identify_boundary_vertices(editor) if protect_boundary else set()
    total_ops = 0
    
    for iteration in range(max_iterations):
        n_splits = 0
        edges_to_check = []
        
        for edge in list(editor.edge_map.keys()):
            u, v = edge
            if protect_boundary and u in boundary_verts and v in boundary_verts:
                continue
            metric_len = compute_metric_edge_length(editor, edge, metric_fn)
            edges_to_check.append((edge, metric_len))
        
        edges_to_check.sort(key=lambda x: abs(x[1] - h_target), reverse=True)
        
        for edge, metric_len in edges_to_check:
            if metric_len > 1.5 * h_target:
                if editor.split_edge_delaunay(edge):
                    n_splits += 1
                    if n_splits >= 20:
                        break
        
        total_ops += n_splits
        print(f"Iteration {iteration + 1}: {n_splits} splits")
        
        if n_splits == 0:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return total_ops


def plot_mesh_with_metric(editor, metric_fn, title, filename, 
                          show_metric_ellipses=True, n_ellipses=30):
    """Visualize mesh with metric field."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    for tri in editor.triangles:
        if np.all(tri == -1):
            continue
        vertices = editor.points[tri]
        tri_closed = np.vstack([vertices, vertices[0]])
        ax.plot(tri_closed[:, 0], tri_closed[:, 1], 'b-', linewidth=0.5, alpha=0.4)
    
    boundary_edges = boundary_edges_from_map(editor.edge_map)
    for u, v in boundary_edges:
        p0, p1 = editor.points[u], editor.points[v]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=2.0)
    
    if show_metric_ellipses:
        x_min, y_min = editor.points.min(axis=0)
        x_max, y_max = editor.points.max(axis=0)
        x_samples = np.linspace(x_min, x_max, int(np.sqrt(n_ellipses)))
        y_samples = np.linspace(y_min, y_max, int(np.sqrt(n_ellipses)))
        
        for x_pos in x_samples:
            for y_pos in y_samples:
                pos = np.array([x_pos, y_pos])
                M = metric_fn(pos)
                eigenvalues, eigenvectors = np.linalg.eigh(M)
                width = 2.0 / math.sqrt(eigenvalues[0])
                height = 2.0 / math.sqrt(eigenvalues[1])
                angle = math.degrees(math.atan2(eigenvectors[1, 1], eigenvectors[0, 1]))
                ellipse = mpatches.Ellipse(pos, width, height, angle=angle,
                                          facecolor='yellow', edgecolor='orange',
                                          alpha=0.3, linewidth=1)
                ax.add_patch(ellipse)
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    """Main execution."""
    msh_file = "naca0012.msh"
    
    if not os.path.exists(msh_file):
        print(f"ERROR: Mesh file '{msh_file}' not found!")
        print("\nTo run this example:")
        print("1. Generate a NACA0012 mesh using Gmsh")
        print("2. Save as 'naca0012.msh' in current directory")
        print("3. Mesh should be 2D triangular with airfoil as boundary")
        sys.exit(1)
    
    print("="*70)
    print("SOFIA: Anisotropic Adaptation on NACA0012 Airfoil")
    print("="*70)
    
    print("\n1. Loading mesh from meshes/naca0012_ultra_coarse.msh...")
    points, triangles = read_msh(msh_file, verbose=True)
    print(f"   Loaded: {len(points)} vertices, {len(triangles)} triangles")
    
    editor = PatchBasedMeshEditor(points, triangles, 
                                   use_incremental_structures=True,
                                   reject_crossing_edges=True)
    
    print("\n2. Identifying airfoil boundary...")
    airfoil_points = extract_airfoil_boundary_points(editor)
    print(f"   Found {len(airfoil_points)} boundary vertices")
    
    metric_fn = lambda x: naca0012_metric(x, airfoil_points, boundary_thickness=0.05)
    
    initial_min_angle = mesh_min_angle(editor.points, editor.triangles)
    print(f"\n3. Initial mesh quality:")
    print(f"   Min angle: {initial_min_angle:.2f}°")
    print(f"   Triangles: {len(editor.triangles)}")
    
    print("\n4. Plotting initial mesh...")
    plot_mesh_with_metric(editor, metric_fn, 
                         "Initial NACA0012 Mesh with Metric Field",
                         "naca0012_initial.png",
                         show_metric_ellipses=True, n_ellipses=20)
    
    print("\n5. Adapting mesh with anisotropic metric...")
    n_ops = adapt_mesh_with_metric(editor, metric_fn, 
                                   h_target=1.0, 
                                   max_iterations=30,
                                   protect_boundary=True)
    print(f"   Total operations: {n_ops}")
    
    if editor.has_tombstones():
        editor.compact_triangle_indices()
    
    final_min_angle = mesh_min_angle(editor.points, editor.triangles)
    print(f"\n6. Final mesh quality:")
    print(f"   Min angle: {final_min_angle:.2f}°")
    print(f"   Triangles: {len(editor.triangles)}")
    print(f"   Vertices: {len(editor.points)}")
    
    print("\n7. Plotting adapted mesh...")
    plot_mesh_with_metric(editor, metric_fn,
                         "Adapted NACA0012 Mesh with Metric Field",
                         "naca0012_adapted.png",
                         show_metric_ellipses=True, n_ellipses=20)
    
    print("\n8. Exporting to VTK...")
    from sofia.core.geometry import triangles_min_angles
    min_angles = triangles_min_angles(editor.points, editor.triangles)
    write_vtk("naca0012_adapted.vtk", editor.points, editor.triangles,
              cell_data={'min_angle': min_angles},
              title="NACA0012 Adapted Mesh")
    print("   Saved: naca0012_adapted.vtk (open in ParaView)")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Initial: {initial_min_angle:.2f}° → Final: {final_min_angle:.2f}°")
    print(f"Adaptation operations: {n_ops}")
    print("\nOutput files:")
    print("  - naca0012_initial.png")
    print("  - naca0012_adapted.png")
    print("  - naca0012_adapted.vtk")


if __name__ == "__main__":
    main()
