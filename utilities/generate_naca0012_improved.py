#!/usr/bin/env python3
"""
Generate a high-quality NACA0012 airfoil mesh with proper boundary layer clustering.

Based on NASA TMR best practices:
- Corrected scaled NACA 0012 formula with sharp trailing edge
- Exponential clustering near airfoil surface
- Structured-like point distribution in boundary layer
- Farfield at ~10-20 chord lengths
"""

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


def naca0012_nasa_tmr(n_points=150):
    """Generate NACA0012 airfoil using NASA TMR corrected formula.
    
    NASA TMR scaled NACA 0012 with sharp trailing edge:
    y = ± 0.594689181*[0.298222773*sqrt(x) - 0.127125232*x - 0.357907906*x^2 
                        + 0.291984971*x^3 - 0.105174606*x^4]
    
    Returns:
        x, y: Arrays of coordinates going counter-clockwise from TE
    """
    # Use higher-order cosine clustering for excellent LE/TE resolution
    theta = np.linspace(0, np.pi, n_points)
    # Double cosine for extra clustering at LE
    beta = 0.5 * (1 - np.cos(theta))
    x = 0.5 * (1 - np.cos(beta * np.pi))
    
    # NASA TMR corrected coefficients
    a = np.array([0.298222773, -0.127125232, -0.357907906, 0.291984971, -0.105174606])
    scale = 0.594689181
    
    # Thickness distribution
    yt = scale * (a[0] * np.sqrt(x) + a[1] * x + a[2] * x**2 + a[3] * x**3 + a[4] * x**4)
    
    # Create closed airfoil: TE upper -> LE -> TE lower
    x_upper = x[::-1]  # Reverse for clockwise from TE
    y_upper = yt[::-1]
    
    x_lower = x[1:]  # Skip LE duplicate
    y_lower = -yt[1:]
    
    x_airfoil = np.concatenate([x_upper, x_lower])
    y_airfoil = np.concatenate([y_upper, y_lower])
    
    return x_airfoil, y_airfoil


def generate_boundary_layer_points(x_airfoil, y_airfoil, n_layers=15, 
                                   first_height=0.005, growth_rate=1.2):
    """Generate structured boundary layer points around airfoil.
    
    Args:
        x_airfoil, y_airfoil: Airfoil surface coordinates
        n_layers: Number of boundary layer mesh layers
        first_height: Height of first cell (wall-adjacent)
        growth_rate: Geometric growth rate for layer spacing
    
    Returns:
        points: Array of (x, y) coordinates
    """
    n_surf = len(x_airfoil)
    
    # Compute normals at each surface point
    dx = np.gradient(x_airfoil)
    dy = np.gradient(y_airfoil)
    
    # Normal vectors (perpendicular to tangent, pointing outward)
    nx = -dy / np.sqrt(dx**2 + dy**2)
    ny = dx / np.sqrt(dx**2 + dy**2)
    
    # Generate layers with geometric growth
    points = []
    for i in range(n_surf):
        for layer in range(n_layers):
            if layer == 0:
                height = 0  # On surface
            else:
                # Geometric progression
                height = first_height * (growth_rate**(layer - 1) - 1) / (growth_rate - 1)
                height += first_height
            
            x_bl = x_airfoil[i] + height * nx[i]
            y_bl = y_airfoil[i] + height * ny[i]
            points.append([x_bl, y_bl])
    
    return np.array(points)


def generate_farfield_points(n_points=40, radius=10.0):
    """Generate circular farfield boundary points.
    
    Args:
        n_points: Number of points on farfield
        radius: Farfield radius (in chord lengths)
    
    Returns:
        Array of (x, y) farfield points
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y])


def generate_wake_refinement_points(n_points=100):
    """Generate points in wake region for refinement.
    
    Args:
        n_points: Number of wake points
    
    Returns:
        Array of (x, y) wake region points
    """
    # Create points in rectangle behind airfoil
    rng = np.random.RandomState(42)
    
    # Wake region: x in [1, 3], y in [-0.5, 0.5]
    x_wake = rng.uniform(1.0, 3.0, n_points)
    y_wake = rng.uniform(-0.5, 0.5, n_points)
    
    return np.column_stack([x_wake, y_wake])


def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def create_naca0012_mesh_improved():
    """Create high-quality NACA0012 mesh with boundary layer clustering."""
    
    print("Generating improved NACA0012 mesh...")
    
    # 1. Generate airfoil surface
    x_airfoil, y_airfoil = naca0012_nasa_tmr(n_points=150)
    
    # Translate to origin
    x_center = 0.5
    x_airfoil = x_airfoil - x_center
    
    # 2. Generate boundary layer points
    bl_points = generate_boundary_layer_points(
        x_airfoil, y_airfoil, 
        n_layers=12, 
        first_height=0.003,  # Very fine first layer
        growth_rate=1.25      # Moderate growth
    )
    
    # 3. Generate farfield
    ff_points = generate_farfield_points(n_points=50, radius=15.0)
    
    # 4. Generate wake refinement
    wake_points = generate_wake_refinement_points(n_points=150)
    
    # 5. Generate field points (outside BL, inside farfield)
    rng = np.random.RandomState(123)
    field_points = []
    n_field = 400
    attempts = 0
    max_attempts = n_field * 20
    
    airfoil_polygon = np.column_stack([x_airfoil, y_airfoil])
    
    while len(field_points) < n_field and attempts < max_attempts:
        attempts += 1
        
        # Random point in farfield circle
        r = rng.uniform(0.5, 14.0)
        theta = rng.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Check if outside airfoil and reasonable distance from BL
        if not point_in_polygon([x, y], airfoil_polygon):
            dists_surf = np.sqrt((x - x_airfoil)**2 + (y - y_airfoil)**2)
            if np.min(dists_surf) > 0.3:  # Outside boundary layer region
                field_points.append([x, y])
    
    field_points = np.array(field_points)
    
    # 6. Combine all points
    all_points = np.vstack([
        bl_points,
        ff_points,
        wake_points,
        field_points
    ])
    
    print(f"  Total points before triangulation: {len(all_points)}")
    
    # 7. Delaunay triangulation
    tri = Delaunay(all_points)
    triangles = tri.simplices
    
    print(f"  Triangles from Delaunay: {len(triangles)}")
    
    # 8. Filter triangles: keep only those outside airfoil
    valid_triangles = []
    for triangle in triangles:
        tri_pts = all_points[triangle]
        centroid = tri_pts.mean(axis=0)
        
        # Keep if centroid outside airfoil
        if not point_in_polygon(centroid, airfoil_polygon):
            valid_triangles.append(triangle)
    
    triangles = np.array(valid_triangles, dtype=np.int32)
    
    print(f"\nGenerated mesh:")
    print(f"  Vertices: {len(all_points)}")
    print(f"  Triangles: {len(triangles)}")
    
    return all_points, triangles


def write_gmsh_msh(filename, points, triangles):
    """Write mesh in Gmsh v2.2 ASCII format."""
    with open(filename, 'w') as f:
        # Header
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # Nodes
        f.write("$Nodes\n")
        f.write(f"{len(points)}\n")
        for i, (x, y) in enumerate(points, 1):
            f.write(f"{i} {x:.16e} {y:.16e} 0.0\n")
        f.write("$EndNodes\n")
        
        # Elements (triangles only, element type 2)
        f.write("$Elements\n")
        f.write(f"{len(triangles)}\n")
        for i, tri in enumerate(triangles, 1):
            # Format: elm-number elm-type number-of-tags <tags> node-indices
            # Type 2 = 3-node triangle
            # Tags: physical-tag geometric-tag
            f.write(f"{i} 2 2 1 1 {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        f.write("$EndElements\n")


def plot_mesh_preview(points, triangles, filename="naca0012_mesh_preview.png"):
    """Create visualization of generated mesh."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Full mesh
    ax1.triplot(points[:, 0], points[:, 1], triangles, 'b-', lw=0.3, alpha=0.4)
    ax1.plot(points[:, 0], points[:, 1], 'k.', ms=0.5, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Full Mesh')
    ax1.set_xlabel('x/c')
    ax1.set_ylabel('y/c')
    ax1.grid(True, alpha=0.3)
    
    # Zoom on airfoil
    ax2.triplot(points[:, 0], points[:, 1], triangles, 'b-', lw=0.5, alpha=0.5)
    ax2.plot(points[:, 0], points[:, 1], 'r.', ms=1, alpha=0.5)
    ax2.set_xlim([-0.6, 0.6])
    ax2.set_ylim([-0.3, 0.3])
    ax2.set_aspect('equal')
    ax2.set_title('Airfoil Region (Zoomed)')
    ax2.set_xlabel('x/c')
    ax2.set_ylabel('y/c')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved preview: {filename}")


if __name__ == "__main__":
    # Generate mesh
    points, triangles = create_naca0012_mesh_improved()
    
    # Write to file
    output_file = "meshes/naca0012.msh"
    write_gmsh_msh(output_file, points, triangles)
    print(f"\nSaved to: {output_file}")
    
    # Create preview
    plot_mesh_preview(points, triangles)
    
    print("\nYou can now run:")
    print("  python examples/anisotropic_boundary_adaptation2.py")
