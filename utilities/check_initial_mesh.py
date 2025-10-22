import numpy as np
from sofia.core.conformity import check_mesh_conformity

def mesh_with_boundary_vertex_degree_three():
    pts = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1 (boundary, target to remove)
        [1.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [0.5, 0.3],  # 4 interior near bottom
        [0.7, 0.6],  # 5 interior near right
    ], dtype=float)
    tris = np.array([
        [4, 0, 1],  # involves 1
        [4, 1, 5],  # involves 1
        [5, 1, 2],  # involves 1
        [4, 0, 3],
        [4, 3, 2],
        [5, 2, 4],
    ], dtype=int)
    return pts, tris

pts, tris = mesh_with_boundary_vertex_degree_three()
print("Testing initial mesh conformity...")
ok, msgs = check_mesh_conformity(pts, tris, allow_marked=False)
print(f"Result: ok={ok}")
if not ok:
    print(f"Errors: {msgs}")
    
# Check specifically for edges (2,4) and (1,5)
print(f"\nEdge (2,4): {pts[2]} -> {pts[4]}")
print(f"Edge (1,5): {pts[1]} -> {pts[5]}")

# Check which triangles contain these edges
print("\nTriangles containing edge (2,4):")
for i, t in enumerate(tris):
    edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
    for e in edges:
        if set(e) == {2, 4}:
            print(f"  Triangle {i}: {t}")

print("\nTriangles containing edge (1,5):")
for i, t in enumerate(tris):
    edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
    for e in edges:
        if set(e) == {1, 5}:
            print(f"  Triangle {i}: {t}")
