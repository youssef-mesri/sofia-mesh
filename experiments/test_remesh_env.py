import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from remesh_environment import MeshPatchEnvCustom

# -----------------------
# Build initial Delaunay mesh
# -----------------------
def build_random_delaunay(npts=40, seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.rand(npts, 2)
    corners = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    pts = np.vstack((pts, corners))
    tri = Delaunay(pts)
    return pts, tri.simplices.copy()

# -----------------------
# Demo
# -----------------------
if __name__ == "__main__":
    pts, tris = build_random_delaunay(40, seed=2025)

    env = MeshPatchEnvCustom(pts, tris, max_steps=50, max_points=150)
    obs = env.reset()

    print("Initial obs:", obs)

    plt.figure(figsize=(5,5))
    plt.triplot(env.points[:,0], env.points[:,1], env.triangles)
    plt.scatter(env.points[:,0], env.points[:,1], s=6)
    plt.gca().set_aspect("equal")
    plt.title("Initial mesh")
    plt.savefig("mesh_initial.png", dpi=150)
    plt.close()

    for step in range(10):
        mask = env.get_action_mask()
        valid_actions = np.where(mask == 1)[0]
        action = np.random.choice(valid_actions)  # choose valid action
        obs, reward, done, info = env.step(action)
        print(f"Step {step}: action={env.action_space.actions[action]}, reward={reward:.3f}, success={info['success']}")

        if done:
            break

    plt.figure(figsize=(5,5))
    plt.triplot(env.points[:,0], env.points[:,1], env.triangles)
    plt.scatter(env.points[:,0], env.points[:,1], s=6)
    plt.gca().set_aspect("equal")
    plt.title("Final mesh after 10 actions")
    plt.savefig("mesh_final.png", dpi=150)
    plt.close()

    print("Images saved: mesh_initial.png, mesh_final.png")
