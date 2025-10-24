import numpy as np
from geometry import EPS_AREA
from scipy.spatial import Delaunay
from sofia.core.mesh_modifier2 import (
    PatchBasedMeshEditor,
    build_edge_to_tri_map,
    build_vertex_to_tri_map,
    retriangulate_patch_strict,
    remove_node_star,
)
from sofia.core.quality import mesh_min_angle

# -----------------------
# Action space
# -----------------------
class ActionSpace:
    def __init__(self):
        # noop, flip, split, add, remove
        self.actions = ["noop", "flip", "split", "add", "remove"]
        self.total_size = len(self.actions)

    def sample(self):
        return np.random.randint(0, self.total_size)

    def __len__(self):
        return self.total_size


# -----------------------
# Mesh optimization environment
# -----------------------
class MeshPatchEnvCustom:
    def __init__(self, points, triangles, max_steps=200, max_points=200):
        #self.orig_points = np.asarray(points).copy()
        #self.orig_triangles = np.asarray(triangles, dtype=int).copy()
        self.points = None
        self.triangles = None
        self.editor = PatchBasedMeshEditor(points, triangles)
        
        self.max_steps = max_steps
        self.max_points = max_points
        self.step_count = 0

        self.action_space = ActionSpace()
        self.action_space.total_size = 5  # {flip, split, add, remove, noop}
        # Observation locale
        #self.obs_size = len(self.editor.points) * 2 + len(self.editor.triangles) * 3
        # Observation globale = [global_min_angle, n_points, n_triangles]
        self.obs_size = 3

        self.reset()

    def reset(self):
        self.points = self.editor.points
        self.triangles = self.editor.triangles
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        #pts = self.editor.points.flatten()
        #tris = self.editor.triangles.flatten()
        #obs = np.concatenate([pts, tris])
        #return obs[:self.obs_size]
        min_angle = mesh_min_angle(self.points, self.triangles)
        return np.array([min_angle, len(self.points), len(self.triangles)], dtype=np.float32)

    def get_action_mask(self):
        mask = np.zeros(self.action_space.total_size, dtype=np.int32)

        # noop always valid
        mask[0] = 1

        # flip valid if there is at least one interior edge
        edge_map = build_edge_to_tri_map(self.triangles)
        if any(len(lst) == 2 for lst in edge_map.values()):
            mask[1] = 1

        # split valid if at least one edge
        if len(edge_map) > 0 and len(self.points) < self.max_points:
            mask[2] = 1

        # add always valid if below max_points
        if len(self.points) < self.max_points:
            mask[3] = 1

        # remove valid if there exists an interior vertex
        vmap = build_vertex_to_tri_map(self.triangles)
        for v, incident_tris in vmap.items():
            if not incident_tris:
                continue
            incident_edges = []
            for t in incident_tris:
                tri = self.triangles[t]
                for i in range(3):
                    a = int(tri[i])
                    b = int(tri[(i+1) % 3])
                    if v in (a, b):
                        incident_edges.append(tuple(sorted((a, b))))
            if incident_edges and all(len(edge_map[e]) == 2 for e in incident_edges):
                mask[4] = 1
                break

        return mask

    # -----------------------
    # Local ops
    # -----------------------
    def _edge_flip(self, edge):
        """Flip edge if possible, return True on success"""
        edge_map = build_edge_to_tri_map(self.triangles)
        if edge not in edge_map or len(edge_map[edge]) != 2:
            return False

        t1, t2 = edge_map[edge]
        tri1 = self.triangles[t1].tolist()
        tri2 = self.triangles[t2].tolist()

        a, b = edge
        opp1 = [v for v in tri1 if v not in edge][0]
        opp2 = [v for v in tri2 if v not in edge][0]

        new_tri1 = [opp1, opp2, a]
        new_tri2 = [opp1, opp2, b]

        # Area check
        def area(tri):
            p0, p1, p2 = [self.points[int(i)] for i in tri]
            return 0.5 * np.cross(p1 - p0, p2 - p0)

        if abs(area(new_tri1)) < EPS_AREA or abs(area(new_tri2)) < EPS_AREA:
            return False

        new_tris = self.triangles.copy()
        new_tris[t1] = new_tri1
        new_tris[t2] = new_tri2
        self.triangles = new_tris
        return True

    # -----------------------
    # Step
    # -----------------------
    def step(self, action):
        prev_quality = mesh_min_angle(self.points, self.triangles)
        success = False

        if action == 0:  # noop
            success = True

        elif action == 1:  # flip
            edge_map = build_edge_to_tri_map(self.triangles)
            interior_edges = [e for e, l in edge_map.items() if len(l) == 2]
            if interior_edges:
                e = interior_edges[np.random.randint(len(interior_edges))]
                success = self._edge_flip(e)

        elif action == 2:  # split
            edge_map = build_edge_to_tri_map(self.triangles)
            if edge_map:
                e = list(edge_map.keys())[np.random.randint(len(edge_map))]
                a, b = e
                mid = 0.5 * (self.points[a] + self.points[b])
                vmap = build_vertex_to_tri_map(self.triangles)
                tri_indices = sorted(set(vmap.get(int(a), []) + vmap.get(int(b), [])))
                new_pts, new_tris, ok = retriangulate_patch_strict(
                    self.points, self.triangles, tri_indices, new_point_coords=[mid], strict_mode="all_vertices"
                )
                if ok:
                    self.points, self.triangles = new_pts, new_tris
                    success = True

        elif action == 3:  # add
            tri_idx = np.random.randint(len(self.triangles))
            tri = self.triangles[tri_idx]
            centroid = np.mean(self.points[tri], axis=0)
            vmap = build_vertex_to_tri_map(self.triangles)
            tri_indices = sorted(
                set(vmap.get(int(tri[0]), []) + vmap.get(int(tri[1]), []) + vmap.get(int(tri[2]), []))
            )
            new_pts, new_tris, ok = retriangulate_patch_strict(
                self.points, self.triangles, tri_indices, new_point_coords=[centroid], strict_mode="all_vertices"
            )
            if ok:
                self.points, self.triangles = new_pts, new_tris
                success = True

        elif action == 4:  # remove
            vmap = build_vertex_to_tri_map(self.triangles)
            interior_vertices = []
            edge_map = build_edge_to_tri_map(self.triangles)
            for v, tri_list in vmap.items():
                incident_edges = []
                for t in tri_list:
                    tri = self.triangles[t]
                    for i in range(3):
                        a = int(tri[i])
                        b = int(tri[(i+1) % 3])
                        if v in (a, b):
                            incident_edges.append(tuple(sorted((a, b))))
                if incident_edges and all(len(edge_map[e]) == 2 for e in incident_edges):
                    interior_vertices.append(v)
            if interior_vertices:
                v = np.random.choice(interior_vertices)
                succ, msg, new_pts, new_tris = remove_node_star(self.points, self.triangles, v)
                if succ:
                    self.points, self.triangles = new_pts, new_tris
                    success = True

        new_quality = mesh_min_angle(self.points, self.triangles)
        reward = new_quality - prev_quality
        reward -= 0.001  # petit coût temporel

        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_obs(), float(reward), done, {"success": success}
