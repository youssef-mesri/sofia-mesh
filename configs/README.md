# Sofia Demo Configs

This folder contains example JSON configuration files for the demo drivers.
Both files follow the same pattern used by `remesh_driver`: you can either
wrap the configuration under a `{"type": ..., "config": { ... }}` envelope
or provide the raw config object directly.


## Files

- `coarsening_scenario.json` — A small scenario to coarsen a mesh by remove a node  or collapsing an edge; used by `demos/coarsening_scenario.py`.
- `coarsening_scenario2_h2.json`
- `coarsening_scenario_h2.json`
- `coarsening_scenario_h4.json`
- `generate_scenario_h_quad.json` — A small scenario to generate a mesh from boundary ploygons by fill_pocket and edge_split; used by `demos/generate_scenario.py`.
- `letter_o_scenario.json` — A letter O polygon shape to triangulate
- `sample_polygon_pentagon.json` — A pentagon shape to triangulate
- `sample_polygon_square.json` — A square shape to triangulate
- `o_polygon.json` — A O shape to triangulate
- `s_polygon.json` — A S shape to triangulate
- `u_polygon.json` — A U shape to triangulate
- `patch_cfg.json`
- `patch_config.json` — Example config for the patch/batch driver (topology mode in the partition demo). Keys mirror `PatchDriverConfig`.
- `refinement_scenario.json` — A small scenario to refine a mesh by adding a node at a triangle centroid and splitting an edge; used by `demos/refinement_scenario.py`.
- `refinement_scenario_h2.json` — Drives the mesh size h to h/2 using the simplified refinement loop.
- `refinement_scenario_h2_quad.json`
- `refinement_scenario_h2_smooth.json`
- `greedy_config.json` — Example config for the greedy smoother used in the partitioned demo (greedy mode). Keys mirror `GreedyConfig`.
- `greedy_cfg.json`


## Using with the partition demo

Greedy (faster, no topology changes):

```bash
python demos/partition_parallel.py \
  --npts 200 --parts 4 --workers 4 \
  --log-level INFO \
  --config-json configs/greedy_config.json
```

Topology (slower, per-part patch driver + merge):

```bash
python demos/partition_parallel.py \
  --npts 200 --parts 4 --workers 4 \
  --allow-topology \
  --log-level INFO \
  --config-json configs/patch_config.json
```
- Refinement scenario (explicit ops add_node/split_edge):

```bash
python demos/refinement_scenario.py --scenario configs/refinement_scenario.json --log-level INFO
```

- Target h refinement (simple scan):

```bash
python demos/refinement_scenario.py --scenario configs/refinement_scenario_h2.json --log-level INFO
```


Notes:
- The demo loads the JSON and filters unknown fields based on the target config.
- In greedy mode, only a `GreedyConfig` is consumed. In topology mode, a
  `PatchDriverConfig` is consumed.
- You may also pass raw config objects without the `type`/`config` envelope.
  The demo will infer the type from present keys.

## Key Options

### Refinement Scenario (auto)
- Standard phases (ignored when `target_h_factor` is set):
  - Optional legacy steps: `refine_large_tris` (centroid insert) and `split_long_edges` (average-based), with `order` and per-pass caps (`max_tri_ops`, `max_edge_ops`).
- Collapse (optional, global):
  - `collapse_shortest_edges` (bool): attempt collapsing the shortest interior edges
  - `collapse_k` (int): target number of collapses (best-effort, stops early if quality guard blocks)
  - `collapse_order` ("before"|"after"): run the collapse phase before or after the refine/split phases (default after)
- Target h refinement (simple scan; skips standard phases when enabled):
  - `target_h_factor` (float|null): if set (e.g., 0.5 for h/2, 0.25 for h/4), drives the mesh size h toward `initial_h * factor` by splitting long internal edges.
  - `h_metric` (string): which h to measure/drive.
  - `max_h_iters` (int): max iterations (default 10)
  - `max_h_splits_per_iter` (int|null): limit splits per iteration (optional)
  - `h_tolerance` (float): tolerance around target h (default 1e-6)
  - `enforce_split_quality` (bool): if true, `split_edge` enforces a non-worsening worst-angle gate; for refinement, set false to allow more progress.
- Smoothing (optional, global):
  - `move_to_barycenter` (bool): after add/split, move interior vertices to the average of their neighbors
  - `barycenter_passes` (int): how many smoothing passes to run (default 1)

#### Supported h_metric values

- `avg_internal_edge_length`: average length of interior edges (default)
- `median_internal_edge_length`: median interior edge length
- `avg_longest_edge`: for each triangle, take its longest edge; average over all triangles
- `median_longest_edge`: median of the per-triangle longest-edge lengths
- `avg_equilateral_h`: average equilateral side s computed from triangle area A via $s = \sqrt{\frac{4A}{\sqrt{3}}}$

### GreedyConfig
- `max_vertex_passes` (int): number of vertex passes
- `max_edge_passes` (int): number of edge passes
- `strict` (bool): enable strict conformity checks (loops/crossings) during ops
- `reject_crossings` (bool): simulate crossings and reject if any
- `reject_new_loops` (bool): reject if operation increases boundary loop count
- `force_pocket_fill` (bool): force pocket fill post-run
- `verbose` (bool): more detailed logs
- `gif_capture`, `gif_dir`, `gif_out`, `gif_fps`: GIF capture options

### PatchDriverConfig (subset)
- `threshold` (float): minimum angle target (deg)
- `max_iterations` (int): max patch iterations per worker
- `patch_radius` (int): patch neighborhood radius
- `top_k` (int): candidate selection size per iteration
- `disjoint_on` ("tri"|"vertex"): disjoint constraint
- `allow_overlap` (bool): allow overlapping patches
- `batch_attempts` (int): attempts per batch
- `min_triangle_area` (float): area guard
- `reject_min_angle_deg` (float|null): reject if below
- `reject_new_loops`, `reject_crossings` (bool): guards
- `auto_fill_pockets` (bool): pocket filling
- `angle_unit` ("deg"|"rad")
- `plot_every` (int): plotting interval (large for off)
- `use_greedy_remesh` (bool): optional greedy fallback per-iteration
- `gif_capture`, `gif_dir`, `gif_out`, `gif_fps`

## Tips
- Start with small `max_iterations` in topology mode (e.g., 5–10) to validate
  the end-to-end merge quickly.
- The partition demo colors the full domain per-part before/after; to visualize
  only processed interior triangles, we can add an overlay on request.