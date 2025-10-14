# Scripts

This folder contains small utilities for fuzzing, tracing, reproducing, and summarizing remeshing runs. Most scripts assume you run them from the repository root so relative paths like `diagnostics/` resolve correctly.

Quick check:
- Run `python scripts/doctor.py` to verify prerequisites (imports, `./diagnostics` folder, expected NPZ files) and get next steps if something’s missing.

## Prerequisites
- Python 3.8+ and NumPy installed.
- Run from the repo root: `python scripts/<script>.py`
- Some scripts expect inputs in `diagnostics/` (e.g., `gfail_seed_0.npz`, `gtrace_seed_0.npz`). Create the folder first if it doesn’t exist.

## Diagnostics directory
- Location: `./diagnostics/`
- Typical files:
  - `gfail_seed_<seed>.npz`: Before/after failing meshes from fuzzing.
  - `gtrace_seed_<seed>.npz`: Per-operation compacted snapshots from a traced run.
  - Per-commit failure files created by “per_commit_*” scripts.

---

## Script reference

### compact_check_runner_seed0.py
Run `greedy_remesh` starting from `diagnostics/gfail_seed_0.npz` and perform a compact+strict check after each pass via a hook. On failure, writes a compacted snapshot and exits.

Usage:
- python scripts/compact_check_runner_seed0.py

Requires:
- diagnostics/gfail_seed_0.npz

---

### find_matching_snapshot.py
Find the earliest per-op compacted snapshot in `gtrace` that matches the final failing mesh from `gfail_seed_0.npz`. Tries exact index equality, then geometry set equality.

Usage:
- python scripts/find_matching_snapshot.py

Requires:
- diagnostics/gfail_seed_0.npz
- diagnostics/gtrace_seed_0.npz

---

### fuzz_greedy_remesh.py
Fuzz `greedy_remesh` across random seeds and save diagnostics for failing cases. Writes `gfail_seed_<seed>.npz` into `./diagnostics/`.

Usage:
- python scripts/fuzz_greedy_remesh.py

Notes:
- Edit the default parameters in the `run_fuzz(...)` call at the bottom to change seed count, passes, etc.

---

### generate_boundary_screenshots.py
Generate fresh screenshots demonstrating boundary loop rendering modes into `docs/`.

Usage:
- python scripts/generate_boundary_screenshots.py

Outputs:
- docs/boundary_loops_per_loop.png
- docs/boundary_loops_uniform_labeled.png

---

### generate_test_snapshots.py
Generate additional “before/after” snapshots useful for docs and tests.

Usage:
- python scripts/generate_test_snapshots.py

Outputs (in docs/):
- greedy_skinny_before.png / greedy_skinny_after.png
- greedy_heptagon_before.png / greedy_heptagon_after.png
- crossing_rejection.png
- pocket_fill_before.png / pocket_fill_after.png

---

### per_commit_compact_check_seed0.py
Replay `greedy_remesh` from `gfail_seed_0.npz` and run a compact+strict check after every committing operation. On first compacted failure, saves a diagnostic NPZ and exits.

Usage:
- python scripts/per_commit_compact_check_seed0.py

Requires:
- diagnostics/gfail_seed_0.npz

---

### per_commit_prepost_save_seed0.py
Enhanced per-commit checker: when a compacted failure is detected, save both pre- and post-commit compacted meshes, raw arrays, and op details.

Usage:
- python scripts/per_commit_prepost_save_seed0.py

Requires:
- diagnostics/gfail_seed_0.npz

---

### replay_gfail_seed0_stepby_step.py
Replay a trace stored in `diagnostics/trace_seed0.npz` step-by-step, applying recorded actions and checking compacted conformity after each commit. Writes the first offending step if detected.

Usage:
- python scripts/replay_gfail_seed0_stepby_step.py

Requires:
- diagnostics/trace_seed0.npz (created by `trace_seed0.py`)

---

### reproduce_gfail_seed0_from_fuzz.py
Re-run `greedy_remesh` starting from the failing snapshot `gfail_seed_0.npz` and wrap `apply_patch_operation` to detect the first committing op that causes compacted non-conformity. Saves a diagnostic NPZ and exits when found.

Usage:
- python scripts/reproduce_gfail_seed0_from_fuzz.py

Requires:
- diagnostics/gfail_seed_0.npz

---

### run_greedy_full_op_tracer.py
Run `greedy_remesh` for a given seed and record every `apply_patch_operation` outcome and compacted snapshot. Saves `diagnostics/gtrace_seed_<seed>.npz`.

Usage:
- python scripts/run_greedy_full_op_tracer.py [seed]

Requires:
- diagnostics/gfail_seed_<seed>.npz (provides the starting mesh)

---

### summarize_dumps.py
Summarize pocket and “reject-min-angle” dumps at the repo root, printing counts and simple boundary-edge stats.

Usage:
- python scripts/summarize_dumps.py

---

### trace_seed0.py
Trace operations on a random mesh (seed=0) until a duplication or non-manifold condition is observed (flips disabled by default). Saves `diagnostics/trace_seed0.npz` with per-op snapshots and issue summaries.

Usage:
- python scripts/trace_seed0.py

Outputs:
- diagnostics/trace_seed0.npz

---

### trace_seed0_replay.py
Replay a batch/attempt loop akin to the driver but instrumented; stop at the first detected issue. Configurable via CLI flags.

Usage:
- python scripts/trace_seed0_replay.py --seed 0 [--allow-flips] [--max-iters 50] [--batch-attempts 2] [--patch-radius 1] [--top-k 80]

Outputs:
- diagnostics/trace_seed<seed>_replay_flips_{on|off}.npz (or `_noissue` variant)

---

## Tips
- Always seed Python and NumPy before reproducing a trace to ensure deterministic behavior.
- If you use compact-check hooks, remember the driver reads them from a per-run context. Some scripts set both a legacy global and the run context for safety.
- If an import fails, ensure you’re running from the repo root so `sofia/` can be imported, or add the root to `PYTHONPATH`.
