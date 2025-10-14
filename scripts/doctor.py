#!/usr/bin/env python3
"""Tiny scripts doctor: verify expected inputs and print next steps.

Checks common prerequisites for the scripts in this folder and suggests
what to run to generate missing artifacts.

Examples:
  python scripts/doctor.py                  # check defaults (seed=0)
  python scripts/doctor.py --seed 7         # check for seed 7
  python scripts/doctor.py --create-diagnostics  # create ./diagnostics if missing
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0, help='Seed to check for diagnostics artifacts (default: 0)')
    p.add_argument('--create-diagnostics', action='store_true', help='Create ./diagnostics if it does not exist')
    p.add_argument('--verbose', action='store_true', help='Verbose output')
    args = p.parse_args()

    here = Path(__file__).resolve()
    root = here.parents[1]
    diag = root / 'diagnostics'

    print('Scripts doctor')
    print('- repo root:', root)
    print('- working dir:', Path.cwd())
    if Path.cwd() != root:
        print('[WARN] You are not running from the repo root. Some scripts assume CWD == repo root.')
        print(f'       Consider:\n         - cd {root}')

    # Import check
    def try_import() -> bool:
        try:
            import sofia.sofia.remesh_driver  # noqa: F401
            import sofia.sofia.mesh_modifier2  # noqa: F401
            return True
        except Exception:
            return False

    ok_import = try_import()
    if not ok_import:
        # try adding repo root to sys.path once
        sys.path.insert(0, str(root))
        ok_import = try_import()
    if ok_import:
        print('[OK] sofia package importable')
    else:
        print('[MISSING] Could not import sofia.* modules.')
        print('  Next step: run from repo root or add it to PYTHONPATH:')
        print('  export PYTHONPATH="', root, ':$PYTHONPATH"', sep='')

    # Diagnostics dir
    if diag.exists():
        print('[OK] diagnostics directory exists ->', diag)
    else:
        print('[MISSING] diagnostics directory ->', diag)
        if args.create_diagnostics:
            try:
                diag.mkdir(parents=True, exist_ok=True)
                print('  [CREATED] diagnostics directory')
            except Exception as e:
                print('  [ERROR] could not create diagnostics directory:', e)

    # Seeded artifacts
    seed = args.seed
    gfail = diag / f'gfail_seed_{seed}.npz'
    gtrace = diag / f'gtrace_seed_{seed}.npz'
    trace_seed0 = diag / 'trace_seed0.npz'

    # gfail
    if gfail.exists():
        print(f'[OK] {gfail.name} present')
    else:
        print(f'[MISSING] {gfail.name}')
        print('  Next step options:')
        print('  - Generate via fuzzing:')
        print(f'    python scripts/fuzz_greedy_remesh.py')
        print('  - Or create your own starting mesh and save NPZ with pts_before/tris_before')

    # gtrace
    if gtrace.exists():
        print(f'[OK] {gtrace.name} present')
    else:
        print(f'[MISSING] {gtrace.name}')
        if gfail.exists():
            print('  Next step: record per-op trace from gfail start:')
            print(f'  python scripts/run_greedy_full_op_tracer.py {seed}')
        else:
            print('  Note: tracer expects a gfail file for the chosen seed')

    # trace_seed0 for step-by-step replay example
    if trace_seed0.exists():
        print(f'[OK] {trace_seed0.name} present')
    else:
        print(f'[MISSING] {trace_seed0.name}')
        print('  Next step: create a simple trace without flips:')
        print('  python scripts/trace_seed0.py')

    # Per-commit checkers readiness
    if not gfail.exists():
        print('[INFO] Per-commit checkers require gfail_seed_*.npz as input:')
        print('  - python scripts/per_commit_compact_check_seed0.py')
        print('  - python scripts/per_commit_prepost_save_seed0.py')

    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
