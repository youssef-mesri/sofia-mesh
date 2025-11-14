# Contributing to Sofia

Thanks for your interest in improving Sofia! This document describes how to set up a development environment, coding style, how to run tests, and the preferred PR workflow.

## Getting started

- Fork this repository on GitHub and clone your fork (or clone directly if you have push rights).
- Use a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev]
```

- Run the test suite to verify your environment:

```bash
pytest -q
```

CI runs on GitHub Actions (workflow "CI"). The main branch is protected and requires a green CI to merge.

## Branching and commits

- Create a feature branch from `main`:
  - `feat/<short-topic>` for new features
  - `fix/<short-topic>` for bug fixes
  - `docs/<short-topic>` for documentation-only changes
  - `chore/<short-topic>` for non-functional changes (tooling, CI, etc.)
- Keep PRs small and focused. Prefer several cohesive PRs over a very large one.
- Commit messages should be clear and imperative (optionally follow Conventional Commits, e.g., `feat: …`, `fix: …`).

## Coding style and guidelines

- Follow PEP 8 (line length ~88–100 is OK) and write docstrings for public APIs.
- Prefer type hints for new/edited public functions (PEP 484). If you have Python ≥ 3.10, you can run `mypy` locally.
- Lint locally before pushing:

```bash
flake8
```

- Constants/tolerances: reuse values from `sofia/sofia/constants.py` (avoid hard-coded tolerance literals). Tests enforce this (see `sofia/tests/test_no_raw_tolerance_literals.py`).
- Data formats: the editor assumes canonical storage
  - points: float64 (N,2), C-contiguous
  - triangles: int32 (M,3), C-contiguous
  Use the existing helpers (e.g., `ensure_positive_orientation`) and property setters.
- Performance: prefer vectorized NumPy operations and batched checks; avoid unnecessary compactions—use the editor’s amortization helpers and `has_tombstones()` gates.

## Tests

- Add unit tests in `sofia/tests/` for new features or behavior changes.
- Run all tests locally:

```bash
pytest -q
```

- Run a subset (example):

```bash
pytest sofia/tests/test_greedy_remesh.py::test_greedy_remesh_preserves_conformity_and_nonregression -q
```

- Headless plotting is enabled via Matplotlib Agg; tests should not require a GUI.

## Documentation & examples

- Update `README.md` when adding public-facing features or workflows.
- If you add new demo scripts, consider linking them in the README “Reproduce” section.
- Place documentation images in `docs/` (PNG). The repo tracks only `docs/*.png`.

## Pull Requests

Before opening a PR, please check:

- [ ] Tests added/updated and passing locally (`pytest -q`)
- [ ] Lint clean (`flake8`) and (optional) type-checks (`mypy`) for Python ≥ 3.10
- [ ] CI is green on your branch (GitHub Actions)
- [ ] Docs/README examples updated if behavior changed
- [ ] Clear description of the change and rationale

Thanks for contributing! 