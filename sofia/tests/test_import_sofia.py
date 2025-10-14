"""Smoke test to ensure top-level package import works without triggering
circular import errors. This guards against regressions in the flat API
layer (`sofia/__init__.py`).
"""

def test_import_sofia_smoke():
    import sofia  # noqa: F401
    # A couple of light sanity checks on expected public symbols
    assert hasattr(sofia, 'triangle_area')
    assert hasattr(sofia, 'greedy_remesh')  # lazy proxy should resolve
