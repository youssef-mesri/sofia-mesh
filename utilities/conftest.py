import sys
import logging
import io
import os
import datetime
import pathlib
import pytest
if sys.version_info < (3, 8):
    pytest.exit("Python >= 3.8 is required to run tests. Current version: {}".format(sys.version.replace("\n", " ")))


LOG_DIR = pathlib.Path(__file__).parent / "test-logs"
LOG_DIR.mkdir(exist_ok=True)


import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Attach the TestReport (with .outcome) to the item so fixtures can see the
    # outcome in teardown. Use a hookwrapper to get the report object.
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(autouse=True)
def capture_test_logs(request):
    """Capture logging for each test into an in-memory buffer and write it to
    a file only when the test fails.

    This keeps successful test runs quiet while preserving full debug logs for
    debugging failing tests.
    """
    root = logging.getLogger()
    # Save and remove existing handlers so we control where logs go during the test
    prev_handlers = list(root.handlers)
    for h in prev_handlers:
        root.removeHandler(h)

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    root.addHandler(handler)

    # Force debug so we capture everything into our buffer; pytest's CLI logging
    # may be enabled by user flags, but by temporarily replacing handlers we
    # avoid noisy console printing for each test.
    prev_level = root.level
    root.setLevel(logging.DEBUG)

    try:
        yield
    finally:
        # restore logger state first
        root.removeHandler(handler)
        root.setLevel(prev_level)
        for h in prev_handlers:
            root.addHandler(h)

        # Decide whether to persist logs: write only when the test call phase failed
        rep = getattr(request.node, "rep_call", None)
        if rep is not None and getattr(rep, "outcome", None) == "failed":
            # build a safe filename from nodeid and timestamp
            nodeid = request.node.nodeid.replace("::", "__").replace("/", "_")
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            fname = LOG_DIR / ("{}__{}.log".format(nodeid, ts))
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    f.write("=== Test: {}\n".format(request.node.nodeid))
                    f.write("=== Timestamp: {}\n\n".format(ts))
                    f.write(buf.getvalue())
            except Exception:
                # Avoid raising during fixture teardown; just ignore file write errors
                pass