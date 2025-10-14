import pathlib
import re

FORBIDDEN = [r"1e-12", r"-1e-12", r"1e-9"]
ALLOW_FILES = {
    'geometry.py',   # re-exports constants
    'constants.py',  # defines the constants
}

def test_no_raw_tolerance_literals():
    root = pathlib.Path(__file__).resolve().parent.parent
    py_files = [p for p in root.rglob('*.py') if 'tests' not in p.parts]
    pattern = re.compile('|'.join(FORBIDDEN))
    offenders = []
    for f in py_files:
        if f.name in ALLOW_FILES:
            continue
        text = f.read_text(encoding='utf-8', errors='ignore')
        # Skip deprecated shim modules that intentionally contain legacy code patterns
        if f.name in {'mesh_modifier.py','editor.py','rl-ym.py','pocket_fill.py'}:
            continue
        for m in pattern.finditer(text):
            offenders.append((str(f.relative_to(root)), m.group(0)))
    assert not offenders, "Raw tolerance literals found (use geometry.EPS_AREA / EPS_MIN_ANGLE_DEG / EPS_IMPROVEMENT):\n" + '\n'.join(f"{f}: {lit}" for f, lit in offenders)