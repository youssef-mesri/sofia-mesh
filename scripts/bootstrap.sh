#!/usr/bin/env bash
# scripts/bootstrap.sh
# Create a Python virtual environment in .venv, activate it (in this script) and install dev requirements.
# Note: sourcing this script won't activate the venv in your interactive shell. Use the printed activation command
# after the script finishes: `source .venv/bin/activate`.
set -euo pipefail
VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}
REQ_FILE="requirements-dev.txt"

echo "Bootstrapping dev environment (venv=${VENV_DIR}) using ${PYTHON}..."

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: ${PYTHON} not found in PATH. Install Python 3.8+ or set PYTHON env var to the desired interpreter." >&2
  exit 2
fi

if [ -d "$VENV_DIR" ]; then
  echo "Using existing virtualenv at $VENV_DIR"
else
  echo "Creating virtualenv..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate for the duration of this script so pip installs into the venv
# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

if [ -f "$REQ_FILE" ]; then
  echo "Installing developer requirements from $REQ_FILE..."
  pip install -r "$REQ_FILE"
else
  echo "Warning: $REQ_FILE not found. Skipping dependency installation." >&2
fi

echo "Bootstrap complete. To start working run:\n  source $VENV_DIR/bin/activate"
