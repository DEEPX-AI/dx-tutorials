#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$TUTORIAL_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python environment was not found: $PYTHON_BIN" >&2
    echo "Start from the dx-tutorials environment, or set PYTHON_BIN explicitly." >&2
    exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/camera_app.py" "$@"

