#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TUTORIAL_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
# Section 7.1 of the notebook creates workspace/.venv-ocr (dx_engine wheel + app requirements).
# PYTHON_BIN overrides it; the repository .venv is the last fallback.
OCR_ENV_PYTHON="$SCRIPT_DIR/../workspace/.venv-ocr/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$OCR_ENV_PYTHON" ]]; then PYTHON_BIN="$OCR_ENV_PYTHON"; else PYTHON_BIN="$TUTORIAL_ROOT/.venv/bin/python"; fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python environment was not found: $PYTHON_BIN" >&2
    echo "Run section 7.1 of the notebook (creates workspace/.venv-ocr), or set PYTHON_BIN explicitly." >&2
    exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/camera_app.py" "$@"

