#!/usr/bin/env bash

set -euo pipefail

export ROOT_PATH=$(pwd)

# Virtual Env Name
VENV_DIR=".venv"

if ! command -v uv >/dev/null 2>&1; then
    echo "--- uv is not installed. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh ---"
    exit 1
fi

# Check if the virtual env exists
if [ ! -d "$VENV_DIR" ]; then
    echo "--- No '$VENV_DIR'. Create a new virtual env! ---"

    # Create a virtual env
    uv venv "$VENV_DIR"

    # Activate
    source "$VENV_DIR/bin/activate"

    # Install required packages
    echo "--- Install packages from requirements.txt with uv ---"
    uv pip install -r requirements.txt

    echo "--- Complete! ---"
else
    echo "--- Reusing the existing virtual env '$VENV_DIR'! ---"
    source "$VENV_DIR/bin/activate"
fi

jupyter lab
