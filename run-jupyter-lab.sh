#!/usr/bin/env bash

set -euo pipefail

export ROOT_PATH=$(pwd)

# Virtual Env Name
VENV_DIR=".venv"

if ! command -v uv >/dev/null 2>&1; then
    {
        echo "[ERROR] Required command 'uv' was not found."
        echo
        echo "uv is used to create the Python virtual environment and install"
        echo "the packages required by JupyterLab."
        echo
        echo "Install uv:"
        echo
        if command -v curl >/dev/null 2>&1; then
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        elif command -v wget >/dev/null 2>&1; then
            echo "  wget -qO- https://astral.sh/uv/install.sh | sh"
        else
            echo "  curl and wget are not available. Install curl first:"
            echo
            echo "    sudo apt update"
            echo "    sudo apt install -y curl"
            echo
            echo "  Then install uv:"
            echo
            echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
        fi
        echo
        echo "After installation:"
        echo
        echo "  1. Restart your terminal, or follow the PATH instructions"
        echo "     printed by the installer."
        echo "  2. Verify the installation: uv --version"
        echo "  3. Run this script again: ./run-jupyter-lab.sh"
        echo
        echo "Installation guide:"
        echo "  https://docs.astral.sh/uv/getting-started/installation/"
    } >&2
    exit 127
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
