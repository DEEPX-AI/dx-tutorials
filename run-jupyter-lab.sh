#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
export ROOT_PATH="${SCRIPT_DIR}"
cd "${ROOT_PATH}"

# Virtual Env Name
VENV_DIR="${ROOT_PATH}/.venv"

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

# Create the virtual environment when it does not exist.
if [ ! -d "$VENV_DIR" ]; then
    echo "--- No '$VENV_DIR'. Create a new virtual env! ---"
    uv venv "$VENV_DIR"
else
    echo "--- Reusing the existing virtual env '$VENV_DIR'! ---"
fi

# Always synchronize the required packages. This also updates an existing
# environment when requirements.txt changes between tutorial releases.
source "$VENV_DIR/bin/activate"
echo "--- Synchronize packages from requirements.txt with uv ---"
uv pip install -r "${ROOT_PATH}/requirements.txt"
echo "--- Environment is ready! ---"

jupyter lab
