#!/bin/bash

export ROOT_PATH=$(pwd)

# Virtual Env Name
VENV_DIR=".venv"

# Check if the virtual env exists
if [ ! -d "$VENV_DIR" ]; then
    echo "--- No '$VENV_DIR'. Create a new virtual env! ---"

    # Create a virtual env
    python3 -m venv "$VENV_DIR"

    # Activate
    source "$VENV_DIR/bin/activate"

    # Upgrade pip & install required packages
    echo "--- Upgrade pip and install package ---"
    pip install -U pip
    pip install -r requirements.txt

    echo "--- Complete! ---"
else
    echo "--- Reusing the existing virtual env '$VENV_DIR'! ---"
    source "$VENV_DIR/bin/activate"
fi

jupyter lab
