#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")

cd "${SCRIPT_DIR}"
exec ./build/pidnet_cityscapes \
    --camera 0 \
    --width 1280 \
    --height 720 \
    --fps 30 \
    --full-screen \
    "$@"
