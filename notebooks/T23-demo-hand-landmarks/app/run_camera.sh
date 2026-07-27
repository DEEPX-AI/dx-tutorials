#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")

cd "${SCRIPT_DIR}"
exec ./build/hand_landmarks \
    --camera 0 \
    --width 640 \
    --height 480 \
    --fps 30 \
    --landmark-only \
    "$@"
