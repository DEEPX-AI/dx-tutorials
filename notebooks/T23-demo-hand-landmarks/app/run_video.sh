#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
VIDEO_PATH="../assets/videos/hands.mp4"

cd "${SCRIPT_DIR}"
exec ./build/hand_landmarks \
    --video "${VIDEO_PATH}" \
    --loop \
    --landmark-only \
    "$@"
