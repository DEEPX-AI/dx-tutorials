#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
VIDEO_PATH="../assets/videos/CLIP-demo.mp4"

TEXTS=(
    "Cars are driving on the road"
    "A car accident occurred on the road"
)

cd "${SCRIPT_DIR}"
exec ./build/clip_single \
    --input "${VIDEO_PATH}" \
    --texts "${TEXTS[@]}" \
    --full-screen \
    --exit-btn \
    "$@"
