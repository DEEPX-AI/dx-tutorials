#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
VIDEO_PATH="../assets/videos/pidnet.mp4"

cd "${SCRIPT_DIR}"
exec ./build/pidnet_cityscapes \
    --video "${VIDEO_PATH}" \
    --loop \
    --full-screen \
    "$@"
