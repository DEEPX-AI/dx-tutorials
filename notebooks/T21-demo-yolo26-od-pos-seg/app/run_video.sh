#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")

VIDEO_PATH='../assets/videos/dance-960-540.mp4'

cd "${SCRIPT_DIR}"
exec ./build/yolo26s_3 \
    --video "${VIDEO_PATH}" \
    --exit-btn \
    "$@"

