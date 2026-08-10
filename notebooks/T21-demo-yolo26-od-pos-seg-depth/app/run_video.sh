#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")

VIDEO_PATH="${SCRIPT_DIR}/../assets/videos/dance-960-540.mp4"

# Use the first positional argument as the input video. If it is omitted,
# keep the bundled sample video as the default. Remaining arguments are
# forwarded to the application.
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    VIDEO_PATH="$1"
    shift

    if [[ "${VIDEO_PATH}" != /* ]]; then
        VIDEO_PATH=$(realpath "${VIDEO_PATH}")
    fi
fi

cd "${SCRIPT_DIR}"
exec ./build/yolo26s_4 \
    --video "${VIDEO_PATH}" \
    --exit-btn \
    "$@"
