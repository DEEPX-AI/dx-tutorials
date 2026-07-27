#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
cd "${SCRIPT_DIR}"

exec ./build/yolo26s_3 --exit-btn "$@"

