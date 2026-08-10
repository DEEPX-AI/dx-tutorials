#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
RESOURCE_DIR="${SCRIPT_DIR}/assets"
ARCHIVE_PATH="${SCRIPT_DIR}/archive.tar.gz"
DOWNLOAD_URL="https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/yolo26-od-pos-seg-depth.tar.gz"
REQUIRED_FILES=(
    "${RESOURCE_DIR}/models/yolo26-s_640x640.dxnn"
    "${RESOURCE_DIR}/models/yolo26-s-pose_640x640.dxnn"
    "${RESOURCE_DIR}/models/yolo26-s-seg_640x640.dxnn"
    "${RESOURCE_DIR}/models/yolo26-depth-s_768x768_q-lite.dxnn"
    "${RESOURCE_DIR}/yolo26.png"
    "${RESOURCE_DIR}/videos/dance-960-540.mp4"
)

resources_ready() {
    local path
    for path in "${REQUIRED_FILES[@]}"; do
        [[ -s "${path}" ]] || return 1
    done
}

if resources_ready; then
    echo "All required resources are already available. Skipping download."
    exit 0
fi

mkdir -p "${RESOURCE_DIR}"

echo "Downloading AI models and videos..."
curl --fail --location --output "${ARCHIVE_PATH}.part" "${DOWNLOAD_URL}"
mv "${ARCHIVE_PATH}.part" "${ARCHIVE_PATH}"

echo "Extracting assets to ${RESOURCE_DIR}..."
tar -xzf "${ARCHIVE_PATH}" -C "${RESOURCE_DIR}"

if ! resources_ready; then
    echo "ERROR: The archive did not provide all required resources:" >&2
    for path in "${REQUIRED_FILES[@]}"; do
        [[ -s "${path}" ]] || echo "  missing: ${path}" >&2
    done
    exit 1
fi

rm -f "${ARCHIVE_PATH}"

echo "Resource setup complete."
