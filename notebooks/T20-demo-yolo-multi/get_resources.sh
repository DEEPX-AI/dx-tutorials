#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
RESOURCE_DIR="${SCRIPT_DIR}/assets"
ARCHIVE_PATH="${SCRIPT_DIR}/archive.tar.gz"
DOWNLOAD_URL="https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/yolo-multi.tar.gz"
REQUIRED_FILES=(
    "${RESOURCE_DIR}/models/YOLOV5S_PPU.dxnn"
    "${RESOURCE_DIR}/videos/10_360p.mp4"
    "${RESOURCE_DIR}/videos/11_360p.mp4"
    "${RESOURCE_DIR}/videos/12_360p.mp4"
    "${RESOURCE_DIR}/videos/13_360p.mp4"
    "${RESOURCE_DIR}/videos/14_360p.mp4"
    "${RESOURCE_DIR}/videos/15_360p.mp4"
    "${RESOURCE_DIR}/videos/16_360p.mp4"
    "${RESOURCE_DIR}/videos/17_360p.mp4"
    "${RESOURCE_DIR}/videos/18_360p.mp4"
    "${RESOURCE_DIR}/videos/19_360p.mp4"
    "${RESOURCE_DIR}/videos/20_360p.mp4"
    "${RESOURCE_DIR}/videos/22_360p.mp4"
    "${RESOURCE_DIR}/videos/23_360p.mp4"
    "${RESOURCE_DIR}/videos/24_360p.mp4"
    "${RESOURCE_DIR}/videos/25_360p.mp4"
    "${RESOURCE_DIR}/videos/26_360p.mp4"
    "${RESOURCE_DIR}/videos/27_360p.mp4"
    "${RESOURCE_DIR}/videos/28_360p.mp4"
    "${RESOURCE_DIR}/videos/29_360p.mp4"
    "${RESOURCE_DIR}/videos/2_360p.mp4"
    "${RESOURCE_DIR}/videos/30_360p.mp4"
    "${RESOURCE_DIR}/videos/31_360p.mp4"
    "${RESOURCE_DIR}/videos/3_360p.mp4"
    "${RESOURCE_DIR}/videos/4_360p.mp4"
    "${RESOURCE_DIR}/videos/50_360p.mp4"
    "${RESOURCE_DIR}/videos/51_360p.mp4"
    "${RESOURCE_DIR}/videos/52_360p.mp4"
    "${RESOURCE_DIR}/videos/53_360p.mp4"
    "${RESOURCE_DIR}/videos/54_360p.mp4"
    "${RESOURCE_DIR}/videos/55_360p.mp4"
    "${RESOURCE_DIR}/videos/57_360p.mp4"
    "${RESOURCE_DIR}/videos/5_360p.mp4"
    "${RESOURCE_DIR}/videos/6_360p.mp4"
    "${RESOURCE_DIR}/videos/7_360p.mp4"
    "${RESOURCE_DIR}/videos/8_360p.mp4"
    "${RESOURCE_DIR}/videos/dance-group.mp4"
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
