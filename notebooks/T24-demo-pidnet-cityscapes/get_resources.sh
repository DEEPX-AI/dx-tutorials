#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
RESOURCE_DIR="${SCRIPT_DIR}/assets"
ARCHIVE_PATH="${SCRIPT_DIR}/archive.tar.gz"
DOWNLOAD_URL="https://cs.deepx.ai/_deepx_fae_archive/dx-tutorials/pidnet.tar.gz"

mkdir -p "${RESOURCE_DIR}"

echo "Downloading AI model & Videos..."
curl --fail --location --output "${ARCHIVE_PATH}" "${DOWNLOAD_URL}"

echo "Extracting assets to ${RESOURCE_DIR}..."
tar -xzf "${ARCHIVE_PATH}" -C "${RESOURCE_DIR}"

rm -f "${ARCHIVE_PATH}"

echo "Resource setup complete."
