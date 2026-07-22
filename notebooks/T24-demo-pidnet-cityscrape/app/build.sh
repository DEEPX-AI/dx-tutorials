#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
BUILD_DIR="${SCRIPT_DIR}/build"

show_help() {
    echo "Usage: $0 [--clean]"
    echo "  --clean    Remove the build directory before building"
}

if (( $# > 1 )); then
    show_help >&2
    exit 1
fi

if (( $# == 1 )); then
    case "$1" in
        --clean)
            rm -rf "${BUILD_DIR}"
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -DCMAKE_BUILD_TYPE=Release "${SCRIPT_DIR}"
make -j"$(nproc)"
