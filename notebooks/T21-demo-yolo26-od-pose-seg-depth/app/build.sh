#!/usr/bin/env bash
# Build this demo in Release mode.
#   ./build.sh                 incremental build into ./build
#   ./build.sh --clean         remove ./build first
#   ./build.sh -DVAR=VALUE ... extra options are passed to cmake, for example
#                              -DCMAKE_PREFIX_PATH=/opt/dx-rt when DX-RT lives under a custom prefix

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
BUILD_DIR="${SCRIPT_DIR}/build"
CMAKE_ARGS=()

show_help() {
    echo "Usage: $0 [--clean] [-D<var>=<value> ...]"
    echo "  --clean           Remove the build directory before building"
    echo "  -D<var>=<value>   Passed to cmake (e.g. -DCMAKE_PREFIX_PATH=<dx-rt prefix>)"
}

for arg in "$@"; do
    case "${arg}" in
        --clean) rm -rf "${BUILD_DIR}" ;;
        --help|-h) show_help; exit 0 ;;
        -D*) CMAKE_ARGS+=("${arg}") ;;
        *) echo "Unknown option: ${arg}" >&2; show_help >&2; exit 1 ;;
    esac
done

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -DCMAKE_BUILD_TYPE=Release "${CMAKE_ARGS[@]+"${CMAKE_ARGS[@]}"}" "${SCRIPT_DIR}"
make -j"$(nproc)"
