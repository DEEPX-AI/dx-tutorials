#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")

TEXTS=(
    "A person giving a thumbs up"
    "A person clapping hands"
    "A person making a hand heart"
    "A person making a V sign with fingers"
    "A person holding a cup"
    "A person signaling OK with fingers"
)

cd "${SCRIPT_DIR}"
exec ./build/clip_single \
    --texts "${TEXTS[@]}" \
    --full-screen \
    --exit-btn \
    "$@"
