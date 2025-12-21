#!/usr/bin/env bash

set -euo pipefail

GROUP="sudo"
SUDO_FILE="/etc/sudoers.d/${GROUP}-nopasswd"

TMP="$(mktemp)"
{
  echo "# Created $(date)"
  echo "%${GROUP} ALL=(ALL) NOPASSWD: ALL"
} > "$TMP"

sudo visudo -cf "$TMP" >/dev/null
sudo install -m 0440 "$TMP" "$SUDO_FILE"
rm -f "$TMP"

echo "Complete: $SUDO_FILE"
echo "Test: sudo -n true && echo OK || echo NG"
