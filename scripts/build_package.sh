#!/bin/bash
set -euo pipefail

PACKAGE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd -P )"

log() { echo -e "\e[92m[OK]\e[39m $@"; }
err() { echo -e "\e[91m[ERR]\e[39m $@" >&2; }

if ! python3 -m pip list | grep build >/dev/null 2>/dev/null; then
    log "Missing ruff Installing..."
     python3 -m pip install build -U
fi

cd "$PACKAGE_PATH"
exec python3 -m build "$PACKAGE_PATH"  "$@"