#!/bin/bash
set -euxo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd -P )"
BUILD_CONTEXT="${BUILD_CONTEXT:-dev}"

log() { echo -e "\e[92m[OK]\e[39m $@"; }
err() { echo -e "\e[91m[ERR]\e[39m $@" >&2; }

if ! python3 -m pip list | grep mypy >/dev/null 2>/dev/null; then
    log "Missing ruff Installing..."
     python3 -m pip install mypy
fi

cd "$PROJECT_ROOT"
exec  mypy "$PROJECT_ROOT" "$@"