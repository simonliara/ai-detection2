#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd -P )"
VENV_ROOT="$PROJECT_ROOT/.venv"

log() { echo -e "\e[92m[OK]\e[39m $@"; }
err() { echo -e "\e[91m[ERR]\e[39m $@" >&2; }

if [ ! -e "$VENV_ROOT/bin/activate" ]; then
    log "Missing venv. Creating..."
    python3 -m venv "$VENV_ROOT"
    log venv ready
fi

source "$VENV_ROOT/bin/activate"
export PYTHONDONTWRITEBYTECODE=1


if ! python3 -m pip list | grep cyclonedds>/dev/null 2>/dev/null; then
    log "Installing cyclonedds"
    $PROJECT_ROOT/external/data-bus/scripts/install_cyclonedds_python.sh
fi

if ! python3 -m pip list | grep ara.data_bus>/dev/null 2>/dev/null; then
  python3 -m pip install -e $PROJECT_ROOT/external/data-bus/data_bus_py
fi

exec "$@"
