#!/bin/bash
set -euxo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd -P )"

log() { echo -e "\e[92m[OK]\e[39m $@"; }
err() { echo -e "\e[91m[ERR]\e[39m $@" >&2; }

pip install -e .[tests]

python3 -m pytest -v $PROJECT_ROOT/tests/ "$@"
