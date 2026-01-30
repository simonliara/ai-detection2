#!/bin/bash
set -euo pipefail

FILE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
log() { echo -e "\e[92m[OK]\e[39m $@"; }
err() { echo -e "\e[91m[ERR]\e[39m $@" >&2; }

$FILE_PATH/with_docker.sh /bin/bash
