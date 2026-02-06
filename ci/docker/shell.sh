#!/bin/bash
set -euo pipefail

# toolings
FILE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

log() { echo -e "\e[92m[OK]\e[39m $@"; }
err() { echo -e "\e[91m[ERR]\e[39m $@" >&2; }

# Check if at least one argument (TARGET) is provided
if [ $# -lt 1 ]; then
    echo "Error: Target argument is missing."
    echo "Usage: $0 <ARCH-OS>"
    echo "Example: $0 amd64-bionic "
    exit 1
fi

TARGET=$1

$FILE_PATH/with_docker.sh "$TARGET" /bin/bash
