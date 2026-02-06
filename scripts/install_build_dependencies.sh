#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd -P )"
INSTALL_DIR="$PROJECT_ROOT/external/eigen_339"

sudo apt update && sudo apt install -y ninja-build wget cmake libeigen3-dev

ldconfig
if ! ldconfig -p | grep -q libddscxx; then
    echo "cyclonedds cxx not installed!"
    $PROJECT_ROOT/external/data-bus/scripts/install_cyclonedds_cxx.sh
fi