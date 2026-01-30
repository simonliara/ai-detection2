#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd -P )"

apt update && apt install -y libeigen3-dev ninja-build libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-base uuid-dev gstreamer1.0-rtsp

ldconfig
if ! ldconfig -p | grep -q libddscxx; then
    echo "cyclonedds cxx not installed!"
    $PROJECT_ROOT/external/data-bus/scripts/install_cyclonedds_cxx.sh
fi