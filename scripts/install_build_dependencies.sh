#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd -P )"
INSTALL_DIR="$PROJECT_ROOT/external/eigen_339"

sudo apt update && sudo apt install -y ninja-build wget cmake

if [ ! -d "$INSTALL_DIR/include/eigen3" ]; then
    echo "Installing Eigen 3.3.9 locally to $INSTALL_DIR..."
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
    tar -xzvf eigen-3.3.9.tar.gz
    cd eigen-3.3.9
    
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
    make install -j$(nproc)
    
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_DIR"
else
    echo "Local Eigen 3.3.9 already installed, skipping..."
fi

ldconfig
if ! ldconfig -p | grep -q libddscxx; then
    echo "cyclonedds cxx not installed!"
    $PROJECT_ROOT/external/data-bus/scripts/install_cyclonedds_cxx.sh
fi