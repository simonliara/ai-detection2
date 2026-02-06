#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd -P )"

DEFAULT_TARGETS="
    ai-detection
    ai-detection.deb
    webcam
    onnx2engine
"

# parse args
TARGET=""
VERBOSE="-DCMAKE_VERBOSE_MAKEFILE=TRUE"
OPTIONS=
LONGOPTS=target:,noverbose
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "Theres an error with your arguments..." >&2; exit 2
fi
eval set -- "$PARSED"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --target)       TARGET=$2; shift ;;
        --noverbose)    VERBOSE="" ;;
        --) shift; break ;;
        *) break ;;
    esac
    shift
done
[ -z "$TARGET" ] && [ "$#" -ge 1 ] && TARGET="$@" && shift
[ -z "$TARGET" ] && TARGET=$DEFAULT_TARGETS

set -x

mkdir -p $PROJECT_ROOT/build
cd $PROJECT_ROOT/build

cmake -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  $VERBOSE \
  -S $PROJECT_ROOT \
  -B $PROJECT_ROOT/build

cmake --build . --target $TARGET -- -j $(nproc)

