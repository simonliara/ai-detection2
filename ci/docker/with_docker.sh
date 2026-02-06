#!/bin/bash
set -euo pipefail

# toolings
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. >/dev/null 2>&1 && pwd)"
FILE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Check if at least one argument (TARGET) is provided
if [ $# -lt 1 ]; then
    echo "Error: Target argument is missing."
    echo "Usage: $0 <ARCH-OS> <COMMAND>"
    echo "Example: $0 amd64-bionic \"bash\""
    exit 1
fi

TARGET=$1
shift # Now safe to shift because we verified $# >= 1

COMMAND="$@"
ARCH=$(echo "$TARGET" | cut -d'-' -f1)
OS=$(echo "$TARGET" | cut -d'-' -f2)


log() { echo -e "\e[92m[OK]\e[39m $@"; }
err() { echo -e "\e[91m[ERR]\e[39m $@" >&2; }

CI_DOCKER_DIR=$FILE_PATH
DOCKERFILE="${CI_DOCKER_DIR}/${ARCH}/${OS}/Dockerfile"
DOCKER_IMAGE="ara/ai-detection:${ARCH}-${OS}"

log $PROJECT_ROOT
docker build  --network=host -f "$DOCKERFILE" -t $DOCKER_IMAGE "$PROJECT_ROOT"

docker run --rm -it \
  --network=host \
  -v $PROJECT_ROOT:$PROJECT_ROOT \
  -w $(pwd) \
  $DOCKER_IMAGE \
  $COMMAND
