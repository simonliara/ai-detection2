#!/bin/bash
set -euo pipefail

# toolings
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../.. >/dev/null 2>&1 && pwd -P )"
FILE_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DOCKER_IMAGE_TAG='ara/ai-detection'

USER_UID=$(id -u ${USER})
USER_GID=$(id -g ${USER})


docker build -f "$FILE_PATH/Dockerfile" --build-arg="USER_UID=$USER_UID" --build-arg="USER_GID=$USER_GID"  -t $DOCKER_IMAGE_TAG "$PROJECT_ROOT"
docker run --rm -it \
  --network=host \
  -u $USER_UID:$USER_GID \
  -v $PROJECT_ROOT:$PROJECT_ROOT \
  -w $(pwd) \
  $DOCKER_IMAGE_TAG \
  "$@"
