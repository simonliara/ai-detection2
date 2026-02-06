#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd -P )"

exec find  $PROJECT_ROOT/ai_detection_cxx  -regex '.*\.\(cpp\|h\|hpp\)' | xargs clang-format -i -style=file  $@
