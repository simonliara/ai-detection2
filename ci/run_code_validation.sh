#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null 2>&1 && pwd -P )"


$PROJECT_ROOT/scripts/with_dev_venv.sh $PROJECT_ROOT/scripts/run_format.sh --check --diff
$PROJECT_ROOT/scripts/with_dev_venv.sh $PROJECT_ROOT/scripts/run_lint.sh
$PROJECT_ROOT/scripts/with_dev_venv.sh $PROJECT_ROOT/scripts/run_static_typing.sh
