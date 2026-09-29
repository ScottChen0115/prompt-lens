#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/project-env.sh"
cd "$PROJECT_ROOT"
exec uv run --locked "$@"
