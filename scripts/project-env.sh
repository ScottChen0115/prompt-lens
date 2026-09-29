#!/usr/bin/env bash
# Source this before project commands; caches stay inside the project.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
