#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/eval/tasks_qwen3.json}"
shift || true

sepquant-eval-tasks --config "${CONFIG_PATH}" "$@"

