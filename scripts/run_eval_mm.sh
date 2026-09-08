#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/eval/mm_qwen3_vl_mme.json}"
shift || true

sepquant-eval-mm --config "${CONFIG_PATH}" "$@"
