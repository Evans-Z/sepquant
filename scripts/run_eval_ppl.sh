#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/eval/ppl_qwen3_wikitext2.json}"
shift || true

sepquant-eval-ppl --config "${CONFIG_PATH}" "$@"

