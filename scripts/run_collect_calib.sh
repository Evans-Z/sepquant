#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/calib/qwen3_wikitext2.json}"
shift || true

sepquant-collect-calib --config "${CONFIG_PATH}" "$@"

