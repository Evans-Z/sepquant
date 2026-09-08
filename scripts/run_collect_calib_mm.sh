#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/calib/qwen3_vl_coco.json}"
shift || true

sepquant-collect-calib-mm --config "${CONFIG_PATH}" "$@"
