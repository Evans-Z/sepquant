#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/optimize/opt125m_weight_format_search.json}"
shift || true

sepquant-optimize-layers --config "${CONFIG_PATH}" "$@"

