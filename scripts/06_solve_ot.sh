#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/toy_qwen_1p5b_to_0p5b.yaml"
OVERWRITE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONFIG="$2"; shift 2 ;;
    --overwrite) OVERWRITE="--overwrite"; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

python -m src.ot.solve_ot --config "$CONFIG" $OVERWRITE "${EXTRA_ARGS[@]}"
python -m src.ot.linear_baseline --config "$CONFIG" $OVERWRITE "${EXTRA_ARGS[@]}"
