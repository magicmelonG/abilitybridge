#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/abilitybridge_v4_rotation_grid.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/root/restored_envs/abilitybridge_new/bin/python}"
ROOT_DIR="${ROOT_DIR:-outputs/abilitybridge_v4_feature_rotation_pruning}"
WORKER="${WORKER:-gpu0}"
GPU="${GPU:-0}"
INCLUDE_STAGE2=0
RUN_PRUNING="${RUN_PRUNING:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worker) WORKER="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --include-stage2) INCLUDE_STAGE2=1; shift ;;
    --run-pruning) RUN_PRUNING=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$ROOT_DIR/logs"
LOG_FILE="$ROOT_DIR/logs/${WORKER}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

EXTRA=()
if [[ "$INCLUDE_STAGE2" == "1" ]]; then
  EXTRA+=(--include-stage2)
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] worker=$WORKER gpu=$GPU root=$ROOT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m src.analysis.abilitybridge_v4_feature_rotation train-rotation \
  --config "$CONFIG" \
  --root-dir "$ROOT_DIR" \
  --worker "$WORKER" \
  "${EXTRA[@]}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m src.analysis.abilitybridge_v4_feature_rotation score-features \
  --config "$CONFIG" \
  --root-dir "$ROOT_DIR" \
  --worker "$WORKER" \
  --train-missing \
  "${EXTRA[@]}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m src.analysis.abilitybridge_v4_feature_rotation run-feature-causal \
  --config "$CONFIG" \
  --root-dir "$ROOT_DIR" \
  --worker "$WORKER" \
  --train-missing \
  "${EXTRA[@]}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m src.analysis.abilitybridge_v4_feature_rotation build-mlp-map \
  --config "$CONFIG" \
  --root-dir "$ROOT_DIR" \
  --worker "$WORKER" \
  --train-missing \
  "${EXTRA[@]}"

if [[ "$RUN_PRUNING" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m src.analysis.abilitybridge_v4_feature_rotation run-pruning \
    --config "$CONFIG" \
    --root-dir "$ROOT_DIR" \
    --worker "$WORKER" \
    --train-missing \
    "${EXTRA[@]}"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] worker=$WORKER complete"
