#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/abilitybridge_v4_rotation_grid.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/root/restored_envs/abilitybridge_new/bin/python}"
ROOT_DIR="${ROOT_DIR:-outputs/abilitybridge_v4_feature_rotation_pruning}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(pwd)}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RESUME=0
INCLUDE_STAGE2=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --include-stage2) INCLUDE_STAGE2=1; shift ;;
    --gpu0) GPU0="$2"; shift 2 ;;
    --gpu1) GPU1="$2"; shift 2 ;;
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --workspace-root) WORKSPACE_ROOT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$WORKSPACE_ROOT"
mkdir -p "$ROOT_DIR"/{logs,status,configs}
cp -f "$CONFIG" "$ROOT_DIR/configs/$(basename "$CONFIG")"
cp -f "$0" "$ROOT_DIR/scripts_run_v4_feature_rotation_grid_2gpu.sh" 2>/dev/null || true

LOG_FILE="$ROOT_DIR/logs/orchestrator.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] v4 feature rotation grid start root=$ROOT_DIR gpu0=$GPU0 gpu1=$GPU1 resume=$RESUME include_stage2=$INCLUDE_STAGE2"

EXTRA=()
if [[ "$INCLUDE_STAGE2" == "1" ]]; then
  EXTRA+=(--include-stage2)
fi

"$PYTHON_BIN" -m src.analysis.abilitybridge_v4_feature_rotation list-grid \
  --config "$CONFIG" \
  --root-dir "$ROOT_DIR" \
  "${EXTRA[@]}"

CUDA_VISIBLE_DEVICES="$GPU0" bash scripts/run_v4_feature_rotation_worker.sh \
  --worker gpu0 \
  --gpu "$GPU0" \
  --root-dir "$ROOT_DIR" \
  --config "$CONFIG" \
  "${EXTRA[@]}" &
PID0=$!

CUDA_VISIBLE_DEVICES="$GPU1" bash scripts/run_v4_feature_rotation_worker.sh \
  --worker gpu1 \
  --gpu "$GPU1" \
  --root-dir "$ROOT_DIR" \
  --config "$CONFIG" \
  "${EXTRA[@]}" &
PID1=$!

wait "$PID0"
wait "$PID1"

"$PYTHON_BIN" -m src.analysis.abilitybridge_v4_feature_rotation summarize \
  --config "$CONFIG" \
  --root-dir "$ROOT_DIR" \
  "${EXTRA[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] v4 feature rotation grid complete"
