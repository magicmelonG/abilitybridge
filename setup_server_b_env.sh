#!/usr/bin/env bash
set -euo pipefail

# Bootstrap/check the AbilityBridge runtime on a less-prepared second server.
# Run from the project root. The script is idempotent and can be re-run.

ENV_NAME="${ENV_NAME:-abilitybridge}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
LOCAL_WHEEL_DIR="${LOCAL_WHEEL_DIR:-/opt/data/private/abilitybridge}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${PROJECT_ROOT}/requirements.txt}"
SKIP_TORCH="${SKIP_TORCH:-0}"
INSTALL_TORCHAUDIO="${INSTALL_TORCHAUDIO:-0}"
OFFLINE_CHECK_ONLY="${OFFLINE_CHECK_ONLY:-0}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

log() {
  echo "[setup_server_b_env] $*"
}

die() {
  echo "[setup_server_b_env] ERROR: $*" >&2
  exit 1
}

cd "$PROJECT_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  die "conda not found. Install Miniconda/Anaconda first, then re-run this script."
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  die "requirements file not found: $REQUIREMENTS_FILE"
fi

log "project_root=$PROJECT_ROOT"
log "env_name=$ENV_NAME python=$PYTHON_VERSION"
log "requirements=$REQUIREMENTS_FILE"

if [[ "$OFFLINE_CHECK_ONLY" != "1" ]]; then
  ENV_NAME="$ENV_NAME" \
  PYTHON_VERSION="$PYTHON_VERSION" \
  CUDA_INDEX_URL="$CUDA_INDEX_URL" \
  LOCAL_WHEEL_DIR="$LOCAL_WHEEL_DIR" \
  REQUIREMENTS_FILE="$REQUIREMENTS_FILE" \
  SKIP_TORCH="$SKIP_TORCH" \
  INSTALL_TORCHAUDIO="$INSTALL_TORCHAUDIO" \
  FORCE_RECREATE="$FORCE_RECREATE" \
    bash scripts/install_env.sh
else
  log "OFFLINE_CHECK_ONLY=1, skipping package installation"
fi

PYTHON_BIN="$(conda run -n "$ENV_NAME" which python)"
export PYTHON_BIN

log "python_bin=$PYTHON_BIN"
log "running package/GPU check"
PYTHON_BIN="$PYTHON_BIN" bash scripts/check_env.sh

log "checking project imports needed by fixed-layer 4GPU experiment"
"$PYTHON_BIN" - <<'PY'
import importlib

mods = [
    "src.compression.v2_fixed_layer_4gpu",
    "src.analysis.v2_fixed_layer_summary",
    "src.analysis.abilitybridge_v4_feature_rotation",
    "src.compression.abilitybridge_v2_minimal",
    "src.eval.eval_gsm8k",
    "src.eval.eval_all",
]
for mod in mods:
    importlib.import_module(mod)
    print(f"ok import {mod}")
PY

log "checking local model/cache paths"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path("configs/toy_qwen_1p5b_to_0p5b.yaml").read_text())
paths = {
    "teacher": Path(cfg["models"]["teacher"]["path"]),
    "student": Path(cfg["models"]["student"]["path"]),
}
missing = []
for name, path in paths.items():
    exists = path.exists()
    print(f"{name}_path={path} exists={exists}")
    if not exists:
        missing.append(str(path))

cache = Path.home() / ".cache" / "huggingface" / "datasets"
print(f"hf_dataset_cache={cache} exists={cache.exists()}")
if missing:
    print("missing_model_paths=" + ",".join(missing))
    sys.exit(2)
PY

log "checking fixed eval data preparation in offline mode"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
"$PYTHON_BIN" -m src.data.prepare_fixed_eval_sets \
  --config configs/toy_qwen_1p5b_to_0p5b.yaml \
  --shared-root outputs/server_b_env_check/shared \
  --train-size 8 \
  --eval-size 4 \
  --math500-samples 2 \
  --wikitext-samples 2

cat <<EOF

[setup_server_b_env] done

Use this environment for the experiment:

cd $PROJECT_ROOT
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PATH=$(dirname "$PYTHON_BIN"):\$PATH
export PYTHON_BIN=$PYTHON_BIN

Smoke server B:
bash scripts/run_v2_fixed_layer_server.sh \\
  --server-tag server_B \\
  --layer 20 \\
  --gpu0 0 \\
  --gpu1 1 \\
  --root-dir outputs/abilitybridge_v2_fixed_layer_4gpu/server_B \\
  --shared-root outputs/abilitybridge_v2_fixed_layer_4gpu \\
  --prepare-if-missing \\
  --smoke

Formal server B:
bash scripts/run_v2_fixed_layer_server.sh \\
  --server-tag server_B \\
  --layer 20 \\
  --gpu0 0 \\
  --gpu1 1 \\
  --seeds 7,13,23,37 \\
  --root-dir outputs/abilitybridge_v2_fixed_layer_4gpu/server_B \\
  --shared-root outputs/abilitybridge_v2_fixed_layer_4gpu \\
  --prepare-if-missing \\
  --resume
EOF
