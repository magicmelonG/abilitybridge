#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-abilitybridge}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"
LOCAL_WHEEL_DIR="${LOCAL_WHEEL_DIR:-/opt/data/private/abilitybridge}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs}"
MAX_RETRIES="${MAX_RETRIES:-5}"
SLEEP_SECONDS="${SLEEP_SECONDS:-15}"
SKIP_TORCH="${SKIP_TORCH:-0}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
INSTALL_TORCHAUDIO="${INSTALL_TORCHAUDIO:-0}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/abilitybridge-install/${ENV_NAME}}"
FILTERED_REQUIREMENTS_FILE="${FILTERED_REQUIREMENTS_FILE:-$STATE_DIR/requirements.filtered.txt}"

export PIP_CACHE_DIR
export CONDA_PKGS_DIRS
mkdir -p "$STATE_DIR"

retry() {
  local attempt=1
  until "$@"; do
    local exit_code=$?
    if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
      echo "[install_env] command failed after ${attempt} attempts: $*"
      return "$exit_code"
    fi
    echo "[install_env] attempt ${attempt}/${MAX_RETRIES} failed with exit_code=${exit_code}: $*"
    echo "[install_env] sleeping ${SLEEP_SECONDS}s before retry"
    sleep "$SLEEP_SECONDS"
    attempt=$((attempt + 1))
  done
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

first_match() {
  local pattern="$1"
  find "$LOCAL_WHEEL_DIR" -maxdepth 1 -type f -name "$pattern" | sort | head -n 1
}

mark_done() {
  local step="$1"
  touch "${STATE_DIR}/${step}.done"
}

is_done() {
  local step="$1"
  [[ -f "${STATE_DIR}/${step}.done" ]]
}

build_filtered_requirements() {
  python - "$REQUIREMENTS_FILE" "$FILTERED_REQUIREMENTS_FILE" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
skip = {"torch", "torchvision", "torchaudio"}
lines = []
for raw in src.read_text(encoding="utf-8").splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        lines.append(raw)
        continue
    pkg = stripped.split(";", 1)[0].split("[", 1)[0].split("=", 1)[0].strip().lower()
    if pkg in skip:
        continue
    lines.append(raw)
dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

if [[ "$FORCE_RECREATE" == "1" ]] && conda_env_exists; then
  echo "[install_env] removing existing env=${ENV_NAME}"
  retry conda env remove -n "$ENV_NAME" -y
  rm -rf "$STATE_DIR"
  mkdir -p "$STATE_DIR"
fi

if ! conda_env_exists; then
  echo "[install_env] creating env=${ENV_NAME} python=${PYTHON_VERSION}"
  retry conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip -y
else
  echo "[install_env] reusing existing env=${ENV_NAME}"
fi

if ! is_done pip_bootstrap; then
  echo "[install_env] upgrading pip/setuptools/wheel"
  retry conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
  mark_done pip_bootstrap
else
  echo "[install_env] skip pip bootstrap"
fi

if [[ "$SKIP_TORCH" != "1" ]]; then
  TORCH_WHEEL="$(first_match 'torch-*.whl' || true)"
  TORCHVISION_WHEEL="$(first_match 'torchvision-*.whl' || true)"
  TORCHAUDIO_WHEEL="$(first_match 'torchaudio-*.whl' || true)"
  CUDNN_WHEEL="$(first_match 'nvidia_cudnn_cu12-*.whl' || true)"

  if [[ -n "${CUDNN_WHEEL}" ]] && ! is_done cudnn_local; then
    echo "[install_env] installing local cudnn wheel=${CUDNN_WHEEL}"
    retry conda run -n "$ENV_NAME" python -m pip install "$CUDNN_WHEEL"
    mark_done cudnn_local
  elif [[ -n "${CUDNN_WHEEL}" ]]; then
    echo "[install_env] skip local cudnn"
  fi

  if [[ -n "${TORCH_WHEEL}" ]] && ! is_done torch_local; then
    echo "[install_env] installing local torch wheel=${TORCH_WHEEL}"
    retry conda run -n "$ENV_NAME" python -m pip install "$TORCH_WHEEL"
    mark_done torch_local
  elif [[ -n "${TORCH_WHEEL}" ]]; then
    echo "[install_env] skip local torch"
  else
    if ! is_done torch_remote; then
      echo "[install_env] installing torch from ${CUDA_INDEX_URL}"
      retry conda run -n "$ENV_NAME" python -m pip install --index-url "$CUDA_INDEX_URL" torch
      mark_done torch_remote
    else
      echo "[install_env] skip remote torch"
    fi
  fi

  if [[ -n "${TORCHVISION_WHEEL}" ]] && ! is_done torchvision_local; then
    echo "[install_env] installing local torchvision wheel=${TORCHVISION_WHEEL}"
    retry conda run -n "$ENV_NAME" python -m pip install "$TORCHVISION_WHEEL"
    mark_done torchvision_local
  elif [[ -n "${TORCHVISION_WHEEL}" ]]; then
    echo "[install_env] skip local torchvision"
  else
    if ! is_done torchvision_remote; then
      echo "[install_env] installing torchvision from ${CUDA_INDEX_URL}"
      retry conda run -n "$ENV_NAME" python -m pip install --index-url "$CUDA_INDEX_URL" torchvision
      mark_done torchvision_remote
    else
      echo "[install_env] skip remote torchvision"
    fi
  fi

  if [[ "$INSTALL_TORCHAUDIO" == "1" ]]; then
    if [[ -n "${TORCHAUDIO_WHEEL}" ]] && ! is_done torchaudio_local; then
      echo "[install_env] installing local torchaudio wheel=${TORCHAUDIO_WHEEL}"
      retry conda run -n "$ENV_NAME" python -m pip install "$TORCHAUDIO_WHEEL"
      mark_done torchaudio_local
    elif [[ -n "${TORCHAUDIO_WHEEL}" ]]; then
      echo "[install_env] skip local torchaudio"
    else
      if ! is_done torchaudio_remote; then
        echo "[install_env] installing torchaudio from ${CUDA_INDEX_URL}"
        retry conda run -n "$ENV_NAME" python -m pip install --index-url "$CUDA_INDEX_URL" torchaudio
        mark_done torchaudio_remote
      else
        echo "[install_env] skip remote torchaudio"
      fi
    fi
  else
    echo "[install_env] skipping torchaudio by default"
  fi
fi

if ! is_done requirements; then
  build_filtered_requirements
  echo "[install_env] installing project requirements from ${REQUIREMENTS_FILE}"
  retry conda run -n "$ENV_NAME" python -m pip install --find-links "$LOCAL_WHEEL_DIR" -r "$FILTERED_REQUIREMENTS_FILE"
  mark_done requirements
else
  echo "[install_env] skip requirements"
fi

echo "[install_env] validating core packages"
retry conda run -n "$ENV_NAME" python - <<'PY'
import importlib
mods = ["torch", "transformers", "peft", "bitsandbytes", "ot", "accelerate"]
for name in mods:
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'unknown')}")
PY

echo "[install_env] done"
echo "[install_env] activate with: conda activate ${ENV_NAME}"
echo "[install_env] verify with: PYTHON_BIN=$(conda run -n ${ENV_NAME} which python) bash scripts/check_env.sh"
