#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"

echo "[check_env] python=$($python_bin -c 'import sys; print(sys.executable)')"
echo "[check_env] python_version=$($python_bin -c "import sys; print(sys.version.replace('\\n', ' '))")"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[check_env] nvidia-smi:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  echo "[check_env] nvidia-smi not found"
fi

"$python_bin" - <<'PY'
import importlib
import json
import sys

checks = {}

try:
    import torch
    checks["torch"] = {
        "ok": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            checks["torch"]["gpus"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                }
            )
except Exception as exc:
    checks["torch"] = {"ok": False, "error": repr(exc)}

for name in ["transformers", "peft", "bitsandbytes", "ot", "accelerate"]:
    try:
        module = importlib.import_module(name)
        checks[name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        checks[name] = {"ok": False, "error": repr(exc)}

required = ["torch", "transformers", "peft", "bitsandbytes", "ot"]
failed = [name for name in required if not checks.get(name, {}).get("ok")]
torch_info = checks.get("torch", {})
if torch_info.get("ok") and (not torch_info.get("cuda_available") or int(torch_info.get("gpu_count", 0)) < 2):
    failed.append("cuda_or_gpu_count")

print("[check_env] package_report=" + json.dumps(checks, ensure_ascii=False))
if failed:
    print("[check_env] missing_or_broken=" + ",".join(failed))
    sys.exit(1)
PY
