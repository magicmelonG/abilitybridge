from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path


def choose_dtype(dtype: str | None = "auto") -> str | torch.dtype:
    if dtype in (None, "auto"):
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype, dtype)


def load_tokenizer(model_path: str | Path, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def infer_load_in_4bit(cfg: dict[str, Any] | None) -> bool:
    if not cfg:
        return False
    train_cfg = cfg.get("train", {})
    runtime_cfg = cfg.get("runtime", {})
    return bool(train_cfg.get("load_in_4bit", runtime_cfg.get("load_in_4bit", False)))


def infer_dtype(dtype: str | torch.dtype, cfg: dict[str, Any] | None) -> str | torch.dtype:
    if not isinstance(dtype, str) or dtype != "auto" or not cfg:
        return dtype
    train_cfg = cfg.get("train", {})
    runtime_cfg = cfg.get("runtime", {})
    if bool(train_cfg.get("bf16", runtime_cfg.get("bf16", False))):
        return torch.bfloat16
    if bool(train_cfg.get("fp16", runtime_cfg.get("fp16", False))):
        return torch.float16
    return dtype


def load_causal_lm(
    model_path: str | Path,
    dtype: str | torch.dtype = "auto",
    device_map: str | dict[str, Any] | None = "auto",
    trust_remote_code: bool = True,
    load_in_4bit: bool = False,
):
    tokenizer = load_tokenizer(model_path, trust_remote_code=trust_remote_code)
    load_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "local_files_only": True,
    }
    chosen_dtype = choose_dtype(dtype) if isinstance(dtype, str) else dtype
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=chosen_dtype if isinstance(chosen_dtype, torch.dtype) else torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["torch_dtype"] = chosen_dtype
    else:
        load_kwargs["torch_dtype"] = chosen_dtype
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        **load_kwargs,
    )
    model.eval()
    return model, tokenizer


def model_path_from_config(cfg: dict[str, Any], role: str) -> Path:
    return resolve_path(cfg, cfg["models"][role]["path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test local Qwen model loading.")
    add_common_args(parser)
    parser.add_argument("--role", choices=["teacher", "student"], default="student")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    path = model_path_from_config(cfg, args.role)
    model, tokenizer = load_causal_lm(
        path,
        dtype=infer_dtype(args.dtype, cfg),
        device_map=args.device_map,
        load_in_4bit=infer_load_in_4bit(cfg),
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"[load_model] role={args.role} path={path}")
    print(f"[load_model] model_type={model.config.model_type} hidden_size={model.config.hidden_size} layers={model.config.num_hidden_layers}")
    print(f"[load_model] params={params:,} tokenizer_vocab={len(tokenizer)} device={next(model.parameters()).device}")


if __name__ == "__main__":
    main()
