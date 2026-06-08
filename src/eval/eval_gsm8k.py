from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.models.load_model import load_causal_lm, model_path_from_config
from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path
from src.utils.io import read_jsonl, write_json, write_jsonl


NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def build_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a careful math solver. Answer step by step and put the final answer after ####."},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {question}\nAnswer step by step. Final answer after #### "


def extract_answer(text: str) -> str:
    if "####" in text:
        tail = text.split("####")[-1]
        nums = NUMBER_RE.findall(tail.replace(",", ""))
        if nums:
            return nums[-1]
    nums = NUMBER_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else ""


def normalize_num(text: str) -> str:
    nums = NUMBER_RE.findall(str(text).replace(",", ""))
    if not nums:
        return str(text).strip()
    val = nums[-1]
    if "." in val:
        try:
            as_float = float(val)
            if as_float.is_integer():
                return str(int(as_float))
        except ValueError:
            pass
    return val.lstrip("+")


@torch.inference_mode()
def generate_one(model, tokenizer, question: str, max_new_tokens: int, do_sample: bool, temperature: float) -> str:
    prompt = build_prompt(question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
    output = model.generate(**inputs, **kwargs)
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local causal LM on prepared GSM8K jsonl.")
    add_common_args(parser)
    parser.add_argument("--role", choices=["teacher", "student"], default="student")
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    eval_cfg = cfg["eval"]
    input_path = resolve_path(cfg, args.input_jsonl or cfg["data"]["eval_jsonl"])
    results_dir = resolve_path(cfg, eval_cfg["results_dir"])
    output_path = resolve_path(
        cfg,
        args.output_jsonl or results_dir / f"{args.role}_base_predictions.jsonl",
    )
    metrics_path = resolve_path(
        cfg,
        args.metrics_json or results_dir / f"{args.role}_base_metrics.json",
    )

    if output_path.exists() and metrics_path.exists() and not args.overwrite:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"[eval_gsm8k] skip existing outputs: {output_path}, {metrics_path}")
        print(f"[eval_gsm8k] accuracy={metrics.get('accuracy')} n={metrics.get('n')}")
        return

    rows = read_jsonl(input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    model_path = model_path_from_config(cfg, args.role)
    model, tokenizer = load_causal_lm(model_path, dtype=args.dtype, device_map=args.device_map)

    max_new_tokens = args.max_new_tokens or int(eval_cfg["max_new_tokens"])
    do_sample = bool(eval_cfg.get("do_sample", False))
    temperature = float(eval_cfg.get("temperature", 0.0))
    predictions = []
    correct = 0
    started = time.time()
    for row in tqdm(rows, desc=f"eval-{args.role}"):
        response = generate_one(model, tokenizer, row["question"], max_new_tokens, do_sample, temperature)
        pred = normalize_num(extract_answer(response))
        target = normalize_num(row["target"])
        ok = pred == target
        correct += int(ok)
        predictions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "target": target,
                "prediction": pred,
                "correct": ok,
                "response": response,
            }
        )

    n = len(predictions)
    metrics = {
        "role": args.role,
        "model_path": str(model_path),
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_jsonl(output_path, predictions)
    write_json(metrics_path, metrics)
    print(f"[eval_gsm8k] saved predictions={output_path} rows={n}")
    print(f"[eval_gsm8k] saved metrics={metrics_path} accuracy={metrics['accuracy']:.4f} correct={correct}/{n}")


if __name__ == "__main__":
    main()
