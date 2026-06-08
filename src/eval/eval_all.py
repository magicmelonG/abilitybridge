from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm

from src.eval.eval_gsm8k import build_prompt, extract_answer, generate_one, normalize_num
from src.models.load_model import load_causal_lm, model_path_from_config
from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path
from src.utils.io import ensure_parent, read_jsonl, write_json, write_jsonl


METHODS = [
    "base_student",
    "sft_lora",
    "sft_linear_rep",
    "sft_vanilla_ot_rep",
    "sft_ability_ot_rep",
    "sft_random_mask_rep",
]


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def method_checkpoint(cfg: dict[str, Any], method: str) -> Path | None:
    base = resolve_path(cfg, cfg["train"]["output_dir"])
    mapping = {
        "sft_lora": base / "sft_lora" / "checkpoint",
        "sft_linear_rep": base / "linear_rep_lora" / "checkpoint",
        "sft_vanilla_ot_rep": base / "vanilla_ot_rep_lora" / "checkpoint",
        "sft_ability_ot_rep": base / "ability_ot_rep_lora" / "checkpoint",
        "sft_random_mask_rep": base / "random_mask_rep_lora" / "checkpoint",
    }
    return mapping.get(method)


def load_eval_model(cfg: dict[str, Any], method: str, dtype: str, device_map: str):
    model, tokenizer = load_causal_lm(model_path_from_config(cfg, "student"), dtype=dtype, device_map=device_map)
    ckpt = method_checkpoint(cfg, method)
    if ckpt is not None and ckpt.exists():
        model = PeftModel.from_pretrained(model, str(ckpt))
        model.eval()
    elif method != "base_student":
        raise FileNotFoundError(f"Checkpoint for {method} not found: {ckpt}")
    return model, tokenizer


def answer_from_math500(solution: str) -> str:
    matches = BOXED_RE.findall(solution)
    if matches:
        return matches[-1].strip()
    if "####" in solution:
        return solution.split("####")[-1].strip()
    return extract_answer(solution)


def simple_exact(pred: str, target: str) -> bool:
    pred_n = normalize_num(pred)
    target_n = normalize_num(target)
    if pred_n and target_n and pred_n == target_n:
        return True
    clean = lambda x: str(x).strip().strip("$").replace(" ", "")
    return clean(pred) == clean(target)


def load_math500_rows(max_samples: int, seed: int) -> list[dict]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if max_samples:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    rows = []
    for i, row in enumerate(ds):
        rows.append(
            {
                "id": f"math500-{i}",
                "question": row.get("problem", row.get("question", "")),
                "target": answer_from_math500(row.get("solution", row.get("answer", ""))),
            }
        )
    return rows


@torch.inference_mode()
def eval_exact_rows(model, tokenizer, rows: list[dict], max_new_tokens: int, task: str) -> tuple[list[dict], dict]:
    preds = []
    correct = 0
    gen_lens = []
    for row in tqdm(rows, desc=f"eval-{task}"):
        response = generate_one(model, tokenizer, row["question"], max_new_tokens, False, 0.0)
        pred = extract_answer(response)
        ok = simple_exact(pred, row["target"])
        correct += int(ok)
        gen_lens.append(len(tokenizer(response, add_special_tokens=False)["input_ids"]))
        preds.append(
            {
                "id": row["id"],
                "question": row["question"],
                "target": row["target"],
                "prediction": pred,
                "correct": ok,
                "response": response,
            }
        )
    n = len(rows)
    return preds, {"n": n, "correct": correct, "accuracy": correct / n if n else None, "avg_generation_length": sum(gen_lens) / len(gen_lens) if gen_lens else None}


def load_wikitext_texts(max_samples: int, seed: int) -> list[str]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [row["text"] for row in ds if row["text"].strip()]
    if max_samples:
        texts = texts[:max_samples]
    return texts


@torch.inference_mode()
def eval_wikitext_ppl(model, tokenizer, texts: list[str], max_seq_len: int) -> dict:
    total_loss = 0.0
    total_tokens = 0
    for text in tqdm(texts, desc="eval-wikitext"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        if enc["input_ids"].shape[1] < 2:
            continue
        enc = enc.to(model.device)
        outputs = model(**enc, labels=enc["input_ids"])
        tokens = int(enc["attention_mask"].sum().item() - 1)
        total_loss += float(outputs.loss.detach().cpu()) * tokens
        total_tokens += tokens
    mean_loss = total_loss / total_tokens if total_tokens else None
    return {"n": len(texts), "tokens": total_tokens, "loss": mean_loss, "perplexity": math.exp(mean_loss) if mean_loss is not None and mean_loss < 100 else None}


def save_task_metrics(results_dir: Path, method: str, task: str, metrics: dict, predictions: list[dict] | None = None) -> None:
    metrics_path = results_dir / f"{method}_{task}_metrics.json"
    write_json(metrics_path, {"method": method, "task": task, **metrics})
    if predictions is not None:
        write_jsonl(results_dir / f"{method}_{task}_predictions.jsonl", predictions)
    print(f"[eval_all] saved {task} metrics={metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all ablation methods and build result tables.")
    add_common_args(parser)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--aggregate-only", action="store_true", help="Only build tables from existing metrics.")
    parser.add_argument("--gsm8k-samples", type=int, default=None)
    parser.add_argument("--math500-samples", type=int, default=None)
    parser.add_argument("--wikitext-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    eval_cfg = cfg["eval"]
    results_dir = resolve_path(cfg, eval_cfg["results_dir"])
    ensure_parent(results_dir / ".keep")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    max_new_tokens = args.max_new_tokens or int(eval_cfg.get("max_new_tokens", 256))
    gsm8k_samples = args.gsm8k_samples if args.gsm8k_samples is not None else eval_cfg.get("gsm8k_samples")
    math500_samples = args.math500_samples if args.math500_samples is not None else int(eval_cfg.get("math500_samples", 100))
    wikitext_samples = args.wikitext_samples if args.wikitext_samples is not None else int(eval_cfg.get("wikitext_samples", 128))
    seed = int(cfg["experiment"].get("seed", 42))

    if not args.aggregate_only:
        gsm8k_rows = read_jsonl(resolve_path(cfg, cfg["data"]["eval_jsonl"]))
        if gsm8k_samples:
            gsm8k_rows = gsm8k_rows[: int(gsm8k_samples)]
        math_rows = load_math500_rows(int(math500_samples), seed) if math500_samples != 0 else []
        wiki_texts = load_wikitext_texts(int(wikitext_samples), seed) if wikitext_samples != 0 else []

        for method in methods:
            started = time.time()
            try:
                model, tokenizer = load_eval_model(cfg, method, dtype=args.dtype, device_map=args.device_map)
            except FileNotFoundError as exc:
                print(f"[eval_all] skip {method}: {exc}")
                continue
            if gsm8k_rows:
                out_pred = results_dir / f"{method}_gsm8k_predictions.jsonl"
                out_metrics = results_dir / f"{method}_gsm8k_metrics.json"
                if out_metrics.exists() and out_pred.exists() and not args.overwrite:
                    print(f"[eval_all] skip existing GSM8K metrics for {method}")
                else:
                    preds, metrics = eval_exact_rows(model, tokenizer, gsm8k_rows, max_new_tokens, "gsm8k")
                    save_task_metrics(results_dir, method, "gsm8k", metrics, preds)
            if math_rows:
                out_pred = results_dir / f"{method}_math500_predictions.jsonl"
                out_metrics = results_dir / f"{method}_math500_metrics.json"
                if out_metrics.exists() and out_pred.exists() and not args.overwrite:
                    print(f"[eval_all] skip existing MATH-500 metrics for {method}")
                else:
                    preds, metrics = eval_exact_rows(model, tokenizer, math_rows, max_new_tokens, "math500")
                    save_task_metrics(results_dir, method, "math500", metrics, preds)
            if wiki_texts:
                out_metrics = results_dir / f"{method}_wikitext_metrics.json"
                if out_metrics.exists() and not args.overwrite:
                    print(f"[eval_all] skip existing WikiText metrics for {method}")
                else:
                    metrics = eval_wikitext_ppl(model, tokenizer, wiki_texts, int(cfg["train"].get("max_seq_len", 768)))
                    save_task_metrics(results_dir, method, "wikitext", metrics)
            print(f"[eval_all] method={method} elapsed_sec={time.time() - started:.3f}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    from src.analysis.make_tables import build_tables

    outputs = build_tables(cfg, methods=methods)
    print(f"[eval_all] saved csv={outputs['csv']}")
    print(f"[eval_all] saved markdown={outputs['markdown']}")
    print(f"[eval_all] saved figure={outputs['figure']}")


if __name__ == "__main__":
    main()
