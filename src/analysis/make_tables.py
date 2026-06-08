from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.eval.eval_all import METHODS, method_checkpoint
from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path
from src.utils.io import ensure_parent


NA = "NA"


DISPLAY_NAMES = {
    "base_student": "base_student",
    "sft_lora": "sft_lora",
    "sft_linear_rep": "sft_linear_rep",
    "sft_vanilla_ot_rep": "sft_vanilla_ot_rep",
    "sft_ability_ot_rep": "sft_ability_ot_rep",
    "sft_random_mask_rep": "sft_random_mask_rep",
}


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def avg_generation_length_from_predictions(path: Path) -> float | str:
    if not path.exists():
        return NA
    lengths = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            response = str(row.get("response", ""))
            lengths.append(len(response.split()))
    return round(sum(lengths) / len(lengths), 3) if lengths else NA


def read_train_info(cfg: dict[str, Any], method: str) -> tuple[Any, Any]:
    if method == "base_student":
        return NA, NA
    ckpt = method_checkpoint(cfg, method)
    if ckpt is None:
        return NA, NA
    run_dir = ckpt.parent
    meta = read_json_if_exists(run_dir / "run_metadata.json")
    train_time = NA
    peak_mem = NA
    if meta:
        train_time = meta.get("elapsed_sec", NA)
        peak_mem = meta.get("peak_gpu_memory_mb", NA)
    log_path = run_dir / "train_log.csv"
    if peak_mem == NA and log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and "peak_gpu_memory_mb" in rows[-1]:
            peak_mem = rows[-1].get("peak_gpu_memory_mb") or NA
    return peak_mem, train_time


def fmt(value: Any) -> Any:
    if value is None:
        return NA
    if isinstance(value, float):
        return round(value, 6)
    return value


def build_rows(cfg: dict[str, Any], methods: list[str]) -> list[dict[str, Any]]:
    results_dir = resolve_path(cfg, cfg["eval"]["results_dir"])
    rows = []
    for method in methods:
        gsm8k = read_json_if_exists(results_dir / f"{method}_gsm8k_metrics.json") or {}
        if method == "base_student" and not gsm8k:
            gsm8k = read_json_if_exists(results_dir / "student_base_metrics.json") or {}
        math500 = read_json_if_exists(results_dir / f"{method}_math500_metrics.json") or {}
        wikitext = read_json_if_exists(results_dir / f"{method}_wikitext_metrics.json") or {}
        avg_len = gsm8k.get("avg_generation_length", None)
        if avg_len is None:
            avg_len = avg_generation_length_from_predictions(results_dir / f"{method}_gsm8k_predictions.jsonl")
        if avg_len == NA and method == "base_student":
            avg_len = avg_generation_length_from_predictions(results_dir / "student_base_predictions.jsonl")
        peak_mem, train_time = read_train_info(cfg, method)
        rows.append(
            {
                "method": DISPLAY_NAMES.get(method, method),
                "gsm8k_exact_match": fmt(gsm8k.get("accuracy", NA)),
                "math500_exact_match": fmt(math500.get("accuracy", NA)),
                "wikitext_perplexity": fmt(wikitext.get("perplexity", NA)),
                "avg_generation_length": fmt(avg_len),
                "train_peak_gpu_memory_mb": fmt(peak_mem),
                "train_time_sec": fmt(train_time),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_float_or_none(value: Any) -> float | None:
    try:
        if value == NA:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def write_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    labels = [row["method"] for row in rows]
    gsm = [to_float_or_none(row["gsm8k_exact_match"]) for row in rows]
    math500 = [to_float_or_none(row["math500_exact_match"]) for row in rows]
    x = list(range(len(labels)))
    width = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar([i - width / 2 for i in x], [v if v is not None else 0 for v in gsm], width=width, label="GSM8K")
    plt.bar([i + width / 2 for i in x], [v if v is not None else 0 for v in math500], width=width, label="MATH-500")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Exact match accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def build_tables(cfg: dict[str, Any], methods: list[str] | None = None) -> dict[str, Path]:
    methods = methods or METHODS
    rows = build_rows(cfg, methods)
    ablation_dir = resolve_path(cfg, cfg.get("analysis", {}).get("output_dir", "outputs/ablations"))
    figure_dir = resolve_path(cfg, cfg.get("analysis", {}).get("figure_dir", "outputs/figures"))
    csv_path = ablation_dir / "main_results.csv"
    md_path = ablation_dir / "main_results.md"
    fig_path = figure_dir / "main_bar.png"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    write_bar(fig_path, rows)
    return {"csv": csv_path, "markdown": md_path, "figure": fig_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build main ablation tables and figure from eval/train metrics.")
    add_common_args(parser)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    outputs = build_tables(cfg, methods=methods)
    print(f"[make_tables] saved csv={outputs['csv']}")
    print(f"[make_tables] saved markdown={outputs['markdown']}")
    print(f"[make_tables] saved figure={outputs['figure']}")


if __name__ == "__main__":
    main()
