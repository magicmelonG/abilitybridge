from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from datasets import load_dataset

from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path
from src.utils.io import write_jsonl


ANSWER_RE = re.compile(r"####\s*(.+)\s*$")


def normalize_answer(answer: str) -> str:
    match = ANSWER_RE.search(answer)
    raw = match.group(1) if match else answer
    return raw.replace(",", "").strip()


def format_row(row: dict, idx: int, split: str) -> dict:
    return {
        "id": f"gsm8k-{split}-{idx}",
        "question": row["question"].strip(),
        "answer": row["answer"].strip(),
        "target": normalize_answer(row["answer"]),
    }


def fallback_rows(split: str) -> list[dict]:
    examples = [
        {
            "question": "A store has 12 apples and sells 5. How many apples remain?",
            "answer": "The store has 12 - 5 = 7 apples left. #### 7",
        },
        {
            "question": "Tom has 3 bags with 4 marbles each. How many marbles does he have?",
            "answer": "Tom has 3 * 4 = 12 marbles. #### 12",
        },
        {
            "question": "A car travels 60 miles per hour for 2 hours. How far does it travel?",
            "answer": "It travels 60 * 2 = 120 miles. #### 120",
        },
    ]
    return [format_row(row, i, split) for i, row in enumerate(examples)]


def sample_split(dataset_name: str, dataset_config: str, split: str, size: int, seed: int) -> list[dict]:
    try:
        ds = load_dataset(dataset_name, dataset_config, split=split)
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        indices = indices[: min(size, len(indices))]
        return [format_row(ds[int(i)], pos, split) for pos, i in enumerate(indices)]
    except Exception as exc:
        print(f"[prepare_data] Falling back to built-in toy data for split={split}: {exc}")
        rows = fallback_rows(split)
        return rows[: min(size, len(rows))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small GSM8K jsonl dataset.")
    add_common_args(parser)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    data_cfg = cfg["data"]
    train_size = args.train_size or int(data_cfg["train_size"])
    eval_size = args.eval_size or int(data_cfg["eval_size"])

    train_path = resolve_path(cfg, data_cfg["train_jsonl"])
    eval_path = resolve_path(cfg, data_cfg["eval_jsonl"])
    if train_path.exists() and eval_path.exists() and not args.overwrite:
        print(f"[prepare_data] skip existing outputs: {train_path}, {eval_path}")
        print(f"[prepare_data] train={sum(1 for _ in train_path.open(encoding='utf-8'))} eval={sum(1 for _ in eval_path.open(encoding='utf-8'))}")
        return

    Path(resolve_path(cfg, data_cfg["output_dir"])).mkdir(parents=True, exist_ok=True)
    seed = int(data_cfg.get("seed", cfg["experiment"].get("seed", 42)))
    train_rows = sample_split(data_cfg["dataset_name"], data_cfg["dataset_config"], data_cfg["train_split"], train_size, seed)
    eval_rows = sample_split(data_cfg["dataset_name"], data_cfg["dataset_config"], data_cfg["eval_split"], eval_size, seed + 1)

    n_train = write_jsonl(train_path, train_rows)
    n_eval = write_jsonl(eval_path, eval_rows)
    print(f"[prepare_data] saved train_jsonl={train_path} rows={n_train}")
    print(f"[prepare_data] saved eval_jsonl={eval_path} rows={n_eval}")


if __name__ == "__main__":
    main()
