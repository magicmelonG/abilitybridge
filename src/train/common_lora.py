from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset

from src.eval.eval_gsm8k import build_prompt
from src.models.load_model import load_causal_lm, model_path_from_config
from src.utils.config import resolve_path
from src.utils.io import ensure_parent, read_jsonl


def choose_train_file(cfg: dict[str, Any], explicit_path: str | None = None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(resolve_path(cfg, explicit_path))
    candidates.extend(
        [
            resolve_path(cfg, "data/processed/math_train.jsonl"),
            resolve_path(cfg, "data/processed/hard_train.jsonl"),
            resolve_path(cfg, cfg["data"]["train_jsonl"]),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No train jsonl found. Tried: {[str(p) for p in candidates]}")


def load_student_lora(cfg: dict[str, Any], dtype: str = "auto", device_map: str | dict | None = "auto"):
    model, tokenizer = load_causal_lm(model_path_from_config(cfg, "student"), dtype=dtype, device_map=device_map)
    tokenizer.padding_side = "right"
    lora_cfg = cfg["train"].get("lora", {})
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model = get_peft_model(model, peft_config)
    model.train()
    return model, tokenizer


def format_answer(row: dict) -> str:
    answer = str(row.get("answer", "")).strip()
    if answer:
        return answer
    target = str(row.get("target", "")).strip()
    return f"#### {target}"


def encode_sft_example(row: dict, tokenizer, max_seq_len: int) -> dict[str, torch.Tensor | str]:
    prompt = build_prompt(row["question"], tokenizer)
    answer = format_answer(row)
    if tokenizer.eos_token and not answer.endswith(tokenizer.eos_token):
        answer = answer + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[-max_seq_len:]
    prompt_len = max(0, min(len(prompt_ids), len(input_ids) - 1))
    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len
    attention_mask = [1] * len(input_ids)
    return {
        "id": str(row["id"]),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_seq_len: int):
        self.examples = [encode_sft_example(row, tokenizer, max_seq_len) for row in rows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


@dataclass
class SFTCollator:
    tokenizer: Any

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        ids = [item["id"] for item in batch]
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "ids": ids,
            "input_ids": padded_inputs,
            "attention_mask": padded_mask,
            "labels": padded_labels,
        }


def last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.long().sum(dim=1).clamp_min(1) - 1


def append_train_log(path: Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_config_used(config_path: str | Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "config_used.yaml")


def read_train_rows(cfg: dict[str, Any], train_file: str | None, max_samples: int | None = None) -> tuple[Path, list[dict]]:
    path = choose_train_file(cfg, train_file)
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]
    return path, rows
