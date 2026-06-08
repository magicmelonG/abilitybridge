from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.train.common_lora import (
    SFTCollator,
    SFTDataset,
    append_train_log,
    last_nonpad_indices,
    load_student_lora,
    read_train_rows,
    save_config_used,
)
from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path


def load_teacher_cache(path: Path, teacher_layer: int) -> tuple[dict[str, int], torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    hidden = payload["hidden"]
    if teacher_layer in hidden:
        tensor = hidden[teacher_layer].float()
    elif str(teacher_layer) in hidden:
        tensor = hidden[str(teacher_layer)].float()
    else:
        raise KeyError(f"Teacher layer {teacher_layer} not found in {path}.")
    ids = [str(x) for x in payload.get("ids", [])]
    if not ids:
        ids = [str(i) for i in range(tensor.shape[0])]
    return {sample_id: i for i, sample_id in enumerate(ids)}, tensor


class OTDistillDataset(Dataset):
    def __init__(self, base: SFTDataset, id_to_pos: dict[str, int], teacher_hidden: torch.Tensor):
        self.base = base
        self.id_to_pos = id_to_pos
        self.teacher_hidden = teacher_hidden

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        sample_id = str(item["id"])
        if sample_id not in self.id_to_pos:
            raise KeyError(f"Sample id {sample_id} missing from teacher hidden cache.")
        out = dict(item)
        out["teacher_hidden"] = self.teacher_hidden[self.id_to_pos[sample_id]]
        return out


class OTDistillCollator(SFTCollator):
    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        out = super().__call__(batch)
        out["teacher_hidden"] = torch.stack([item["teacher_hidden"].float() for item in batch], dim=0)
        return out


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_linear_mapping(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    metadata = read_json(path.parent / "metadata.json")
    return {
        "weight": payload["weight"].float(),
        "bias": payload["bias"].float(),
        "teacher_dims": payload["teacher_dims"].long(),
        "student_dims": payload["student_dims"].long(),
        "metadata": metadata,
    }


def load_ot_mapping(method: str, output_dir: Path) -> dict[str, Any]:
    method_dir = output_dir / ("vanilla" if method == "vanilla_ot" else "ability_aware")
    plan = torch.load(method_dir / "ot_matrix.pt", map_location="cpu").float()
    metadata = read_json(method_dir / "metadata.json")
    return {
        "plan": plan,
        "teacher_dims": torch.tensor(metadata["teacher_dims"], dtype=torch.long),
        "student_dims": torch.tensor(metadata["student_dims"], dtype=torch.long),
        "metadata": metadata,
    }


def map_teacher_to_student(teacher_hidden: torch.Tensor, mapping: dict[str, Any], method: str) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_dims = mapping["teacher_dims"].to(teacher_hidden.device)
    student_dims = mapping["student_dims"].to(teacher_hidden.device)
    teacher_sel = teacher_hidden[:, teacher_dims]
    if method == "linear":
        weight = mapping["weight"].to(teacher_hidden.device)
        bias = mapping["bias"].to(teacher_hidden.device)
        target = teacher_sel @ weight.T + bias
    elif method in {"vanilla_ot", "ability_ot"}:
        plan = mapping["plan"].to(teacher_hidden.device)
        row_mass = plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
        row_normalized = plan / row_mass
        target = teacher_sel @ row_normalized.T
    else:
        raise ValueError(f"Unknown method: {method}")
    return target.detach(), student_dims


def get_student_rep(outputs, attention_mask: torch.Tensor, student_layer: int, student_dims: torch.Tensor) -> torch.Tensor:
    hidden_states = outputs.hidden_states
    if student_layer < 0 or student_layer >= len(hidden_states):
        raise ValueError(f"Student layer {student_layer} out of range for hidden_states length {len(hidden_states)}.")
    idx = last_nonpad_indices(attention_mask).to(hidden_states[student_layer].device)
    reps = hidden_states[student_layer][torch.arange(hidden_states[student_layer].shape[0], device=idx.device), idx]
    return reps[:, student_dims.to(reps.device)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA with SFT + OT representation distillation.")
    add_common_args(parser)
    parser.add_argument("--method", choices=["linear", "vanilla_ot", "ability_ot"], default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--teacher-hidden", default=None)
    parser.add_argument("--alignment-dir", default=None)
    parser.add_argument("--linear-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--teacher-layer", type=int, default=None)
    parser.add_argument("--student-layer", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    train_cfg = cfg["train"]
    ot_cfg = cfg["ot"]
    method = args.method or str(train_cfg.get("method", "ability_ot"))
    beta = float(args.beta if args.beta is not None else train_cfg.get("beta", 1.0))
    target_layers = train_cfg.get("target_layers") or [{"teacher": ot_cfg.get("teacher_layer", cfg["activations"]["teacher_layers"][0]), "student": ot_cfg.get("student_layer", cfg["activations"]["student_layers"][0])}]
    first_pair = target_layers[0]
    teacher_layer = args.teacher_layer if args.teacher_layer is not None else int(first_pair["teacher"])
    student_layer = args.student_layer if args.student_layer is not None else int(first_pair["student"])
    output_dir = resolve_path(cfg, args.output_dir or Path(train_cfg["output_dir"]) / f"{method}_rep_lora")
    checkpoint_dir = output_dir / "checkpoint"
    log_path = output_dir / "train_log.csv"
    if checkpoint_dir.exists() and not args.overwrite:
        print(f"[train_ot_rep_lora] skip existing checkpoint={checkpoint_dir}")
        return

    act_dir = resolve_path(cfg, cfg["activations"]["output_dir"])
    teacher_hidden_path = resolve_path(cfg, args.teacher_hidden or act_dir / "teacher_math_hidden.pt")
    alignment_dir = resolve_path(cfg, args.alignment_dir) if args.alignment_dir else resolve_path(cfg, ot_cfg["output_dir"]) / "alignment"
    linear_dir = resolve_path(cfg, args.linear_dir) if args.linear_dir else resolve_path(cfg, ot_cfg["output_dir"]) / "linear_projector"
    if method == "linear":
        mapping = load_linear_mapping(linear_dir / "projector.pt")
    else:
        mapping = load_ot_mapping(method, alignment_dir)

    id_to_pos, teacher_hidden = load_teacher_cache(teacher_hidden_path, teacher_layer)
    model, tokenizer = load_student_lora(cfg, dtype=args.dtype, device_map=args.device_map)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_file, rows = read_train_rows(cfg, args.train_file or train_cfg.get("train_file"), args.max_samples)
    max_seq_len = int(train_cfg.get("max_seq_len", train_cfg.get("max_seq_length", 768)))
    base_dataset = SFTDataset(rows, tokenizer, max_seq_len=max_seq_len)
    dataset = OTDistillDataset(base_dataset, id_to_pos, teacher_hidden)
    collator = OTDistillCollator(tokenizer)
    batch_size = int(train_cfg.get("batch_size", train_cfg.get("per_device_train_batch_size", 1)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    lr = float(train_cfg.get("learning_rate", 2e-5))
    num_epochs = int(train_cfg.get("num_epochs", train_cfg.get("num_train_epochs", 1)))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    device = next(model.parameters()).device

    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_used(cfg["_config_path"], output_dir)
    model.print_trainable_parameters()
    started = time.time()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    total_update_steps = math.ceil(len(dataloader) * num_epochs / grad_accum)
    progress = tqdm(total=total_update_steps, desc=f"{method}-rep-lora")
    stop = False
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            ids = batch.pop("ids")
            teacher_batch = batch.pop("teacher_hidden").to(device)
            batch_t = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch_t, output_hidden_states=True)
            sft_loss = outputs.loss
            target_rep, student_dims = map_teacher_to_student(teacher_batch, mapping, method)
            student_rep = get_student_rep(outputs, batch_t["attention_mask"], student_layer, student_dims)
            rep_loss = torch.nn.functional.mse_loss(student_rep.float(), target_rep.float())
            loss = (sft_loss + beta * rep_loss) / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0 or step + 1 == len(dataloader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                total_loss = float((sft_loss + beta * rep_loss).detach().cpu())
                sft_val = float(sft_loss.detach().cpu())
                rep_val = float(rep_loss.detach().cpu())
                append_train_log(
                    log_path,
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": total_loss,
                        "sft_loss": sft_val,
                        "rep_loss": rep_val,
                        "beta": beta,
                        "learning_rate": lr,
                        "method": method,
                    },
                )
                progress.update(1)
                progress.set_postfix(loss=f"{total_loss:.4f}", rep=f"{rep_val:.4f}")
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop = True
                    break
            _ = ids
        if stop:
            break
    progress.close()
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    run_meta = {
        "method": method,
        "beta": beta,
        "teacher_layer": teacher_layer,
        "student_layer": student_layer,
        "teacher_hidden": str(teacher_hidden_path),
        "alignment_dir": str(alignment_dir),
        "linear_dir": str(linear_dir),
        "train_file": str(train_file),
        "rows": len(rows),
        "steps": global_step,
        "elapsed_sec": round(time.time() - started, 3),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 3) if torch.cuda.is_available() else "NA",
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"[train_ot_rep_lora] saved checkpoint={checkpoint_dir}")
    print(f"[train_ot_rep_lora] saved train_log={log_path}")
    print(f"[train_ot_rep_lora] saved run_metadata={output_dir / 'run_metadata.json'}")
    print(f"[train_ot_rep_lora] method={method} beta={beta} rows={len(rows)} steps={global_step} elapsed_sec={run_meta['elapsed_sec']}")


if __name__ == "__main__":
    main()
