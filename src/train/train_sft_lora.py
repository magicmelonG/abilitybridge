from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train.common_lora import SFTCollator, SFTDataset, append_train_log, load_student_lora, read_train_rows, save_config_used
from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline SFT LoRA student.")
    add_common_args(parser)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    train_cfg = cfg["train"]
    output_dir = resolve_path(cfg, args.output_dir or Path(train_cfg["output_dir"]) / "sft_lora")
    checkpoint_dir = output_dir / "checkpoint"
    log_path = output_dir / "train_log.csv"
    if checkpoint_dir.exists() and not args.overwrite:
        print(f"[train_sft_lora] skip existing checkpoint={checkpoint_dir}")
        return

    model, tokenizer = load_student_lora(cfg, dtype=args.dtype, device_map=args.device_map)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_file, rows = read_train_rows(cfg, args.train_file or train_cfg.get("train_file"), args.max_samples)
    max_seq_len = int(train_cfg.get("max_seq_len", train_cfg.get("max_seq_length", 768)))
    dataset = SFTDataset(rows, tokenizer, max_seq_len=max_seq_len)
    collator = SFTCollator(tokenizer)
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
    progress = tqdm(total=total_update_steps, desc="sft-lora")
    stop = False
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            batch_t = {k: v.to(device) for k, v in batch.items() if k != "ids"}
            outputs = model(**batch_t)
            loss = outputs.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0 or step + 1 == len(dataloader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                raw_loss = float(loss.detach().cpu()) * grad_accum
                append_train_log(
                    log_path,
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": raw_loss,
                        "sft_loss": raw_loss,
                        "rep_loss": 0.0,
                        "learning_rate": lr,
                    },
                )
                progress.update(1)
                progress.set_postfix(loss=f"{raw_loss:.4f}")
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop = True
                    break
        if stop:
            break
    progress.close()
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    peak_mem = round(torch.cuda.max_memory_allocated() / (1024**2), 3) if torch.cuda.is_available() else "NA"
    run_meta = {
        "method": "sft_lora",
        "train_file": str(train_file),
        "rows": len(rows),
        "steps": global_step,
        "elapsed_sec": round(time.time() - started, 3),
        "peak_gpu_memory_mb": peak_mem,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"[train_sft_lora] saved checkpoint={checkpoint_dir}")
    print(f"[train_sft_lora] saved train_log={log_path}")
    print(f"[train_sft_lora] saved run_metadata={output_dir / 'run_metadata.json'}")
    print(f"[train_sft_lora] train_file={train_file} rows={len(rows)} steps={global_step} elapsed_sec={run_meta['elapsed_sec']}")


if __name__ == "__main__":
    main()
