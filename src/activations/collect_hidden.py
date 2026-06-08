from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.eval.eval_gsm8k import build_prompt
from src.models.load_model import load_causal_lm, model_path_from_config
from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path
from src.utils.io import ensure_parent, read_jsonl


def parse_layers(raw: str | None, default_layers: list[int]) -> list[int]:
    if raw is None:
        return [int(x) for x in default_layers]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def control_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "You copy text carefully without solving math problems."},
        {"role": "user", "content": f"Repeat this text exactly: {question}"},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Repeat this text exactly:\n{question}\n"


def prompt_for_kind(row: dict, tokenizer, kind: str) -> str:
    if kind == "math":
        return build_prompt(row["question"], tokenizer)
    if kind == "control":
        return control_prompt(row["question"], tokenizer)
    raise ValueError(f"Unknown activation kind: {kind}")


def last_nonpad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    indices = []
    for mask in attention_mask:
        nonzero = torch.nonzero(mask, as_tuple=False).flatten()
        if len(nonzero) == 0:
            indices.append(torch.tensor(mask.shape[0] - 1, device=mask.device))
        else:
            indices.append(nonzero[-1])
    return torch.stack(indices)


@torch.inference_mode()
def collect_hidden_vectors(model, tokenizer, rows: list[dict], layers: list[int], kind: str, max_length: int) -> dict[int, torch.Tensor]:
    buckets: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for row in tqdm(rows, desc=f"collect-{kind}"):
        prompt = prompt_for_kind(row, tokenizer, kind)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(model.device)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        last_idx = last_nonpad_indices(inputs["attention_mask"])
        for layer in layers:
            if layer < 0 or layer >= len(hidden_states):
                raise ValueError(f"Layer {layer} is out of range for hidden_states length {len(hidden_states)}.")
            vec = hidden_states[layer][torch.arange(hidden_states[layer].shape[0], device=model.device), last_idx]
            buckets[layer].append(vec.squeeze(0).detach().float().cpu())
    return {layer: torch.stack(vectors, dim=0) for layer, vectors in buckets.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect selected Qwen hidden states for math/control prompts.")
    add_common_args(parser)
    parser.add_argument("--role", choices=["teacher", "student"], default="student")
    parser.add_argument("--kind", choices=["math", "control"], default="math")
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-pt", default=None)
    parser.add_argument("--layers", default=None, help="Comma-separated hidden_state indices. 0 is embedding; 1..N are blocks.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    act_cfg = cfg["activations"]
    input_path = resolve_path(cfg, args.input_jsonl or cfg["data"]["train_jsonl"])
    role_layers = act_cfg[f"{args.role}_layers"]
    layers = parse_layers(args.layers, role_layers)
    output_dir = resolve_path(cfg, act_cfg["output_dir"])
    output_path = resolve_path(cfg, args.output_pt or output_dir / f"{args.role}_{args.kind}_hidden.pt")

    if output_path.exists() and not args.overwrite:
        payload = torch.load(output_path, map_location="cpu")
        stats = {int(k): list(v.shape) for k, v in payload["hidden"].items()}
        print(f"[collect_hidden] skip existing output={output_path}")
        print(f"[collect_hidden] stats={json.dumps(stats)}")
        return

    rows = read_jsonl(input_path)
    max_samples = args.max_samples if args.max_samples is not None else int(act_cfg.get("max_samples", len(rows)))
    rows = rows[:max_samples]
    model, tokenizer = load_causal_lm(model_path_from_config(cfg, args.role), dtype=args.dtype, device_map=args.device_map)
    max_length = args.max_length or int(act_cfg.get("max_length", cfg["train"].get("max_seq_length", 768)))

    started = time.time()
    hidden = collect_hidden_vectors(model, tokenizer, rows, layers, args.kind, max_length)
    payload = {
        "role": args.role,
        "kind": args.kind,
        "input_jsonl": str(input_path),
        "layers": layers,
        "ids": [row["id"] for row in rows],
        "hidden": hidden,
        "meta": {
            "n": len(rows),
            "max_length": max_length,
            "elapsed_sec": round(time.time() - started, 3),
            "token_strategy": "last_non_padding",
        },
    }
    ensure_parent(output_path)
    torch.save(payload, output_path)
    stats = {int(k): list(v.shape) for k, v in hidden.items()}
    print(f"[collect_hidden] saved output={output_path}")
    print(f"[collect_hidden] role={args.role} kind={args.kind} rows={len(rows)} stats={json.dumps(stats)}")


if __name__ == "__main__":
    main()
