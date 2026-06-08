from __future__ import annotations

import argparse
import csv
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from tqdm import tqdm

from src.eval.eval_gsm8k import build_prompt, extract_answer, generate_one, normalize_num
from src.models.load_model import load_causal_lm, model_path_from_config
from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path
from src.utils.io import ensure_parent, read_jsonl


def load_topk(path: Path) -> dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return {int(layer): entry["indices"].long() for layer, entry in payload["topk_by_layer"].items()}


def get_layers_module(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Could not find transformer block list on model.")


def make_zero_hook(indices: torch.Tensor):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        idx = indices.to(hidden.device)
        patched = hidden.clone()
        patched[..., idx] = 0
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    return hook


@contextmanager
def zeroout_hook(model, hidden_state_layer: int, indices: torch.Tensor):
    if hidden_state_layer <= 0:
        raise ValueError("Intervention supports transformer block outputs only: use layer >= 1.")
    blocks = get_layers_module(model)
    block_index = hidden_state_layer - 1
    if block_index < 0 or block_index >= len(blocks):
        raise ValueError(f"Layer {hidden_state_layer} is out of range for {len(blocks)} blocks.")
    handle = blocks[block_index].register_forward_hook(make_zero_hook(indices))
    try:
        yield
    finally:
        handle.remove()


def answer_logprob(model, tokenizer, row: dict, prompt_kind: str, max_length: int) -> float:
    if prompt_kind == "math":
        prompt = build_prompt(row["question"], tokenizer)
        answer = " " + str(row["target"])
    elif prompt_kind == "control":
        prompt = f"Question text: {row['question']}\nCopy the final numeric answer only: "
        answer = str(row["target"])
    else:
        raise ValueError(f"Unknown prompt_kind: {prompt_kind}")

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    input_ids = torch.cat([prompt_ids, answer_ids], dim=0)[-max_length:].unsqueeze(0).to(model.device)
    prompt_len = min(prompt_ids.numel(), input_ids.shape[1] - answer_ids.numel())
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]
        answer_start = max(prompt_len - 1, 0)
        answer_token_log_probs = log_probs[:, answer_start:, :].gather(-1, labels[:, answer_start:].unsqueeze(-1)).squeeze(-1)
    return float(answer_token_log_probs.sum().detach().cpu())


def exact_match_score(model, tokenizer, rows: list[dict], max_new_tokens: int) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        response = generate_one(model, tokenizer, row["question"], max_new_tokens, False, 0.0)
        correct += int(normalize_num(extract_answer(response)) == normalize_num(row["target"]))
    return correct / len(rows)


def mean_logprob(model, tokenizer, rows: list[dict], prompt_kind: str, max_length: int) -> float:
    if not rows:
        return 0.0
    vals = [answer_logprob(model, tokenizer, row, prompt_kind, max_length) for row in rows]
    return sum(vals) / len(vals)


def random_indices(width: int, k: int, seed: int, layer: int) -> torch.Tensor:
    rng = random.Random(seed + layer * 1009 + k)
    return torch.tensor(rng.sample(range(width), k), dtype=torch.long)


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "k", "method", "math_drop", "ctrl_drop"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero out top-k hidden dimensions and measure math/control drops.")
    add_common_args(parser)
    parser.add_argument("--role", choices=["teacher", "student"], default="student")
    parser.add_argument("--topk-pt", default=None)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--metric", choices=["logprob", "exact_match"], default=None)
    parser.add_argument("--k-list", default=None, help="Comma-separated absolute k values. Default: use full top-k from WFS.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    ot_dir = resolve_path(cfg, cfg["ot"]["output_dir"])
    topk_path = resolve_path(cfg, args.topk_pt or ot_dir / f"{args.role}_wfs_topk.pt")
    intervention_cfg = cfg.get("intervention", {})
    output_csv = resolve_path(cfg, args.output_csv or intervention_cfg.get("output_csv", ot_dir / f"{args.role}_zeroout_intervention.csv"))
    input_path = resolve_path(cfg, args.input_jsonl or cfg["data"]["eval_jsonl"])

    if output_csv.exists() and not args.overwrite:
        print(f"[intervention_zeroout] skip existing output={output_csv}")
        return

    max_samples = args.max_samples if args.max_samples is not None else int(intervention_cfg.get("max_samples", 16))
    metric = args.metric or str(intervention_cfg.get("metric", "logprob"))
    k_list_raw = args.k_list if args.k_list is not None else intervention_cfg.get("k_list")
    rows = read_jsonl(input_path)[:max_samples]
    topk_by_layer = load_topk(topk_path)
    model, tokenizer = load_causal_lm(model_path_from_config(cfg, args.role), dtype=args.dtype, device_map=args.device_map)
    max_length = args.max_length or int(cfg["train"].get("max_seq_length", 768))
    seed = args.seed if args.seed is not None else int(cfg["experiment"].get("seed", 42))
    width = int(model.config.hidden_size)

    if metric == "logprob":
        base_math = mean_logprob(model, tokenizer, rows, "math", max_length)
        base_ctrl = mean_logprob(model, tokenizer, rows, "control", max_length)
    else:
        base_math = exact_match_score(model, tokenizer, rows, args.max_new_tokens)
        base_ctrl = base_math

    out_rows: list[dict] = []
    started = time.time()
    for layer, ranked_indices in tqdm(sorted(topk_by_layer.items()), desc="zeroout-layers"):
        default_k = int(ranked_indices.numel())
        k_values = [int(x.strip()) for x in str(k_list_raw).split(",") if x.strip()] if k_list_raw else [default_k]
        for k in k_values:
            k = min(k, default_k)
            for method, indices in [
                ("wfs_topk", ranked_indices[:k]),
                ("random_topk", random_indices(width, k, seed, layer)),
            ]:
                with zeroout_hook(model, layer, indices):
                    if metric == "logprob":
                        math_score = mean_logprob(model, tokenizer, rows, "math", max_length)
                        ctrl_score = mean_logprob(model, tokenizer, rows, "control", max_length)
                    else:
                        math_score = exact_match_score(model, tokenizer, rows, args.max_new_tokens)
                        ctrl_score = math_score
                out_rows.append(
                    {
                        "layer": layer,
                        "k": k,
                        "method": method,
                        "math_drop": base_math - math_score,
                        "ctrl_drop": base_ctrl - ctrl_score,
                    }
                )

    write_csv(output_csv, out_rows)
    print(f"[intervention_zeroout] saved csv={output_csv} rows={len(out_rows)}")
    print(f"[intervention_zeroout] metric={metric} samples={len(rows)} base_math={base_math:.6f} base_ctrl={base_ctrl:.6f} elapsed_sec={time.time() - started:.3f}")


if __name__ == "__main__":
    main()
