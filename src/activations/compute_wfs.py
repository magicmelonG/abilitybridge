from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path
from src.utils.io import ensure_parent, write_jsonl


def topk_count(width: int, raw_k: float) -> int:
    if raw_k <= 0:
        raise ValueError("top-k must be positive.")
    if raw_k <= 1:
        return max(1, int(round(width * raw_k)))
    return min(width, int(raw_k))


def layer_stats(x: torch.Tensor, threshold: float) -> dict[str, torch.Tensor]:
    x = x.float()
    pos = x > threshold
    freq = pos.float().mean(dim=0)
    pos_sum = torch.where(pos, x, torch.zeros_like(x)).sum(dim=0)
    pos_count = pos.sum(dim=0).clamp_min(1)
    mean_pos = pos_sum / pos_count
    wfs = freq * mean_pos
    return {
        "activation_frequency": freq.cpu(),
        "mean_positive_activation": mean_pos.cpu(),
        "wfs": wfs.cpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WFS scores from math/control hidden activations.")
    add_common_args(parser)
    parser.add_argument("--role", choices=["teacher", "student"], default="student")
    parser.add_argument("--math-pt", default=None)
    parser.add_argument("--control-pt", default=None)
    parser.add_argument("--output-npz", default=None)
    parser.add_argument("--topk-jsonl", default=None)
    parser.add_argument("--topk-pt", default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--topk", type=float, default=None, help="Fraction <=1 or absolute dimension count.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    act_dir = resolve_path(cfg, cfg["activations"]["output_dir"])
    ot_dir = resolve_path(cfg, cfg["ot"]["output_dir"])
    role = args.role
    math_path = resolve_path(cfg, args.math_pt or act_dir / f"{role}_math_hidden.pt")
    control_path = resolve_path(cfg, args.control_pt or act_dir / f"{role}_control_hidden.pt")
    output_npz = resolve_path(cfg, args.output_npz or ot_dir / f"{role}_wfs_stats.npz")
    topk_jsonl = resolve_path(cfg, args.topk_jsonl or ot_dir / f"{role}_wfs_topk.jsonl")
    topk_pt = resolve_path(cfg, args.topk_pt or ot_dir / f"{role}_wfs_topk.pt")

    if output_npz.exists() and topk_jsonl.exists() and topk_pt.exists() and not args.overwrite:
        print(f"[compute_wfs] skip existing outputs: {output_npz}, {topk_jsonl}, {topk_pt}")
        return

    math_payload = torch.load(math_path, map_location="cpu")
    ctrl_payload = torch.load(control_path, map_location="cpu")
    math_hidden: dict[int, torch.Tensor] = {int(k): v for k, v in math_payload["hidden"].items()}
    ctrl_hidden: dict[int, torch.Tensor] = {int(k): v for k, v in ctrl_payload["hidden"].items()}
    layers = sorted(set(math_hidden) & set(ctrl_hidden))
    if not layers:
        raise ValueError("No overlapping layers found between math and control activation files.")

    topk_raw = args.topk if args.topk is not None else float(cfg.get("wfs", {}).get("topk", cfg["ot"].get("ability_mask_topk", 0.5)))
    gamma = float(args.gamma if args.gamma is not None else cfg.get("wfs", {}).get("gamma", 1.0))
    threshold = float(args.threshold if args.threshold is not None else cfg.get("wfs", {}).get("threshold", 0.0))
    arrays: dict[str, np.ndarray] = {}
    top_rows: list[dict] = []
    top_payload: dict[int, dict[str, torch.Tensor]] = {}

    for layer in layers:
        math_stats = layer_stats(math_hidden[layer], threshold)
        ctrl_stats = layer_stats(ctrl_hidden[layer], threshold)
        score = math_stats["wfs"] - gamma * ctrl_stats["wfs"]
        k = topk_count(score.numel(), topk_raw)
        top_scores, top_idx = torch.topk(score, k=k, largest=True, sorted=True)

        arrays[f"layer_{layer}_math_activation_frequency"] = math_stats["activation_frequency"].numpy()
        arrays[f"layer_{layer}_math_mean_positive_activation"] = math_stats["mean_positive_activation"].numpy()
        arrays[f"layer_{layer}_wfs_math"] = math_stats["wfs"].numpy()
        arrays[f"layer_{layer}_ctrl_activation_frequency"] = ctrl_stats["activation_frequency"].numpy()
        arrays[f"layer_{layer}_ctrl_mean_positive_activation"] = ctrl_stats["mean_positive_activation"].numpy()
        arrays[f"layer_{layer}_wfs_ctrl"] = ctrl_stats["wfs"].numpy()
        arrays[f"layer_{layer}_score"] = score.numpy()
        top_payload[layer] = {"indices": top_idx.cpu(), "scores": top_scores.cpu(), "k": torch.tensor(k)}

        for rank, (idx, val) in enumerate(zip(top_idx.tolist(), top_scores.tolist()), start=1):
            top_rows.append({"layer": layer, "rank": rank, "dim": int(idx), "score": float(val), "k": k, "gamma": gamma})

    ensure_parent(output_npz)
    np.savez_compressed(output_npz, **arrays, layers=np.array(layers), gamma=np.array(gamma), threshold=np.array(threshold))
    write_jsonl(topk_jsonl, top_rows)
    ensure_parent(topk_pt)
    torch.save(
        {
            "math_pt": str(math_path),
            "control_pt": str(control_path),
            "layers": layers,
            "gamma": gamma,
            "threshold": threshold,
            "topk": topk_raw,
            "topk_by_layer": top_payload,
        },
        topk_pt,
    )
    summary = {layer: int(top_payload[layer]["k"]) for layer in layers}
    print(f"[compute_wfs] saved stats_npz={output_npz}")
    print(f"[compute_wfs] saved topk_jsonl={topk_jsonl} rows={len(top_rows)}")
    print(f"[compute_wfs] saved topk_pt={topk_pt} k_by_layer={json.dumps(summary)}")


if __name__ == "__main__":
    main()
