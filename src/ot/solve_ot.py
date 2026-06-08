from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import ot
import torch

from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path
from src.utils.io import ensure_parent, write_json


def _layer_key_candidates(layer: int) -> list[str]:
    return [
        f"layer_{layer}",
        f"layer_{layer}_hidden",
        f"hidden_layer_{layer}",
        f"layer_{layer}_activations",
        str(layer),
    ]


def load_layer_activations(path: str | Path, layer: int) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "hidden" in payload:
            hidden = payload["hidden"]
            if layer in hidden:
                return hidden[layer].float()
            if str(layer) in hidden:
                return hidden[str(layer)].float()
        for key in _layer_key_candidates(layer):
            if isinstance(payload, dict) and key in payload:
                return payload[key].float()
        raise KeyError(f"Could not find layer {layer} activations in {path}.")

    if path.suffix == ".npz":
        payload = np.load(path)
        for key in _layer_key_candidates(layer):
            if key in payload:
                return torch.from_numpy(payload[key]).float()
        raise KeyError(f"Could not find layer {layer} activations in {path}. Available keys: {list(payload.keys())[:20]}")

    raise ValueError(f"Unsupported activation file extension: {path.suffix}")


def load_wfs_score(path: str | Path, layer: int) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "topk_by_layer" in payload:
            entry = payload["topk_by_layer"].get(layer) or payload["topk_by_layer"].get(str(layer))
            if entry and "scores" in entry:
                return entry["scores"].float()
        if isinstance(payload, dict):
            for key in [f"layer_{layer}_score", f"score_layer_{layer}", f"score_{layer}", str(layer)]:
                if key in payload:
                    return payload[key].float()
        raise KeyError(f"Could not find WFS layer {layer} score in {path}.")

    if path.suffix == ".npz":
        payload = np.load(path)
        for key in [f"layer_{layer}_score", f"score_layer_{layer}", f"score_{layer}", str(layer)]:
            if key in payload:
                return torch.from_numpy(payload[key]).float()
        raise KeyError(f"Could not find WFS layer {layer} score in {path}. Available keys: {list(payload.keys())[:20]}")

    raise ValueError(f"Unsupported WFS file extension: {path.suffix}")


def normalize_score(score: torch.Tensor) -> torch.Tensor:
    score = torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lo = score.min()
    hi = score.max()
    if float(hi - lo) < 1e-12:
        return torch.zeros_like(score)
    return (score - lo) / (hi - lo)


def select_top_dims(score: torch.Tensor, top_k_dims: float | None) -> torch.Tensor:
    width = score.numel()
    if top_k_dims is None or top_k_dims <= 0:
        return torch.arange(width, dtype=torch.long)
    if top_k_dims <= 1:
        k = max(1, int(round(width * top_k_dims)))
    else:
        k = min(width, int(top_k_dims))
    return torch.topk(score, k=k, largest=True, sorted=True).indices.long()


def correlation_cost(student_hidden: torch.Tensor, teacher_hidden: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if student_hidden.shape[0] != teacher_hidden.shape[0]:
        n = min(student_hidden.shape[0], teacher_hidden.shape[0])
        student_hidden = student_hidden[:n]
        teacher_hidden = teacher_hidden[:n]
    s = student_hidden - student_hidden.mean(dim=0, keepdim=True)
    t = teacher_hidden - teacher_hidden.mean(dim=0, keepdim=True)
    s = s / s.norm(dim=0, keepdim=True).clamp_min(eps)
    t = t / t.norm(dim=0, keepdim=True).clamp_min(eps)
    corr = s.transpose(0, 1).matmul(t).clamp(-1.0, 1.0)
    return 1.0 - corr


def ability_cost(base_cost: torch.Tensor, student_score: torch.Tensor, teacher_score: torch.Tensor, lam: float, beta: float) -> torch.Tensor:
    s = normalize_score(student_score).view(-1, 1)
    t = normalize_score(teacher_score).view(1, -1)
    return base_cost + lam * torch.abs(s - t) - beta * (s * t)


def solve_sinkhorn(cost: torch.Tensor, reg: float, max_iter: int) -> tuple[torch.Tensor, dict[str, float]]:
    cost_np = cost.detach().cpu().double().numpy()
    a = np.ones(cost_np.shape[0], dtype=np.float64) / cost_np.shape[0]
    b = np.ones(cost_np.shape[1], dtype=np.float64) / cost_np.shape[1]
    plan = ot.sinkhorn(a, b, cost_np, reg=reg, numItermax=max_iter, stopThr=1e-9, warn=False)
    plan_t = torch.from_numpy(plan).float()
    row_err = float(np.abs(plan.sum(axis=1) - a).max())
    col_err = float(np.abs(plan.sum(axis=0) - b).max())
    avg_cost = float((plan * cost_np).sum())
    sanity = {
        "row_marginal_linf": row_err,
        "column_marginal_linf": col_err,
        "avg_matching_cost": avg_cost,
        "transport_mass": float(plan.sum()),
    }
    return plan_t, sanity


def save_solution(
    output_dir: Path,
    method: str,
    plan: torch.Tensor,
    cost: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    method_dir = output_dir / method
    ensure_parent(method_dir / "metadata.json")
    torch.save(plan, method_dir / "ot_matrix.pt")
    torch.save(cost.float().cpu(), method_dir / "cost_matrix.pt")
    write_json(method_dir / "metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve vanilla and ability-aware OT maps between teacher/student hidden dimensions.")
    add_common_args(parser)
    parser.add_argument("--teacher-activation", default=None)
    parser.add_argument("--student-activation", default=None)
    parser.add_argument("--teacher-wfs", default=None)
    parser.add_argument("--student-wfs", default=None)
    parser.add_argument("--teacher-layer", type=int, default=None)
    parser.add_argument("--student-layer", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k-dims", type=float, default=None)
    parser.add_argument("--reg", type=float, default=None)
    parser.add_argument("--lambda-ability", type=float, default=None)
    parser.add_argument("--beta-ability", type=float, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    ot_cfg = cfg["ot"]
    act_dir = resolve_path(cfg, cfg["activations"]["output_dir"])
    output_dir = resolve_path(cfg, args.output_dir) if args.output_dir else resolve_path(cfg, ot_cfg["output_dir"]) / "alignment"
    teacher_layer = args.teacher_layer if args.teacher_layer is not None else int(ot_cfg.get("teacher_layer", cfg["activations"]["teacher_layers"][0]))
    student_layer = args.student_layer if args.student_layer is not None else int(ot_cfg.get("student_layer", cfg["activations"]["student_layers"][0]))
    teacher_activation = resolve_path(cfg, args.teacher_activation or act_dir / "teacher_math_hidden.pt")
    student_activation = resolve_path(cfg, args.student_activation or act_dir / "student_math_hidden.pt")
    teacher_wfs = resolve_path(cfg, args.teacher_wfs or ot_cfg.get("teacher_wfs", resolve_path(cfg, ot_cfg["output_dir"]) / "teacher_wfs_stats.npz"))
    student_wfs = resolve_path(cfg, args.student_wfs or ot_cfg.get("student_wfs", resolve_path(cfg, ot_cfg["output_dir"]) / "student_wfs_stats.npz"))
    reg = float(args.reg if args.reg is not None else ot_cfg.get("reg", ot_cfg.get("epsilon", 0.05)))
    max_iter = int(ot_cfg.get("max_iter", 200))
    lam = float(args.lambda_ability if args.lambda_ability is not None else ot_cfg.get("lambda_ability", 0.25))
    beta = float(args.beta_ability if args.beta_ability is not None else ot_cfg.get("beta_ability", 0.25))
    top_k_dims = args.top_k_dims if args.top_k_dims is not None else ot_cfg.get("top_k_dims")

    vanilla_meta_path = output_dir / "vanilla" / "metadata.json"
    ability_meta_path = output_dir / "ability_aware" / "metadata.json"
    if vanilla_meta_path.exists() and ability_meta_path.exists() and not args.overwrite:
        print(f"[solve_ot] skip existing outputs: {output_dir / 'vanilla'}, {output_dir / 'ability_aware'}")
        return

    teacher_hidden = load_layer_activations(teacher_activation, teacher_layer)
    student_hidden = load_layer_activations(student_activation, student_layer)
    teacher_score = load_wfs_score(teacher_wfs, teacher_layer)
    student_score = load_wfs_score(student_wfs, student_layer)

    teacher_idx = select_top_dims(teacher_score, top_k_dims)
    student_idx = select_top_dims(student_score, top_k_dims)
    teacher_hidden_sel = teacher_hidden[:, teacher_idx]
    student_hidden_sel = student_hidden[:, student_idx]
    teacher_score_sel = teacher_score[teacher_idx]
    student_score_sel = student_score[student_idx]

    base_cost = correlation_cost(student_hidden_sel, teacher_hidden_sel)
    aware_cost = ability_cost(base_cost, student_score_sel, teacher_score_sel, lam=lam, beta=beta)
    aware_cost = aware_cost - aware_cost.min()
    base_cost = base_cost - base_cost.min()

    vanilla_plan, vanilla_sanity = solve_sinkhorn(base_cost, reg=reg, max_iter=max_iter)
    aware_plan, aware_sanity = solve_sinkhorn(aware_cost, reg=reg, max_iter=max_iter)

    shared_meta = {
        "teacher_activation": str(teacher_activation),
        "student_activation": str(student_activation),
        "teacher_wfs": str(teacher_wfs),
        "student_wfs": str(student_wfs),
        "teacher_layer": teacher_layer,
        "student_layer": student_layer,
        "teacher_dims": teacher_idx.tolist(),
        "student_dims": student_idx.tolist(),
        "teacher_hidden_shape": list(teacher_hidden.shape),
        "student_hidden_shape": list(student_hidden.shape),
        "selected_teacher_shape": list(teacher_hidden_sel.shape),
        "selected_student_shape": list(student_hidden_sel.shape),
        "reg": reg,
        "max_iter": max_iter,
        "top_k_dims": top_k_dims,
    }
    save_solution(output_dir, "vanilla", vanilla_plan, base_cost, {**shared_meta, "method": "vanilla", "sanity": vanilla_sanity})
    save_solution(
        output_dir,
        "ability_aware",
        aware_plan,
        aware_cost,
        {**shared_meta, "method": "ability_aware", "lambda_ability": lam, "beta_ability": beta, "sanity": aware_sanity},
    )

    print(f"[solve_ot] saved vanilla_dir={output_dir / 'vanilla'}")
    print(f"[solve_ot] vanilla sanity={json.dumps(vanilla_sanity)}")
    print(f"[solve_ot] saved ability_aware_dir={output_dir / 'ability_aware'}")
    print(f"[solve_ot] ability_aware sanity={json.dumps(aware_sanity)}")


if __name__ == "__main__":
    main()
