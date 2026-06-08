from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.ot.solve_ot import load_layer_activations, load_wfs_score, select_top_dims
from src.utils.config import add_common_args, apply_overrides, load_config, resolve_path
from src.utils.io import ensure_parent, write_json


def fit_ridge_projector(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if x.shape[0] != y.shape[0]:
        n = min(x.shape[0], y.shape[0])
        x = x[:n]
        y = y[:n]
    x = x.float()
    y = y.float()
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype)
    design = torch.cat([x, ones], dim=1)
    eye = torch.eye(design.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    lhs = design.T @ design + alpha * eye
    rhs = design.T @ y
    coef = torch.linalg.solve(lhs, rhs)
    weight = coef[:-1].T.contiguous()
    bias = coef[-1].contiguous()
    pred = x @ weight.T + bias
    mse = torch.mean((pred - y) ** 2).item()
    denom = torch.var(y, unbiased=False).clamp_min(1e-12).item()
    r2_like = 1.0 - mse / denom
    return weight, bias, {"mse": float(mse), "r2_like": float(r2_like), "n": int(x.shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a ridge linear projector baseline from teacher hidden dims to student hidden dims.")
    add_common_args(parser)
    parser.add_argument("--teacher-activation", default=None)
    parser.add_argument("--student-activation", default=None)
    parser.add_argument("--teacher-wfs", default=None)
    parser.add_argument("--student-wfs", default=None)
    parser.add_argument("--teacher-layer", type=int, default=None)
    parser.add_argument("--student-layer", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k-dims", type=float, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="YAML override key=value.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    ot_cfg = cfg["ot"]
    act_dir = resolve_path(cfg, cfg["activations"]["output_dir"])
    output_dir = resolve_path(cfg, args.output_dir) if args.output_dir else resolve_path(cfg, ot_cfg["output_dir"]) / "linear_projector"
    projector_path = output_dir / "projector.pt"
    metadata_path = output_dir / "metadata.json"
    if projector_path.exists() and metadata_path.exists() and not args.overwrite:
        print(f"[linear_baseline] skip existing outputs: {projector_path}, {metadata_path}")
        return

    teacher_layer = args.teacher_layer if args.teacher_layer is not None else int(ot_cfg.get("teacher_layer", cfg["activations"]["teacher_layers"][0]))
    student_layer = args.student_layer if args.student_layer is not None else int(ot_cfg.get("student_layer", cfg["activations"]["student_layers"][0]))
    teacher_activation = resolve_path(cfg, args.teacher_activation or act_dir / "teacher_math_hidden.pt")
    student_activation = resolve_path(cfg, args.student_activation or act_dir / "student_math_hidden.pt")
    teacher_wfs = resolve_path(cfg, args.teacher_wfs or ot_cfg.get("teacher_wfs", resolve_path(cfg, ot_cfg["output_dir"]) / "teacher_wfs_stats.npz"))
    student_wfs = resolve_path(cfg, args.student_wfs or ot_cfg.get("student_wfs", resolve_path(cfg, ot_cfg["output_dir"]) / "student_wfs_stats.npz"))
    top_k_dims = args.top_k_dims if args.top_k_dims is not None else ot_cfg.get("top_k_dims")
    alpha = float(args.ridge_alpha if args.ridge_alpha is not None else ot_cfg.get("ridge_alpha", 1.0))

    teacher_hidden = load_layer_activations(teacher_activation, teacher_layer)
    student_hidden = load_layer_activations(student_activation, student_layer)
    teacher_score = load_wfs_score(teacher_wfs, teacher_layer)
    student_score = load_wfs_score(student_wfs, student_layer)
    teacher_idx = select_top_dims(teacher_score, top_k_dims)
    student_idx = select_top_dims(student_score, top_k_dims)
    x = teacher_hidden[:, teacher_idx]
    y = student_hidden[:, student_idx]

    weight, bias, sanity = fit_ridge_projector(x, y, alpha=alpha)
    ensure_parent(projector_path)
    torch.save(
        {
            "weight": weight,
            "bias": bias,
            "teacher_dims": teacher_idx,
            "student_dims": student_idx,
            "teacher_layer": teacher_layer,
            "student_layer": student_layer,
            "ridge_alpha": alpha,
        },
        projector_path,
    )
    metadata = {
        "method": "ridge_linear_projector",
        "teacher_activation": str(teacher_activation),
        "student_activation": str(student_activation),
        "teacher_wfs": str(teacher_wfs),
        "student_wfs": str(student_wfs),
        "teacher_layer": teacher_layer,
        "student_layer": student_layer,
        "teacher_dims": teacher_idx.tolist(),
        "student_dims": student_idx.tolist(),
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "ridge_alpha": alpha,
        "top_k_dims": top_k_dims,
        "sanity": sanity,
    }
    write_json(metadata_path, metadata)
    print(f"[linear_baseline] saved projector={projector_path}")
    print(f"[linear_baseline] saved metadata={metadata_path}")
    print(f"[linear_baseline] sanity={json.dumps(sanity)}")


if __name__ == "__main__":
    main()
