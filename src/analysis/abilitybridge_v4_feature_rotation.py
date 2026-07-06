from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.activations.intervention_zeroout import get_layers_module
from src.models.load_model import infer_dtype, infer_load_in_4bit, load_causal_lm
from src.utils.config import apply_overrides, load_config, resolve_path
from src.utils.io import ensure_parent, read_jsonl, write_json, write_jsonl


EPS = 1e-8
PROTOCOL_VERSION = "abilitybridge_v4_feature_rotation_pruning_v1"
TARGET_METHODS = {"omcr", "aagr", "subspace", "ot_displacement"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def deep_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_v4_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = apply_overrides(load_config(path), overrides or [])
    cfg["_config_sha256"] = sha256_file(Path(path))
    return cfg


def root_dirs(root: Path) -> None:
    for name in ["rotations", "features", "causal", "maps", "pruning", "figures", "summary", "logs", "status", "configs"]:
        (root / name).mkdir(parents=True, exist_ok=True)


def status(root: Path, stage: str, message: str, **extra: Any) -> None:
    root_dirs(root)
    write_json(root / "status" / "queue_status.json", {"stage": stage, "message": message, "updated_at": now(), **extra})


def cfg_path(cfg: dict[str, Any], key: str) -> Path:
    return resolve_path(cfg, deep_get(cfg, key))


def output_root(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    return Path(args.root_dir or cfg_path(cfg, "paths.output_root"))


def protocol_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    v3_root = cfg_path(cfg, "paths.v3_root")
    v3_protocol = v3_root / "protocol" / "protocol_manifest.json"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "config_sha256": cfg["_config_sha256"],
        "v3_root": str(v3_root),
        "v3_protocol": str(v3_protocol),
        "v3_protocol_sha256": sha256_file(v3_protocol) if v3_protocol.exists() else None,
        "model_path": str(cfg_path(cfg, "paths.model_path")),
        "scope": {
            "stage1": deep_get(cfg, "grid"),
            "no_distillation": True,
            "ot_role": "feature_distribution_scoring_only",
        },
    }
    payload["protocol_hash"] = sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    payload["created_at"] = now()
    return payload


def ensure_protocol(root: Path, cfg: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    root_dirs(root)
    out = root / "protocol.json"
    payload = protocol_payload(cfg)
    if out.exists() and not overwrite:
        old = read_json(out)
        if old.get("protocol_hash") != payload["protocol_hash"]:
            raise RuntimeError(f"Protocol hash mismatch: existing={old.get('protocol_hash')} new={payload['protocol_hash']}")
        return old
    write_json(out, payload)
    return payload


class SparseAE(nn.Module):
    def __init__(self, d_in: int, d_latent: int, kind: str):
        super().__init__()
        self.kind = kind
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)
        self.gate = nn.Linear(d_in, d_latent) if kind in {"gated", "gap_gated"} else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mag = F.relu(self.encoder(x))
        if self.gate is None:
            return mag
        return mag * (self.gate(x) > 0).to(mag.dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z


def load_sae(path: Path, device: str | torch.device) -> SparseAE:
    payload = torch.load(path, map_location="cpu")
    model = SparseAE(int(payload["d_in"]), int(payload["d_latent"]), str(payload["kind"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def sae_paths(cfg: dict[str, Any], site: str, sae_kind: str) -> tuple[Path, Path]:
    base = cfg_path(cfg, "paths.v3_root") / "sae" / site / sae_kind
    return base / "checkpoint.pt", base / "feature_activation_cache.pt"


def load_feature_cache(cfg: dict[str, Any], site: str, sae_kind: str) -> tuple[torch.Tensor, list[str]]:
    _ckpt, cache = sae_paths(cfg, site, sae_kind)
    if not cache.exists():
        raise FileNotFoundError(f"Missing v3 feature cache: {cache}")
    payload = torch.load(cache, map_location="cpu")
    return payload["features"].float(), list(payload["labels"])


def grid_configs(cfg: dict[str, Any], include_stage2: bool = False) -> list[dict[str, Any]]:
    grid = deep_get(cfg, "grid", {})
    rows: list[dict[str, Any]] = []
    for layer in grid.get("layer", [4]):
        for site in grid.get("site", ["mlp"]):
            for sae_kind in grid.get("sae_kind", ["gated"]):
                for rotation in grid.get("rotation", ["omcr"]):
                    for dim in grid.get("dim", [4]):
                        for k in grid.get("k", [64]):
                            for ablation in grid.get("ablation", ["zero"]):
                                for seed in grid.get("seed", [42]):
                                    rows.append(
                                        {
                                            "layer": int(layer),
                                            "site": str(site),
                                            "sae_kind": str(sae_kind),
                                            "rotation": str(rotation),
                                            "dim": int(dim),
                                            "k": int(k),
                                            "ablation": str(ablation),
                                            "seed": int(seed),
                                        }
                                    )
    if include_stage2 and bool(deep_get(cfg, "stage2_extra.enabled", False)):
        extra = deep_get(cfg, "stage2_extra", {})
        for site in extra.get("site", ["mlp"]):
            for dim in extra.get("dim", [2, 4, 8, 16, 32]):
                for sparsity in extra.get("sparsity", [0.0]):
                    for ppl_penalty in extra.get("ppl_penalty", [0.3]):
                        for base in list(rows):
                            r = dict(base)
                            r.update({"site": site, "dim": int(dim), "sparsity": float(sparsity), "ppl_penalty": float(ppl_penalty), "stage2": True})
                            rows.append(r)
    for idx, row in enumerate(rows):
        row["config_id"] = config_id(row)
    return rows


def config_id(row: dict[str, Any]) -> str:
    keys = ["layer", "site", "sae_kind", "rotation", "dim", "k", "ablation", "seed"]
    return "_".join(str(row[k]).replace(".", "p") for k in keys)


def split_rows(rows: list[dict[str, Any]], worker: str) -> list[dict[str, Any]]:
    if worker in {"all", ""}:
        return rows
    idx = 0 if worker in {"gpu0", "0"} else 1
    return [row for i, row in enumerate(rows) if i % 2 == idx]


def group_masks(labels: list[str]) -> dict[str, torch.Tensor]:
    return {g: torch.tensor([lab == g for lab in labels], dtype=torch.bool) for g in sorted(set(labels))}


def group_mean(features: torch.Tensor, masks: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    mask = masks.get(name)
    if mask is None or int(mask.sum()) == 0:
        return torch.zeros(features.shape[1])
    return features[mask].mean(dim=0)


def centered_cov(x: torch.Tensor, max_rows: int = 4096) -> torch.Tensor:
    if x.shape[0] == 0:
        return torch.eye(x.shape[1])
    if x.shape[0] > max_rows:
        gen = torch.Generator().manual_seed(123)
        idx = torch.randperm(x.shape[0], generator=gen)[:max_rows]
        x = x[idx]
    x = x - x.mean(dim=0, keepdim=True)
    return (x.T @ x) / max(1, x.shape[0] - 1)


def orthonormal_rows(mat: torch.Tensor, dim: int) -> torch.Tensor:
    mat = torch.nan_to_num(mat.float())
    if mat.ndim == 1:
        mat = mat.unsqueeze(0)
    q, _ = torch.linalg.qr(mat.T, mode="reduced")
    r = q[:, : min(dim, q.shape[1])].T.contiguous()
    if r.shape[0] < dim:
        gen = torch.Generator().manual_seed(777)
        extra = torch.randn(dim - r.shape[0], mat.shape[1], generator=gen)
        r = torch.cat([r, orthonormal_rows(extra, dim - r.shape[0])], dim=0)
        r = orthonormal_rows(r, dim)
    return r[:dim]


def train_omcr(features: torch.Tensor, labels: list[str], dim: int, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    masks = group_masks(labels)
    mu_gap = group_mean(features, masks, "math_gap")
    if torch.allclose(mu_gap, torch.zeros_like(mu_gap)):
        mu_gap = group_mean(features, masks, "math_gap_proxy")
    mu_easy = group_mean(features, masks, "math_easy")
    mu_control = group_mean(features, masks, "control")
    mu_lm = group_mean(features, masks, "lm")
    beta = float(deep_get(cfg, "rotation.beta_easy", 0.5))
    lc = float(deep_get(cfg, "rotation.lambda_control", 0.3))
    ll = float(deep_get(cfg, "rotation.lambda_lm", 0.3))
    ridge = float(deep_get(cfg, "rotation.ridge", 1e-3))
    signal = torch.outer(mu_gap - mu_control, mu_gap - mu_control) + beta * torch.outer(mu_gap - mu_easy, mu_gap - mu_easy)
    cov_control = centered_cov(features[masks.get("control", torch.zeros(len(labels), dtype=torch.bool))])
    cov_lm = centered_cov(features[masks.get("lm", torch.zeros(len(labels), dtype=torch.bool))])
    mat = signal - lc * cov_control - ll * cov_lm + ridge * torch.eye(features.shape[1])
    eigvals, eigvecs = torch.linalg.eigh(mat)
    order = torch.argsort(eigvals, descending=True)
    r = eigvecs[:, order[:dim]].T.contiguous()
    direction_scores = eigvals[order[:dim]].float()
    return r, direction_scores, {"objective": "omcr", "eig_max": float(eigvals.max()), "eig_min": float(eigvals.min())}


def train_subspace(features: torch.Tensor, labels: list[str], dim: int, _cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    masks = group_masks(labels)
    means = [
        group_mean(features, masks, "math_gap") - group_mean(features, masks, "control"),
        group_mean(features, masks, "math_gap") - group_mean(features, masks, "lm"),
        group_mean(features, masks, "math_easy") - group_mean(features, masks, "control"),
    ]
    mat = torch.stack(means)
    if mat.shape[0] < dim:
        mat = torch.cat([mat, centered_cov(features).diagonal().unsqueeze(0)], dim=0)
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    r = orthonormal_rows(vh[: min(dim, vh.shape[0])], dim)
    scores = torch.zeros(r.shape[0])
    scores[: min(len(s), len(scores))] = s[: min(len(s), len(scores))]
    return r, scores, {"objective": "subspace", "singular_values": [float(x) for x in s[:dim]]}


def train_ot_displacement(features: torch.Tensor, labels: list[str], dim: int, _cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    masks = group_masks(labels)
    gap = group_mean(features, masks, "math_gap")
    if torch.allclose(gap, torch.zeros_like(gap)):
        gap = group_mean(features, masks, "math_gap_proxy")
    control = group_mean(features, masks, "control")
    lm = group_mean(features, masks, "lm")
    displacement = (gap - control).abs() - 0.5 * (control - lm).abs()
    order = torch.argsort(displacement.abs(), descending=True)
    r = torch.zeros(dim, features.shape[1])
    for i, j in enumerate(order[:dim]):
        r[i, j] = 1.0 if displacement[j] >= 0 else -1.0
    return r, displacement[order[:dim]].abs(), {"objective": "ot_displacement_proxy", "note": "Uses axis-wise transport displacement proxy for long grid ranking."}


def train_aagr(features: torch.Tensor, labels: list[str], dim: int, cfg: dict[str, Any], seed: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    gen = torch.Generator().manual_seed(seed)
    n = features.shape[0]
    y = torch.zeros(n, 1)
    weights = torch.ones(n, 1)
    for i, lab in enumerate(labels):
        if lab.startswith("math_gap"):
            y[i] = 1.0
            weights[i] = 2.0
        elif lab == "math_easy":
            y[i] = 0.75
            weights[i] = 1.25
        elif lab in {"control", "lm"}:
            y[i] = 0.0
            weights[i] = 1.0
    x = (features - features.mean(dim=0, keepdim=True)) / (features.std(dim=0, keepdim=True) + EPS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.to(device)
    y = y.to(device)
    weights = weights.to(device)
    r = torch.randn(dim, x.shape[1], generator=gen, device=device) / math.sqrt(x.shape[1])
    r.requires_grad_(True)
    head = torch.zeros(dim, 1, device=device, requires_grad=True)
    opt = torch.optim.AdamW([r, head], lr=float(deep_get(cfg, "rotation.aagr_lr", 0.03)))
    steps = int(deep_get(cfg, "rotation.aagr_steps", 800))
    batch_size = int(deep_get(cfg, "rotation.batch_size", 512))
    l1 = float(deep_get(cfg, "rotation.aagr_l1", 1e-4))
    orth = float(deep_get(cfg, "rotation.aagr_orthogonal", 0.05))
    for _ in tqdm(range(steps), desc="train-aagr"):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        z = x[idx] @ r.T
        logits = z @ head
        bce = F.binary_cross_entropy_with_logits(logits, y[idx], weight=weights[idx])
        gram = r @ r.T
        loss = bce + l1 * r.abs().mean() + orth * ((gram - torch.eye(dim, device=device)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        rr = orthonormal_rows(r.detach().cpu(), dim)
        scores = head.detach().abs().flatten().cpu()
    return rr, scores, {"objective": "aagr_proxy", "steps": steps, "uses_label_proxy": True}


def feature_scores_from_rotation(features: torch.Tensor, labels: list[str], rotation: torch.Tensor, direction_scores: torch.Tensor) -> torch.Tensor:
    masks = group_masks(labels)
    gap = group_mean(features, masks, "math_gap")
    if torch.allclose(gap, torch.zeros_like(gap)):
        gap = group_mean(features, masks, "math_gap_proxy")
    control = group_mean(features, masks, "control")
    lm = group_mean(features, masks, "lm")
    load = rotation.abs()
    dscore = direction_scores.abs()
    if dscore.numel() < rotation.shape[0]:
        dscore = F.pad(dscore, (0, rotation.shape[0] - dscore.numel()), value=1.0)
    base = (load * dscore[: rotation.shape[0]].view(-1, 1)).sum(dim=0)
    specificity = (gap - control).abs() - 0.25 * (control - lm).abs()
    return base * torch.clamp(specificity, min=0.0)


def rotation_artifact_path(root: Path, row: dict[str, Any]) -> Path:
    return root / "rotations" / row["site"] / row["sae_kind"] / f"{row['rotation']}_d{row['dim']}_seed{row['seed']}.pt"


def manifest_path(root: Path, row: dict[str, Any]) -> Path:
    return root / "features" / row["site"] / row["sae_kind"] / f"{row['rotation']}_d{row['dim']}_k{row['k']}_{row['ablation']}_seed{row['seed']}.pt"


def train_rotation(args: argparse.Namespace) -> None:
    cfg = load_v4_config(args.config, args.overrides)
    root = output_root(args, cfg)
    ensure_protocol(root, cfg, args.overwrite)
    rows = grid_configs(cfg, args.include_stage2)
    rows = [r for r in rows if (not args.config_id or r["config_id"] == args.config_id)]
    rows = split_rows(rows, args.worker)
    for row in rows:
        out = rotation_artifact_path(root, row)
        if out.exists() and not args.overwrite:
            continue
        status(root, "train-rotation", f"{row['config_id']}", worker=args.worker)
        features, labels = load_feature_cache(cfg, row["site"], row["sae_kind"])
        method = row["rotation"]
        if method == "omcr":
            r, ds, meta = train_omcr(features, labels, row["dim"], cfg)
        elif method == "aagr":
            r, ds, meta = train_aagr(features, labels, row["dim"], cfg, row["seed"])
        elif method == "subspace":
            r, ds, meta = train_subspace(features, labels, row["dim"], cfg)
        elif method == "ot_displacement":
            r, ds, meta = train_ot_displacement(features, labels, row["dim"], cfg)
        else:
            raise ValueError(f"Unknown rotation method: {method}")
        scores = feature_scores_from_rotation(features, labels, r, ds)
        ensure_parent(out)
        torch.save({"rotation": r, "direction_scores": ds, "feature_scores": scores, "config": row, "meta": meta, "protocol_hash": read_json(root / "protocol.json")["protocol_hash"]}, out)


def score_features(args: argparse.Namespace) -> None:
    cfg = load_v4_config(args.config, args.overrides)
    root = output_root(args, cfg)
    ensure_protocol(root, cfg, args.overwrite)
    rows = split_rows(grid_configs(cfg, args.include_stage2), args.worker)
    rows = [r for r in rows if (not args.config_id or r["config_id"] == args.config_id)]
    for row in rows:
        art = rotation_artifact_path(root, row)
        if not art.exists():
            if args.train_missing:
                one = argparse.Namespace(**vars(args))
                one.config_id = row["config_id"]
                train_rotation(one)
            else:
                raise FileNotFoundError(f"Missing rotation artifact: {art}")
        out = manifest_path(root, row)
        if out.exists() and not args.overwrite:
            continue
        payload = torch.load(art, map_location="cpu")
        scores = payload["feature_scores"].float()
        order = torch.argsort(scores, descending=True)
        k = min(int(row["k"]), int(scores.numel()))
        idx = order[:k].long()
        features, _labels = load_feature_cache(cfg, row["site"], row["sae_kind"])
        mean_values = features[:, idx].mean(dim=0)
        ensure_parent(out)
        torch.save({"feature_indices": idx, "mean_values": mean_values, "scores": scores[idx], "config": row, "protocol_hash": payload["protocol_hash"]}, out)


def blocks(model):
    return get_layers_module(model)


def mlp_module(block):
    if hasattr(block, "mlp"):
        return block.mlp
    raise ValueError("Could not find block.mlp")


def simple_prompt(question: str, tokenizer) -> str:
    messages = [{"role": "user", "content": f"Solve the problem. Put the final answer after ####.\\n\\n{question}"}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"] + "\nAnswer:"


def extract_answer(text: str) -> str:
    if "####" in text:
        text = text.split("####")[-1]
    boxed = re.findall(r"\\boxed\\{([^{}]+)\\}", text)
    if boxed:
        text = boxed[-1]
    nums = re.findall(r"[-+]?\\d+(?:\\.\\d+)?(?:/\\d+)?", text.replace(",", ""))
    return nums[-1] if nums else text.strip().splitlines()[-1].strip() if text.strip() else ""


def norm_answer(text: str) -> str:
    text = extract_answer(str(text)).strip().lower()
    text = text.replace(",", "").replace("$", "")
    text = re.sub(r"\\s+", "", text)
    return text


@torch.inference_mode()
def eval_math_rows(model, tokenizer, rows: list[dict[str, Any]], max_new_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    correct = 0
    for row in tqdm(rows, desc="eval-math500"):
        prompt = simple_prompt(str(row["question"]), tokenizer)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = norm_answer(response)
        target = norm_answer(str(row["target"]))
        ok = pred == target
        correct += int(ok)
        preds.append({"id": row["id"], "question": row["question"], "target": row["target"], "prediction": pred, "correct": ok, "response": response})
    n = len(preds)
    return preds, {"n": n, "correct": correct, "accuracy": correct / n if n else None}


@torch.inference_mode()
def eval_wikitext(model, tokenizer, texts: list[str], max_seq_len: int) -> dict[str, Any]:
    total_loss = 0.0
    total_tokens = 0
    for text in tqdm(texts, desc="eval-wikitext"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        if enc["input_ids"].shape[1] < 2:
            continue
        enc = enc.to(model.device)
        out = model(**enc, labels=enc["input_ids"])
        tokens = int(enc["attention_mask"].sum().item() - 1)
        total_loss += float(out.loss.detach().cpu()) * tokens
        total_tokens += tokens
    loss = total_loss / total_tokens if total_tokens else None
    return {"n": len(texts), "tokens": total_tokens, "loss": loss, "perplexity": math.exp(loss) if loss is not None and loss < 100 else None}


@contextmanager
def sae_feature_hook(model, sae: SparseAE, layer: int, site: str, feature_indices: torch.Tensor, ablation: str, mean_values: torch.Tensor | None):
    block = blocks(model)[layer - 1]
    module = block if site == "resid" else mlp_module(block)
    device = next(sae.parameters()).device
    idx = feature_indices.to(device)
    means = mean_values.to(device) if mean_values is not None else None

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1]).to(device).float()
        with torch.no_grad():
            z = sae.encode(flat)
            if ablation == "mean" and means is not None:
                z[:, idx] = means.to(z.dtype)
            else:
                z[:, idx] = 0
            recon = sae.decoder(z).to(hidden.device, dtype=hidden.dtype).reshape(shape)
        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def load_eval_data(cfg: dict[str, Any], math_samples: int, wiki_samples: int) -> tuple[list[dict[str, Any]], list[str]]:
    shared = cfg_path(cfg, "paths.formal_root") / "shared"
    math_rows = read_jsonl(shared / "math500.jsonl")[:math_samples]
    wiki_rows = read_jsonl(shared / "wikitext.jsonl")[:wiki_samples]
    wiki_texts = [str(r.get("text", "")) for r in wiki_rows if str(r.get("text", "")).strip()]
    return math_rows, wiki_texts


def run_feature_causal(args: argparse.Namespace) -> None:
    cfg = load_v4_config(args.config, args.overrides)
    root = output_root(args, cfg)
    ensure_protocol(root, cfg, args.overwrite)
    rows = grid_configs(cfg, args.include_stage2)
    if args.smoke:
        smoke = deep_get(cfg, "smoke", {})
        rows = [
            {
                "layer": 4,
                "site": "mlp",
                "sae_kind": smoke.get("sae_kind", "gated"),
                "rotation": smoke.get("rotation", "omcr"),
                "dim": int(smoke.get("dim", 4)),
                "k": int(smoke.get("k", 8)),
                "ablation": "zero",
                "seed": int(deep_get(cfg, "experiment.seed", 42)),
            }
        ]
        for row in rows:
            row["config_id"] = config_id(row)
    rows = split_rows(rows, args.worker)
    rows = [r for r in rows if (not args.config_id or r["config_id"] == args.config_id)]
    max_new_tokens = int(deep_get(cfg, "runtime.max_new_tokens", 256))
    max_seq_len = int(deep_get(cfg, "runtime.max_seq_length", 768))
    math_samples = int(args.max_samples or (deep_get(cfg, "smoke.max_samples") if args.smoke else deep_get(cfg, "causal.math_samples", 500)))
    wiki_samples = int(args.wikitext_samples or (deep_get(cfg, "smoke.wikitext_samples") if args.smoke else deep_get(cfg, "causal.wikitext_samples", 256)))
    model_path = cfg_path(cfg, "paths.model_path")
    model, tokenizer = None, None
    loaded_sae: dict[tuple[str, str], SparseAE] = {}
    try:
        for row in rows:
            run_dir = root / "causal" / "runs" / row["config_id"]
            complete = run_dir / "run_complete.json"
            if complete.exists() and not args.overwrite:
                continue
            score_features(argparse.Namespace(**{**vars(args), "config_id": row["config_id"], "train_missing": True}))
            manifest = torch.load(manifest_path(root, row), map_location="cpu")
            if model is None:
                model, tokenizer = load_causal_lm(
                    model_path,
                    dtype=infer_dtype(str(deep_get(cfg, "runtime.dtype", "auto")), cfg),
                    device_map=deep_get(cfg, "runtime.device_map", "auto"),
                    load_in_4bit=bool(deep_get(cfg, "runtime.load_in_4bit", infer_load_in_4bit(cfg))),
                )
            key = (row["site"], row["sae_kind"])
            if key not in loaded_sae:
                ckpt, _cache = sae_paths(cfg, row["site"], row["sae_kind"])
                loaded_sae[key] = load_sae(ckpt, "cuda" if torch.cuda.is_available() else "cpu")
            math_rows, wiki_texts = load_eval_data(cfg, math_samples, wiki_samples)
            status(root, "feature-causal", row["config_id"], worker=args.worker)
            with sae_feature_hook(model, loaded_sae[key], int(row["layer"]), row["site"], manifest["feature_indices"], row["ablation"], manifest.get("mean_values")):
                preds, math_metrics = eval_math_rows(model, tokenizer, math_rows, max_new_tokens)
                ppl_metrics = eval_wikitext(model, tokenizer, wiki_texts, max_seq_len)
            pred_path = root / "causal" / "preds" / f"{row['config_id']}.jsonl"
            write_jsonl(pred_path, preds)
            write_json(
                complete,
                {
                    **row,
                    "math500_accuracy": math_metrics["accuracy"],
                    "math500_n": math_metrics["n"],
                    "wikitext_perplexity": ppl_metrics["perplexity"],
                    "heldout_lm_loss": ppl_metrics["loss"],
                    "feature_score_mean": float(manifest["scores"].float().mean()),
                    "completed_at": now(),
                    "protocol_hash": read_json(root / "protocol.json")["protocol_hash"],
                },
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def decoder_group_scores(cfg: dict[str, Any], row: dict[str, Any], feature_indices: torch.Tensor) -> torch.Tensor:
    ckpt, _cache = sae_paths(cfg, row["site"], row["sae_kind"])
    payload = torch.load(ckpt, map_location="cpu")
    decoder_w = payload["state_dict"]["decoder.weight"].float()
    signal = decoder_w[:, feature_indices].abs().mean(dim=1)
    return signal


def build_mlp_map(args: argparse.Namespace) -> None:
    cfg = load_v4_config(args.config, args.overrides)
    root = output_root(args, cfg)
    ensure_protocol(root, cfg, args.overwrite)
    rows = split_rows(grid_configs(cfg, args.include_stage2), args.worker)
    rows = [r for r in rows if (not args.config_id or r["config_id"] == args.config_id)]
    top_n = int(deep_get(cfg, "map.top_groups", 256))
    bottom_n = int(deep_get(cfg, "map.bottom_groups", 256))
    random_n = int(deep_get(cfg, "map.random_groups", 256))
    for row in rows:
        out = root / "maps" / f"{row['config_id']}_groups.csv"
        if out.exists() and not args.overwrite:
            continue
        score_features(argparse.Namespace(**{**vars(args), "config_id": row["config_id"], "train_missing": True}))
        manifest = torch.load(manifest_path(root, row), map_location="cpu")
        scores = decoder_group_scores(cfg, row, manifest["feature_indices"])
        order = torch.argsort(scores, descending=True)
        gen = torch.Generator().manual_seed(int(row["seed"]) + 991)
        rand = torch.randperm(scores.numel(), generator=gen)[: min(random_n, scores.numel())]
        rows_out: list[dict[str, Any]] = []
        for rank, idx in enumerate(order[: min(top_n, scores.numel())]):
            rows_out.append({"group": int(idx), "score": float(scores[idx]), "bucket": "top_preserve", "rank": rank, **row})
        for rank, idx in enumerate(torch.flip(order, dims=[0])[: min(bottom_n, scores.numel())]):
            rows_out.append({"group": int(idx), "score": float(scores[idx]), "bucket": "bottom_prunable", "rank": rank, **row})
        for rank, idx in enumerate(rand):
            rows_out.append({"group": int(idx), "score": float(scores[idx]), "bucket": "random", "rank": rank, **row})
        write_csv(out, rows_out)


@contextmanager
def mlp_group_zero_hook(model, layer: int, group_indices: torch.Tensor):
    block = blocks(model)[layer - 1]
    module = mlp_module(block)
    idx = group_indices.to(model.device)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        idx_clamped = idx[idx < patched.shape[-1]]
        patched[..., idx_clamped] = 0
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def run_pruning(args: argparse.Namespace) -> None:
    cfg = load_v4_config(args.config, args.overrides)
    root = output_root(args, cfg)
    ensure_protocol(root, cfg, args.overwrite)
    rows = split_rows(grid_configs(cfg, args.include_stage2), args.worker)
    rows = [r for r in rows if (not args.config_id or r["config_id"] == args.config_id)]
    ratios = [float(x) for x in deep_get(cfg, "map.ratios", [0.01])]
    methods = list(deep_get(cfg, "map.methods", ["rotation_map", "random"]))
    model_path = cfg_path(cfg, "paths.model_path")
    model, tokenizer = load_causal_lm(model_path, dtype=infer_dtype(str(deep_get(cfg, "runtime.dtype", "auto")), cfg), device_map=deep_get(cfg, "runtime.device_map", "auto"), load_in_4bit=bool(deep_get(cfg, "runtime.load_in_4bit", False)))
    math_rows, wiki_texts = load_eval_data(cfg, int(args.max_samples or deep_get(cfg, "causal.math_samples", 500)), int(args.wikitext_samples or deep_get(cfg, "causal.wikitext_samples", 256)))
    max_new_tokens = int(deep_get(cfg, "runtime.max_new_tokens", 256))
    max_seq_len = int(deep_get(cfg, "runtime.max_seq_length", 768))
    try:
        for row in rows:
            build_mlp_map(argparse.Namespace(**{**vars(args), "config_id": row["config_id"]}))
            map_rows = read_csv(root / "maps" / f"{row['config_id']}_groups.csv")
            if not map_rows:
                continue
            n_groups = max(int(r["group"]) for r in map_rows) + 1
            for method in methods:
                if method == "rotation_map":
                    ordered = [int(r["group"]) for r in map_rows if r["bucket"] == "bottom_prunable"]
                elif method == "decoder_high":
                    ordered = [int(r["group"]) for r in map_rows if r["bucket"] == "top_preserve"]
                elif method == "random":
                    ordered = [int(r["group"]) for r in map_rows if r["bucket"] == "random"]
                else:
                    ordered = [int(r["group"]) for r in map_rows if r["bucket"] == "bottom_prunable"]
                for ratio in ratios:
                    k = max(1, min(len(ordered), int(n_groups * ratio)))
                    run_id = f"{row['config_id']}_{method}_r{str(ratio).replace('.', 'p')}"
                    complete = root / "pruning" / "runs" / run_id / "run_complete.json"
                    if complete.exists() and not args.overwrite:
                        continue
                    status(root, "pruning-eval", run_id, worker=args.worker)
                    with mlp_group_zero_hook(model, int(row["layer"]), torch.tensor(ordered[:k]).long()):
                        preds, math_metrics = eval_math_rows(model, tokenizer, math_rows, max_new_tokens)
                        ppl_metrics = eval_wikitext(model, tokenizer, wiki_texts, max_seq_len)
                    write_jsonl(root / "pruning" / "preds" / f"{run_id}.jsonl", preds)
                    write_json(complete, {**row, "method": method, "ratio": ratio, "n_groups": k, "math500_accuracy": math_metrics["accuracy"], "math500_n": math_metrics["n"], "wikitext_perplexity": ppl_metrics["perplexity"], "heldout_lm_loss": ppl_metrics["loss"], "completed_at": now()})
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def paired_bootstrap(deltas: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    vals = []
    for _ in range(n_boot):
        vals.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    vals.sort()
    return mean(deltas), vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def summarize(args: argparse.Namespace) -> None:
    cfg = load_v4_config(args.config, args.overrides)
    root = output_root(args, cfg)
    protocol = ensure_protocol(root, cfg, args.overwrite)
    causal_rows = []
    for p in sorted((root / "causal" / "runs").glob("*/run_complete.json")):
        row = read_json(p)
        if row.get("protocol_hash") != protocol["protocol_hash"]:
            raise RuntimeError(f"Protocol mismatch in {p}")
        causal_rows.append(row)
    write_csv(root / "summary" / "feature_causal_results.csv", causal_rows)
    prune_rows = [read_json(p) for p in sorted((root / "pruning" / "runs").glob("*/run_complete.json"))]
    write_csv(root / "summary" / "pruning_results.csv", prune_rows)
    lines = [
        "# AbilityBridge-v4 Feature Rotation Summary",
        "",
        f"- Protocol: `{PROTOCOL_VERSION}`",
        f"- Feature causal completed: `{len(causal_rows)}`",
        f"- Pruning eval completed: `{len(prune_rows)}`",
        "",
    ]
    if causal_rows:
        best = sorted(causal_rows, key=lambda r: float(r.get("math500_accuracy") or -1), reverse=True)[:10]
        lines += ["## Top Feature Causal Runs", "", "| config | acc | ppl |", "| --- | ---: | ---: |"]
        for r in best:
            lines.append(f"| `{r['config_id']}` | `{float(r.get('math500_accuracy') or 0):.4f}` | `{float(r.get('wikitext_perplexity') or 0):.2f}` |")
        lines.append("")
    (root / "summary" / "v4_feature_rotation_report.md").write_text("\n".join(lines), encoding="utf-8")


def list_grid(args: argparse.Namespace) -> None:
    cfg = load_v4_config(args.config, args.overrides)
    root = output_root(args, cfg)
    ensure_protocol(root, cfg, args.overwrite)
    rows = grid_configs(cfg, args.include_stage2)
    rows = split_rows(rows, args.worker)
    write_csv(root / "configs" / f"grid_{args.worker}.csv", rows)
    print(json.dumps({"configs": len(rows), "worker": args.worker, "root": str(root), "grid_csv": str(root / "configs" / f"grid_{args.worker}.csv")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="AbilityBridge-v4 feature rotation and pruning grid")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/abilitybridge_v4_rotation_grid.yaml")
    common.add_argument("--root-dir", default=None)
    common.add_argument("--workspace-root", default=None, help="Reserved for script compatibility; paths are resolved from config.")
    common.add_argument("--worker", default="all", choices=["all", "gpu0", "gpu1", "0", "1"])
    common.add_argument("--config-id", default="")
    common.add_argument("--include-stage2", action="store_true")
    common.add_argument("--overwrite", action="store_true")
    common.add_argument("--overrides", action="append", default=[], metavar="key=value")
    common.add_argument("--gpu", default="")

    p = sub.add_parser("list-grid", parents=[common])
    p.set_defaults(func=list_grid)
    p = sub.add_parser("train-rotation", parents=[common])
    p.set_defaults(func=train_rotation)
    p = sub.add_parser("score-features", parents=[common])
    p.add_argument("--train-missing", action="store_true")
    p.set_defaults(func=score_features)
    p = sub.add_parser("run-feature-causal", parents=[common])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--wikitext-samples", type=int, default=None)
    p.add_argument("--train-missing", action="store_true")
    p.set_defaults(func=run_feature_causal)
    p = sub.add_parser("build-mlp-map", parents=[common])
    p.add_argument("--train-missing", action="store_true")
    p.set_defaults(func=build_mlp_map)
    p = sub.add_parser("run-pruning", parents=[common])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--wikitext-samples", type=int, default=None)
    p.add_argument("--train-missing", action="store_true")
    p.set_defaults(func=run_pruning)
    p = sub.add_parser("summarize", parents=[common])
    p.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
