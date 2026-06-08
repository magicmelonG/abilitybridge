# AbilityBridge MVP Pipeline

This document describes the current runnable flow, environment, files, and expected artifacts for the AbilityBridge toy experiment.

## Can The Full Flow Run Now?

Mostly yes for the implemented methods:

- `base_student`
- `sft_lora`
- `sft_linear_rep`
- `sft_vanilla_ot_rep`
- `sft_ability_ot_rep`

The current repository has all major stages implemented: data preparation, base evaluation, hidden-state collection, WFS scoring, zero-out intervention, OT solving, linear projector baseline, LoRA SFT training, OT representation LoRA training, unified evaluation, and result tables.

One important caveat: `sft_random_mask_rep` is listed in the final table, but random-mask representation training is not implemented yet. The intervention script has a random top-k baseline, but the training script currently supports only:

- `linear`
- `vanilla_ot`
- `ability_ot`

So a complete six-method ablation table will show `NA` for `sft_random_mask_rep` until that training path is added.

Another practical caveat: this Windows PowerShell environment currently does not have `bash` available. The `.sh` scripts are written, but on this machine the safer way is to run the Python module commands shown below.

## Environment

Use the conda environment that was created earlier:

```powershell
conda activate abilitybridge
```

Dependencies are listed in:

```text
requirements.txt
```

Refresh them if needed:

```powershell
python -m pip install -r requirements.txt
```

Local model directories:

```text
models/teacher/qwen2p5_math_1p5b_instruct
models/student/qwen2p5_0p5b_instruct
```

Main config:

```text
configs/toy_qwen_1p5b_to_0p5b.yaml
```

The config controls model paths, data paths, layer choices, WFS settings, OT settings, LoRA settings, evaluation sample counts, and output directories.

## Recommended End-To-End Flow

Run these commands from the repository root:

```text
g:\ICLR\workspace
```

### 1. Prepare GSM8K Toy Data

```powershell
python -m src.data.prepare_data --config configs/toy_qwen_1p5b_to_0p5b.yaml
```

Outputs:

```text
data/gsm8k_toy/train.jsonl
data/gsm8k_toy/eval.jsonl
```

The default config prepares `train=128` and `eval=64` samples.

### 2. Evaluate Base Student

```powershell
python -m src.eval.eval_gsm8k --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student
```

Outputs:

```text
outputs/eval/student_base_predictions.jsonl
outputs/eval/student_base_metrics.json
```

This is useful as an early sanity check before the heavier activation and training stages.

### 3. Collect Teacher And Student Hidden States

Teacher math/control:

```powershell
python -m src.activations.collect_hidden --config configs/toy_qwen_1p5b_to_0p5b.yaml --role teacher --kind math
python -m src.activations.collect_hidden --config configs/toy_qwen_1p5b_to_0p5b.yaml --role teacher --kind control
```

Student math/control:

```powershell
python -m src.activations.collect_hidden --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student --kind math
python -m src.activations.collect_hidden --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student --kind control
```

Outputs:

```text
outputs/activations/teacher_math_hidden.pt
outputs/activations/teacher_control_hidden.pt
outputs/activations/student_math_hidden.pt
outputs/activations/student_control_hidden.pt
```

Only the configured layers and the last non-padding token vector are saved. This avoids storing full activation tensors.

### 4. Compute WFS Scores

```powershell
python -m src.activations.compute_wfs --config configs/toy_qwen_1p5b_to_0p5b.yaml --role teacher
python -m src.activations.compute_wfs --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student
```

Outputs:

```text
outputs/ot/teacher_wfs_stats.npz
outputs/ot/teacher_wfs_topk.jsonl
outputs/ot/teacher_wfs_topk.pt
outputs/ot/student_wfs_stats.npz
outputs/ot/student_wfs_topk.jsonl
outputs/ot/student_wfs_topk.pt
```

For each layer and hidden dimension, WFS stores:

- activation frequency
- mean positive activation
- `WFS_math`
- `WFS_ctrl`
- `score = WFS_math - gamma * WFS_ctrl`

### 5. Optional Zero-Out Check

```powershell
python -m src.activations.intervention_zeroout --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student
```

Output:

```text
outputs/ot/student_zeroout_intervention.csv
```

CSV columns:

```text
layer,k,method,math_drop,ctrl_drop
```

This compares WFS top-k zero-out against a random top-k baseline.

### 6. Solve OT And Linear Projector

```powershell
python -m src.ot.solve_ot --config configs/toy_qwen_1p5b_to_0p5b.yaml `
  --teacher-activation outputs/activations/teacher_math_hidden.pt `
  --student-activation outputs/activations/student_math_hidden.pt `
  --teacher-wfs outputs/ot/teacher_wfs_stats.npz `
  --student-wfs outputs/ot/student_wfs_stats.npz `
  --teacher-layer 8 `
  --student-layer 4
```

Outputs:

```text
outputs/ot/alignment/vanilla/ot_matrix.pt
outputs/ot/alignment/vanilla/cost_matrix.pt
outputs/ot/alignment/vanilla/metadata.json
outputs/ot/alignment/ability_aware/ot_matrix.pt
outputs/ot/alignment/ability_aware/cost_matrix.pt
outputs/ot/alignment/ability_aware/metadata.json
```

Vanilla OT cost:

```text
cost = 1 - correlation
```

Ability-aware OT cost:

```text
cost = 1 - correlation
     + lambda * abs(norm_s_student - norm_s_teacher)
     - beta * norm_s_student * norm_s_teacher
```

Then fit the linear projector baseline:

```powershell
python -m src.ot.linear_baseline --config configs/toy_qwen_1p5b_to_0p5b.yaml `
  --teacher-activation outputs/activations/teacher_math_hidden.pt `
  --student-activation outputs/activations/student_math_hidden.pt `
  --teacher-wfs outputs/ot/teacher_wfs_stats.npz `
  --student-wfs outputs/ot/student_wfs_stats.npz `
  --teacher-layer 8 `
  --student-layer 4
```

Outputs:

```text
outputs/ot/linear_projector/projector.pt
outputs/ot/linear_projector/metadata.json
```

### 7. Train SFT LoRA Baseline

```powershell
python -m src.train.train_sft_lora --config configs/toy_qwen_1p5b_to_0p5b.yaml
```

Outputs:

```text
outputs/checkpoints/sft_lora/checkpoint
outputs/checkpoints/sft_lora/train_log.csv
outputs/checkpoints/sft_lora/config_used.yaml
outputs/checkpoints/sft_lora/run_metadata.json
```

Only LoRA parameters are trainable.

### 8. Train Representation-Distillation LoRA Variants

Linear projector:

```powershell
python -m src.train.train_ot_rep_lora --config configs/toy_qwen_1p5b_to_0p5b.yaml `
  --method linear `
  --teacher-hidden outputs/activations/teacher_math_hidden.pt `
  --linear-dir outputs/ot/linear_projector `
  --output-dir outputs/checkpoints/linear_rep_lora
```

Vanilla OT:

```powershell
python -m src.train.train_ot_rep_lora --config configs/toy_qwen_1p5b_to_0p5b.yaml `
  --method vanilla_ot `
  --teacher-hidden outputs/activations/teacher_math_hidden.pt `
  --alignment-dir outputs/ot/alignment `
  --output-dir outputs/checkpoints/vanilla_ot_rep_lora
```

Ability-aware OT:

```powershell
python -m src.train.train_ot_rep_lora --config configs/toy_qwen_1p5b_to_0p5b.yaml `
  --method ability_ot `
  --teacher-hidden outputs/activations/teacher_math_hidden.pt `
  --alignment-dir outputs/ot/alignment `
  --output-dir outputs/checkpoints/ability_ot_rep_lora
```

Loss:

```text
loss = sft_loss + beta * rep_loss
```

Outputs for each method:

```text
outputs/checkpoints/<method_dir>/checkpoint
outputs/checkpoints/<method_dir>/train_log.csv
outputs/checkpoints/<method_dir>/config_used.yaml
outputs/checkpoints/<method_dir>/run_metadata.json
```

### 9. Unified Evaluation And Tables

Full evaluation:

```powershell
python -m src.eval.eval_all --config configs/toy_qwen_1p5b_to_0p5b.yaml
```

Evaluation sample counts can be controlled:

```powershell
python -m src.eval.eval_all --config configs/toy_qwen_1p5b_to_0p5b.yaml `
  --gsm8k-samples 64 `
  --math500-samples 100 `
  --wikitext-samples 128
```

Only rebuild result tables from existing metrics:

```powershell
python -m src.eval.eval_all --config configs/toy_qwen_1p5b_to_0p5b.yaml --aggregate-only --overwrite
```

Final outputs:

```text
outputs/ablations/main_results.csv
outputs/ablations/main_results.md
outputs/figures/main_bar.png
```

## Script Entry Points

The shell scripts exist under `scripts/`, but this machine currently lacks `bash`. Use Python module commands in PowerShell, or run these scripts in a shell that has bash.

| Script | Python module | Purpose |
| --- | --- | --- |
| `scripts/01_prepare_data.sh` | `src.data.prepare_data` | Build small GSM8K train/eval JSONL files |
| `scripts/02_eval_base.sh` | `src.eval.eval_gsm8k` | Evaluate teacher or student base model on GSM8K |
| `scripts/03_collect_activations.sh` | `src.activations.collect_hidden` | Collect teacher/student math/control hidden vectors |
| `scripts/04_compute_wfs.sh` | `src.activations.compute_wfs` | Compute teacher/student WFS scores |
| `scripts/05_zeroout_check.sh` | `src.activations.intervention_zeroout` | Zero-out WFS top-k vs random top-k dimensions |
| `scripts/06_solve_ot.sh` | `src.ot.solve_ot`, `src.ot.linear_baseline` | Solve vanilla OT, ability-aware OT, and linear projector |
| `scripts/07_train_sft.sh` | `src.train.train_sft_lora` | Train SFT LoRA baseline |
| `scripts/08_train_ot_rep.sh` | `src.train.train_ot_rep_lora` | Train one representation-distillation LoRA method |
| `scripts/09_eval_all.sh` | `src.eval.eval_all` | Evaluate methods and build final tables/figure |

## Source File Roles

### `src/data`

| File | Role |
| --- | --- |
| `prepare_data.py` | Downloads/loads GSM8K, samples a toy train/eval split, writes JSONL |

### `src/models`

| File | Role |
| --- | --- |
| `load_model.py` | Loads local Qwen causal LM and tokenizer with `AutoModelForCausalLM` |

### `src/eval`

| File | Role |
| --- | --- |
| `eval_gsm8k.py` | GSM8K exact-match generation evaluation |
| `eval_all.py` | Unified evaluation for all methods on GSM8K, MATH-500, WikiText, then calls table generation |

### `src/activations`

| File | Role |
| --- | --- |
| `collect_hidden.py` | Saves selected-layer, last-non-padding-token hidden vectors |
| `compute_wfs.py` | Computes WFS statistics and top-k ability dimensions |
| `intervention_zeroout.py` | Runs WFS/random top-k zero-out intervention checks |
| `collect_activations.py` | Older placeholder; superseded by `collect_hidden.py` |

### `src/ot`

| File | Role |
| --- | --- |
| `solve_ot.py` | Builds correlation cost, ability-aware cost, solves POT Sinkhorn OT |
| `linear_baseline.py` | Fits ridge linear projector from teacher hidden dims to student hidden dims |

### `src/train`

| File | Role |
| --- | --- |
| `common_lora.py` | Shared dataset, collator, LoRA loading, logging helpers |
| `train_sft_lora.py` | SFT LoRA baseline training |
| `train_ot_rep_lora.py` | SFT + representation distillation LoRA training |
| `train_sft.py` | Older placeholder |
| `train_ot_rep.py` | Older placeholder |

### `src/analysis`

| File | Role |
| --- | --- |
| `make_tables.py` | Builds `main_results.csv`, `main_results.md`, and `main_bar.png` |
| `compute_wfs.py` | Older placeholder; superseded by `src.activations.compute_wfs` |
| `zeroout_check.py` | Older placeholder; superseded by `src.activations.intervention_zeroout` |

### `src/utils`

| File | Role |
| --- | --- |
| `config.py` | YAML config loading, CLI override handling, path resolution |
| `io.py` | JSONL, JSON, CSV, and path utility helpers |

## Output Directory Roles

| Directory | Contents |
| --- | --- |
| `data/gsm8k_toy` | Prepared toy GSM8K JSONL files |
| `outputs/activations` | Cached teacher/student hidden states |
| `outputs/ot` | WFS scores, OT matrices, cost matrices, projector, zero-out CSV |
| `outputs/checkpoints` | LoRA checkpoints and train logs |
| `outputs/eval` | Prediction JSONL and metrics JSON files |
| `outputs/ablations` | Final result CSV/Markdown tables |
| `outputs/figures` | Final plots |

## Resume And Overwrite Behavior

Most scripts skip existing outputs by default. To recompute a stage:

```powershell
--overwrite
```

Examples:

```powershell
python -m src.activations.compute_wfs --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student --overwrite
python -m src.eval.eval_all --config configs/toy_qwen_1p5b_to_0p5b.yaml --aggregate-only --overwrite
```

## Current Known Gaps

1. `sft_random_mask_rep` training is not implemented yet.
2. The bash wrappers cannot run in the current PowerShell environment unless bash is installed.
3. `collect_activations.py`, `src.analysis.compute_wfs.py`, `src.analysis.zeroout_check.py`, `train_sft.py`, and `train_ot_rep.py` are older placeholders kept for compatibility; use the newer files listed above.
4. Full evaluation can be slow on CPU. Prefer configuring sample counts first.

