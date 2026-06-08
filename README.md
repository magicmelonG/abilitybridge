# AbilityBridge MVP

Minimal experiment scaffold for testing ability-aware OT representation distillation on a small GSM8K setup.

This first pass implements the runnable foundation:

- project layout under `src/data`, `src/models`, `src/eval`, `src/activations`, `src/ot`, `src/train`, `src/analysis`, `src/utils`, `scripts`, `configs`
- local Qwen model loading from `models/teacher/qwen2p5_math_1p5b_instruct` and `models/student/qwen2p5_0p5b_instruct`
- GSM8K small-sample preparation to JSONL
- base GSM8K evaluation with JSONL predictions and JSON metrics
- resumable outputs by default, with `--overwrite` to regenerate

## Environment

Use the environment already created:

```powershell
conda activate abilitybridge
```

Install or refresh dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

## Config

The toy config is:

```text
configs/toy_qwen_1p5b_to_0p5b.yaml
```

It intentionally targets the 1.5B teacher and 0.5B student only. No 7B-heavy path is included.

## Run

Prepare data:

```powershell
bash scripts/01_prepare_data.sh --config configs/toy_qwen_1p5b_to_0p5b.yaml
```

Smoke-test local model loading:

```powershell
python -m src.models.load_model --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student
```

Evaluate the base student:

```powershell
bash scripts/02_eval_base.sh --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student
```

Evaluate the base teacher:

```powershell
bash scripts/02_eval_base.sh --config configs/toy_qwen_1p5b_to_0p5b.yaml --role teacher
```

Override config values from CLI:

```powershell
python -m src.eval.eval_gsm8k --role student --set eval.max_new_tokens=128 --set data.eval_jsonl=data/gsm8k_toy/eval.jsonl
```

Collect student math/control hidden states:

```powershell
python -m src.activations.collect_hidden --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student --kind math
python -m src.activations.collect_hidden --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student --kind control
```

Compute WFS top-k dimensions:

```powershell
python -m src.activations.compute_wfs --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student
```

Run zero-out intervention with the WFS top-k and random top-k baseline:

```powershell
python -m src.activations.intervention_zeroout --config configs/toy_qwen_1p5b_to_0p5b.yaml --role student
```

Solve vanilla OT, ability-aware OT, and the ridge linear projector baseline:

```powershell
python -m src.ot.solve_ot --config configs/toy_qwen_1p5b_to_0p5b.yaml `
  --teacher-activation outputs/activations/teacher_math_hidden.pt `
  --student-activation outputs/activations/student_math_hidden.pt `
  --teacher-wfs outputs/ot/teacher_wfs_stats.npz `
  --student-wfs outputs/ot/student_wfs_stats.npz `
  --teacher-layer 8 --student-layer 4

python -m src.ot.linear_baseline --config configs/toy_qwen_1p5b_to_0p5b.yaml `
  --teacher-activation outputs/activations/teacher_math_hidden.pt `
  --student-activation outputs/activations/student_math_hidden.pt `
  --teacher-wfs outputs/ot/teacher_wfs_stats.npz `
  --student-wfs outputs/ot/student_wfs_stats.npz `
  --teacher-layer 8 --student-layer 4
```

Train LoRA baselines:

```powershell
python -m src.train.train_sft_lora --config configs/toy_qwen_1p5b_to_0p5b.yaml

python -m src.train.train_ot_rep_lora --config configs/toy_qwen_1p5b_to_0p5b.yaml `
  --method ability_ot `
  --teacher-hidden outputs/activations/teacher_math_hidden.pt `
  --alignment-dir outputs/ot/alignment
```

## Outputs

Default output paths:

- `data/gsm8k_toy/train.jsonl`
- `data/gsm8k_toy/eval.jsonl`
- `outputs/activations/student_math_hidden.pt`
- `outputs/activations/student_control_hidden.pt`
- `outputs/ot/student_wfs_stats.npz`
- `outputs/ot/student_wfs_topk.jsonl`
- `outputs/ot/student_wfs_topk.pt`
- `outputs/ot/student_zeroout_intervention.csv`
- `outputs/ot/alignment/vanilla/ot_matrix.pt`
- `outputs/ot/alignment/vanilla/cost_matrix.pt`
- `outputs/ot/alignment/vanilla/metadata.json`
- `outputs/ot/alignment/ability_aware/ot_matrix.pt`
- `outputs/ot/alignment/ability_aware/cost_matrix.pt`
- `outputs/ot/alignment/ability_aware/metadata.json`
- `outputs/ot/linear_projector/projector.pt`
- `outputs/ot/linear_projector/metadata.json`
- `outputs/checkpoints/sft_lora/checkpoint`
- `outputs/checkpoints/sft_lora/train_log.csv`
- `outputs/checkpoints/sft_lora/config_used.yaml`
- `outputs/checkpoints/ability_ot_rep_lora/checkpoint`
- `outputs/checkpoints/ability_ot_rep_lora/train_log.csv`
- `outputs/checkpoints/ability_ot_rep_lora/config_used.yaml`
- `outputs/eval/student_base_predictions.jsonl`
- `outputs/eval/student_base_metrics.json`
- `outputs/eval/teacher_base_predictions.jsonl`
- `outputs/eval/teacher_base_metrics.json`

Existing outputs are skipped unless `--overwrite` is passed.

## Current Status

The activation/WFS/intervention part is implemented:

- `scripts/03_collect_activations.sh`
- `scripts/04_compute_wfs.sh`
- `scripts/05_zeroout_check.sh`

The OT alignment part is implemented:

- `scripts/06_solve_ot.sh`

The LoRA training part is implemented:

- `scripts/07_train_sft.sh`
- `scripts/08_train_ot_rep.sh`

The unified evaluation and result table part is implemented:

- `scripts/09_eval_all.sh`

All scripts parse `--config`, support `--overwrite`, and print saved paths/key statistics.
