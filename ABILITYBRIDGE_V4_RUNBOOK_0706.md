# AbilityBridge-v4 Feature Rotation 运行文档

目标：在两张 H200 上手动跑 AbilityBridge-v4 feature rotation 长实验。Codex 不在本地启动正式实验；本文件给同伴照着运行。

## 机器前置条件

在同伴机器上需要有：

- 当前 GitHub 仓库代码。
- 本地模型目录：`outputs/abilitybridge_v2_layer_boundary/server_A/shared/pruned_checkpoints/drop_layer_16/model`
- v2/v3 缓存目录，至少包含：
  - `outputs/abilitybridge_v2_ability_validation/shared/math500.jsonl`
  - `outputs/abilitybridge_v2_ability_validation/shared/wikitext.jsonl`
  - `outputs/abilitybridge_v2_layer4_protocol_v2/baseline/layer16/run_complete.json`
  - `outputs/abilitybridge_v3_sparse_feature_mvp/protocol/protocol_manifest.json`
  - `outputs/abilitybridge_v3_sparse_feature_mvp/sae/mlp/gated/checkpoint.pt`
  - `outputs/abilitybridge_v3_sparse_feature_mvp/sae/mlp/gated/feature_activation_cache.pt`
  - `outputs/abilitybridge_v3_sparse_feature_mvp/sae/mlp/gap_gated/checkpoint.pt`
  - `outputs/abilitybridge_v3_sparse_feature_mvp/sae/mlp/gap_gated/feature_activation_cache.pt`

这些缓存很大，不建议进 Git。用 `rsync` 或共享磁盘同步。

## 推荐目录

```bash
cd /opt/data/private/abilitybridge/workspace
git pull
```

如果同伴目录不是这个路径，设置：

```bash
export WORKSPACE_ROOT=/path/to/abilitybridge/workspace
cd "$WORKSPACE_ROOT"
```

## 环境检查

```bash
export PYTHON_BIN=/root/restored_envs/abilitybridge_new/bin/python
$PYTHON_BIN -m py_compile src/analysis/abilitybridge_v4_feature_rotation.py
$PYTHON_BIN -m src.analysis.abilitybridge_v4_feature_rotation list-grid \
  --config configs/abilitybridge_v4_rotation_grid.yaml \
  --root-dir outputs/abilitybridge_v4_feature_rotation_pruning
```

Stage 1 默认配置数：

- `site=mlp`
- `sae_kind=gated,gap_gated`
- `rotation=omcr,aagr,subspace,ot_displacement`
- `dim=4,8,16`
- `k=64,128`
- `ablation=zero,mean`
- 总计 `96` 个 feature causal 配置，双卡各 48 个。

## Smoke 测试

只跑 2 个 MATH 样本和 2 个 WikiText 样本，确认能读取 v3 cache、训练 rotation、做 SAE 写回干预。

```bash
ROOT=outputs/abilitybridge_v4_feature_rotation_pruning_smoke
$PYTHON_BIN -m src.analysis.abilitybridge_v4_feature_rotation run-feature-causal \
  --config configs/abilitybridge_v4_rotation_grid.yaml \
  --root-dir "$ROOT" \
  --smoke \
  --max-samples 2 \
  --wikitext-samples 2

$PYTHON_BIN -m src.analysis.abilitybridge_v4_feature_rotation summarize \
  --config configs/abilitybridge_v4_rotation_grid.yaml \
  --root-dir "$ROOT"
```

查看：

```bash
cat "$ROOT/status/queue_status.json"
find "$ROOT/causal/runs" -name run_complete.json -print
cat "$ROOT/summary/v4_feature_rotation_report.md"
```

## 正式长跑：两张 H200

这个命令会顺序执行：

1. `train-rotation`
2. `score-features`
3. `run-feature-causal`
4. `build-mlp-map`
5. `summarize`

默认不跑 pruning。feature causal 会加载模型并逐配置评测 MATH500 + WikiText，预计会跑很久，适合 10-20 小时级别任务。

```bash
tmux new -s abilitybridge_v4
cd /opt/data/private/abilitybridge/workspace
export PYTHON_BIN=/root/restored_envs/abilitybridge_new/bin/python

bash scripts/run_v4_feature_rotation_grid_2gpu.sh \
  --root-dir outputs/abilitybridge_v4_feature_rotation_pruning \
  --config configs/abilitybridge_v4_rotation_grid.yaml \
  --gpu0 0 \
  --gpu1 1 \
  --resume
```

## 两条线分开跑

如果想手动分配两张 H200 到两个 tmux：

GPU0：

```bash
tmux new -s abilitybridge_v4_gpu0
cd /opt/data/private/abilitybridge/workspace
export PYTHON_BIN=/root/restored_envs/abilitybridge_new/bin/python
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v4_feature_rotation_worker.sh \
  --worker gpu0 \
  --gpu 0 \
  --root-dir outputs/abilitybridge_v4_feature_rotation_pruning \
  --config configs/abilitybridge_v4_rotation_grid.yaml
```

GPU1：

```bash
tmux new -s abilitybridge_v4_gpu1
cd /opt/data/private/abilitybridge/workspace
export PYTHON_BIN=/root/restored_envs/abilitybridge_new/bin/python
CUDA_VISIBLE_DEVICES=1 bash scripts/run_v4_feature_rotation_worker.sh \
  --worker gpu1 \
  --gpu 1 \
  --root-dir outputs/abilitybridge_v4_feature_rotation_pruning \
  --config configs/abilitybridge_v4_rotation_grid.yaml
```

两边都完成后：

```bash
$PYTHON_BIN -m src.analysis.abilitybridge_v4_feature_rotation summarize \
  --config configs/abilitybridge_v4_rotation_grid.yaml \
  --root-dir outputs/abilitybridge_v4_feature_rotation_pruning
```

## 可选 pruning 长跑

只有 feature causal 有正信号后再跑。命令会对 map 产生的 group 做 no-recovery zero-out pruning evaluation，不做 SFT/GKD。

```bash
RUN_PRUNING=1 bash scripts/run_v4_feature_rotation_grid_2gpu.sh \
  --root-dir outputs/abilitybridge_v4_feature_rotation_pruning \
  --config configs/abilitybridge_v4_rotation_grid.yaml \
  --gpu0 0 \
  --gpu1 1 \
  --resume
```

## Stage 2 扩展

Stage 1 有正信号后再加：

```bash
bash scripts/run_v4_feature_rotation_grid_2gpu.sh \
  --root-dir outputs/abilitybridge_v4_feature_rotation_stage2 \
  --config configs/abilitybridge_v4_rotation_grid.yaml \
  --gpu0 0 \
  --gpu1 1 \
  --include-stage2 \
  --resume
```

## 查看状态

```bash
ROOT=outputs/abilitybridge_v4_feature_rotation_pruning
cat "$ROOT/status/queue_status.json"
tail -n 100 -f "$ROOT/logs/orchestrator.log"
tail -n 100 -f "$ROOT/logs/gpu0.log"
tail -n 100 -f "$ROOT/logs/gpu1.log"
find "$ROOT/causal/runs" -name run_complete.json | wc -l
find "$ROOT/maps" -name '*_groups.csv' | wc -l
nvidia-smi
```

## 停止与续跑

停止：

```bash
pkill -f abilitybridge_v4_feature_rotation
```

续跑同一个 root 即可。已有 `run_complete.json`、rotation artifact、feature manifest 默认会跳过。

```bash
bash scripts/run_v4_feature_rotation_grid_2gpu.sh \
  --root-dir outputs/abilitybridge_v4_feature_rotation_pruning \
  --config configs/abilitybridge_v4_rotation_grid.yaml \
  --gpu0 0 --gpu1 1 --resume
```

## 结果文件

- `outputs/abilitybridge_v4_feature_rotation_pruning/configs/grid_all.csv`
- `outputs/abilitybridge_v4_feature_rotation_pruning/rotations/.../*.pt`
- `outputs/abilitybridge_v4_feature_rotation_pruning/features/.../*.pt`
- `outputs/abilitybridge_v4_feature_rotation_pruning/causal/runs/*/run_complete.json`
- `outputs/abilitybridge_v4_feature_rotation_pruning/maps/*_groups.csv`
- `outputs/abilitybridge_v4_feature_rotation_pruning/summary/feature_causal_results.csv`
- `outputs/abilitybridge_v4_feature_rotation_pruning/summary/pruning_results.csv`
- `outputs/abilitybridge_v4_feature_rotation_pruning/summary/v4_feature_rotation_report.md`

## 判断标准

先看 feature gate 信号，不要直接跳到 pruning：

- rotation candidate 的 MATH drop 是否稳定高于 random/control。
- WikiText PPL damage 是否不明显更坏。
- MLP map 是否出现稳定 Pareto 区域。
- 若 Stage 1 没有正信号，停止 pruning，回到 rotation objective 和 data construction。
