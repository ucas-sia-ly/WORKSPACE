# CCR-VPR Stage 1

`--synthetic-aux` 默认关闭。关闭时保持已有 real / mix / replace 训练行为。
开启后主批次固定为 P 个地点、每地点 K 张真实图，原始 MS loss 不变；
`--synthetic-fraction` 不再控制主批次，生效值记录为 0。
Stage 1 暂不与 `--synthetic-mode replace` 或 `--reliability-ot` 组合，入口会明确拒绝。

## 精确源图辅助采样

`SyntheticAuxSampler.sample(main_sampling_metadata)` 只查看本主批次地点及其当前 K 张真实图。
若某地点在这 K 张中有 manifest 验收通过的生成版本，以 `p_exp` 的概率选中该地点，
均匀选一个可用源图，再均匀选 condition/domain 及其中的版本，每地点最多一张。
若这些真实图没有生成版本，直接跳过；不另外加载同地点源图，也不伪造配对。

返回字段：

| 字段 | 语义 |
|---|---|
| `synthetic_images` | `[M,3,H,W]`，0 ≤ M ≤ P |
| `synthetic_labels` | `[M]`，与主批次真实源图完全相同的地点标签 |
| `source_indices` | `[M]`，展平主批次 `[P*K,...]` 中的精确源图下标 |
| `synthetic_paths` / `source_paths` | 对应的 manifest 生成图 / 精确源图路径 |
| `condition` | manifest 的 condition；缺失时记作 `unknown` |
| `pair_valid` | `[M]` bool；当前实现只返回精确匹配，全部为 true |

采样和生成图的随机图像增强使用独立 `random.Random`，不会消耗主采样/增强的
Python、NumPy 或 Torch RNG。该 RNG 写入 checkpoint 的 `rng_state.synthetic_aux`。
辅助图用自己的增强，不要求与真实图共享像素级变换；本阶段只有全局描述子损失。

没有 manifest 或没有可用生成图时，仍可正常训练。`p_exp=0` 不解码生成图、
不执行生成前向，回退为相同数据、地点范围、配置和种子下的真实基线。

## 损失及训练顺序

设 q 是生成 query，r 是当前真实主批次，按地点标签确定 positives：

```text
L_real = multi_similarity_loss(real_descriptors, real_labels)
s(q, r) = cosine(q, stop_gradient(r)) / temperature
L_syn = mean_q [logsumexp_r s(q,r) - logsumexp_{r: label(r)=label(q)} s(q,r)]
L = L_real + synthetic_global_weight * L_syn
```

分子为同地点全部真实图，包括精确源图；分母为全部当前真实图。
无效配对或没有正样本的 query 不参与均值。空辅助批次视为辅助损失 0。
归一化、相似度和 logsumexp 都在禁用 autocast 的 FP32 块内计算。

训练顺序固定为：zero_grad → 真实前向及 backward → 保存 detached reference、
释放真实计算图 → 生成微批次前向及加权 backward → 一次 unscale、梯度裁剪、
optimizer step、scaler update。每步只有一个辅助微批次，最多 P 张图。
损失非有限时终止该步，不更新参数；FP16 梯度溢出由 GradScaler 跳过更新并降低 scale，
FP32/BF16 的非有限梯度由裁剪检查报错。

**detach 只阻断辅助损失经过真实 reference 的直接梯度。生成 query 分支仍会更新共享学生主干和聚合器。**

新增参数默认值：

| 参数 | 默认值 |
|---|---|
| `--synthetic-aux` | 关闭 |
| `--synthetic-exposure-prob` | 0.5 |
| `--synthetic-global-weight` | 0.1 |
| `--synthetic-global-temperature` | 0.1 |

## 记录和恢复

`training_log.jsonl` 中的 `synthetic_aux` 记录：

- `main_places` / `places_with_synthetic` / `eligible_places`：主批次地点、全池有生成图地点、当前 K 张存在精确配对的地点访问数。
- `actual_trigger_rate`：实际生成图数 / 当前可配对地点数；`trigger_rate_all_places` 的分母为全部主批次地点数。
- `available_source_fraction`：当前真实图中具有合格生成版本的比例。
- `condition_exposure`：按天气 condition 统计的实际生成曝光数。
- `synthetic_fraction_observed`：辅助生成图数 /（真实主图数 + 辅助生成图数），不是 `p_exp`。
- `loss_real` / `loss_syn_per_step`：按所有训练步平均的真实、辅助损失；后者包含空辅助步的零值。
- `loss_syn_nonempty` / `nonempty_batches`：非空辅助步的平均损失及数量。

顶层 `real_exposure` / `synthetic_exposure` 仍描述主批次，C 模式的顶层
`synthetic_exposure` 为 0；辅助曝光见 `synthetic_aux.triggered_places`。

resume 校验数据与原训练参数，以及辅助开关、p_exp、权重和温度，恢复独立辅助 RNG。
不允许用 `--resume` 切换 A/B/C；切换实验应使用 `--init-checkpoint` 开始新优化器和输出目录。
恢复时 `--epochs` 必须为原定总轮数。

## A / B / C 短程 GPU 对照命令

从工作区根目录执行。以下三组使用相同预训练权重、相同两个 Bangkok 地点和完整真实图池，
均限制为 2 轮 × 每轮 1 batch，**不会启动全城市训练**。
这里 A 也读取 manifest，只用于将地点范围统一到同样的两个地点，不会读取生成图参与训练。

```bash
CCR_PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
CCR_OUT=$(mktemp -d "$PWD/salad/outputs/ccr_stage1_compare_XXXXXX")
CCR_COMMON=(
  --real-data dataset/gsv-cities --cities Bangkok
  --synthetic-manifest salad/outputs/reliability_ot_smoke/manifest.jsonl
  --synthetic-places-only
  --init-checkpoint salad/checkpoint/dino_salad.ckpt
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main
  --epochs 2 --max-batches-per-epoch 1
  --batch-size 2 --images-per-place 4 --min-images-per-place 4
  --image-size 126 126 --num-trainable-blocks 1
  --real-sampling coverage_pk --seed 13
  --learning-rate 1e-6 --weight-decay 0
  --num-workers 0 --no-augment --device cuda --precision 32
)

# A：纯真实 MS loss
"$CCR_PY" salad/train_salad.py "${CCR_COMMON[@]}" \
  --output-dir "$CCR_OUT/A_real" --synthetic-fraction 0

# B：已有 mix（主批次含生成图，只有原 MS loss）
"$CCR_PY" salad/train_salad.py "${CCR_COMMON[@]}" \
  --output-dir "$CCR_OUT/B_mix" --synthetic-mode mix --synthetic-fraction 0.5

# C：纯真实主 MS loss + 非对称生成辅助 loss
"$CCR_PY" salad/train_salad.py "${CCR_COMMON[@]}" \
  --output-dir "$CCR_OUT/C_aux" --synthetic-aux \
  --synthetic-exposure-prob 1 \
  --synthetic-global-weight 0.1 --synthetic-global-temperature 0.1
```

Smoke 使用 p_exp=1 确保有精确配对时必触发，仍不会保证每个地点每轮都有可用配对。
检查 AMP 时，将 `--precision 32` 分别改为 `16` 和 `bf16`，使用新的输出目录。
正式实验可采用 p_exp=0.5，但本次不启动正式实验。

针对性 CPU 回归命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=salad:salad/tests /home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  -m unittest test_synthetic_aux test_coverage_sampling test_training_data \
  test_source_replacement test_reliability_workflow -v
```

## 本次验证

真实 DINOv2 + SALAD GPU smoke 已完成，输入 126×126，训练最后一个主干 block：

- FP16、BF16、FP32 各完成 2 个训练步，每步确认为两次 backward、一次 optimizer.step，梯度有限。
- 每轮真实曝光 8 张；辅助曝光分别为 2 张和 1 张。天气曝光为第一轮 night/rain 各 1，第二轮 night 1。
- PyTorch 峰值 allocated 显存约 548–550 MiB（不包含其他进程及 CUDA context）。
- FP32 从第 1 轮 checkpoint 恢复后，最终参数与连续训练逐位一致，辅助 RNG 和覆盖统计也一致。
- 这些结果只验证训练与恢复链路，不代表检索精度提升。

本次日志及完整统计位于 `salad/outputs/ccr_stage1_smoke_0ec3z27d/summary.json`。
