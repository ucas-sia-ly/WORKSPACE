# CCR-VPR Stage 3：局部蒸馏与 Reliability

实现与数值回归通过；**本次短程效果验收未通过**。两步更新后 Reliability 仍接近
常数，留出受控遮挡区域没有得到更低可靠性，真实检索没有改善。这里不把程序能跑、
辅助 Loss 有梯度或下降当作检索有效性的证据，也不能据此断言更长训练一定失败。

## 训练与证据

复用现有 Reliability V2 head/context、`predict_reliability`、dustbin 偏置和原
Sinkhorn。没有更改聚合器参数、OT 迭代或旧 checkpoint 格式；原 V2 的真实图
Copy-Paste 及区域外 detached-clean consistency 均保留。

Stage 3 在 Stage 2C 固定 Resize 的精确真实/生成配对分支中添加：

| 状态 | 证据 | 直接可靠性监督 |
|---|---|---|
| Positive | Stage 2B 的 A 类对应与 `positive_conf` | soft target=0.95 的 BCE |
| Negative | manifest 明确提供的遮挡/破坏 mask | soft target=0.05 的 BCE，权重为该 Patch 的 mask 面积比例 |
| Unknown | 没有上述证据，包括未核验 B 候选 | 零直接监督 |

目标端点沿用 V2；不是把 q 当作标签或令负置信度等于 `1-q`。
正负类别分别按证据质量归一化，再对实际存在的类别平均；分母覆盖整个辅助批次，
不会把小覆盖图像和大覆盖图像等权。置信度 detach，BCE/归约使用 FP32，检查有限值。
同一 Patch 的正负证据冲突会报错。普通 edit mask 仅阻止 Positive，不能制造 Negative。

低 DINO 相似度、LightGlue 缺失/失配、暗区、天气名称、教师 Dustbin 高都不是
负标签来源。无 Negative 时保持零负标签，统计中的负类均值为 `null`，不伪报为 0。
没有生成辅助图时仍正常训练，并保留独立真实图受控负监督。
Unknown 的“零监督”仅指直接三态 BCE；共享 head 仍可通过全局检索和 OT 路径更新。

新增参数默认均为 0：

- `--reliability-generated-weight`：以上三态生成图监督的权重。
- `--reliability-generated-occlusion-weight`：额外生成图受控遮挡副本的负监督权重。

两者需要 `--reliability-ot --synthetic-aux --teacher-checkpoint --pair-calibration`。
生成图正标签始终使用真实证据强度，与 Feature/OT 的 uniform/verified 消融独立。
固定 head 模式不允许非零辅助监督权重。现有 V2 主批次入口仍可独立运行。

C3 的受控遮挡是显式代理任务：在**生成图副本**上用归一化均值颜色覆盖一个
2×2 Patch 矩形，mask 只标记像素实际发生变化的位置；不更改原图、不调用 Qwen。
它不代表原始 Qwen 图发生了幻觉，也不会自动把遮挡以外 Unknown 变成 Positive。
该生成副本只计算 mask 内的负 BCE；真实图 V2 Copy-Paste 的外部一致性机制保持不变。
各自记录 `generated_reliability`、`controlled_generated_occlusion` 和
`reliability.corruption_negative_patches`，不能合并成“生成错误检出率”。

主图与全局生成 query 的 augmentation policy 保留；局部配对仍使用相同的全帧
Resize。生成受控遮挡发生在该固定输入上，mask 与 Patch 精确对应，且不用于修改
原始配对的 Feature/Score/OT 目标。遮挡随机数按 seed/global_step/pair_index 独立生成。
多个 backward 后只做一次 unscale、clip、optimizer.step、scaler.update。

`--reliability-detach-features` 继续生效：阻断 predictor 输入到 Backbone 的直接
head 梯度，不阻断主检索、Feature、Score 等其他路径。真实图受控 Copy-Paste
沿用 V2 的 detached Backbone 特征。生成三态/遮挡分支可通过
`--no-reliability-detach-features` 做对照。单元测试使用非零末层权重检查两条梯度
路径，避免新 head 零初始化导致“有路径但梯度恰为零”的假通过。
本次两步 detach 对照没有可辨别的检索收益差异，尚不能决定最终是否解除隔离。

## Dustbin 与诊断含义

保留现有计算：`dustbin_score[k] = base + lambda * (1 - reliability[k])`。
总边际目标仍为有效簇各 1、Dustbin 为 N−M，每个 Patch 的总分配质量为 1。
空间均匀的 head 偏置会在行归一化中抵消；调高 lambda 不能解决常数输出。
即使 r=0，其有效簇概率仍可大于零，不等于硬删除 Patch。

原模型只做 3 次 Sinkhorn 迭代，尖锐 logits 下行边际仍有误差。留出四张生成图
的目标 Dustbin 总质量为 4×(256−64)=768，C3 实际为 807.528；这不代表修改了
拒绝率目标。单元测试同时检查收敛后的质量约束、均匀偏置抵消及非均匀重分配。
本阶段没有为了消除诊断误差更改原 Sinkhorn。

新增统计包括 Positive/Negative 可靠性均值、各类数量、Unknown 比例、图像内
可靠性标准差、各类 Dustbin 均值和差值。均值按 Patch 数累计，缺失类别保留 null。
训练统计在固定局部分支记录，GPU smoke 还在独立地点上做更新后的诊断。
**两类 Dustbin 均值之差也受原 Score 影响，不能单独作为 Reliability 的因果贡献。**

## 本次验证

222 项 CPU 测试通过，包括旧 V2、Stage 1/2A/2B/2C、新三态标签、独立 RNG、
正确 mask、Unknown 零梯度、detach、FP16/BF16/FP32 BCE、一次更新和精确恢复。
另将 Git HEAD 中原 V2 入口与新入口直接比较：新增权重关闭时，33 个模型 state
key、配置、Python/Torch RNG 完全一致。原 V2 输出目录 25 个文件的 SHA256
前后逐一一致；没有修改旧 checkpoint 或实验记录。

GPU smoke 共 6 组×2 次更新，FP16，ViT-B/14，224×224，P=2/K=4，末尾一个
Backbone block 可训练。教师来自 Stage 2C 已训练 checkpoint，保持冻结 FP32。
学生新 head 使用 context、lr=1e-4，其余 lr=1e-6。每组初始化与更新预算一致。

为了保留历史对照，又能隔离局部蒸馏，输出分为：

| 请求中的组合 | 内容 | 实际输出子目录 |
|---|---|---|
| C0 | B5（Global+Feature+OT+q），无 Reliability | `C0` |
| C1 | 原 V2：mix 主批次、冻结标签缓存、真实 Copy-Paste | `C1_legacy` |
| 额外 C1 matched | 改用 Stage 1 全真实主批次+全局 query，V2 head/Copy-Paste+可信生成 BCE，无局部蒸馏 | `C1` |
| C2 | C0+V2 head/Copy-Paste+可信生成 BCE | `C2` |
| C3 | C2+显式生成遮挡副本负监督 | `C3` |
| detach 对照 | C3 的 predictor 不 detach | `C3_no_detach` |

C1 legacy 保留旧 mix/no-augment 协议，与 C2 的差别不只有局部 Loss；不能把它们
的差值全部归因于蒸馏。额外的 C1 matched 与 C2 才是同采样/增强协议的局部损失对照。
原始输出名字已记录在 `ablation_mapping.json`，不改写已运行日志。
正式 C1 使用的原 V2 缓存还保留其原始教师（`dino_salad.ckpt`），不同于 C2/C3
的 Stage 2A 教师；不能将历史缓存对照解释为单变量实验。Smoke 的 C1 legacy
小缓存则在新目录用相同初始 checkpoint 重建。

四个训练地点共曝光 16 张主真实图、4 张辅助生成图（C1 legacy 使用原 mix）。
C2/C3 原生成图提供 143 个 A 正 Patch、**0 个负 Patch**。所有 head 组保留
65 个真实 Copy-Paste 负 Patch；C3 另提供 16 个生成受控遮挡负 Patch。

留出诊断使用另外四个地点、四种天气；原图共有 132 个 Positive、0 个 Negative、
892 个 Unknown，Unknown=87.109%。以下为更新后结果：

| 组合 | 原生成 Positive 平均 r | 原生成空间 std | 受控生成遮挡区 r | 受控副本 Positive r | 遮挡 Negative−Positive Dustbin |
|---|---:|---:|---:|---:|---:|
| C1 legacy | .899296 | .0000884 | .899351 | .899292 | −.254931 |
| C1 matched | .899276 | .0000942 | .899333 | .899272 | −.255007 |
| C2 | .899271 | .0000944 | .899329 | .899268 | −.255161 |
| C3 | .899254 | .0000937 | .899310 | .899251 | −.255159 |
| C3 不 detach | .899254 | .0000937 | .899310 | .899251 | −.255162 |

初始 head 恒为 0.9；现在仍几乎只发生整体偏移。遮挡区 r 反而略高于可靠正区域，
不满足希望得到的排序。C3 的真实 Copy-Paste 留出负区 r=.899159、未改动区
r=.899242，差距也很小。**本次空间判别效果验收失败**，不能宣称解决了常数退化。

真实检索回归：四城市各 16 个本次微调未使用的地点，64 reference、64 query，
同地点其他真实视图作为正样本。初始模型以及所有组合均 R@1=62/64=96.875%，
R@5/R@10=100%；没有观测到下降，也没有观测到提升。
这些地点不保证未被历史初始化模型训练过；该小集合只是本次微调回归，不能替代
独立标准验证集，且 R@1 的单查询分辨率为 1.5625 pp。未用它选择阈值或调参。

| 组合 | L_feat | L_OT | 生成三态 BCE | 生成遮挡 BCE | 峰值 allocated MiB | ms/step |
|---|---:|---:|---:|---:|---:|---:|
| C0 | .066104 | 1.480835 | — | — | 1120.73 | 197.61 |
| C1 legacy | — | — | — | — | 573.01 | 128.99 |
| C1 matched | .066102* | 1.480968* | .215326 | — | 912.30 | 836.36 |
| C2 | .066104 | 1.480757 | .215326 | — | 912.30 | 132.60 |
| C3 | .066104 | 1.480762 | .215326 | 2.191306 | 912.30 | 146.59 |
| C3 不 detach | .066104 | 1.480762 | .215326 | 2.191306 | 911.64 | 151.69 |

`*` 仅诊断，C1 matched 的局部权重为 0。所有 Loss 为未乘权重均值。
显存为当前进程 allocator allocated，时间从主图传输后到 scaler.update，含在线
教师与局部准备，不含检索评估与写盘。无独立预热、每组只有两步，不能进行速度排名。
测量设备 RTX 4090，驱动 570.195.03。

完整记录：`salad/outputs/ccr_stage3_smoke/summary.json`、`effectiveness_verdict.json`、
`real_holdout.json`、`spatial_holdout.jsonl`、`v2_exact_regression.json`、
`v2_artifact_integrity.json` 和每个组合的配置/结果/日志。Smoke 只保存 JSON，
拦截了 checkpoint 序列化，没有生成可恢复训练的学生权重。

## 单元测试和 Smoke 命令

```bash
cd /home/admin123/github/WORKSPACE
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
TEACHER="$PWD/outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1/generated_8to1/checkpoint.pt"
DINO=/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main
DIAG="$PWD/salad/outputs/ccr_stage2b_diagnostics"

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=salad:salad/tests "$PY" -m unittest discover -s salad/tests -q

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" salad/tests/smoke_stage3_reliability.py \
  --teacher-checkpoint "$TEACHER" --backbone-repo "$DINO" --diagnostic-dir "$DIAG" \
  --training-manifest "$PWD/salad/outputs/ccr_stage2c_smoke/manifest.jsonl" \
  --real-data "$PWD/dataset/gsv-cities" \
  --output-dir "$PWD/salad/outputs/ccr_stage3_smoke_repeat"
```

Smoke 输出目录必须尚未存在。它只执行 12 次 optimizer 更新，并在新目录构建
C1 legacy 的小型 target/confidence 缓存，不创建教师特征缓存或覆盖 V2 文件。

## 正式消融命令（本次未启动）

以下固定已有 V2 cohort：四城市、700 行原 manifest 中通过审核的 455 张生成图。
每组同一初值、seed、总 epoch；正式训练写 checkpoint。C1 legacy 的协议差异已在
上表说明，额外运行 C1 matched 才能隔离局部蒸馏贡献。权重是 smoke 初值，未验证最优。
先执行上面的路径变量赋值，然后定义：

```bash
MANIFEST="$PWD/salad/outputs/reliability_v2/fixed_700.jsonl"
COMMON=(
  --real-data "$PWD/dataset/gsv-cities" --synthetic-manifest "$MANIFEST"
  --cities Bangkok BuenosAires LosAngeles Medellin --synthetic-places-only
  --init-checkpoint "$TEACHER" --backbone-repo "$DINO"
  --image-size 224 224 --num-trainable-blocks 1
  --epochs 3 --batch-size 8 --images-per-place 4 --min-images-per-place 4
  --real-sampling coverage_pk --seed 42 --num-workers 0
  --learning-rate 1e-6 --weight-decay 0 --precision 16 --device cuda --profile-training
)
AUX=(--synthetic-aux --synthetic-exposure-prob .5 --synthetic-global-weight .1)
PAIR=(--teacher-checkpoint "$TEACHER" --teacher-batch-size 1
  --pair-calibration "$DIAG/calibration.json" --pair-geometry-cache "$DIAG/geometry"
  --pair-confidence-mode verified)
DISTILL=(--feature-distill-weight .1 --ot-distill-weight .1 --score-distill-weight 0)
HEAD=(--reliability-ot --reliability-context --reliability-detach-features
  --reliability-head-lr 1e-4 --reliability-corruption-weight .05
  --reliability-corruption-max-images 4)
run_stage3() {
  local name="$1"
  shift
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" salad/train_salad.py "${COMMON[@]}" \
    --output-dir "$PWD/salad/outputs/ccr_stage3_formal/$name" "$@"
}
```

按需要选择单条运行，不会自动串行调度：

```bash
# C0：已验证执行路径的局部蒸馏基线，无 Reliability
run_stage3 C0 "${AUX[@]}" "${PAIR[@]}" "${DISTILL[@]}"

# C1：原 Reliability V2，沿用已有只读 target/confidence 缓存
run_stage3 C1 "${HEAD[@]}" --synthetic-mode mix --synthetic-fraction .5 --no-augment \
  --reliability-target-cache "$PWD/salad/outputs/reliability_v2/frozen_targets"

# C2：局部蒸馏 + V2 head/真实 Copy-Paste + 三态生成证据
run_stage3 C2 "${AUX[@]}" "${PAIR[@]}" "${DISTILL[@]}" "${HEAD[@]}" \
  --reliability-generated-weight .1

# C3：另加有精确 mask 的生成受控遮挡副本
run_stage3 C3 "${AUX[@]}" "${PAIR[@]}" "${DISTILL[@]}" "${HEAD[@]}" \
  --reliability-generated-weight .1 --reliability-generated-occlusion-weight .05

# 与 C2 相同采样/增强/三态规则，局部蒸馏关闭
run_stage3 C1_matched "${AUX[@]}" "${PAIR[@]}" "${HEAD[@]}" \
  --reliability-generated-weight .1

# 与 C3 相同，仅解除 predictor 输入 detach
run_stage3 C3_no_detach "${AUX[@]}" "${PAIR[@]}" "${DISTILL[@]}" "${HEAD[@]}" \
  --reliability-generated-weight .1 --reliability-generated-occlusion-weight .05 \
  --no-reliability-detach-features
```

需要先检查正式配置时加 `--check-data`；需要短跑则换新输出目录，附加
`--epochs 1 --max-batches-per-epoch 2`。恢复时以相同配置把 init-checkpoint
换成 resume，保持原 cosine schedule 总 epoch；新权重与 detach 设置也受恢复校验。

正式真实图回归命令（对 C0/C1/C2/C3 分别执行），固定现有的 64 地点协议：

```bash
"$PY" salad/evaluate_salad.py \
  --checkpoint "$PWD/salad/outputs/ccr_stage3_formal/C3/checkpoint.pt" \
  --dataset manifest --dataset-root "$PWD/dataset/gsv-cities" \
  --eval-manifest "$PWD/salad/outputs/ccr_stage3_smoke/real_holdout.json" \
  --backbone-repo "$DINO" --batch-size 8 --num-workers 0 --device cuda \
  --output "$PWD/salad/outputs/ccr_stage3_formal/C3/real_regression.json"
```

正式 cohort 更大，必须另外核对上述回归地点是否与正式训练重叠；它是固定行为回归，
不是正式泛化验证集。检索收益仍需使用事先固定、与训练隔离的标准评估协议，不能
根据辅助 Loss 或此小集合选模型后宣称提高 Recall。本次没有启动正式训练或大规模评估。

已有 SVOX 验证集的独立真实图评估命令如下；所有组及初始化模型使用同样的
224×224、25m 正样本半径和完整 gallery，结果写各自新目录：

```bash
"$PY" salad/evaluate_salad.py \
  --checkpoint "$PWD/salad/outputs/ccr_stage3_formal/C3/checkpoint.pt" \
  --dataset SVOX --dataset-root "$PWD/dataset/svox" --split val \
  --positive-radius 25 --backbone-repo "$DINO" --image-size 224 224 \
  --batch-size 8 --num-workers 0 --device cuda \
  --output "$PWD/salad/outputs/ccr_stage3_formal/C3/svox_val.json"
```
