# CCR-VPR Stage 2C：局部 Feature / Score / Conditional OT 蒸馏

三个局部权重默认均为零。真实主批次仍为 P 个地点、每地点 K 张真实图，
MS Loss 和主图 augmentation policy 不变；Stage 1 非对称全局生成损失保留。
新增实现位于 `workflow/synthetic_losses.py` 和 `workflow/local_distillation.py`。
本阶段没有启用 Reliability Head，也没有重新生成图片。

## 损失与梯度

`match_index[k]` 表示生成图 Patch k 对应的真实源图 Patch j。
三个接口均支持动态网格、真实与生成侧不同 Patch 数，并沿源图索引 gather：

| 接口 | 每个有效 Patch 的损失 |
|---|---|
| `feature_distillation_loss` | `1 - cosine(teacher_backbone[j], student_backbone[k])` |
| `score_distillation_loss` | `KL(softmax(teacher_score[j]) || softmax(student_score[k]))` |
| `conditional_ot_distillation_loss` | 完整 log_assignment 去掉末尾 dustbin 后，沿有效簇 log_softmax，再计算 `KL(teacher || student)` |

每项按 `sum(q * patch_loss) / (sum(q) + 1e-8)` 归一化。
局部学生以一对一个微批次处理，但分母来自**当前整个生成辅助批次**，
不把不同有效 Patch 数的图像等权平均。总损失为：

```text
L = L_real + lambda_g * L_syn
           + lambda_f * L_feat + lambda_s * L_score + lambda_o * L_route
```

教师输出和 q 均 detach。相似度、log_softmax、KL、加权归约在禁用 autocast 的
FP32 中计算；KL 使用 log_target，避免先 exp 再 log。输入、单项损失与总损失
均检查有限值。q 全零时返回可反向传播的精确零；训练中没有有效目标的微批次可跳过。

Conditional OT 不直接对 dustbin 做 KL。教师有效簇质量相对总质量极小时，
只排除该 Patch 的 OT 项，不排除其 Feature/Score 项；学生自身 Keep 低不能关闭监督。
`--ot-min-teacher-keep` 默认是 FP32 epsilon（约 1.19e-7），仅作数值保护，
不是经过验证的语义可信度阈值。以 logsumexp 计算质量比并记录排除数量。
不直接监督 dustbin 并不代表 Sinkhorn 的共享参数或耦合路径永远不会变化。
同样，detach 仅隔离真实 reference 的直接辅助梯度，生成分支仍更新共享学生参数。

## 几何、可信度与在线教师

局部输入来自 Stage 1 辅助采样器已选中的精确 manifest 源图/生成图，
不额外从同地点取其他源图。主真实图与全局生成 query 沿用原有随机增强；
局部监督另行读取同一对原文件，双方使用全帧固定 Resize、RGB、PIL bilinear、
ImageNet normalization。这样局部学生与教师处于明确的同一坐标约定。
对应仍由 Stage 2B 的双向 radius=1 搜索、歧义检查和几何证据确定，
不假设天气编辑严格逐像素对齐。现有 mask 同样投影到此网格。

第一版仅采用 Stage 2B 的 **A 类有效对应**。B 类仍是未经进一步核验的困难候选，
单独记录 `excluded_b_patches`，不自动升级为监督。两种模式使用完全相同的 A 支持集：

- `uniform`：有效 A 对应的 q=1，其余 q=0。
- `verified`：有效 A 对应使用 Stage 2B 的 `positive_conf`，其余 q=0。

`verified` 是证据加权模式名，不表示人工真值或已校准的正确概率。
Unknown 和明确遮挡/破坏区域不会被强制正向蒸馏，也不在本阶段添加负样本损失。

局部蒸馏必须提供 `--teacher-checkpoint`、`--synthetic-aux`、`--pair-calibration`。
校准文件可为单个对象或数组，`--pair-calibration-index` 默认 0。
使用 Stage 2B 已测操作点，不添加固定余弦/差距阈值。
校准的教师 SHA256、预处理、radius 必须匹配当前设置。
Feature Loss 还要求至少一个可训练 Backbone block。

`--pair-geometry-cache` 可指向现有几何 JSON 目录。图像以原始 RGB 哈希和尺寸
匹配证据；正确处理原图、匹配器输入画布与 Patch 坐标。缺少几何证据时仅保留
通过校准教师结构规则的 A 对应，不把 LightGlue 失配当作破坏。
训练期间不启动 LightGlue，不创建教师特征磁盘缓存。

冻结教师在线处理真实源图和生成图，使用 eval/no_grad/FP32，不进入 optimizer。
`--teacher-batch-size 1` 将教师输入分块；仍保留 Stage 2A OOM 自动缩小分块机制。
仅当前辅助批次目标临时放在 CPU，学生局部前向逐对执行并释放计算图。

每步顺序：真实 MS backward → 全局生成 backward → 各局部微批次 backward →
一次 unscale / clip / optimizer.step / scaler.update。所有分支使用同一个 scaler；
局部目标或损失非有限时清除本步梯度并报错，AMP 梯度溢出由原 scaler 跳过更新。
原辅助采样器独立 RNG 保留；局部目标构建不使用随机增强或采样 RNG。

checkpoint 配置记录各权重、模式、固定变换、教师与校准文件 SHA256、几何缓存
整体摘要及 OT 数值下限。resume 必须匹配这些设置，且恢复原有 RNG/scaler/optimizer。
三个权重全零时完全跳过局部模块，不加载校准或几何缓存。

## B0–B5 有界运行命令

以下均从工作区根目录运行，每次最多 **1 epoch、2 次更新**，使用已有四天气
诊断子集，每天气一个不同地点。权重 0.1 仅用于 smoke，尚未做检索性能调参。

| 组合 | Global | Feature | Score | Conditional OT | q |
|---|---:|---:|---:|---:|---|
| B0 | 0.1 | 0 | 0 | 0 | 无局部监督 |
| B1 | 0.1 | 0.1 | 0 | 0 | uniform |
| B2 | 0.1 | 0 | 0.1 | 0 | uniform |
| B3 | 0.1 | 0 | 0 | 0.1 | uniform |
| B4 | 0.1 | 0.1 | **0** | 0.1 | uniform |
| B5 | 0.1 | 0.1 | **0** | 0.1 | verified |

复现带梯度断言的完整 GPU smoke（8 组，共 16 次更新，包括 B5 的 BF16/FP32）：

```bash
cd /home/admin123/github/WORKSPACE
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
TEACHER="$PWD/outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1/generated_8to1/checkpoint.pt"
DINO=/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main
DIAG="$PWD/salad/outputs/ccr_stage2b_diagnostics"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" salad/tests/smoke_local_distillation.py \
  --teacher-checkpoint "$TEACHER" --backbone-repo "$DINO" \
  --diagnostic-dir "$DIAG" --real-data "$PWD/dataset/gsv-cities" \
  --output-dir "$PWD/salad/outputs/ccr_stage2c_smoke_repeat"
```

输出目录须未存在。此测试入口拦截 checkpoint 序列化，只写配置、统计和梯度断言结果，
**不产出可恢复训练的学生权重**。它从已有诊断配对构造小 manifest，并选择使当前 K
包含精确源图的 seed，不绕过正式采样器。原始实测目录为 `outputs/ccr_stage2c_smoke`。

下面通过正式入口单独运行某个组合，正常保存 checkpoint。先执行上面的变量赋值；
本机已有原始 smoke 子集 manifest，repeat 不覆盖它。主图增强保持开启。

```bash
COMMON=(
  --real-data "$PWD/dataset/gsv-cities"
  --synthetic-manifest "$PWD/salad/outputs/ccr_stage2c_smoke/manifest.jsonl"
  --synthetic-places-only --cities BuenosAires Chicago MexicoCity TRT
  --init-checkpoint "$TEACHER" --backbone-repo "$DINO"
  --image-size 224 224 --num-trainable-blocks 1
  --epochs 1 --max-batches-per-epoch 2 --batch-size 2
  --images-per-place 4 --min-images-per-place 4
  --real-sampling coverage_pk --seed 18 --num-workers 0
  --synthetic-aux --synthetic-exposure-prob 1
  --synthetic-global-weight .1 --synthetic-global-temperature .1
  --learning-rate 1e-6 --precision 16 --device cuda --profile-training
)
LOCAL=(
  --teacher-checkpoint "$TEACHER" --teacher-batch-size 1
  --pair-calibration "$DIAG/calibration.json" --pair-calibration-index 0
  --pair-geometry-cache "$DIAG/geometry"
)
run_stage2c() {
  local name="$1"
  shift
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" salad/train_salad.py "${COMMON[@]}" \
    --output-dir "$PWD/salad/outputs/ccr_stage2c_manual/$name" "$@"
}

# 按需要单独执行，不会启动全城市训练。
run_stage2c B0
run_stage2c B1 "${LOCAL[@]}" --feature-distill-weight .1
run_stage2c B2 "${LOCAL[@]}" --score-distill-weight .1
run_stage2c B3 "${LOCAL[@]}" --ot-distill-weight .1
run_stage2c B4 "${LOCAL[@]}" --feature-distill-weight .1 --ot-distill-weight .1
run_stage2c B5 "${LOCAL[@]}" --feature-distill-weight .1 --ot-distill-weight .1 \
  --pair-confidence-mode verified
```

`p_exp=1` 用于确保 smoke 经过辅助分支，不是最终生成占比。
恢复尚未完成的正式运行时，使用相同配置，把 `--init-checkpoint` 换为
`--resume path/to/checkpoint.pt`，保留原总 epochs；不能从已结束的一轮 smoke
直接扩展 cosine schedule。更改实验组合应使用新运行而不是 resume。

## 2026-10-18 验证与实测

60 项 CPU 回归通过，覆盖损失公式、KL 方向、精确 gather、动态尺寸、q=0/1、
teacher/q 无梯度、学生非零梯度、极端 logits/AMP、微批次归约、空辅助批次、
非有限损失不更新、恢复与配置检查，以及 Stage 1/2A/2B 相关回归：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=salad:salad/tests "$PY" -m unittest \
  test_local_distillation test_synthetic_aux test_pair_correspondence \
  test_frozen_teacher test_pretrained_initialization -v
```

另直接加载修改前 Git HEAD 的训练入口，与三个新权重均零的新入口比较。
开启增强、运行两轮后，26 个模型 state key 逐元素完全相同，training_config、
Python/Torch/辅助 RNG 状态相同，见 `stage1_exact_regression.json`。
该严格对照使用可控小模型；真实 ViT-B/14 的 GPU 路径由以下 smoke 验证。

实测设备 NVIDIA GeForce RTX 4090，驱动 570.195.03，PyTorch 2.8.0。
使用同一个完整已训练 SALAD checkpoint 初始化学生和教师，SHA256：
`f06772e6b22c5335efccdf0afd23a1afe18442703299281d23b0641286589513`。
ViT-B/14，224×224，P=2/K=4，最后一个 Backbone block 可训练，教师 FP32。
每组两步、16 张真实主图、4 张生成辅助图，各天气一次。

| 组合 | 精度 | L_real | L_syn | L_feat | Score KL | OT KL | 特征余弦 | 峰值 allocated MiB | ms/step |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | FP16 | .613736 | .072991 | — | — | — | — | 784.0 | 442.4 |
| B1 | FP16 | .615555 | .073468 | .099077 | 1.141011 | 1.477296 | .900923 | 1119.3 | 395.8 |
| B2 | FP16 | .615558 | .073460 | .099071 | 1.140369 | 1.477077 | .900929 | 1119.3 | 354.4 |
| B3 | FP16 | .615550 | .073472 | .099083 | 1.141163 | 1.477092 | .900917 | 1119.3 | 369.0 |
| B4 | FP16 | .615553 | .073482 | .099082 | 1.141114 | 1.477099 | .900918 | 1119.3 | 380.3 |
| B5 | FP16 | .615556 | .073480 | .066104 | .873696 | 1.480835 | .933896 | 1119.3 | 373.1 |
| B5 | BF16 | .615453 | .073057 | .066082 | .875891 | 1.486040 | .933918 | 1119.3 | 480.2 |
| B5 | FP32 | .547120 | .067744 | .066117 | .858193 | 1.366163 | .933883 | 1155.9 | 466.3 |

表中损失是未乘权重的逐步均值。局部启用后同时记录三项诊断，**Score KL 有日志
不代表启用 Score Loss**；B4/B5 的 Score 权重仍为零。
所有局部组合支持集相同：143/1024 Patch（13.9648%），排除 B 候选 151 个，
4 对均命中几何缓存，此小样本没有触发极低教师 Keep 过滤。
q 加权后的余弦更高不能解释为检索效果改善，因为统计权重已变化。

每组均断言实际 optimizer.step 恰好两次；B0 共四次 backward，其他组共八次。
教师无梯度，启用的学生局部分支存在非零且有限梯度，三种精度均完成更新。
这里只确认数值与训练路径，不报告检索收益或配对准确率。

时间范围从主图输入传到 GPU 后到 scaler.update 完成，包含局部解码、在线教师、
对应搜索和学生训练，排除 DataLoader 等待、初始主图拷贝及 checkpoint 写盘。
CUDA 同步计时；显存为 PyTorch allocator 的最大 allocated，包含常驻模型/教师，
不是整卡占用。reserved 原始值也保存在 JSON。每组仅两步、无独立预热，存在
冷启动和调度噪声，不能据此判断 B0 更慢或局部蒸馏更快。

完整证据位于 `salad/outputs/ccr_stage2c_smoke/summary.json`、各组合的
`checkpoint_summary.json` / `train.log`、`regression.log`、`gpu_smoke.log`。
实验输出和本地 checkpoint 不纳入源码版本管理。
