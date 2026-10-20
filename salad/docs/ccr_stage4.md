# CCR-VPR 第四阶段：辅助损失归因与局部消融

入口 `salad/run_ccr_stage4.py`，配置 `salad/configs/ccr_stage4.json`，训练适配层 `salad/train_ccr4.py`。新目录为 `outputs/qwen_curriculum/state/finetuning/ccr_stage4_v1/`。默认只准备计划；本阶段不启用 Reliability，也不改原始训练入口、损失公式、Sinkhorn 或老实验输出。

## 先检查依赖

第四阶段读取第三阶段冻结的数据快照、完整真实地点池、140 个地点排除清单和 seeds。**必须已有第三阶段 `formal_protocol.json` 才能启动任何本阶段真实数据训练**。不把第三阶段初始中心配置冒充选参结果。

本次交付预检查时，第三阶段尚未冻结，因此报告状态为 `awaiting_stage3_freeze`。代码与短程测试已完成，但没有可用于研究结论的第四阶段正式训练结果。

教师固定为第三阶段初始完整 checkpoint；不随学生更新。校准只从原 12 个 calibration 地点中每地点取一个已审核配对，用不同地点的 null 和同图 copy controls 拟合阈值；`tail_probability=0.01` 是预声明的校准统计设置，不是手填 cosine/匹配置信度阈值。诊断配对从原 validation 保留地点按天气固定选取，与训练及 calibration 地点分开。此处生成配对仅是诊断数据，不是自然天气检索验证集。

SVOX 仅使用第三阶段冻结的真实 val；不读历史 test。当前缺少真实 Night/Rain 等 val 域，会明确列出，不用合成天气替代。

## 实验次序和门槛

### 1. RNG 归因对照

| 名称 | 辅助前向 | Global 权重 | 模型 RNG |
|---|---|---:|---|
| A | 无 | 0 | 主分支原轨迹 |
| B0-zero | 有 | 0 | 与主分支共享 |
| B0-zero-isolated | 有 | 0 | 隔离 |
| B0-original | 有 | 0.1 | 与主分支共享 |
| B0-isolated | 有 | 0.1 | 隔离 |

以上在 screen seed=42 上各执行一个完整 epoch，保持第三阶段完整 cosine schedule 的第一轮。辅助采样依然有独立且被 checkpoint 保存的 RNG。模型辅助 RNG 则采用全局 seed、optimizer step、分支编号的确定性 hash；只临时设置 CPU 和当前 CUDA device 的生成器，退出后恢复主 RNG。每步 mask 正常变化，不改变 Python 数据增强或主地点采样轨迹。局部分支采用另一个分支编号，同样隔离。

`B0-zero` 通过零权重 backward 累积零梯度，仍执行相同辅助前向；用来观察 RNG 消耗本身。`B0-zero-isolated` 是额外的严格控制：在相同硬件/精度条件下应恢复 A 的参数轨迹。报告比较**参数内容哈希**，不比较包含不同配置元数据的 checkpoint 文件哈希。所有策略写入 training_config/checkpoint 的 `stage4` 字段。

### 2. Global 权重筛选与 B0 确认

`global-screen` 在同一个 screen seed 上单独运行 A，以及隔离 RNG 的 global_weight=0/0.01/0.03/0.1。每个候选一个完整 epoch。候选只是研究起点，不假设小权重更好，也不以 loss 比值判定梯度主导。

A 必须通过第二阶段的完整真实曝光和 mining 门槛；非零候选相对 A 的总体 val R@1 下降不超过预声明的 0.1pp，才进入候选集合。集合中取 val R@1 最高者；精确并列按权重升序。零权重作为控制不被命名为有效 B0。若没有合格非零候选，停止正式局部研究。

选定后，正式 A/B0 使用继承的 seeds=43/44/45，分别从初始 checkpoint 开始训练全部预算。必须全部完成且通过以下工程检查，才允许 `--phase local`：

- A 每轮完整真实曝光、有足够 mined pairs，并通过原有效 anchor / 空 mining batch 门槛。
- A、B0 每个 seed 的真实 val R@1 相对初始下降均不超过第二阶段固定容限。
- 每个 seed 的 B0−A 不低于 −0.1pp；三个配对增量的标准差不超过 0.1pp。
- A 的 seed 间标准差不超过第二阶段容限。

这些是“行为可接受”的工程门槛，不是收益显著性的证明。实际配对检验与跨 seed 分布仍要独立检查。B0 未通过时，代码允许两 batch 的局部 smoke，拒绝正式局部训练。

### 3. 局部消融

| 组合 | Global | Feature | Conditional OT | q | 局部学生模式 |
|---|---|---|---|---|---|
| B0 | 冻结选定权重 | 0 | 0 | — | — |
| B-Feature | 同 B0 | 0.1 | 0 | A 支持上为 1 | 仅关闭 SALAD Dropout |
| B-OT | 同 B0 | 0 | 0.1 | A 支持上为 1 | 仅关闭 SALAD Dropout |
| B4 | 同 B0 | 0.1 | 0.1 | A 支持上为 1 | 仅关闭 SALAD Dropout |
| B5 | 同 B0 | 0.1 | 0.1 | verified positive_conf | 仅关闭 SALAD Dropout |
| B-OT-train | 同 B0 | 0 | 0.1 | A 支持上为 1 | 原始 Dropout |

Feature/OT 权重 0.1 是可配置、预先声明的起点，不声称两者梯度已平衡。修改它们应使用新 output_dir，不能在已冻结实验中改权重。B4 不加 Score Loss；Score KL 仅作为只读诊断。

真实主 batch 保留原 augmentation。局部源图和生成图都做同样固定 Resize；教师 FP32 eval 且 no_grad。Feature 使用 `backbone_local_features`；OT 对完整 log_assignment 的有效簇做条件 log_softmax，保留原 dustbin/Sinkhorn。没有新增语义门控，仅沿用 FP32 epsilon 的近零教师有效簇质量数值保护，并报告被排除数。

仅关闭 SALAD Dropout 时，保留 Backbone 的原 train/eval 状态及 autograd；这与整个学生 eval 不同。**Backbone Feature 差异不能归因于 SALAD score/content Dropout。** 教师和学生全 eval 用于只读诊断，不替换主训练模式。

B4/B5 的 A 支持来自同一初始教师、同一校准、相同几何缓存和同一精确源图配对。每次消费配对记录索引/支持 mask 的哈希，完成后逐 seed/step/图像比较；不相同则报告 `invalid_B4_B5_support_mismatch`。Unknown、B 类候选和无效索引不升级为 Positive。

## 记录什么

每一步记录天气辅助 loss sum、query 数及已消费的局部支持；默认每 100 optimizer steps 额外记录梯度诊断，smoke 每步记录。诊断不增加学生前向，不消耗模型 RNG；局部梯度测量通过相同计算图的额外 autograd.grad 完成，随后正常 backward 和一次 optimizer.step。

- `g_real`、已加权 `g_syn`、Feature/OT 已加权梯度范数；real/syn cosine、范数比、各 DINO block / SALAD score、content、token 等分组。真实梯度为 0 时 ratio/cosine 标为未定义，不能填一个误导性有限值。
- 实际总梯度裁剪率来自每个训练 batch，而非采样梯度对裁剪的推测。
- 每图和每天气的 A 支持数、覆盖率、q mass、有效样本量 `(sum q)^2/sum(q^2)`、支持内 Feature cosine、全图同坐标 cosine、q 与匹配特征一致性的相关性。
- confidence bands 是按已有支持中 q 的排序分成四组，属于描述统计，不参与监督筛选。记录各组的 Feature/OT numerator 与实际加权 loss 贡献。
- 在采样 step 的同一学生输出上，另外计算 uniform 和 verified 的梯度对照，使用同一 A 支持与整辅助批次归一化。该项是**逐配对条件对照**，不是把不同训练后模型的梯度差都归因于 q。
- 无有效支持仍记录覆盖率为 0，loss/梯度监督为 0；loss 汇总不能把这些零当成学得更好。
- GSV 真实图没有可靠天气标注，曝光报告为 `unannotated_GSV`。不能从其配对生成图的天气倒推真实图天气。
- 每轮 MS/global/local loss、mining、LR、更新数、epoch、训练时间、GPU 峰值和受监控的进程树 RSS；最终参数 SHA256 与 checkpoint SHA256、Backbone/Aggregator L2 位移；固定保留源图上的 FP32 eval 描述子位移。

位移是观察到的训练结果，不能单独替代梯度归因。norm 分组统计见 audit，原始采样 Gram 矩阵与 confidence bands 留在 attribution.jsonl。

## 只读局部诊断怎么解释

`diagnose` 固定教师和对应支持，重复比较 FP32/BF16、学生 train/eval，并再跑一次只屏蔽 SALAD Dropout 的对照。初始 checkpoint 的诊断也采用第三阶段选定的可训练 block 配置，避免错误的 train-mode 下限。

- `source_self` / `generated_self` 对同一个学生的 FP32 eval 输出：主要检查随机性和精度影响。
- 同图学生对初始教师：还包含参数漂移。
- 真实—生成对：进一步包含跨域差异。
- same-coordinate 与 matched-support：呈现对应策略依赖；没有真实对应 ground truth 时，差值不能直接等同于“匹配错误量”。

复用 LightGlue 几何缓存及确实存在的 mask；没有证据时允许无监督。所有结果只用于分解与比较，不从 loss 中扣除噪声下限。学生/教师参数版本和教师无梯度状态受检查，诊断不创建 optimizer。

## 手动命令

在工作区根目录：

```bash
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
RUN=salad/run_ccr_stage4.py

# CPU 预检查，当前可以运行
CUDA_VISIBLE_DEVICES='' "$PY" "$RUN"
```

先完成第三阶段选参并执行其 `--phase freeze`。然后冻结第四阶段依赖：

```bash
"$PY" "$RUN" --phase freeze
```

以下命令均需你显式启动，未在本次开发中自动执行：

```bash
# RNG 控制与 global 权重候选，各自只跑 screen seed 的一轮
"$PY" "$RUN" --phase controls --execute
"$PY" "$RUN" --phase global-screen --execute
"$PY" "$RUN" --phase freeze-global

# 独立保留地点校准与初始 checkpoint 只读诊断
"$PY" "$RUN" --phase calibrate --execute
"$PY" "$RUN" --phase diagnose --execute

# 两个 batch 的真实数据执行链路；不进入正式统计，不绕过第三阶段冻结依赖
"$PY" "$RUN" --phase smoke --method B4 --execute

# 三 seeds 的正式 A/B0，共六组；使用冻结的完整更新预算
"$PY" "$RUN" --phase confirm-b0 --execute

# 只有 A/B0 门槛通过才允许下面的正式局部训练
# 此命令将串行运行五个局部组合 × 三 seeds，共十五组
"$PY" "$RUN" --phase local --execute
```

可以逐组运行、恢复，无需一次全部启动：

```bash
"$PY" "$RUN" --phase confirm-b0 --method B0 --seed 43 --execute
"$PY" "$RUN" --phase local --method B-Feature --seed 43 --execute
"$PY" "$RUN" --phase local --method B5 --seed 43 --execute
```

重复原命令会恢复各自 checkpoint，保留 optimizer、cosine、模型/采样 RNG、覆盖状态和测量策略。未保存的半轮重跑；attribution 中尚未提交至 checkpoint 的记录被移除后重算，避免曝光重复计数。改变 RNG、局部模式、权重、数据、seed 或测量频率会被拒绝。所有方法从同一初始模型开始，不从 A 或 B0 的最终模型继续训练局部组合。

对训练后 checkpoint 做相同只读诊断：

```bash
"$PY" "$RUN" --phase diagnose \
  --checkpoint outputs/qwen_curriculum/state/finetuning/ccr_stage4_v1/formal/43/B0/checkpoint.pt \
  --execute

CUDA_VISIBLE_DEVICES='' "$PY" "$RUN" --phase report
```

## 报告与决策

输出包括 `plan.json`、`protocol.json`、`global_selection.json`、`local_protocol.json`、`calibration.json`、`diagnostics/`、每组 `contract.json` / `metrics_epoch_*.json` / `attribution.jsonl` / `audit.json` / checkpoint / validation，以及 `report.md/json`。

`report.json` 的主要部分：

| 字段 | 用途 |
|---|---|
| `dependencies` / `status` | 第三阶段冻结、校准及实际完成情况 |
| `pilot_runs` / `rng_controls` | 原始/隔离/零权重的参数轨迹、验证值和梯度诊断 |
| `B0_gate` | 是否允许正式局部研究及失败原因 |
| `runs[].audit` | objective、预算、曝光、loss/mining、梯度分布、覆盖、位移和成本 |
| `paired_statistics` | 各 seed/域/R@1/5/10，query 修好/改坏数、精确 McNemar、Holm p |
| `seed_distributions` | 逐 seed 增量、均值/标准差、条件于当前验证集的 seed bootstrap 区间 |
| `support_comparison` | B4/B5 精确共同支持检查 |
| `module_assessment` | Global、Feature、OT、组合增量、q 和 Dropout 的直接对照及当前证据状态 |

所有正式方法对 initial，B0 对 A，局部方法对 A/B0；另外明确比较 B5−B4、B4−Feature、B4−OT、OT-train−OT-off。Holm 比较族固定为全部计划 seed/实际 val 域/Recall cutoff/这些对照，缺失项按 p=1 保留，不因少跑某组放宽校正。只有一个 val 域、三 seeds 时共 198 项。不同 seeds 的 query 不合并成独立样本。

继承第三阶段提供的真实地点/轨迹 group metadata 时，增加每 seed 的配对分组 bootstrap；没有元数据时保持 null。三个 seeds 的经验 bootstrap 区间分辨率很低，且不覆盖新的 query/天气分布，不能当作普遍泛化证明。

继续研究的依据应是：A/B0 行为可接受；Feature−B0、OT−B0 或 B5−B4 在独立 seeds 上有可重复增量；支持/梯度诊断可以解释该变化；成本可接受。loss 下降、辅助梯度变大、q 有非零方差、单 seed 好转都不充分。B0 退化、q 支持不一致或局部仅拟合随机性时，应明确报告失败/证据不足，而不是继续加入 Reliability。

## 本次验证范围

CPU 覆盖 RNG 恢复、零权重、独立 Feature/OT、SALAD-only eval 梯度、共同 A 支持、q 数值与梯度、无效零监督、FP16/BF16/FP32 loss、全局/局部完整 checkpoint 恢复、runner audit 与协议门槛。

另外执行了两项显式微型 CUDA smoke：RNG/两次 backward 一次 step，以及微型 SALAD BF16 的真实+全局+Feature/OT 一步更新。后者测得 allocator allocated 峰值 40,393,728 bytes、reserved 峰值 48,234,496 bytes；这只是 28×28 合成测试夹具，不是实际 ViT-B/14 全量训练的资源估计。本次没有启动真实数据正式 GPU 训练，也没有生成新图片。
