# CCR-VPR 第三阶段：A/M 多种子对照

本阶段只比较 A（纯真实 MS）与 M（真实/生成图共同进入同一个 MS）。代码不启用 synthetic_aux、教师、Feature/Score/OT 蒸馏或 Reliability。入口 `salad/run_am_multiseed.py`；配置 `salad/configs/am_multiseed.json`。默认仅 CPU 预检查和准备，GPU 必须显式 `--execute`。

输出使用新目录 `outputs/qwen_curriculum/state/finetuning/am_multiseed_v1/`，不覆盖第二阶段或历史 A/M。共享的 `train_salad.py`、采样与模型代码保持原样；`train_am.py` 只增加已消费样本的天气计数，复用原训练过程。

## 划分、数据和可比范围

沿用第二阶段的 `configs/splits/ccr_reserved_places.json`：128 个 validation 地点、12 个 calibration 地点均排除全部真实视角和对应生成图。完整真实训练池不按是否有生成图筛选。本轮不训练校准器，也不用这 140 个地点的合成图选择超参数。

预检查读取 `outputs/qwen_curriculum/state/training_manifest.jsonl` 与 `images/accepted` 的交集，验证审核标志、精确源图和文件 SHA256，冻结到 `inputs/`，再按固定地点清单过滤出 `train_manifest.jsonl`。以后增加图片不会改变已冻结实验；纳入增量必须使用新 output_dir。生成标签来自精确源图在 GSV 元数据中的地点，不从文件名猜配对。

所有方法使用同一个 `dino_salad.ckpt`，从头初始化各自 optimizer。M 不加载 A 训练后的权重。每个 seed 的 A/M 使用相同真实地点池、地点访问顺序、K=4、可训练 blocks、LR、AdamW、weight decay=0、MS miner margin=0.1、224×224 预处理和原 augmentation policy。三轮全量 `coverage_pk`，每轮每地点一次，完整 epoch 后 cosine 更新一次，`T_max=3`。

M 保留原 **mix** 语义：每地点生成槽数为 `min(floor(K × nominal), 合格生成图数, K−1)`，缺图时保留真实图。不是 exposure probability，也不额外增加主 batch。A/M 更新次数和总图片曝光相同，M 的真实曝光数减少。生成图不要求正好来自本轮选中的真实视角；其地点仍由 manifest 中精确源图确定。这是主 batch 混合 MS，不是精确配对辅助分支。

本次冻结预检查：62,374 个真实地点、523,498 张真实图；13,710 张 accepted 中排除保留地点的 205 张，训练可用 13,505 张，分布于 10,818 个地点。每轮总曝光 249,496：M25 生成 10,818 次（4.336%），M50 生成 13,505 次（5.413%）。这是完整 epoch 下由库存和采样规则决定的计划值；运行时另行统计已实际消费的曝光。训练生成库存为 Night 7,629、Rain 1,214、Snow 2,679、Fog 1,983，不把这些库存计数当作各天气实际曝光或真实验证域。

## 顺序选参

只训练 A，screen seed=42。继承第二阶段正式 seeds=43/44/45，避免选参 seed 混入正式确认。

| 阶段 | 变化量 | 固定量 |
|---|---|---|
| P | 8、16、32 | LR=3e-6，最后 2 blocks |
| LR | 1e-6、3e-6、1e-5 | 上阶段选出的 P，2 blocks |
| blocks | 最后 1、2、4 | 前两阶段选出的 P/LR |

依据：旧 epoch-050 checkpoint 的诊断提示 MS 饱和、增加负样本有帮助；原始 dino checkpoint 的已有 A/42 却已有约 9.8%–12.1% 有效 anchors，不能混用两份权重的饱和结论。因此从中间 LR/block 控制变量开始，不假设 P/LR/block 越大越好。

每个配置训练一个完整 epoch，使用预设三轮 cosine 的第一轮。重复配置直接复用相同 checkpoint/评估，最多 **7 个不同配置**，不是 27 项网格搜索。不同 P 的筛选有相同总图片曝光，但更新次数和负样本池不同；正式 A/M 在选定 P 后更新次数完全一致。若某个阶段没有合格配置，停止，不自动提高 LR、放宽门槛或跳到下阶段。

筛选规则固定为：先通过训练信号门槛，再按 **SVOX val 总体 R@1** 最高选择；精确并列按配置给定顺序。信号门槛沿用第二阶段：每轮完整真实曝光、生成曝光 0、有限正 MS loss、有效 anchor ≥5%、空 mining batch ≤10%、正负 mined pairs 均存在；val R@1 相对初始下降不超过 0.1pp。这些是预注册的工程门槛，不是理论最优值或显著性证明。所有候选与选择依据保存到 `selection/*.json`，最终 `formal_protocol.json` 冻结预算、种子、方法及选择结果。

## 验证集与统计边界

仅允许 `dataset/svox/images/val/` 内的真实图像。总体目录是 `queries`；只有在 val 下实际存在 `queries_night/rain/snow/sun/overcast` 时才评估对应自然天气验证域。图片或目录若通过 symlink 指向 test 会被拒绝。当前缺失的域写入 `missing_natural_weather_validation`，不会使用合成天气代理补齐，也不会拿 test 天气目录充当 val。

**本入口没有 test 执行阶段，不读取历史 test 分数。** 已查看的 SVOX test 继续只属于历史报告。现有 val 已被使用过；新训练 seed 的复现不能把它变成全新、未使用过的天气测试集。

固定最后三轮预算的 checkpoint 是主要比较对象，不按 val/test 最佳 epoch 挑模型。默认正式计划为三 seeds × A/M25/M50，共 **9 组**；M50 为预设主要比较，M25 为比例消融。也可分次执行，报告明确标记未完成项。

报告逐 seed、逐实际验证域、逐 R@1/5/10 给出：

- 有效 query 数、Recall、相对初始及相对同 seed A 的增量。
- 错→对、对→错计数及双侧精确 McNemar p 值。
- 固定 Holm 比较族：全部正式 seeds × 验证域 × 三个 Recall cutoff ×（三个方法对 initial，以及两个 M 对 A）。只有总体域时共 45 项；未完成比较按 p=1 占位，不因少跑实验缩小校正范围。筛选结果不作为正式显著性证据。
- M−A 的各 seed 增量、均值、样本标准差、最小/最大值、正向 seed 数。不同 seed 共享 query，不合并成更多独立 query 来计算一个夸大的显著性。

McNemar 的独立 query 假设在轨迹相邻图像上可能不成立。若有真实地点/轨迹元数据，可在**准备前**配置 `group_manifest`：

```json
{
  "provenance": "说明分组元数据的实际来源",
  "domains": {
    "svox_val": {"exact_query_id_1": "trajectory_1", "exact_query_id_2": "trajectory_1"}
  }
}
```

每个给定域必须覆盖全部精确 query IDs，不能根据检索结果临时分组。代码执行成组重采样，给出每个训练 seed 条件下的配对 percentile 95% 区间；它不是跨 seed 总体区间，也未经多重区间校正。没有分组元数据时明确输出 null，不用 query ID 或地理坐标猜测轨迹。

## 命令：从工作区根目录运行

先等待当前 GPU 作业结束。定义 Python 与入口：

```bash
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
RUN=salad/run_am_multiseed.py
```

预检查、冻结当前图片和数据清单，不启动模型：

```bash
CUDA_VISIBLE_DEVICES='' "$PY" "$RUN"
```

按顺序执行三个选参阶段。每次命令只运行指定阶段：

```bash
"$PY" "$RUN" --phase screen-p --execute
"$PY" "$RUN" --phase screen-lr --execute
"$PY" "$RUN" --phase screen-blocks --execute
```

审核 `report.json` 和 `selection/` 后冻结正式协议；此命令不训练：

```bash
"$PY" "$RUN" --phase freeze
```

分别启动正式 A 与主要 M50；每条命令串行处理三个 seeds：

```bash
"$PY" "$RUN" --phase train --method A --execute
"$PY" "$RUN" --phase train --method M50 --execute
```

比例消融 M25：

```bash
"$PY" "$RUN" --phase train --method M25 --execute
```

需要一次串行运行所有 **9 组** 时：

```bash
"$PY" "$RUN" --phase train --execute
```

中断恢复直接重复原命令；也可只恢复指定 seed/method：

```bash
"$PY" "$RUN" --phase train --method M50 --seed 43 --execute
```

完整 epoch checkpoint 恢复 optimizer、scheduler、scaler、Python/NumPy/Torch/CUDA/DataLoader RNG 与覆盖状态。未保存的半轮重跑。方法、种子、初始化、元数据哈希、workers、预算或 schedule 不一致会报错。BF16/FP32 下没有 FP16 scaler 跳步导致的更新预算歧义。不要更改冻结 config；更改方案用新目录。

只汇总现有结果，不启动 GPU：

```bash
CUDA_VISIBLE_DEVICES='' "$PY" "$RUN" --phase report
```

## 输出与资源

- `plan.json`、`inputs/`、`real_pool/excluded_places.json`、`real_pool/exclusion_report.json`：来源、哈希、固定地点和图像快照；计划同时给出当前库存下的每轮理论生成曝光上限/数量。
- `selection/*.json`、`formal_protocol.json`、`formal_commands.txt`：顺序搜索结果、选择理由、冻结正式配置和逐组运行命令。
- `runs/<seed>/<A|M25|M50>/`：每轮指标、训练与资源日志、checkpoint、最终 validation、audit。
- `audit.json`：实际 optimizer 更新数、epoch、LR 曲线、每轮 MS/mining、真实/生成/天气曝光、Backbone/Aggregator 位移及相对 L2、训练时间、GPU 峰值、进程树内存记录、最终 checkpoint SHA256 和数据/config 摘要。位移按对应分支浮点 state tensors 计算，包含冻结部分，不是梯度范数。
- `report.md/json`、`seed_metrics.csv`、`paired_statistics.csv`：状态、A 的工程检查、逐 seed 和配对统计。部分实验未完成时不称为稳定提升。

资源设置沿用 48GB 显存/32GB 内存方案：BF16、4 workers、prefetch=1；PyTorch allocator 上限 40GiB并保留当前空闲显存中的 4GiB，进程树 RSS 限 20GiB且系统 available 至少 3GiB。监控不是 cgroup 硬隔离；进程树 RSS 可能重复计算共享页。每轮串行训练/评估，验证共享 gallery。参数位移计算在单独受监控的 CPU 进程完成，不把第二份模型放上 GPU。旧实验文件和共享训练代码不修改。

## 如何判断研究问题

1. **A 有信号吗？** 看每轮有效 anchors、真实正负 mined pairs、空 mining batch、有限 MS loss以及实际参数位移。再看三 seeds 的 val 是否基本保留初始水平。非零 loss/位移本身不能保证 Recall 获益。
2. **M 优于 A 吗？** 主要看同 seed、同最终预算 M50−A 的真实 val 差值、修好/改坏数与 Holm 后证据，不用 M−initial 代替 M−A。M25 属于预设比例对照，不事后只挑最好的一组。
3. **跨 seed 稳定吗？** 要三组都完成；看方向一致性、均值、标准差和最差 seed。只有三个 seeds 或未经校正的单域 p<0.05，都不足以宣称普遍泛化改善。
4. **曝光率如何解释？** 比较 nominal 与实际生成比例、天气计数和 M25/M50 的增量。库存稀疏时 nominal 50% 通常远低于实际 50%；同时真实曝光在减少。两比例的变化不能直接证明线性剂量效应。
5. **是否继续 B0/局部蒸馏？** 若 A 仍饱和/明显退化，先修基线；若 A 稳定而 M 无跨 seed 收益，优先查真实天气验证与数据质量/覆盖；若 M 有稳定正向趋势，再把它作为有信号的参照研究辅助方法。没有独立自然天气验证时，本阶段不能证明夜雨雪鲁棒性，也不能据此证明 B0 或局部蒸馏有效。
