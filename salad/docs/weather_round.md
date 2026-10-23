# 阶段五：天气验证、分层采样与 v3 生成图

本阶段修正前四阶段暴露的四个问题，并给出一套能直接回答"生成图有没有用"的对照。

| 问题 | 证据 | 本阶段处理 |
|---|---|---|
| 选参看不见天气 | SVOX val 只有白天 query；纯真实 A 普通 val +0.1pp，但 test Night −4.5～−5.1pp | 用 SVOX **train** 的五类天气 query 构建自然天气 val，每轮评估 |
| 训练增强偏离原配方 | 原版 `RandAugment(3)`，项目改成翻转+弱色彩；外部复测 Nordland 弱增强 −0.71pp | 训练入口统一使用 RandAugment，删除弱增强及其恢复对照 |
| M 的生成曝光太少且不可调 | M25/M50 实际 4.34%/5.41% | 分层采样：每 batch 固定若干个有生成图的地点，并优先选取其精确源图 |
| B0 辅助梯度过大 | 每 batch 约 2 个辅助 query 被单独取均值再 ×0.1，梯度是真实项的 4.4 倍 | 辅助损失按 query 求和、除以主 batch 图像数，每个生成 query 与一个真实 anchor 同权 |

## 自然天气 validation

`build_weather_validation.py` 读取 `dataset/svox/images/train/queries_{night,rain,snow,sun,overcast}`，
丢弃距任一 SVOX test/val 图像（gallery 或 query）不足 50 m 的 query，正样本为 train gallery 中 25 m 内图像。

| 域 | 保留 | 丢弃 |
|---|---:|---:|
| night | 641 | 61 |
| rain | 685 | 65 |
| snow | 645 | 69 |
| sun | 644 | 68 |
| overcast | 638 | 67 |

它与 SVOX test 同城市、同相机，是目标域 validation，不是未见域测试。每域约 650 个 query，
一个 query 约 0.15pp，读结果时看配对改善/退化数。SVOX test 仍只作历史记录，本阶段不读取。

有效性核对：旧第二阶段 A（seed 43）相对 initial 在这份 val 上 night 88.30%→80.19%（修好/改坏 9/61，
精确 McNemar p<1e-10），sun 91.61%→88.98%（8/25，p=0.005），rain/snow/overcast 小幅下降不显著；
同一模型普通 val 为 +0.15pp。即 test 上看到的天气退化在这里同样可见，前四阶段的选参正是漏掉了它。

外部集（final epoch 才评估）：Nordland 4,096 query 探针、缩减图库 Tokyo247（D/S/N）。
清单由 `build_external_manifests.py` 生成，与旧 `posthoc_external` 逐字节一致。

## 训练改动

- **RandAugment**：`workflow/augmentation.py`，n=3、m=9、bilinear、无翻转。抽样以
  seed/epoch/地点/槽位/文件名为键，在 fork 的 Torch CPU RNG 中执行，与 worker 数无关；
  成对源图/生成图共享同一次抽样。旧 checkpoint 无法用当前代码逐位复现。
- **分层采样** `--synthetic-place-slots S`：每个 batch 的 P 个地点中固定 S 个来自有生成图的地点，
  其余 P−S 个按 coverage 遍历全部真实地点。有生成图的地点沿独立的确定性排列流循环，
  每轮约被访问 `len(sampler)·S / 有生成图地点数` 次。
- **优先源图** `--prefer-synthetic-sources`：每次访问的 K 个真实视角中，最多 K−1 个优先取有生成图的源图。
  A 也使用同一选项，所以 **A/M/B0 的地点流与真实源图完全相同**，唯一区别是这些源图如何使用。
- **B0 归一化**：`asymmetric_multi_positive_infonce(..., denominator=主 batch 图像数)`；
  局部 Feature/OT 损失按同样比例缩放。`--synthetic-global-weight` 默认改为 1.0，含义是
  "一个生成 query 相对一个真实 anchor 的权重"。旧实验的 0.1 与新含义不可直接比较。

## 方法

| 方法 | 主 batch | 额外项 |
|---|---|---|
| A | 真实 | — |
| A_retention | 真实 | 冻结初始模型的描述子保持，权重 0.1 |
| M | 被选中的源图全部替换为其生成图（`replace`，概率 1） | — |
| B0 | 真实 | 每个有生成图的地点一个 InfoNCE query，按 anchor 计权 |

共同设置：P32×K4、LR 3e-6、最后 2 个 block、3 epoch、BF16、S=8、seeds 43/44/45、
140 个保留地点排除。P/LR/block 沿用第二阶段筛选结果，未针对天气 val 重新选参。

GPU 冒烟（每法 4 batch）：A 每 batch 约 1 s、峰值 6.8 GiB；A_retention 约 3.6 倍单步耗时；
M 主 batch 生成占比约 8%（旧 M50 为 5.4%）；B0 每 batch 8 个辅助 query，
未加权辅助损失 0.05，与真实 MS 0.21 同量级，未出现裁剪。

快照：原 13,757 张 accepted + 按 v3 几何门槛重新验收的 2,380 张，共 16,137 张、12,481 个地点。

## 命令（bash）

```bash
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
cd salad

# 只冻结计划与快照；已冻结时重复执行仅校验
$PY run_weather_round.py

# 单个方法/seed；中断后重跑同一命令从最后完整 epoch 继续
$PY run_weather_round.py --phase run --method A --seed 43 --execute

# 全部 4 方法 × 3 seeds（约 12 × 1.5 h，A_retention 更慢）
$PY run_weather_round.py --phase run --execute

$PY run_weather_round.py --phase report
```

`report.md` 给出每个 seed 的最终 R@1（普通 val、五个天气域、外部集分组），各方法相对 initial 与 A 的
跨 seed 平均变化及正向 seed 数，以及实际生成曝光和天气分布。`report.json` 含逐 seed 精确 McNemar
与 Holm 校正 p 值（族 = seeds × 对照 × 分组）。

## 已完成：A，seed 43

| R@1 (%) | val | night | rain | snow | sun | overcast | Nordland | Tokyo N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 96.36 | 88.30 | 95.91 | 95.97 | 91.61 | 97.34 | 81.52 | 82.98 |
| 旧 A（弱增强，seed 43） | 96.51 | 80.19 | 95.18 | 95.19 | 88.98 | 96.55 | — | — |
| 新 A（RandAugment，seed 43） | 96.35 | 89.55 | 96.20 | 95.97 | 91.93 | 97.18 | 81.05 | 80.85 |

新 A 的 night 修好/改坏 16/8（p=0.15），其余天气域变化均不显著；旧 A 为 9/61（p<1e-10）。
两者 P/LR/block/epoch 相同，但新 A 还使用了分层采样与优先源图，所以这不是只改增强的纯消融；
不过两者都只用真实图，最可能的解释仍是增强配方。单 seed，Tokyo 每组仅 141 个 query（N 为净 −3）。
结论：在 RandAugment 下，继续微调不再造成天气遗忘，M/B0 的对照现在有了一个不退化的起点。

## 怎么读结果

1. **先看 A 相对 initial 的天气域**。RandAugment 若有效，A 的 night 不应再大幅下降。若仍明显下降，
   问题在继续微调本身，生成图很难单独补回；先比较 A_retention。
2. **M−A、B0−A 才是生成图的贡献**。两者与 A 的地点流、真实源图完全一致。
3. 普通 val 持平而天气 val 上升，才是期望形态；只看普通 val 会漏掉天气退化，前四阶段即是如此。
4. 外部集是探索性结果，不用于选参。

## 生成图：难度诊断与 v3 提示词

`probe_synthetic_difficulty.py` 在初始 checkpoint 上测量"天气变化本身让描述子离同地点参考图多远"：
生成图为 `cos(源图,参考) − cos(生成图,参考)`；SVOX 为同位置白天 query 与天气 query 相对 gallery 正样本的同一差值；
`day` 对照用相距 2–10 m 的另一张白天 query 代替天气 query，衡量不同拍摄之间的自然差异。

| 来源 | 条件 | 中位位移 | p75 | 位移 > 0.1 |
|---|---|---:|---:|---:|
| 生成 v2 | rain | 0.016 | 0.041 | 4.0% |
| 生成 v2 | night | 0.028 | 0.053 | 6.6% |
| 生成 v2 | snow | 0.031 | 0.059 | 6.9% |
| 生成 v2 | fog | 0.021 | 0.049 | 10.3% |
| SVOX 真实 | day 对照 | 0.035 | 0.260 | 40.6% |
| SVOX 真实 | night | 0.088 | 0.403 | 48.9% |
| SVOX 真实 | rain | 0.074 | 0.355 | 46.2% |
| SVOX 真实 | sun | 0.080 | 0.377 | 47.7% |

结论：

- v2 生成图的天气位移只有真实天气的 1/3～1/4，rain 最弱；而且没有长尾。v2 只是"轻微"的天气编辑。
- 但 SVOX 位移中很大一部分来自拍摄差异本身（day 对照的 p75 已达 0.26）。
  天气额外造成的位移，中位数约 +0.04～0.05。也就是说，"让天气更极端"只能补齐一部分差距；
  另一部分是相机、视角、时间造成的，单张编辑生成不出来。

### 为什么不是简单"更黑、雨更大"

- 把夜晚压暗只是删掉地点证据：几何审核和描述子都会失去建筑结构，检索任务本身变得不可解。
  真实夜间的难点是**光源和传感器响应变了**：钠灯/LED 色温、路灯与车灯眩光、橱窗照明、湿路面反光、高 ISO 噪声。
  v3 的夜晚变体都包含明确光源，不出现无光源的全黑场景。
- 雨天的主要外观变化是湿路面反光、阴天漫射光、镜头水滴，而不是雨丝本身。
- v2 没有 sun；而 SVOX sun 与 Tokyo247 白天都包含逆光、强阴影、过曝，这是 v3 新增的方向。

### v3 提示词库（`AdaptVPR/experiments/qwen_curriculum/prompt_bank.py`）

| 域 | 权重 | 变体 |
|---|---:|---|
| night | 44% | 钠灯、LED、橱窗/招牌、稀疏路灯、车灯眩光、蓝调时刻 |
| rainy_night | 14% | 湿路反光、镜头水滴 |
| rain | 15% | 暴雨阴天、镜头水滴、倾盆+车溅水 |
| snow | 14% | 薄雪、厚积雪、降雪 |
| sun | 8% | 低角度逆光眩光、正午硬阴影、背光剪影 |
| fog | 5% | 浓雾（目标集均无雾） |
| overcast | 0% | 平光灰天（保留审核文本，不再规划） |

权重由下方 100 张 pilot 校准（`qwen_weather_bank_v3.1`）。

下述 pilot 使用每个源图 2 个不同域、每地点 2 个源图。现行全量方案已改为每个合格地点只生成 1 张，独立保存在 `outputs/weather_v3/`，优先完成 night；详见 [当前生成说明](../../AdaptVPR/experiments/qwen_curriculum/WEATHER_V3.md)。每条记录保存 `prompt_variant`，难度探针按变体分组。

### 100 张 pilot 结果

`outputs/qwen_curriculum/weather_v3_pilot`：25 个新地点 × 2 源图 × 2 域，按 v3 审核通过 74 张。
初始 checkpoint 上的描述子位移（同上定义，按域合并）：

| 域 | v2 中位位移 | v3 中位位移 | v3 p75 | SVOX 真实中位位移 | v3 通过 |
|---|---:|---:|---:|---:|---:|
| night | 0.028 | **0.090** | 0.130 | 0.088 | 28/35 |
| rainy_night | — | 0.070 | 0.104 | — | 6/10 |
| snow | 0.031 | 0.057 | 0.099 | 0.075 | 11/13 |
| rain | 0.016 | 0.041 | 0.063 | 0.074 | 12/15 |
| sun | — | 0.013 | 0.037 | 0.080 | 12/17 |
| overcast | — | ≈0 | — | 0.052 | 1/5 |

- v3 夜晚的位移中位数已与 SVOX 真实夜晚相当，是 v2 的 3 倍；车灯、橱窗、钠灯三个变体最强。
  目测：钠灯眩光、LED 冷光、车灯星芒、湿路反光都出现了，建筑结构保留。
- rain、snow 约为 v2 的 2 倍，仍低于真实值；`rain/lens_drops` 目测变化偏弱。
- sun 几乎不移动描述子，部分正午变体被 Qwen 重新构图（拉远）而被几何拒绝；overcast 基本是空操作。
  因此 sun 降到 8%、overcast 不再规划。sun 需要更强的提示词后再提高比例。
- 每个变体只有 2–7 张，单变体数字受随机参考图影响明显，决策只依据按域合并的结果。

### 审核 v3（`quality_v3.py`）

- CLIP 文本覆盖新域，并为 sun 使用"柔和均匀日光"作为中性文本（原先统一用"晴天"，与 sun 自相矛盾）。
  GSV 源图本身常是晴天，sun 的绝对 CLIP 对比接近 0，因此 sun 改为只要求偏移 ≥ 1.5
  （pilot 中 4 张有明显阴影/眩光的 sun 图仅因绝对对比被拒）。
- 几何门槛放宽为 `s_geo ≥ 0.70`、单应位移中位数 ≤ 32 px（原 0.78 / 16 px）。
  抽查原先被几何拒绝的夜晚图，大多是结构忠实、但点亮灯光或轻微重构图的编辑；
  这些重新验收的图描述子位移更大（rain 中位数 0.037 vs 0.021），即 v2 正好丢掉了更难的样本。
  历史 pilot 的重新验收处理只用已记录测量值，得到 2,380 张，已加入本阶段快照；旧 pilot CLI 现已移除。
- 曾尝试在低光域先做亮度均衡再匹配；120 张 v2 几何拒绝的夜晚图中只有 6 张被挽回，没有保留。
- 原 v2 `quality.py`/`weather_signal.py` 未修改，暂停中的 23 城市计划仍可按原指纹继续。

### 当前 v3 生成的控制命令

当前保持暂停。后台统一使用 `night_first_v3.py`，完整计划、来源筛选和中断账本位于 `outputs/weather_v3/state/`；域图片直接放在 `outputs/weather_v3/night/` 等目录。以下命令从工作区根目录执行：

```bash
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python

$PY AdaptVPR/experiments/qwen_curriculum/resume_control.py status
$PY AdaptVPR/experiments/qwen_curriculum/resume_control.py pause
$PY AdaptVPR/experiments/qwen_curriculum/resume_control.py resume

# 按变体测量难度
$PY salad/probe_synthetic_difficulty.py \
  --manifest outputs/weather_v3/state/training_manifest.jsonl \
  --output outputs/qwen_curriculum/state/analysis/difficulty_probe_v3
```

单张约 25 s（pilot 实测 100 张 46 分钟），实际耗时以当前生成日志为准。暂停后可使用 `night_first_v3.py check` 检查已保存输出和恢复状态，不调用 Qwen。旧 pilot CLI 已移除，当前计划统一使用上述入口。

自动审核仍可能漏掉车辆新增和局部拼接缺陷，`accepted` 不代表人工确认无缺陷。后续完成质量复查并确定训练快照后，才将 `outputs/weather_v3/state/training_manifest.jsonl` 加入新的训练配置；旧阶段配置和历史训练结果保持独立。
