# 真实基线遗忘排查：独立 A 对照入口

本轮把问题收敛为：在相同初始化、地点池和更新预算下，增强及真实描述子保持
是否能减少继续训练造成的能力损失。代码不默认加入生成图、InfoNCE、局部蒸馏、
OT 或 Reliability。Stage 4 的共享训练源码、冻结配置和已有结果不修改。

## 对现有分析的修正

- 纯真实 A 已经出现历史天气退化，因此生成数据不是所有退化的必要原因。
  但这不等于排除生成图自身的问题；仍应保留 initial、A、M 三方比较。
- 仓库 `main.py` / `GSVCitiesDataloader.py` 确实配置了
  `Resize → RandAugment(num_ops=3, BILINEAR) → ToTensor → Normalize`。
  这说明训练配方有差异，不能仅凭当前脚本证明下载 checkpoint 的真实训练历史，
  也不能先验认定 RandAugment 就能恢复天气能力。
- 相同步数不代表相同训练剂量：P60 和 P32 的图像曝光及负样本池不同。
  新对照保持 P/LR/block 不变，直接比较预声明更新预算。
- 同一批 query 在不同 seed 下不独立，禁止合并三 seed 的 query 计数当成
  三倍独立样本做 McNemar。新报告逐 seed 配对，另列跨 seed 增量。
- 合成天气代理可用于合成域诊断，不能替代独立真实天气 validation，
  不能据代理分数或普通 val 持平宣称天气鲁棒性恢复。
- 初始学生/教师完全一致时，真实描述子保持项应为零。它是偏移后的恢复约束，
  不是保证每步都有有效训练信号的来源。

## 四个独立对照

| case | 增强 | 真实描述子保持权重 |
|---|---|---:|
| weak | 原弱增强操作 | 0 |
| randaugment | 仓库 RandAugment 配方 | 0 |
| weak_retention | 原弱增强操作 | 0.1 |
| randaugment_retention | 仓库 RandAugment 配方 | 0.1 |

0.1 是研究起点，不是已验证最佳值。四组都从冻结初始 checkpoint 独立开始，
不会用 weak 训练后的模型初始化其他组。

使用 Stage 3 冻结的 P32 / LR1e-6 / 最后1个 block、真实排除清单、
官方 dino_salad 初始权重、BF16。Stage 3 父计划和配置保持原样。
保持这些参数是为了首先隔离增强与保持项的效应；本轮不同时改变 block、
backbone/aggregator 分组 LR，也不默认仅训练聚合器。

RandAugment 明确 num_ops=3、magnitude=9（仓库未传 magnitude 时的默认值）、
bilinear；不额外叠加原弱增强。记录 torchvision/PyTorch 版本。

两种增强都采用以 seed、真实 coverage cycle、city/place 为键的私有 CPU RNG。
增强不消耗学生 Dropout RNG；教师构造也保存/恢复 Torch CPU/CUDA RNG。
这保证保持项对照的随机轨迹更可比，但意味着 weak 是使用原操作的**新受控基线**，
不承诺逐位复现旧训练脚本的历史弱增强序列。

真实保持项复用 `workflow/real_retention.py`：
当前相同增强输入、初始冻结教师、学生 eval 保留梯度、FP32、
按8张微批次累积，与 MS 一次共同 optimizer.step。
eval 并不关闭学生梯度；教师没有梯度。不是 Feature/OT 蒸馏。

## 预算和恢复

固定总 schedule 为1000次更新，每100步一个 checkpoint window。
第500步和第1000步是预声明的观察点；不是看结果后挑出的最佳checkpoint。
第500步不会把LR提前衰减到零，继续至1000时也不会重新启动scheduler。
这是同一轨迹上“是否继续训练”的对照，不是两种独立重新调度的训练方案。

训练日志的 epoch 字段在本入口中表示 checkpoint window，**不代表完整数据epoch**。
窗口采样连续切分原全量池 coverage 顺序，下一窗口接着上一次的位置；
跨完整池边界后才切换真实 coverage cycle。不会每100步重新取前100个batch。
全部地点都有机会被采样，但500/1000步预算不意味着已经遍历全部地点。
真实采样、覆盖统计、optimizer、scheduler、scaler、RNG都随窗口保存。
窗口内中断会从上一个完整100步窗口重放。

默认 workers=0、沿用3 GiB系统可用内存保护和父计划GPU预算。
保持项需要额外前后向，会比纯MS慢；真实耗时、GPU峰值在训练日志中记录，
CPU进程树内存记录在 train.log 的 RESOURCE 行。不预设完整运行能占用多少显存。
若 Stage 4 仍占用GPU，请等它结束后再手动启动本实验。

## 运行命令（fish）

```fish
set PY /home/admin123/miniconda3/envs/AdaptVPR/bin/python
set RUN salad/run_real_recovery.py

# 默认仅预检查/冻结新计划，不加载GPU模型
"$PY" "$RUN"

# 只打印命令，不训练
"$PY" "$RUN" --phase run --case weak --seed 42 --until 500

# 第一对：先只比较增强；任一步失败则停止
"$PY" "$RUN" --phase run --case weak --seed 42 --until 500 --execute
and "$PY" "$RUN" --phase run --case randaugment --seed 42 --until 500 --execute

# 第二对：独立检查保持项，不混同增强变化
"$PY" "$RUN" --phase run --case weak_retention --seed 42 --until 500 --execute
and "$PY" "$RUN" --phase run --case randaugment_retention --seed 42 --until 500 --execute

"$PY" "$RUN" --phase report
```

以上包含手动授权的训练及普通 SVOX val 评估。不会读取 SVOX test 或自动触发
整个四组/三seed训练。

如继续预声明的1000步观察点，对四个case分别执行：

```fish
"$PY" "$RUN" --phase run --case weak --seed 42 --until 1000 --execute
```

将 `weak` 替换为其他case即可。它自动恢复同组 checkpoint；同一命令重跑也能恢复，
无需手改 `--resume`。保留500步checkpoint，并对500/1000预算分别记录结果。
其他可用种子继承43、44、45，只有显式指定才会运行；42作为pilot，跨seed分布
只汇总43/44/45。不存在根据已有test结果自动启动确认实验的机制。

输出目录：
`outputs/qwen_curriculum/state/finetuning/real_recovery_v1`

## 如何读报告

报告包含每组两个固定预算的R@1/5/10、相对initial的配对改善/退化、
相同seed下四个预声明对照、Holm校正、各seed增量及其分布。
不会合并seed-query做显著性检验，不选“最佳epoch”。
audit包含checkpoint SHA256、配置、窗口训练日志、MS/总loss、保持loss、
描述子cosine、mining、裁剪、真实曝光/覆盖及训练性能。

1. `randaugment − weak`：是否存在增强配方效应。
2. `weak_retention − weak`：弱增强下保持项的独立作用。
3. `randaugment_retention − randaugment`：强增强下保持项是否仍有增量。
4. 同一case的500/1000步：继续更新后是否出现指标或保持loss的恶化。
5. 相对initial：能否超过或保留已有能力，不能只看比退化的A好多少。

目前没有独立真实天气validation，报告会始终明确：
`unverified_missing_independent_natural_weather_validation`。
即使所有普通val分数都提高，也不自动宣布“天气不退化的A已建立”。
需另行提供按地点/轨迹隔离的真实天气验证数据并预声明协议，不能把已经反复
查看的SVOX test或合成代理重新命名为真实天气validation。

在这个证据缺口补齐前，不默认提高M到25–30%实际生成曝光：当前稀疏覆盖下
这会涉及地点重采样、减少真实曝光或增加总预算，必须另立公平对照。
