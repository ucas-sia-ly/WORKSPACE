# 阶段二：干净的全量真实 A 基线

入口为 `salad/run_real_baseline.py`，配置为 `salad/configs/real_baseline.json`。默认只准备计划；GPU 训练和评估均须显式指定 `--execute`。所有命令从工作区根目录执行。

初始权重固定为 `salad/checkpoint/dino_salad.ckpt`，SHA256 为 `6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada`。它与此前严重饱和的 `checkpoint_epoch_050.pt` 不同：现有 `ccr_followup_dino_am_fullreal` 的 A/42 三轮日志显示，有效 anchor 为约 9.8%–12.1%，空 mining batch 为约 0.26%–0.36%。其 SVOX val R@1 从 96.360% 到 96.340%，单 seed 基本持平；这不是已经得到稳定 A 的证明，也不能把旧 checkpoint 的零梯度结论直接套用到新初始权重上。

## 固定真实数据与排除地点

- 使用 GSV-Cities 全部城市中至少有 K=4 张真实图的地点，不使用 `synthetic-places-only`。
- `configs/splits/ccr_reserved_places.json` 固定保留 128 个 validation 地点与 12 个 calibration 地点。它们来自已有实验的冻结划分；每个地点的全部真实视角都排除。
- 运行时写出可直接供训练使用的 `excluded_places.json`，以及逐地点列明原因和图片数的 `exclusion_report.json`。重复、重叠或不存在的地点会报错。
- 清单不随 Qwen 新图片到达而变化。后续 M/辅助方法必须沿用相同清单和数据范围，才可与本 A 公平比较。
- 这些排除只约束本轮微调，不代表预训练从未见过这些地点，也不证明 GSV 与 SVOX 没有地理重叠。

本次元数据检查：排除 140 个地点的 1,203 张图片，保留 62,374 个地点、523,498 张真实图片；每轮曝光 249,496 张。P16/P32/P64 每轮分别为 3,899/1,950/975 次更新。

主训练只有原始 MS Loss，保留原 augmentation 与 miner margin=0.1。没有生成 manifest、教师、描述子保持、局部蒸馏或 Reliability。使用 `coverage_pk`，每轮每地点访问一次、取 4 张不同真实图，保留地点完整图片池。默认采样模式本身不作全局修改。

## 筛选与确认

| 候选 | P | LR | 解冻 blocks |
|---|---:|---:|---:|
| 中心配置 | 32 | 1e-5 | 2 |
| 较小 P | 16 | 1e-5 | 2 |
| 较大 P | 64 | 1e-5 | 2 |
| 较小 LR | 32 | 3e-6 | 2 |
| 较大 LR | 32 | 3e-5 | 2 |
| 较少 blocks | 32 | 1e-5 | 1 |
| 较多 blocks | 32 | 1e-5 | 4 |

这是一组单因素筛选，不是穷举所有交互组合。K=4、weight decay=0、初始权重与划分固定。每个候选在 seed=42 上训练一个完整 epoch，使用预设三轮 cosine schedule 的第一轮。所有候选的真实图片曝光相等，但 P 改变了负样本池和更新次数，不能称为相同更新预算。

完成全部七个候选后，只从通过训练信号与 SVOX val 检查的配置中，选择 val R@1 最高者；精确并列按配置顺序选择。然后从同一初始 checkpoint **重新开始**，用独立 seeds=43/44/45 各训练完整三轮，固定比较最后一轮，不挑最佳 epoch 或最佳 test seed。总预算为七个筛选 epoch 加九个确认 epoch。

预先设置的工程验收条件如下，均不是统计证明或效果保证：

- 每轮真实曝光数必须完整，生成曝光为 0，MS loss 有限且大于 0。
- 每轮有效 anchor 比例至少 5%，空 mining batch 不超过 10%，真实正负 mined pair 均非零。
- 筛选及每个确认 seed 的最终 SVOX val R@1，相对初始值下降不超过 0.1 个百分点。
- 三个确认 seed 的最终 R@1 样本标准差不超过 0.1 个百分点。

mining 是训练信号指标，不能替代每个参数的梯度检查。报告还保存 loss、mined pair、裁剪触发、每步时间及显存等原始训练指标。若全部候选失败或确认不稳定，明确报告失败，不自动调宽门槛，不写出通过的 `baseline.json`。若要修改协议，使用新输出目录，避免结果出来后改验收标准。

SVOX test 不参与选择。确认通过后才分别评估 `svox_default`、`svox_night`、`svox_rain`、`svox_snow`、`svox_sun`、`svox_overcast`，展示初始模型及全部三个 seed；共享 gallery 提取，保留逐 query 结果。数据没有 Fog 测试类别。

## 手动运行

先让当前 A/M 作业完成，避免两个实验同时竞争 GPU 和 32GB 主机内存。准备计划不需要 GPU：

```bash
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
"$PY" salad/run_real_baseline.py
```

第一步：串行筛选七个候选，并评估初始权重和各候选的 SVOX val：

```bash
"$PY" salad/run_real_baseline.py --phase screen --execute
```

也可先只运行中心配置，查看资源和训练信号，再运行上面的完整筛选命令；已完成的任务会复用：

```bash
"$PY" salad/run_real_baseline.py --phase screen --candidate p32_lr1e5_b2 --execute
```

第二步：筛选通过后，串行进行三个独立 seed 的完整确认：

```bash
"$PY" salad/run_real_baseline.py --phase confirm --execute
```

第三步：确认通过后，输出六个 SVOX test 类别的独立结果：

```bash
"$PY" salad/run_real_baseline.py --phase test --execute
```

只刷新报告，不启动 GPU：

```bash
"$PY" salad/run_real_baseline.py --phase report
```

同一命令可在中断后重试。按完整 epoch 恢复 optimizer、scaler、随机状态和原三轮 schedule；未保存的半轮重新执行。输入 CSV、排除清单、checkpoint、源代码与验证集清单受到冻结检查，评估缓存校验对应 checkpoint 和结果哈希。

默认输出目录为 `outputs/qwen_curriculum/state/finetuning/real_baseline_v1/`，包含 `plan.json`、排除清单、`report.md/json`、screen/confirm 日志与 checkpoint，以及通过后生成的 `baseline.json`。此目录属于真实基线实验，不依赖 Qwen 图片数量。`screen_commands.txt` 仅供审阅底层命令；建议使用上述入口，以保留资源监控、评估和验收。

## 资源与验证范围

采用 BF16、4 workers、prefetch=1、4 CPU threads；PyTorch 显存预算上限 40GiB，并按当前空闲显存预留 4GiB；进程树 RSS 上限 20GiB，系统至少保留 3GiB available。主机监控是采样保护，CUDA 限制只覆盖 PyTorch allocator，不是系统硬隔离。触发保护会停止本实验子进程，保留最后完整 epoch，不能保证共享机器上绝不发生资源争用。P64 的实际峰值仍以运行日志为准。

开发验证仅使用临时 CPU 小模型，覆盖排除清单、纯真实命令、冻结协议、评估缓存、筛选/确认门槛和逐 epoch 恢复；没有代替真实 GPU 训练结果。
