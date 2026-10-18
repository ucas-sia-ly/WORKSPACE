# 无状态 Coverage PK Sampling

所有命令从工作区根目录执行。此功能的回归测试已编写，尚未运行。

## 采样约定

- `--real-sampling random_pk` 是默认值，保留原来的随机采样、随机数调用顺序和 batch 尾部处理。
- `--real-sampling coverage_pk` 保留每个地点完整的真实图池 `R_i`。`--images-per-place K` 是每次选取数量，不会将池裁剪为 K 张。
- 没有合成图时，每轮每地点默认访问一次，每次选 K 张不同真实图。地点必须至少有 `--min-images-per-place` 张真实图，且该下限不小于 K。
- 不要求一个 epoch 覆盖某地点全部图片。对于连续完整 epoch，前 `ceil(|R_i| / K)` 轮可以覆盖该地点整个真实图池。
- 图片池按文件名排序；置换由 SHA-256 派生的 seed、city、place_id、cycle 确定，不使用 Python `hash()` 或 worker 内部轮换队列。
- 第 e 轮（内部从 0 开始）读取置换流的 `[e*K, (e+1)*K)`。跨边界时，将与上一置换尾段重复的身份移到新置换后部，保证组内唯一，并保留新置换全部图片。
- 直接重建任意 epoch 时，从最近的 `lcm(|R_i|, K)` 对齐边界开始，最多回溯 `K/gcd(|R_i|, K)` 个置换，不需要从第 0 轮重放。
- sampler 把 `(epoch, place_index)` 发送到 Dataset，适用于 `num_workers=0/2/4` 和 persistent workers。直接调用 Dataset 时可以使用 `set_epoch(e)`。
- coverage 模式不会丢弃最后一个地点：最后的单地点 batch 从前一批借一个地点；P=2 时合并成一个三地点 batch。

若启用已有合成图功能，先选 K 个真实源身份，再应用原有 mix/replace 规则；总训练图数仍为 K，实际真实图曝光可以少于 K。真实图覆盖保证针对真实源选择顺序；合成替换后的实际覆盖率按实际真实图统计。Reliability 的辅助源图不计入 metric 训练图曝光。

`--max-batches-per-epoch` 会截断本轮访问。采样仍按 epoch 前进，不补回未访问地点；统计只记录已消费 batch，不包含 worker 预取但未用于训练的图片。

## 训练与恢复

保留现有训练命令的模型、数据和优化参数，新增 `--real-sampling coverage_pk` 即可。以下为真实图训练示例，请替换数据和输出路径：

```bash
python salad/train_salad.py \
  --real-data /path/to/GSV-Cities \
  --output-dir /path/to/coverage_run \
  --epochs 50 --batch-size 32 \
  --images-per-place 4 --min-images-per-place 4 \
  --real-sampling coverage_pk --seed 42 --num-workers 4
```

从已完成第 10 轮的 checkpoint 继续，`--epochs` 仍为原定总轮数 50：

```bash
python salad/train_salad.py \
  --real-data /path/to/GSV-Cities \
  --output-dir /path/to/coverage_resumed \
  --epochs 50 --batch-size 32 \
  --images-per-place 4 --min-images-per-place 4 \
  --real-sampling coverage_pk --seed 42 --num-workers 2 \
  --resume /path/to/coverage_run/checkpoint_epoch_010.pt
```

同一数据、seed、K 下，恢复后的图片身份与连续训练一致；coverage resume 允许改变 num_workers。此保证针对采样身份和地点顺序，不保证改变 worker 数后随机数据增强或模型数值完全相同。

resume 保留原采样模式。旧 checkpoint 缺少该字段时视为 `random_pk`。如要将旧 random 训练切换为 coverage 新实验，使用 `--init-checkpoint` 初始化权重，开启新输出目录；这会重新开始 epoch 和优化器。不要使用 `--resume` 切换采样模式。

## 统计文件

`training_log.jsonl` 的 `sampling` 包含概要；`sampling_coverage.jsonl` 每轮额外包含 `places` 列表，通过 city/place_id 标识每个地点。

| 字段 | 含义 |
|---|---|
| `epoch_unique_real_images` | 本轮不同真实图曝光数 |
| `epoch_new_real_images` | 本轮首次出现在累计记录中的真实图数 |
| `epoch_repeated_real_exposures` | 本轮曝光次数减去本轮不同真实图数 |
| `epoch_previously_seen_real_exposures` | 本轮曝光次数减去本轮首次曝光数，包含历史重复及本轮内部重复 |
| `cumulative_real_coverage` | 累计不同真实图数 / 所有入选地点完整真实图池大小，范围 0–1 |
| `cumulative_repeated_real_exposures` | 累计真实图曝光次数减去累计不同真实图数 |
| `epoch_visits` / `cumulative_visits` | 每地点本轮 / 累计访问次数 |
| `epoch_sampling_frequency` / `cumulative_sampling_frequency` | 每地点本轮 / 累计访问次数占对应全部地点访问次数的比例 |
| `statistics_since_epoch` | 统计起始轮次，从 1 开始 |

checkpoint 的 `sampling_coverage` 保存累计曝光标记与访问次数，resume 直接恢复。它仅用于计数，不影响采样。旧 checkpoint 无曝光记录时，从恢复后的第一轮开始统计，并明确标注起始轮次，不推测历史曝光。

## 稍后执行的验证命令

以下命令仅供手动执行，没有在本次修改中运行。新测试只使用临时小图片和 CPU，不加载模型或下载权重：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=salad python -m unittest discover \
  -s salad/tests -p 'test_coverage_sampling.py' -v
```

测试覆盖不同 N/K 的置换边界、完整图片池、随机模式兼容、每地点一次访问、0/2/4 workers、persistent workers、直接跳到指定 epoch、统计 checkpoint 恢复。

完整现有回归套件（等当前训练结束后再运行）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=salad python -m unittest discover -s salad/tests -v
```
