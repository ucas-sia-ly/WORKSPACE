# CCR-VPR 可行性实验运行入口

入口为 `salad/run_ccr_feasibility.py`，配置为
`salad/configs/ccr_feasibility_pilot.json`。先验证生成辅助和局部蒸馏有没有检索收益，
再决定是否投入 Reliability 实验。默认不启用 Reliability，不重新生成图像。

**4090 48GB 显存 / 32GB 内存机器优先使用
[专用性能与资源配置](ccr_resources_4090.md)**，其输出目录独立于本页原始 pilot。

## 运行与恢复

所有命令从工作区根目录执行，使用现有 AdaptVPR 环境。

```bash
cd /home/admin123/github/WORKSPACE

# 只检查数据、读取 checkpoint 元数据、冻结划分和命令；不启动 GPU 或训练。
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_feasibility_pilot.json

# GPU 空闲后显式执行：校准 → 初始评估 → 五组串行训练与评估 → 汇总。
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_feasibility_pilot.json --execute

# 中断后继续：再次执行上面同一条 --execute 命令。

# 仅根据已经产生的结果重新汇总，不调用模型。
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_feasibility_pilot.json --summarize-only
```

`--execute` 是有界 pilot，不是两步 smoke：默认五组各 **500 次 optimizer 更新**，
总计 2500 次；每组每 100 步暂停、评估、从完整 checkpoint 恢复，保持同一个完整
cosine schedule。每组都从同一初始权重开始，不从上一组继续训练。

仅想先检查本机 GPU 执行链路时，复制配置到同一 `salad/configs/` 目录，
将 `output_dir` 改为一个新的目录、`steps` 改为 2、`evaluation_interval` 改为 1，
再对新配置执行上述命令。其完整 SVOX 评估仍需时间；两步结果只验证执行链路，
不能判断方法有效。不得把 pilot 和 smoke 写入同一目录。

当前默认数据路径使用整理后的 `outputs/qwen_curriculum/state/training_manifest.jsonl`，
初始模型和教师均为 `outputs/coverage_run/checkpoint_epoch_050.pt`。
旧 `generation_1000` 和旧 Stage 2A 教师路径不再作为默认值。
这是已训练的完整 SALAD checkpoint；预检查验证架构、维度、预处理并记录 SHA256。

## 对照与公平性

| 组 | 主批次 | 全局生成损失 | Feature | Score | Conditional OT | 局部 q |
|---|---|---:|---:|---:|---:|---|
| A | 全真实 | 0 | 0 | 0 | 0 | 无 |
| M | 原有 mix，目标比例 0.5 | 0 | 0 | 0 | 0 | 无 |
| B0 | 全真实 | 0.1 | 0 | 0 | 0 | 无 |
| B4 | 全真实 | 0.1 | 0.1 | **0** | 0.1 | 有效 A 对应上 uniform |
| B5 | 全真实 | 0.1 | 0.1 | **0** | 0.1 | 同一支持集上 verified |

权重只是预先声明的 pilot 起点，不是已验证最优值。B4/B5 的区别仅为有效对应上的
置信度权重；uniform 不会把 Unknown 或明确破坏区域强制变成 Positive。
verified 是证据加权策略名称，不代表人工真值或校准后的正确概率。

默认 P=8、K=4、coverage_pk、seed=42、学习率 1e-6、可训练 Backbone block=1，
保留主图随机增强。B0/B4/B5 的生成触发概率 p_exp=0.5，实际曝光另行记录。
这不是最终生成图占比；M 又改变了主批次组成，因此 M 对照不是单纯的损失消融。
所有组使用相同地点池、初始化、更新预算和评估协议。只选有合格生成图的地点池，
A 仍只读取真实图作训练输入。局部教师/学生配对使用固定 Resize。

## 数据划分与校准

默认选择 Bangkok、BuenosAires、LosAngeles、Medellin 四城市。
按固定 partition_seed 和图像路径散列选取地点：

- 每天气 3 个校准地点，共 12 个，四天气之间也不重叠。
- 每天气 8 个验证地点，共 32 个，与校准及本次训练地点均不重叠。
- 其余合格地点进入训练；实际数量记录在 `plan.json`，由 manifest 决定。
- 代理验证共用 32 张真实 gallery，每地点一张其他视角图片；精确生成源图不进入 gallery。
  每天气 8 张生成 query，另有 32 张真实源图 query 用于 real_holdout 对照。

默认 `pair_calibration=null`，执行时先用当前教师重新校准。不同地点真实/生成图
构成 null pairs，同图复制构成 positive controls；调用现有 Stage 2B 校准器，
radius=1、显式 tail_probability=0.01。阈值来自校准数据，不在验证集上调阈值。
12 对控制是小规模 pilot 校准，Patch 之间有相关性，不应把尾概率解读为真实错误率。
更换教师、网格、预处理或校准策略需要新实验目录。

几何证据复用现有 `salad/outputs/ccr_stage2b_diagnostics/geometry`，按图像 RGB 哈希
检查命中；没有缓存的图像依照已有教师结构规则处理，不把失配当作 Negative。
运行期间不启动 LightGlue，也不保存大型教师特征缓存。

注意初始 epoch_050 模型曾在 GSV 训练；这里的地点隔离只保证本次微调的隔离，
不意味着初始模型从未见过这些地点。因此额外配置 **SVOX val** 真实验证，
并且不使用 test 选择参数或 checkpoint。

## 评估与结论

每 100 步评估 real_holdout 和 Night/Rain/Snow/Fog 代理集合。
默认完整 SVOX val 有 17226 张 gallery、14698 张 query，开销较大，只在初始模型
和每组最终模型上评估，共六次。时间、显存以本机日志实测为准，不预设数值。
`external_validations[].cadence` 可显式设为 `every`，但要在新实验开始前确定。

报告固定比较 **最终第 500 步**，不挑中间峰值：

| 比较 | 回答的问题 |
|---|---|
| M − A | 原有混合训练是否有收益？ |
| B0 − A | 非对称全局生成辅助是否有收益？ |
| B4 − B0 | Feature + Conditional OT 是否提供增量？ |
| B5 − B4 | 同一可信对应支持集上的 q 加权是否提供增量？ |
| B5 − A | 完整候选方法相对真实基线是否有收益？ |

默认判定门槛预先设为真实验证平均 R@1 不下降（max_real_drop_pp=0），
四天气代理平均 R@1 增量严格大于 0（minimum_weather_gain_pp=0）。
所有域、所有配置种子齐全后才给出候选信号。报告还列出每天气、每种子的增量、
跨种子标准差和 query 从错到对/从对到错的计数。
门槛是实验决策规则，不是统计显著性声明；真实域按每个域分别检查跨种子均值。

默认每天气只有 8 个 query，单天气 R@1 的最小变化是 12.5 个百分点。
单种子没有跨种子方差。小规模无增益不证明方法永远无效；同样，Loss 下降、
余弦变高或代理验证改善，都不足以证明自然天气检索收益。

若 B4/B5 有稳定的候选信号，复制配置到同一目录，设置新的 output_dir、
`seeds=[42,43,44]`，保持其他参数和固定划分不变，并扩大预先冻结的验证集。
扩大验证集意味着新实验划分，不能继续旧实验。若要加入独立自然天气验证，
提供真实采集数据的 evaluation manifest，例如在 external_validations 中增加：

```json
{
  "name": "natural_night_val",
  "kind": "natural_weather",
  "dataset": "manifest",
  "root": "../../dataset/my_validation",
  "manifest": "../../dataset/my_validation/night_val.json",
  "role": "validation",
  "split": "val",
  "cadence": "final"
}
```

manifest 使用现有评估格式：`references=[{id,path}]`、
`queries=[{id,path,positives:[reference_id]}]`。必须自行确认它是真实天气、
与训练隔离的 validation，而不是生成图或测试集；字段名本身不能证明隔离。
所有配置中的相对路径均相对于配置文件所在目录。

如果有足够曝光却始终没有有效 Positive，先检查新教师校准、对应和几何缓存，
不要用降低阈值或伪造 Negative 的方式强行提高覆盖率。
如果局部 Loss 下降而 B4/B5 检索不改善，应记录为本配置未显示收益，
暂缓增加 Reliability。Reliability 的后续验收仍需空间差异和检索收益。

## 输出与故障恢复

默认输出到 `salad/outputs/ccr_feasibility_pilot/`：

| 文件 | 内容 |
|---|---|
| plan.json | 冻结配置、数据划分、预算、代码/输入 SHA256、评估图片路径/大小/mtime 清单摘要 |
| train_manifest.jsonl、validation_*.json | 独立训练/验证输入 |
| calibration_pairs.json、calibration.json、calibration.receipt.json | 独立校准对、操作点及校验摘要 |
| first_epoch_commands.txt | 各组第一轮完整训练命令；后续由入口自动恢复 |
| status.json | 执行状态、失败原因、恢复提示 |
| initial/ | 初始 checkpoint 各域评估及日志 |
| runs/组/seed/train.log | 完整子进程命令、训练日志与异常 |
| runs/组/seed/checkpoint.pt | 最近完成的 epoch，含 optimizer/scaler/RNG/scheduler 状态 |
| runs/组/seed/metrics_step_*.json | 每个区间损失、真实/生成曝光、局部监督比例、余弦/KL、实测耗时和显存 |
| runs/组/seed/evaluations/step/ | Recall、每 query 正例排名、日志和结果/checkpoint 摘要 |
| training_metrics.csv | 展平的逐区间训练指标，保留原字段名和统计范围 |
| learning_curves.csv | 各域逐步 Recall、相对初始模型增量和 query 改善/退化计数 |
| report.json、report.md | 固定最终预算的配对消融结果和结论限制 |

耗时中 `runtime.step_seconds` 的范围是主输入传输完成后至 scaler.update，
不含全部数据加载和进程启动开销；epoch seconds 可辅助观察总开销。
指标是训练日志中的区间统计，不能把未启用项或无有效监督产生的零当作训练成功。

中断在完整 epoch 后恢复；未完成 epoch 从上一 checkpoint 重跑，不进行近似的
batch 中途恢复。已完成且摘要匹配的评估复用。日志追加保存，目录加互斥锁。
固定预算模式下 AMP 跳过 optimizer 更新会报错，避免把跳过步骤计入预算；
若重复发生，另开新目录改用 BF16/FP32，不能改变同一实验的精度继续比较。

默认每组保留最新 checkpoint 和最终编号 checkpoint，不保存每个评估点权重，
但保留每点评估结果和对应 SHA256。恢复前会检查配置、源代码、输入和数据清单；
manifest 被持续生成任务更新时也会拒绝沿用旧目录。可先复制一份稳定 manifest，
在新配置中引用它并选择新输出目录。不要删除 plan.json 绕过已开始实验的检查。
图片清单摘要使用大小和 mtime，不是对所有真实图片逐像素哈希。

本次交付只做 CPU 回归和预检查，未执行这五组 GPU 实验。检查命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=salad:salad/tests \
  /home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  -m unittest discover -s salad/tests -v
```
