# 使用当前 13,710 张 accepted 图片微调

此文记录第一轮固定数据实验。后续新增图片请使用
[Qwen 通用微调入口](qwen_finetuning.md)，不再复制带图片数量的配置。

配置：`salad/configs/ccr_accepted_13710_4090_48g.json`。
已准备独立数据快照和验证划分；没有启动校准、训练或 GPU 评估。

## 直接运行

```bash
cd /home/admin123/github/WORKSPACE
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_accepted_13710_4090_48g.json --execute
```

中断后重复同一命令。命令依次执行小规模教师校准、初始评估和 A/M/B0/B4/B5
五组微调；每组 3 个完整地点 epoch、4059 次更新，五组共 20295 次更新。
各组从同一个 `outputs/coverage_run/checkpoint_epoch_050.pt` 开始，不串接权重。
**检索评估仅使用 SVOX**：每轮 SVOX val；初始模型和每组最终 checkpoint
分别报告 SVOX test 的全部实际类别。所有任务串行执行。

| 结果名称 | split | query 目录 |
|---|---|---|
| svox_val | val | queries |
| svox_default | test | queries |
| svox_night | test | queries_night |
| svox_rain | test | queries_rain |
| svox_snow | test | queries_snow |
| svox_sun | test | queries_sun |
| svox_overcast | test | queries_overcast |

本地没有 `queries_fog`，不输出虚构的 svox_fog。默认 queries 按目录名标记
为 svox_default，不推断其天气，也不当作其他类别的汇总。
每个类别独立保存 R@1/R@5/R@10 和 query 正例排名，`report.md` 也逐类展示。
同一 test gallery 只提取一次描述子，随后逐类别读取 query、检索并释放其描述子，
避免六次重复提取 gallery，且不同时积累所有类别的 query 描述子。

天气类别仅存在于本地 test/train，不在 val。test 只用于固定预算的初始/最终
结果报告，不用于中途挑 checkpoint、调参数或自动通过/失败判断。
关闭代理验证后，报告给出描述性对比，不套用原先的“代理天气增益”门槛。

只重新汇总已有结果：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_accepted_13710_4090_48g.json --summarize-only
```

## 数据范围

2026-10-19 核验 `outputs/qwen_curriculum/images/accepted/` 中共有 **13,710**
个有效图片入口，全部与当前 manifest 的合格记录一一对应，有精确真实源图映射。
accepted 链接指向 `images/train/` 的训练尺寸图片，因此 manifest 保留解析后的
实际文件路径，读取的就是这批 accepted 图片；不另行生成或复制图像。

| 天气 | accepted 数量 |
|---|---:|
| Night | 7700 |
| Rain | 1254 |
| Snow | 2724 |
| Fog | 2032 |

全部 **23 城市**参与，不再仅限旧 pilot 的四城市。按地点隔离后：

- 训练：10,818 个地点、13,505 张生成图片。
- 校准：12 个独立地点，每天气 3 个。
- 验证：128 个独立地点，每天气 32 个 query，共用 128 张其他真实视角 gallery。
- 校准和验证地点的全部生成版本共 205 张，从训练池整体排除；不是只排除选中的那一张。

真实源图、生成版本和地点标签均来自原 manifest，不用同地点其他真实图伪造精确配对。
生成源图不放入代理验证 gallery。初始权重曾在 GSV 训练，本次地点隔离不能消除
其既有训练曝光。当前只执行 SVOX 检索评估；此前预留的 128 个地点继续留出，
以保持训练池、4059 步预算和数据隔离不变，其 GSV/生成图代理评估文件不会运行。
局部损失需要的教师校准仍使用独立校准地点，这是训练目标准备，不是检索评估。

`salad/outputs/ccr_accepted_13710_inputs/accepted_manifest.jsonl` 是冻结的全量
accepted manifest 快照；同目录 `receipt.json` 记录原 manifest 摘要、天气统计、
城市和 accepted 链接映射。后续生成任务往活动 manifest 增加记录不会改变本实验。
此快照不复制图片，已有图片内容仍须保持不变。

## 微调策略

| 组 | 方法 |
|---|---|
| A | 相同地点池的纯真实主损失 |
| M | 原有 mix，目标生成比例 0.5 |
| B0 | 纯真实主损失 + 非对称全局生成辅助损失 |
| B4 | B0 + Feature + Conditional OT，有效对应均匀加权 |
| B5 | B4 改为有效对应的可信度 q 加权 |

保持 P=8、K=4、coverage_pk、学习率 1e-6、seed=42、一个可训练 Backbone block。
全局/Feature/OT 启用时权重均为 0.1，Score 权重为 0，不启用 Reliability。
冻结教师使用相同完整 checkpoint；执行时为该教师在独立地点重新校准。

每个完整 epoch 访问全部训练地点一次，每次选 K 张不同真实图；这里每 epoch
1353 个主批次，不再截断为旧 pilot 的 100 批。辅助只从当次 K 张真实图的精确
合格生成版本中选，每个可用地点以 p_exp=0.5 最多触发一张，实际触发率另行记录。
全量图片进入候选池不等于三轮内每张生成图都会被使用，也不表示一个 epoch
覆盖了每个地点的所有真实图。没有为了增加生成曝光而改变精确配对约束。
当前天气数量不均衡，保留现有采样策略，通过日志检查各天气实际曝光。

沿用 4090 48GB / 32GB 专用配置：BF16 学生、FP32 教师/评估、4 workers、
每 worker 预取 1 批、评估 batch=64、双图教师前向、2GiB 描述子 GPU 缓存。
PyTorch 显存预算最多 40GiB；实验进程树 RSS 超过 20GiB 或系统可用内存
低于 3GiB 时停止并保留恢复点。保护范围见 [资源配置说明](ccr_resources_4090.md)。

输出目录：`salad/outputs/ccr_accepted_13710_svox_categories/`。
重点查看 `report.md`、`report.json`、`learning_curves.csv`、`training_metrics.csv`。
比较 B0−A、B4−B0、B5−B4 和 B5−A 的各类别结果。SVOX val 用于训练过程检查，
最终 test 结果只做报告；如据此修改方法，应另用独立验证数据，不能反复对同一
test 调参。当前权重和三轮预算是尝试起点，不保证有检索增益。
