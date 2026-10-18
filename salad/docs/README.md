# SALAD 文档索引

所有项目训练和评估命令均从工作区根目录执行。

| 文档 | 内容 |
|---|---|
| [训练与评估](workflow.md) | 原生 PyTorch 入口、真实图与生成图采样、checkpoint、恢复与评估协议 |
| [Coverage PK Sampling](coverage_sampling.md) | 无状态真实图覆盖采样、epoch 与 worker 传递、恢复及曝光统计 |
| [CCR-VPR Stage 1](ccr_vpr_stage1.md) | 真实主批次、精确源图生成辅助微批次、非对称全局损失、短程 A/B/C 命令 |
| [CCR-VPR Stage 2A](ccr_vpr_stage2a.md) | SALAD 内部特征、冻结在线教师、分块前向、实测显存与 OT 回归 |
| [CCR-VPR Stage 2B](ccr_vpr_stage2b.md) | 真实/生成 Patch 对应、A/B 候选、明确负证据、几何缓存与四天气诊断 |
| [Reliability OT](reliability_ot.md) | 可选 patch 可靠性分支、dustbin 偏置、弱监督、单图推理与 v1 兼容性 |
| [Reliability OT v2](reliability_v2.md) | 固定教师缓存、局部几何负监督、轻量上下文与消融设置 |
| [原版 SALAD](../README.md) | 原版环境、数据集、Lightning 训练与论文引用 |

## 代码入口

- [train_salad.py](../train_salad.py)：项目训练入口；可靠性分支默认关闭。
- [evaluate_salad.py](../evaluate_salad.py)：项目 checkpoint 评估入口。
- [cache_reliability_targets.py](../cache_reliability_targets.py)：离线构建可靠性目标缓存。
- [run_finetune_comparison.py](../run_finetune_comparison.py)：串行完成微调前后的评估对照。
- [tests/](../tests/)：训练、评估、采样、checkpoint 和可靠性单元测试。

原版 [main.py](../main.py)、[eval.py](../eval.py) 和
[hubconf.py](../hubconf.py) 保留原有路径。实现位于 [models/](../models/) 与
[workflow/](../workflow/)，实验输出和本地权重不属于源码。
