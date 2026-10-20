# 项目结构

当前主流程是全部 23 城市的 Qwen 天气生成、审核、原始 SALAD 微调与六域 SVOX 测试。源图筛选已经完成，生成继续使用现有计划。

## 当前代码

`AdaptVPR/experiments/qwen_curriculum/` 保留 11 个功能模块及包入口：

| 模块 | 职责 |
|---|---|
| `campaign.py` | 计划校验、任务调度、恢复与完整性检查 |
| `resume_control.py` | 后台暂停、继续与状态 |
| `run.py` | 生成请求、调用账本及审核 |
| `resume_recovery.py` | 中断输出恢复 |
| `image_store.py` | 图片归集、索引及审核通过入口 |
| `quality.py`、`weather_signal.py` | 几何和天气审核 |
| `common.py` | 路径、原子写入及内容校验 |
| `finetune_all_cities.py`、`finetune_pretrained.py` | 原始 SALAD 微调及训练恢复 |
| `evaluate_pretrained_finetune.py` | 原模型与微调模型的六域 SVOX 评估 |

旧批次协调器、一次性迁移和清理脚本、挖掘器、比例与可靠性对照实验已经移除。当前模块直接读取完整的 23 城市状态，不再串联旧实验入口。

`salad/models/` 保存模型，`salad/workflow/` 保存训练、采样、checkpoint 和评估实现。当前质量审核还使用 `AdaptVPR/experiments/generation_diagnosis/verifier_controls.py` 的匹配统计，以及 `AdaptVPR/verification/` 的特征实现。LightX2V 和 VisMatch 分别提供 Qwen 推理与几何匹配。

## 数据与输出

| 路径 | 内容 |
|---|---|
| `dataset/gsv-cities/` | 全部 23 城市真实训练数据 |
| `dataset/svox/` | SVOX 测试集 |
| `salad/checkpoint/dino_salad.ckpt` | 原始 SALAD 预训练权重 |
| `outputs/qwen_curriculum/images/` | 训练 PNG、高清原图、元数据和通过审核的图片入口 |
| `outputs/qwen_curriculum/state/` | 当前计划、账本、审核结果、进度和日志 |
| `outputs/qwen_curriculum/state/training/` | 微调 checkpoint 与六域评估结果 |
| `outputs/qwen_curriculum/state/finetuning/` | CCR A/M 等独立实验；每轮冻结当时的审核通过图片，保留后续增量入口 |

生成与训练分阶段执行。生成服务的临时输出可能包含中断请求的完整图片，由恢复流程核验后收集。

## 操作与验证

- [Qwen 工作流](../AdaptVPR/experiments/qwen_curriculum/README.md)
- [暂停和继续](../AdaptVPR/experiments/qwen_curriculum/RESUME.md)
- [23 城市生成与训练命令](../AdaptVPR/experiments/qwen_curriculum/ALL_23_CITIES.md)
- [SALAD 训练与评估](../salad/docs/workflow.md)
- [Qwen 持续增加图片时的 CCR 微调](../salad/docs/qwen_finetuning.md)
- [固定地点排除与全量真实 A 基线](../salad/docs/real_baseline.md)
- [A/M 多种子公平对照与真实 val 统计](../salad/docs/am_multiseed.md)
- [B0 归因及 Feature/OT/q 局部消融](../salad/docs/ccr_stage4.md)

在工作区根目录、AdaptVPR 环境中运行当前流程测试：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
python -m unittest discover -s AdaptVPR/experiments/qwen_curriculum/tests
```

暂停后核验现有计划、账本和图片，不启动生成模型：

```bash
python AdaptVPR/experiments/qwen_curriculum/campaign.py check
```
