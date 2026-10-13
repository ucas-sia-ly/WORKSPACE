# 项目结构与清理记录

2026-10-12 审计以当前 23 城市 Qwen 增广和 SALAD 训练为主线。
清理 457 个已跟踪冗余文件，合计约 101.4 MiB，另移位整理 10 份文档。
这项统计不包含 Python 缓存，也不包含下文记录的外部 BoQ 目录删除。

## 代码边界

业务入口保留在 `AdaptVPR/experiments/qwen_curriculum/`，通用生成、规划与验证实现
分别位于 `AdaptVPR/generation/`、`prompts/`、`verification/`。
SALAD 的模型实现位于 `salad/models/`，训练、采样、可靠性监督、checkpoint 和评估实现
位于 `salad/workflow/`。每部分的单元测试位于自己的 `tests/`。

LightX2V 和 VisMatch 是实际运行依赖。保留根目录位置可沿用服务配置、安装路径和缓存。
LightX2V 的 `pipeline.py` 会导入并注册多个模型系列，不能仅按模型名删除非 Qwen 源码。
VisMatch 保留匹配器实现、内置 LightGlue 和安装清单引用的测试输入。

## 文档与命名

- [根 README](../README.md) 是整个工作区的入口。
- [Qwen README](../AdaptVPR/experiments/qwen_curriculum/README.md) 只呈现当前流程。
- [Qwen 实验文档](../AdaptVPR/experiments/qwen_curriculum/docs/README.md) 集中保存历史生成批次、比例对照与验证记录，文件名统一为小写下划线形式。
- [SALAD 文档](../salad/docs/README.md) 集中保存训练、评估和两版可靠性说明。
- `test_700_ratio_experiment.py` 实际是历史实验驱动，不是单元测试；它被新实验和冻结配置引用，因此保留原名。新增驱动应使用描述动作的文件名，并将单元测试放到 `tests/`。

用户正在编写的 23 城市入口及 `ALL_23_CITIES.md` 保留原路径和内容。
旧主 README 的历史流程完整移入 [history.md](../AdaptVPR/experiments/qwen_curriculum/docs/history.md)。

## 已清理的内容

- 无调用方的 `AdaptVPR/generation/iclight.py`、旧 IC-Light systemd 模板及重复 `test.md`。
- 当前启动器、预检与公开配置中的 IC-Light 可选分支；启动流程统一为 Qwen。
- AdaptVPR 的展示 GIF 和已生成的十张 demo 展示图，保留 demo 输入、提示词和测试。
- IC-Light 上游 Gradio 示例、背景移除脚本和示例图片，保留许可证与来源说明。
- LightX2V 编辑器技能、文档站、独立应用、示例、测试案例、Docker 示例和展示图片。
- VisMatch 文档站、演示 notebook、生成后的匹配展示图与 LightGlue 演示脚本和图片。
- 源码目录中的 Python 与测试缓存、空编辑器目录和失效的数据目录符号链接。

## 保留的历史依赖

`all_city_campaign.py` 复用 `scale_campaign.py`、`fast_campaign.py`、`extend_plan.py`
和恢复工具；全部城市微调复用预训练微调实现。固定比例与 Reliability OT 对照仍使用其
历史协调器。它们是当前依赖，不按旧实验数量直接删除。

`generation_diagnosis/verifier_controls.py` 被当前质量审核调用。
历史诊断 `runner.py` 还导入 IC-Light adapter；部分旧生成配置绑定了这些源码哈希，
因此保留适配器和必要测试。它们不由当前启动器启动。
旧 100src 配置与清理前分支已经存在源码哈希差异；复跑旧阶段需要相应历史版本，
不能用当前分支冒充原版本。本次保留了现行 Qwen 生成与 SALAD 训练主链的源码字节和路径。

真实数据、生成图片、manifest、模型权重、困难度缓存、可靠性目标和正式训练结果都有
明确用途，保留原路径。服务输出目录可能包含中断请求的恢复图片，也保留。
Git 历史未重写；删除工作树中的示例不会自动缩小已有 `.git/` 历史。

## 本地数据审计中的外部变化

审计开始时 `Bag-of-Queries/data/train/gsv-cities/` 是另一份独立数据副本。
它与 `dataset/gsv-cities/` 的 529,529 个普通文件相对路径和大小一致，23 份 CSV
内容校验一致，前约 350,000 个文件逐 SHA-256 一致。
全量比对期间，另一个操作移除了整个 `Bag-of-Queries/` 目录；本次清理代码没有删除它，
不能把中断的检查记录为全量内容校验成功。当前规范数据目录仍在，生成、训练源码及
已有结果清单未发现引用旧 BoQ 目录的路径。

## 验证

396 项 CPU 单元测试通过：生成服务与路由 54 项，历史诊断兼容 10 项，
Qwen 流程 182 项，SALAD 150 项。992 个剩余 Python 文件通过语法解析，
87 个本地 Markdown 链接有效，`git diff --check` 通过。
863 个现有核心、实验和适配器 Python 文件与清理前 HEAD 字节一致；
用户新增的 5 个文件以及当前 README 的最新说明保持原样。
已有 JSON 配置的源码哈希映射没有指向本次修改或删除的 Python 文件。

使用已安装的 AdaptVPR 环境，从工作区根目录运行：

```bash
PYTHON=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
PYTHONPATH=AdaptVPR CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 $PYTHON -m unittest discover -s AdaptVPR/tests
PYTHONPATH=AdaptVPR CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 $PYTHON -m unittest experiments.generation_diagnosis.test_runner experiments.generation_diagnosis.test_rectify_qwen
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 $PYTHON -m unittest discover -s AdaptVPR/experiments/qwen_curriculum/tests
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 $PYTHON -m unittest discover -s salad/tests
```

本次验证未启动新的图片生成或完整训练。
