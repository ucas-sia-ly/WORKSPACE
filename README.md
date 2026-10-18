# Qwen 天气增广与 SALAD 训练

项目从 GSV-Cities 真实训练图中挖掘困难源图，用 Qwen 生成 night、snow、fog、rain
变体，经结构与天气审核后参与 SALAD 训练。Reliability OT 是默认关闭的可选分支。
当前操作入口是全部 23 城市流程，具体命令见
[23 城市生成与微调](AdaptVPR/experiments/qwen_curriculum/ALL_23_CITIES.md)。

## 从哪里开始

| 工作 | 入口 |
|---|---|
| 23 城市天气生成、审核和恢复 | [Qwen 流程](AdaptVPR/experiments/qwen_curriculum/README.md) |
| SALAD 训练、checkpoint 恢复和评估 | [训练与评估](salad/docs/workflow.md) |
| patch 可靠性与 dustbin | [Reliability OT](salad/docs/reliability_ot.md)、[v2](salad/docs/reliability_v2.md) |
| Qwen 服务、显存配置和依赖安装 | [外部组件](AdaptVPR/docs/EXTERNAL_COMPONENTS.md)、[服务配置](AdaptVPR/configs/systemd/README.md) |

## 目录职责

| 目录 | 用途 |
|---|---|
| `AdaptVPR/` | Qwen 生成服务、审核、恢复和任务协调器 |
| `salad/` | DINOv2 + SALAD、可靠性分支、训练、评估与单元测试 |
| `LightX2V/` | 带本机兼容补丁的 Qwen 推理依赖 |
| `vismatch/` | SuperPoint + LightGlue 几何验收依赖 |
| `services/` | 可选 Qwen3-VL 规划服务；固定天气流程无需此服务 |
| `dataset/` | 本地真实训练与评估数据，Git 忽略 |
| `models/` | 本地模型权重，Git 忽略 |
| `outputs/` | 生成图、清单、缓存、训练结果和评估报告，Git 忽略 |
| `IC-Light/` | 历史上游许可证与来源说明 |
| `docs/` | 当前项目结构说明 |

生成与训练分阶段执行，适配单张 48GB 显卡和 32GB 主存。启动器只启动 Qwen。
当前 Qwen 工作流只保留生成、恢复、图片管理、原始 SALAD 微调及 SVOX 测试所需模块。

[项目结构说明](docs/project_structure.md) 列出当前入口和数据路径。
第三方组件分别保留各自许可证，版本以本工作区 Git 提交及 LightX2V 的来源标记为准。
