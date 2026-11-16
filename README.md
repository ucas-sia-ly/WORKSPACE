# Qwen 夜间增广与 SALAD 训练

当前夜间生成已完成：23 个城市、61,984 个地点，每地点一张精确源图和一张 night 生成图。保留原版 29 类风格，排除汞灯。图片已全部回传第一台，并统一整理到 `outputs/night_dataset_v1/`。

## 查看数据

```bash
./ccr-night status
./ccr-night pair --city Chicago --place-id 5496
./ccr-night manifest
# CPU 文件完整性复核，不启动 GPU
./ccr-night verify
```

[统一数据集说明](salad/docs/night_dataset.md)介绍逐地点目录、原图映射、清单及校验。`pairs.tsv` 可直接查看全部对应关系；`manifest.jsonl` 包含 Prompt、精确源图、生成记录和 SHA256。

## 常用入口

| 工作 | 说明 |
|---|---|
| 当前生成数据 | [数据集与清理说明](salad/docs/night_dataset.md) |
| 生成代码与依赖 | [Qwen 工作流](AdaptVPR/experiments/qwen_curriculum/README.md) |
| SALAD 训练与评估 | [训练与评估](salad/docs/workflow.md)、[文档索引](salad/docs/README.md) |
| 项目目录与清理记录 | [项目结构](docs/project_structure.md)、[清理记录](docs/cleanup_20261113.md) |
| 服务与依赖 | [外部组件](AdaptVPR/docs/EXTERNAL_COMPONENTS.md)、[服务配置](AdaptVPR/configs/systemd/README.md) |

`dataset/` 是真实数据；`models/`、`salad/checkpoint/` 是本地模型；`LightX2V/`、`vismatch/` 是推理与几何匹配依赖。

旧生成目录中仍有本批数据的硬链接和冻结运行记录，用于追溯；它们不占第二份图片空间。旧天气图库、失败试验和回传缓存已清理。最终训练 checkpoint、报告和数据划分保留。生成图仍为待质量审核状态，文件完整不代表已获准训练。没有启动训练或重新生图。
