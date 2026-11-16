# 当前 Qwen 工作流

本批 **61,984 张 night 图片已完成**，包含 46,712 张历史保留图、13,206 张续生成图及 2,066 张汞灯替代图。覆盖 23 个城市，排除汞灯，每地点对应唯一真实源图。

统一入口为 [`outputs/night_dataset_v1`](../../../outputs/night_dataset_v1)，见[数据集说明](../../../salad/docs/night_dataset.md)。

```bash
./ccr-night status
./ccr-night pair --city Chicago --place-id 5496
./ccr-night verify
```

## 当前代码

- `night_dataset.py`：从冻结结果建立唯一地点/图片索引，SHA256 核验、原图链接、生成图硬链接、TSV/JSONL/SQLite 汇总。只使用 CPU。
- `night_cleanup.py`：清理计划与逐文件删除审计，保护当前数据、冻结依赖、模型、最终 checkpoint、训练报告和划分。
- `night_generation_completion.py`：本批双机任务衔接与独立回传。
- `night_original_images.py`、`night_original_queue_recovery.py`：原 Prompt 续生成与远程队列恢复。
- `night_mercury_prompts.py`、`night_mercury_images.py`：原汞灯源图按其余 29 类规划并补生成。
- `night_cluster.py`、`night_cluster_worker.py`、`night_cluster_resources.py`：SSH、持久队列、完整性回收及资源保护。
- `night_place_fast.py`、`night_place_cohort.py`、`night_place_cohort_v4.py`：本批原源图与 Prompt 规划、批次恢复。
- `night_diversity_policy.py`、`night_style_catalog.json`：原版夜景风格、场景适配及覆盖分配。
- `quality.py`、`quality_v3.py`、`content_audit_v4.py`、`content_audit_v42.py`：质量和结构检查实现，供后续独立审核及训练预检查使用。

部分带版本号的文件仍是当前冻结依赖或共用恢复/校验引擎，因此保留原文件及 checkpoint key/哈希。独立旧 V4/V5 生成、Prompt 试验、基准与旧微调入口及其专属测试已删除。

生成服务已停止。完成屏障存在时，`ccr-night pause/resume/sync` 只提示本批已完成，不会意外重启生成。原冻结记录保留在相关运行目录中；新操作以统一数据集为准。
