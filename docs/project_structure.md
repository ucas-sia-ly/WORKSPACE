# 项目结构

## 当前数据入口

`outputs/night_dataset_final/` 是 61,984 对真实/夜间生成图的统一入口。

```text
outputs/night_dataset_final/
  pairs/<city>/<place_id>/
    source.jpg       # 精确真实源图链接
    night.png        # 训练尺寸生成图硬链接
    night_full.png   # 高清生成图硬链接
    generation.json # 原始生成 sidecar
  manifest.jsonl     # Prompt、源图、生成图、哈希、来源记录
  pairs.tsv          # 可直接查看的一一对应表
  index.sqlite       # 按城市/地点或图片 ID 查询
  complete.json      # 数量、城市/风格统计、索引哈希
```

[完整说明](../salad/docs/night_dataset.md)。链接不改变图片像素，也不额外复制整批图。真实数据保存在 `dataset/`，链接依赖该路径。

## 代码

- `ccr-night`：已完成数据集状态、查询和完整性核验。
- `AdaptVPR/experiments/qwen_curriculum/night_dataset.py`：原始配对、查询和校验。
- `night_dataset_final.py`：合并已发布数据与最新重生成结果。
- `night_cleanup.py`：明确清理清单及审计。
- `night_*` 当前双机与 Prompt 模块：本批的冻结策略、恢复与运行来源。
- `AdaptVPR/generation/`、`LightX2V/`：Qwen 推理实现。
- `salad/models/`、`salad/workflow/`、`salad/train_salad.py`、`salad/evaluate_salad.py`：模型、采样、训练、评估。

## 保留的输出

本批 `night_place_*`、`night_cluster_v1`、`night_original_no_mercury_v1`、`night_mercury_replacement_v1` 仍保存冻结记录与历史路径下的图片硬链接。不能把这些目录的 `du` 数字简单相加当作独立图片副本。

`outputs/qwen_curriculum/state/finetuning` 和 `salad/outputs` 保留正式训练报告、最终 checkpoint、数据划分；重复的中间快照、旧图库和临时缓存已清理。旧实验记录中指向已删除旧图库或中间快照的路径属于历史记录，不应直接用于恢复旧实验。

`outputs/coverage_run/checkpoint_epoch_050.pt` 是仍被训练配置引用的初始 checkpoint。Reliability V2 实验继续按先前要求保留。

清理计划、删除清单和实际释放空间记录在 `outputs/maintenance/night_cleanup_20261113/`。见[清理报告](cleanup_20261113.md)。
