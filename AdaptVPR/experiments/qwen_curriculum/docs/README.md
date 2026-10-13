# Qwen 增广文档索引

最新流程见 [项目入口](../README.md) 与 [全部 23 城市操作说明](../ALL_23_CITIES.md)。
以下文档对应已记录的四城市批次和固定实验；文档中的数量、服务状态及“当前”
均属于相应历史阶段。所有终端命令仍从工作区根目录执行。

## 生成与恢复

| 文档 | 内容 |
|---|---|
| [2000 张快速生成](generation_2000_fast.md) | 48GB 显卡配置、零调用恢复及累计批次 |
| [7277 张生成计划](generation_7277.md) | 四城市困难源图扩展与生成预算 |
| [历史流程与状态](history.md) | 原 README 中的四城市流程、服务恢复和旧训练方案 |

## 训练与对照

| 文档 | 内容 |
|---|---|
| [5004 张预训练 SALAD 微调](finetune_pretrained_5004.md) | 审核通过图片池与原预训练模型微调 |
| [700 张比例对照](ratio_700_experiment.md) | 8:1 / 4:1 生成组与配对全真实对照 |
| [Reliability OT 对照](reliability_ot_700_experiment.md) | 相同固定图片池上的第一版可靠性模块实验 |
| [Reliability OT v2 对照](reliability_v2_700_experiment.md) | 固定教师缓存、局部几何监督及版本比较 |
| [2026-10-08 验证记录](validation_20261008.md) | 早期测试、生成审核和数据核验记录 |

SALAD 的训练、评估和可靠性实现说明见 [SALAD 文档索引](../../../../salad/docs/README.md)。
