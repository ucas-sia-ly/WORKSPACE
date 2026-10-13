# Qwen 困难样本增广

**当前入口：全部23城市，不限制困难源图总数。** [mine_all_cities.py](mine_all_cities.py)复用已有四城7277张，逐城筛选其余19城的全部合格困难源图；[all_city_campaign.py](all_city_campaign.py)沿用Qwen配置和审核门限分批生成；[finetune_all_cities.py](finetune_all_cities.py)使用全部23城真实数据与各城审核通过图，微调原始预训练SALAD并自动测试六域SVOX。完整命令、磁盘预估及恢复方式见 [ALL_23_CITIES.md](ALL_23_CITIES.md)。

本地23城训练池已核对：524701张真实图、62514个地点。生成量和审核通过量由数据决定。新入口不固定城市列表、不固定5004/7277训练图数；旧实验目录保留为历史结果。

## 流程与入口

按阶段执行，生成和训练分别加载模型。完整命令以 [全部城市操作说明](ALL_23_CITIES.md) 为准。

1. [mine_all_cities.py](mine_all_cities.py)：逐城挖掘困难真实源图，复用已冻结的四城市记录。
2. [all_city_campaign.py](all_city_campaign.py)：分批生成 night / snow / fog / rain 图像，审核并保存清单。
3. [finetune_all_cities.py](finetune_all_cities.py)：使用真实图与审核通过的对应生成图微调 SALAD，评估 SVOX 六域。

## 目录与职责

| 文件或目录 | 职责 |
|---|---|
| `mine_sources.py`、`extend_plan.py` | 困难度计算与已有计划扩展 |
| `plan.py`、`run.py` | 冻结生成计划、生成、导入和运行状态 |
| `quality.py`、`weather_signal.py` | 结构一致性与天气审核 |
| `common.py` | 路径、原子记录与哈希校验 |
| `tests/` | 采样、生成账本、恢复、训练和评估测试 |
| [docs/](docs/README.md) | 历史生成批次、训练对照和验证记录 |

四城市协调器、固定图片池对照及可靠性模块实验入口保留原路径；
全部城市流程仍复用其公共实现。已冻结实验通过源码路径和 SHA-256 校验，
恢复时以相应配置和历史说明为准。

## 历史实验文档

[文档索引](docs/README.md) 汇总生成批次、预训练微调、比例对照和 Reliability OT 实验。
[历史流程记录](docs/history.md) 保存旧 README 的完整四城市说明。
