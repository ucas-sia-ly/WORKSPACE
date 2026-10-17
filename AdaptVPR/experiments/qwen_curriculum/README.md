# Qwen 困难样本增广

**当前入口：全部23城市，不限制困难源图总数。** [mine_all_cities.py](mine_all_cities.py)复用已有四城7277张，逐城筛选其余19城的全部合格困难源图；[v2_campaign.py](v2_campaign.py)保留原计划和图片，按最新代码审核并接续生成；[finetune_all_cities.py](finetune_all_cities.py)使用全部23城真实数据与各城审核通过图，微调原始预训练SALAD并自动测试六域SVOX。完整命令、磁盘预估及恢复方式见 [ALL_23_CITIES.md](ALL_23_CITIES.md)。

**当前审核规则 v2：** 最少匹配点数为16，移除空间覆盖率门限，保留内点率、homography 位移及天气审核。[v2_campaign.py](v2_campaign.py)将已有图片和诊断数据迁入独立的 `generation_23city_v2`，旧配置、图片和账本保留，新结果明确记录来源，详见 [RESUME.md](RESUME.md)。

**长期生成的暂停与恢复：** 使用 [resume_control.py](resume_control.py) 的 `pause` / `resume` / `status`，自动跟随后台服务当前绑定的 v2 目录。当前协调器 [v2_campaign.py](v2_campaign.py)接续原计划，恢复已落盘图片，独立重试中断任务；暂停同时退出常驻 Qwen 以释放显存。具体命令见 [RESUME.md](RESUME.md)。

本地23城训练池已核对：524701张真实图、62514个地点。生成量和审核通过量由数据决定。新入口不固定城市列表、不固定5004/7277训练图数；旧实验目录保留为历史结果。

## 流程与入口

按阶段执行，生成和训练分别加载模型。完整命令以 [全部城市操作说明](ALL_23_CITIES.md) 为准。

1. [mine_all_cities.py](mine_all_cities.py)：逐城挖掘困难真实源图，复用已冻结的四城市记录。
2. [all_city_campaign.py](all_city_campaign.py) 保留原计划，[v2_campaign.py](v2_campaign.py)迁移并按当前代码接续生成 night / snow / fog / rain 图像，审核并保存清单。
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
