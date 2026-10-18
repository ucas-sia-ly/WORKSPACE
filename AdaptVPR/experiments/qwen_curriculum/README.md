# Qwen 困难样本增广

当前任务覆盖全部 23 个 GSV-Cities 城市，使用 Qwen 生成困难域图片，再以当前审核通过的图片微调原始 SALAD，最后测试六域 SVOX。完整操作见 [ALL_23_CITIES.md](ALL_23_CITIES.md)，暂停和恢复见 [RESUME.md](RESUME.md)。

`outputs/qwen_curriculum/` 顶层只保留两个目录：

- `images/`：统一图片库，`train/` 为训练尺寸 PNG，`raw/` 为高分辨率 PNG，`accepted/` 为审核通过入口。
- `state/`：当前计划、账本、审核结果、进度及日志；`training/` 保存后续训练与评估，`history/legacy.tar.gz` 保留原始历史证据。

生成和完整性检查统一由 [campaign.py](campaign.py) 执行，后台绑定 `--output-dir outputs/qwen_curriculum/state`。

`state/runtime_config.json` 校验当前代码与固定计划。历史配置和源码归档用于溯源，当前任务不再执行旧实验或迁移入口。

整理快照已保存 21010 张，已审核 21009 张，通过 13710 张，1 张等待补审核，还需新生成 58416 张。全计划仍为 79426 张，实时数量见 `state/summary.json`。

## 常用命令

```bash
cd /home/admin123/github/WORKSPACE
conda activate AdaptVPR

python AdaptVPR/experiments/qwen_curriculum/resume_control.py resume
python AdaptVPR/experiments/qwen_curriculum/resume_control.py status
python AdaptVPR/experiments/qwen_curriculum/resume_control.py pause
```

暂停后校验，不启动模型或新增生成调用：

```bash
python AdaptVPR/experiments/qwen_curriculum/campaign.py check
```

已完成图不重复生成；已保存但未审核的图先补审核；中断且没有完整输出的任务，在下一次显式恢复时使用原参数重试。暂停同时退出常驻 Qwen，释放生成占用的显存和内存。

## 生成、审核与训练

源图筛选已完成，保留全部符合规则的困难候选，不限制每城固定张数。完整 23 城训练池为 524701 张真实图、62514 个地点。night/snow/fog/rain 比例为 50%/20%/20%/10%，不生成 overcast。

Qwen-Image-Edit-2511、Lightning 4steps BF16、guidance 1、源宽高比画布及 32 个常驻 DiT blocks 保持不变。审核 v2 使用最少 16 个匹配点、内点率、homography 位移和天气门限，不设空间覆盖率门限。

全部生成完成后，[finetune_all_cities.py](finetune_all_cities.py) 从 `state/data_config.json` 读取原始 `salad/checkpoint/dino_salad.ckpt` 和完整真实数据输入，只以当前通过图片替换对应源图。训练默认输出为 `state/training/`，随后自动评估 SVOX day/night/rain/snow/sun/overcast 六域。

## 代码与文档

| 入口 | 职责 |
|---|---|
| `resume_control.py` | 后台暂停、恢复和状态 |
| `campaign.py` | 计划校验、任务调度、恢复及完整性检查 |
| `run.py`、`resume_recovery.py` | 生成执行、调用账本和中断恢复 |
| `image_store.py` | 统一图库及审核通过入口 |
| `quality.py`、`weather_signal.py` | 当前结构与天气审核 |
| `finetune_all_cities.py`、`finetune_pretrained.py` | 原始 SALAD 微调与训练恢复 |
| `evaluate_pretrained_finetune.py` | 原模型与微调模型的六域 SVOX 评估 |
| `common.py` | 路径、原子写入和校验 |
| `tests/` | 生成、账本、恢复、训练及存储验证 |

完整生成与训练命令见 [ALL_23_CITIES.md](ALL_23_CITIES.md)。
