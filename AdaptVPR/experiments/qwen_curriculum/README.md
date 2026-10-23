# 当前入口：v4 优先覆盖未生成地点

当前使用 `outputs/weather_v4/` 和 `weather_v4_priority.py`，先执行 night，优先覆盖旧两轮未生成的 place；已有 20 张 v4 保留。启动、暂停命令见 [RESUME.md](RESUME.md)，规则见 [WEATHER_V4.md](WEATHER_V4.md)。以下为旧 v3 与旧图库说明。

# Qwen 困难样本增广

**上一轮 v3 计划：新一轮 v3 提示词，优先完成 night；目前保持暂停。** 每个合格 place 一张，覆盖 23 城市的 62499 个地点；比例为 night 44%、rainy_night 14%、rain 15%、snow 14%、sun 8%、fog 5%，不生成 overcast。新图按域存放于独立的 `outputs/weather_v3/`，状态位于 `outputs/weather_v3/state/`。已复用 15 张 pilot 图，night 初始需新增 27489 张。当前操作以 [WEATHER_V3.md](WEATHER_V3.md) 和 [RESUME.md](RESUME.md) 为准。

```text
outputs/
  weather_v3/
    night/ rainy_night/ rain/ snow/ sun/ fog/  本轮训练尺寸图
    raw/<域>/                               本轮高分辨率图
    accepted/<域>/                          本轮自动审核通过入口
    state/                                  本轮计划、进度与 resume 账本
  qwen_curriculum/
    images/train/ images/raw/                旧计划图片
    images/accepted/<域>/                    旧计划审核通过入口，按域分开
    state/                                  旧计划账本和训练实验
```

`accepted` 只代表当时的自动审核结果。局部物体缺陷和源图拼接畸变仍可能漏检；本次整理不修改审核结论，也不自动启动生成。

## 旧计划（暂停保留）

旧计划覆盖全部 23 个 GSV-Cities 城市，使用 Qwen 生成困难域图片，再以审核通过的图片微调原始 SALAD，最后测试六域 SVOX。旧训练说明见 [ALL_23_CITIES.md](ALL_23_CITIES.md)。后台控制命令目前绑定新 v3 计划。

`outputs/qwen_curriculum/` 顶层只保留两个目录：

- `images/`：旧图片库，`train/` 为训练尺寸 PNG，`raw/` 为高分辨率 PNG，`accepted/<域>/` 为审核通过入口。
- `state/`：当前计划、账本、审核结果、进度及日志；`training/` 保存后续训练与评估，`history/legacy.tar.gz` 保留原始历史证据。

旧计划的完整性检查由 [campaign.py](campaign.py) 执行。本轮后台使用 [night_first_v3.py](night_first_v3.py)，绑定 `--output-dir outputs/weather_v3/state`。

`state/runtime_config.json` 校验当前代码与固定计划。历史配置和源码归档用于溯源，当前任务不再执行旧实验或迁移入口。

当前旧库已保存并审核 21077 张，通过 13757 张，没有待补审核图片，还需新生成 58349 张。全计划仍为 79426 张，实时数量见 `outputs/qwen_curriculum/state/summary.json`。

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
python AdaptVPR/experiments/qwen_curriculum/night_first_v3.py check
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
| `night_first_v3.py` | 本轮 place 覆盖、night 优先调度、恢复与完整性检查 |
| `source_screen_v3.py`、`source_reliability_v3.py`、`v3_reuse.py` | 本轮源图筛选、可靠度及精确兼容复用 |
| `campaign.py` | 旧计划恢复与校验；本轮复用的服务控制及配置工具 |
| `run.py`、`resume_recovery.py` | 生成执行、调用账本和中断恢复 |
| `image_store.py` | 统一图库及审核通过入口 |
| `quality_v3.py`、`quality.py`、`weather_signal.py` | 本轮结构与天气审核及旧计划兼容 |
| `finetune_all_cities.py`、`finetune_pretrained.py` | 原始 SALAD 微调与训练恢复 |
| `evaluate_pretrained_finetune.py` | 原模型与微调模型的六域 SVOX 评估 |
| `common.py` | 路径、原子写入和校验 |
| `tests/` | 生成、账本、恢复、训练及存储验证 |

完整生成与训练命令见 [ALL_23_CITIES.md](ALL_23_CITIES.md)。
