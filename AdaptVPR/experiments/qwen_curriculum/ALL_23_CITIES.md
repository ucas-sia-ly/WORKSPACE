# 旧 23 城市计划：原始 SALAD 微调与 SVOX 测试

本文保留旧 79426 张计划的配置和训练说明。新一轮 place 覆盖与 night 优先计划已独立移到 `outputs/weather_v3/`，当前操作见 [WEATHER_V3.md](WEATHER_V3.md) 和 [RESUME.md](RESUME.md)；后台 `resume` 绑定新计划。

旧计划使用 `outputs/qwen_curriculum/images/` 和 `outputs/qwen_curriculum/state/`。旧图统一存放在 `images/train/`、`images/raw/`，通过审核的图片入口为 `images/accepted/<域>/`，例如 `images/accepted/night/`、`images/accepted/rain/`；旧计划、账本、进度及日志位于 `state/`，训练输出为 `state/training/`。分域目录只整理入口，图片内容与原审核记录保持不变。

整理快照：已保存 21010 张、已审核 21009 张、通过 13710 张，另有 1 张已保存图片待审核；还需新生成 58416 张，总计划保持 79426 张。实时进度以 `state/summary.json` 为准。

## 困难源图与生成配置

23 城市真实训练池已核对为 524701 张有效图片、62514 个至少含 4 张真实视角的地点。城市名来自 GSV-Cities 的 CSV，包括 OSL、PRG、PRS、TRT 等原始名称。源图筛选已完成，当前继续使用完整冻结计划，无需重新挖掘或准备旧批次。

困难源图使用原始 SALAD 教师：可靠正样本相似度至少 0.15，困难度位于各城 70%–97.5% 分位，每地点最多 2 张源图，保留全部合格候选，不以每城固定张数截断。新增城市的难度负样本来自城内其他地点，排除 25 米近邻；原四城评分作为历史证据保留。样本 ID、源路径及源图内容全局去重。源图合格不等于生成后审核通过。

天气比例为 night 50%、snow 20%、fog 20%、rain 10%，按完整计划分配，暂停和分批不会重排。生成不包含 overcast。

仍使用 Qwen-Image-Edit-2511、Lightning 4steps BF16、guidance 1 和 `source_aspect_v1` 画布。48GB 显存配置为 `resident_bf16_48g`：32 个 DiT blocks 常驻，其余按原策略流式加载；不减步数、不量化。原始 PNG 保留，训练 PNG 以 LANCZOS 缩回源尺寸。历史稳定速度约 27 秒/张，实际耗时包含加载、审核及停机时间。

当前审核 v2 要求最少 16 个匹配点、内点率至少 0.78、512 尺度的 homography 网格中位位移最多 16 像素，不设空间覆盖率门限。天气审核要求目标天气相对晴天的对比为正，且生成图的该对比高于源图。规则为实验验收标准。

## 旧计划的日志和完整性检查

旧计划保持暂停，不与本轮 v3 同时生成。可单独查看旧日志、检查旧计划完整性：

```bash
tail -f outputs/qwen_curriculum/state/generation.log

python AdaptVPR/experiments/qwen_curriculum/campaign.py check
```

预检不启动模型、不新增生成调用。旧计划仍保留原始 PNG 与训练 PNG。当前后台的暂停、恢复与日志路径见 [RESUME.md](RESUME.md)。

## 原始 SALAD 微调与 SVOX 测试

全部 79426 张生成和审核完成后，先检查训练数据：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
python AdaptVPR/experiments/qwen_curriculum/finetune_all_cities.py check-data \
  --generation-dir outputs/qwen_curriculum/state \
  --output-dir outputs/qwen_curriculum/state/training \
  --epochs 10
```

正式微调及随后自动进行的六域 SVOX 测试：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
python -u AdaptVPR/experiments/qwen_curriculum/finetune_all_cities.py run \
  --generation-dir outputs/qwen_curriculum/state \
  --output-dir outputs/qwen_curriculum/state/training \
  --epochs 10
```

加载指定的原始 `salad/checkpoint/dino_salad.ckpt`，使用完整原骨干和原 SALAD 聚合器，不启用 Reliability。训练为 10 轮、AdamW、lr 1e-5、cosine、weight_decay 1e-4；末 4 个骨干 block、norm 和聚合器可训练。每批 32 地点×4 图，224×224、FP32、workers 0、无额外图像增强。完整 23 城真实池参加训练，只有当前审核通过的源图按对应关系替换为生成图，数量动态读取。

`state/data_config.json` 保存原权重、真实数据、骨干及城市 CSV 的独立输入证明，训练不再读取旧挖掘目录。生成尚未完成、缺少任一城市合格图、来源标签错误或校验失败时，训练入口会停止。

后台训练与前台运行二选一：

```bash
systemctl --user start salad-finetune-23city.service
systemctl --user status salad-finetune-23city.service
```

训练服务日志为 `state/training/coordinator.log`。进度及训练日志：

```bash
cat outputs/qwen_curriculum/state/training/progress.json
tail -f outputs/qwen_curriculum/state/training/training.log
```

训练目录保存 `accepted_manifest.jsonl`、`accepted_summary.json`、`finetuned/checkpoint.pt` 和六域 `comparison.json`。同一训练命令可恢复最近完整 epoch 和已校验的评估结果。

既定最后一轮之后，用同一协议评估原模型与微调模型的 SVOX day/night/rain/snow/sun/overcast 六个完整子集：322×322、FP32、完整图库、25 米正例。SVOX 不参与训练数据筛选或轮次选择。
