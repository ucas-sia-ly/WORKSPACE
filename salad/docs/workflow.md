# AdaptVPR 的 SALAD 训练与评估入口

新增入口为 `salad/train_salad.py` 和 `salad/evaluate_salad.py`，原有
`main.py` / `eval.py` 保持不变。新入口使用原仓库 SALAD 聚合器及 DINOv2，
使用原生 PyTorch 训练、MultiSimilarityLoss 和 MultiSimilarityMiner。
损失参数与原仓库一致：alpha=1、beta=50、base=0、miner epsilon=0.1。
无需 Lightning、FAISS 或 pytorch-metric-learning。

以下命令从 `/home/admin123/github/WORKSPACE` 执行。

可选的 patch 可靠性分支通过 `train_salad.py --reliability-ot` 开启，默认关闭。
实现、弱监督定义、训练/单图推理命令和测试见 [reliability_ot.md](reliability_ot.md)。
该分支只修改 `salad/`，使用已有源图配对，不调用生成模型。
固定教师缓存、轻量上下文及已知局部几何负监督见
[reliability_v2.md](reliability_v2.md)，默认保持旧模式。

## 1. 环境与入口检查

```bash
python -m pip install -r salad/requirements-workflow.txt
python salad/train_salad.py --help
python salad/evaluate_salad.py --help
```

首次默认加载 DINOv2 会使用 Torch Hub 获取代码及预训练 backbone 权重。
离线环境可指定本地 DINOv2 checkout 的 `--backbone-repo` 和预训练
state_dict 的 `--backbone-weights`。评估/恢复时 checkpoint 已包含全部
模型权重，只需要本地代码，不必重复提供 backbone weights。
新入口默认随机初始化 SALAD 聚合器；使用 `--init-checkpoint` 可从完整的
预训练 SALAD checkpoint 开始微调。
CPU 模式在导入 DINOv2 前禁用可选 CUDA xFormers attention。

## 2. 检查训练数据

```bash
python salad/train_salad.py \
  --real-data dataset/gsv-cities \
  --synthetic-manifest outputs/vpr_guidance/round_0/synthetic_manifest.jsonl \
  --output-dir outputs/vpr_guidance/round_0/salad \
  --cities Bangkok \
  --check-data
```

检查 CSV、真实图文件、合成文件及来源对应关系，不下载 backbone、不训练。
省略 `--synthetic-manifest` 就是仅真实图训练。省略 `--cities` 使用全部城市。

合成输入是 `AdaptVPR/scripts/build_manifest.py` 导出的 JSONL，每行类似：

```json
{"source_path":"/actual/gsv/Images/Bangkok/source.jpg","output_path":"/actual/generated/aug.jpg","passed":true,"eligible_for_training":true}
```

仅消费两个 acceptance flag 都严格为 true 的记录；如包含 `plausible` 或
`weather_ok`，这些 flag 也必须严格为 true。增强图继承其原图的
`(city_id, place_id)` 标签；不同城市的同号地点分配不同标签。重复增强图
去重，缺失文件、错误来源与歧义路径明确报错，不生成替代图片。
建议使用绝对路径；相对路径在 manifest 目录及其祖先、当前工作目录中解析，
源图也可相对于 GSV 根目录或 `Images/`。若多个不同文件均匹配则报歧义。

## 3. 训练新的 SALAD

```bash
python salad/train_salad.py \
  --real-data dataset/gsv-cities \
  --synthetic-manifest outputs/vpr_guidance/round_0/synthetic_manifest.jsonl \
  --output-dir outputs/vpr_guidance/round_0/salad \
  --epochs 50 \
  --batch-size 32 \
  --images-per-place 4 \
  --min-images-per-place 4 \
  --synthetic-fraction 0.5 \
  --backbone dinov2_vitb14 \
  --aggregator salad \
  --init-policy random_aggregator_pretrained_backbone \
  --image-size 224 224 \
  --num-trainable-blocks 4 \
  --learning-rate 6e-5 \
  --num-workers 4 \
  --device cuda \
  --seed 42
```

`batch-size` 是每 batch 的地点数：默认 `32 × 4 = 128` 张图。
默认 `--synthetic-mode mix`：每地点使用 `floor(K × synthetic_fraction)` 个合成名额，最多 K−1 个，
始终保留至少一个真实视角；合成不足由不同真实视角补齐。真实视角不足
`min-images-per-place` 的地点不因有合成图而获得准入。
默认增广为随机水平翻转及亮度/对比度/颜色变化，可用 `--no-augment` 关闭。
`--num-trainable-blocks 0` 冻结整个 backbone，仅训练聚合器。

显存不足时减小 batch-size。要求至少两个地点和每地点至少两张图。
图片高宽必须是 14 的倍数，patch 数必须大于 SALAD clusters。
默认 CUDA FP16；CPU 使用 FP32。可显式指定 `--precision 32|16|bf16`。
小规模检查可增加 `--epochs 1 --max-batches-per-epoch 2`，使用独立输出目录。

输出：

- `checkpoint.pt`：最新完整状态，评估使用这个文件。
- `checkpoint_epoch_001.pt` 等：按 `--save-every` 保留的 epoch 状态。
- `training_config.json`：本次配置。
- `data_summary.json`：可用原图/合成图及过滤统计。
- `training_log.jsonl`：每 epoch 的 loss、步数、实际真实图/合成图曝光数。

不指定 resume 时拒绝覆盖已有 `checkpoint.pt`。新一轮换新输出目录。
恢复在 epoch 边界进行：重复原命令并增加
`--resume outputs/vpr_guidance/round_0/salad/checkpoint.pt`。
数据、采样、模型、优化器、workers 和总 epochs 必须保持一致；元数据及
manifest 的 SHA-256 用于检测变化。恢复原定训练，不扩展原 cosine schedule。
完全结束的 checkpoint 不再恢复。checkpoint 包含模型、优化器、scheduler、
scaler 与随机状态；图像内容本身不做全量 checksum。

## 3.1 小规模微调下载的预训练 SALAD

```bash
python salad/train_salad.py \
  --real-data dataset/gsv-cities \
  --synthetic-manifest outputs/vpr_guidance/pretrained_finetune_20261008/accepted_manifest.jsonl \
  --init-checkpoint salad/checkpoint/dino_salad.ckpt \
  --output-dir outputs/vpr_guidance/pretrained_finetune_20261008/finetuned \
  --cities Bangkok \
  --synthetic-places-only \
  --epochs 1 \
  --batch-size 8 \
  --images-per-place 4 \
  --min-images-per-place 4 \
  --synthetic-fraction 0.5 \
  --num-trainable-blocks 0 \
  --image-size 224 224 \
  --learning-rate 1e-6 \
  --weight-decay 0 \
  --no-augment \
  --precision 32 \
  --num-workers 4 \
  --device cuda \
  --seed 42
```

`--init-checkpoint` 严格加载完整 backbone 与聚合器权重，支持原版 raw
state_dict、Lightning checkpoint 和新入口保存的 checkpoint，并从权重推断
模型结构。微调使用新的优化器；`--resume` 则恢复已有训练的优化器和进度，
两者不可同时使用。初始化路径及 SHA-256 会写入训练配置和新 checkpoint。

`--synthetic-places-only` 只保留有验收增强图的地点，仍混合同地点原图。
这里冻结全部 DINOv2，仅以低学习率训练 SALAD 聚合器一个 epoch。
原始 `dino_salad.ckpt` 保留，新权重写入独立的 `finetuned/checkpoint.pt`。
上面的 manifest 是本次实验固定的验收快照；重新实验时应重新整理所需记录，
并使用新的输出目录。

本次一键运行原模型评估、微调、新模型评估的命令是：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/run_finetune_comparison.py \
  --checkpoint salad/checkpoint/dino_salad.ckpt \
  --real-data dataset/gsv-cities \
  --synthetic-manifest outputs/vpr_guidance/pretrained_finetune_20261008/accepted_manifest.jsonl \
  --dataset-root dataset/svox \
  --output-dir outputs/vpr_guidance/pretrained_finetune_20261008 \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --device cuda \
  --train-batch-size 8 \
  --eval-batch-size 32 \
  --num-workers 4 \
  --epochs 1 \
  --learning-rate 1e-6 \
  --seed 42
```

比较脚本分别评估 `queries` 与 `queries_night`，两者使用同一完整测试图库、
25 米正样本半径、224×224 输入和 FP32。每个 checkpoint 提取自己的图库
特征。输出为 `before/`、`after/` 下的结果 JSON、训练日志、实验配置及
`comparison.json`，其中提升量的单位是百分点。已完成的输出目录拒绝覆盖。

## 3.2 当前 23 城市 Qwen 生成与原始 SALAD 微调

当前源图筛选和完整 23 城市计划已经完成。Qwen 按 night/snow/fog/rain =
50%/20%/20%/10% 生成，每张独立源图对应一个天气任务。生成图片统一位于
`outputs/qwen_curriculum/images/`，计划、账本、进度及恢复状态位于
`outputs/qwen_curriculum/state/`。

在工作区根目录、AdaptVPR 环境中操作后台生成：

```bash
python AdaptVPR/experiments/qwen_curriculum/resume_control.py resume
python AdaptVPR/experiments/qwen_curriculum/resume_control.py status
python AdaptVPR/experiments/qwen_curriculum/resume_control.py pause
```

已完成的图不重复生成；已保存但未审核的图先补审核；没有完整输出的中断任务
在显式恢复时使用原参数重试。暂停同时停止 Qwen 服务，释放生成资源。
暂停后可执行完整性检查：

```bash
python AdaptVPR/experiments/qwen_curriculum/campaign.py check
```

全部生成和审核完成后，使用当前审核通过的图片微调原始
`salad/checkpoint/dino_salad.ckpt`。完整 23 城市真实训练池参与训练，
通过的生成图按精确来源替换对应原图，其他来源继续使用真实图。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
python AdaptVPR/experiments/qwen_curriculum/finetune_all_cities.py check-data \
  --generation-dir outputs/qwen_curriculum/state \
  --output-dir outputs/qwen_curriculum/state/training \
  --epochs 10

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
python -u AdaptVPR/experiments/qwen_curriculum/finetune_all_cities.py run \
  --generation-dir outputs/qwen_curriculum/state \
  --output-dir outputs/qwen_curriculum/state/training \
  --epochs 10
```

当前配置为 10 轮、学习率 `1e-5`、末 4 个骨干 block 与聚合器可训练，
每批 32 地点 × 4 图、224×224、FP32、0 workers、无额外图像增强。
训练完成后，用相同 322×322、FP32、25 米正例协议，评估原模型与微调模型的
SVOX day/night/rain/snow/sun/overcast 六个完整子集。SVOX 不参与样本筛选或
checkpoint 选择。同一训练命令可以恢复最近完整 epoch 和已经校验的评估结果。

完整配置和操作说明见 [23 城市生成、微调与测试](../../AdaptVPR/experiments/qwen_curriculum/ALL_23_CITIES.md)，
恢复细节见 [暂停与继续](../../AdaptVPR/experiments/qwen_curriculum/RESUME.md)。

## 4. 评估 SVOX

```bash
python salad/evaluate_salad.py \
  --checkpoint outputs/vpr_guidance/round_0/salad/checkpoint.pt \
  --dataset SVOX \
  --dataset-root dataset/svox \
  --split test \
  --query-subdirs queries_night \
  --positive-radius 25 \
  --output outputs/vpr_guidance/round_0/evaluation/SVOX_night_results.json \
  --save-hard-cases
```

读取 `images/test/gallery/` 和 `images/test/queries_night/`，从文件名
解析 UTM 坐标，以指定米制半径定义 positives。默认 query folder 是
`queries`，可用 `--query-subdirs` 指定多个条件文件夹。
输出中记录所用 protocol；不同 query 条件、半径、split 的结果不可混作同一协议。

## 5. 评估 Nordland

```bash
python salad/evaluate_salad.py \
  --checkpoint outputs/vpr_guidance/round_0/salad/checkpoint.pt \
  --dataset Nordland \
  --dataset-root dataset/nordland \
  --split test \
  --frame-window 10 \
  --output outputs/vpr_guidance/round_0/evaluation/Nordland_results.json \
  --save-hard-cases
```

支持本地 `images/test/database/`、`images/test/queries/` 格式。
按文件中的真实 frame ID 匹配，默认 ±10 帧，参数记录到 protocol。
同时支持旧版 `ref/` / `query/` 布局，使用仓库附带的 Nordland 元数据。
SPED、MSLS、Pittsburgh 可通过对应 `--dataset` 和 `--dataset-root` 使用
仓库附带 .npy 元数据；`--metadata-root` 可覆盖其位置。

所有评估可加 `--check-data` 先检查目录与真值，不加载 checkpoint。
小规模推理可加 `--limit-queries 10 --num-workers 0`。
训练 checkpoint 自动携带模型结构和图片尺寸；旧版 raw state_dict / Lightning
checkpoint 也可加载。旧版尺寸默认 224×224，可用 `--image-size 322 322` 覆盖。

## 6. RobotCar 与 GSV 派生图：显式评估 manifest

当前 RobotCar-Seasons 元数据没有可直接组合的全局检索 positives，因此
`--dataset RobotCar` 要求 `--eval-manifest`，或数据根目录下的
`evaluation_manifest.json`。代码不会把不同 COLMAP 分段坐标硬合并为真值。

所有数据集均可通过显式 manifest 覆盖默认协议。JSON 格式：

```json
{
  "dataset": "GSV-derived",
  "references": [
    {"id": "db1", "path": "references/db1.jpg"},
    {"id": "db2", "path": "references/db2.jpg"}
  ],
  "queries": [
    {
      "id": "aug1",
      "path": "generated/aug1.jpg",
      "positives": ["db1"],
      "source_id": "Bangkok/actual-source-filename.jpg"
    }
  ]
}
```

`path` 相对于 `--dataset-root`，也允许绝对路径；`positives` 使用 reference
ID，可包含多个正确匹配。`source_id` 仅用于有真实来源的 GSV 派生查询，
相对于 GSV 的 `Images/`，需要完整文件名。独立 benchmark 查询省略它。
reference 与 query 不能引用同一图片路径，以免发生自身匹配。

```bash
python salad/evaluate_salad.py \
  --checkpoint outputs/vpr_guidance/round_0/salad/checkpoint.pt \
  --dataset manifest \
  --dataset-root /path/to/evaluation-images \
  --eval-manifest /path/to/evaluation_manifest.json \
  --output outputs/vpr_guidance/round_0/evaluation/feedback_results.json \
  --save-hard-cases
```

评估输出包含 `recall.R@1/5/10`（0–1）、`error_queries`、协议、查询数量，
采用欧氏距离、1-based 精确 rank；多个 positives 取最先检索到的正确项。
分块完整检索不会把 >10 的 rank 截断。相同距离按 reference 顺序处理。
没有 positives 的查询不参与 recall 分母，数量在输出中单独报告。
CPU 描述子仍需内存；`--query-chunk-size` 与 `--reference-chunk-size`
限制距离计算的临时矩阵，可用 `--retrieval-device cpu` 独立指定检索设备。

## 7. 训练池挖掘与评估反馈

当前 23 城市源图筛选已经完成，生成计划保存在
`outputs/qwen_curriculum/state/combined_plan.jsonl`，每项均绑定精确的 GSV
源文件、内容校验和地点标签。继续生成使用现有计划和账本。

外部 benchmark 检索错误不会伪造 GSV `source_id`，不直接用作生成来源。
有真实来源映射的 GSV 派生查询可用于开发阶段诊断；独立保留评估集用于
固定协议下的 baseline、mix、replace 比较。

## 验证

```bash
python -m unittest discover -s salad/tests -v
python -m unittest discover -s AdaptVPR/experiments/qwen_curriculum/tests -v
```

覆盖真实/合成来源标签、曝光、检索排名、多个 positives、源图字段，以及
CPU 微型本地 backbone 下的实际 SALAD 优化、CLI 评估和 epoch 恢复。
微型 backbone 只用于测试，生产入口不提供替代模型开关。
