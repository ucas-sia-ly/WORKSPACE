# AdaptVPR 的 SALAD 训练与评估入口

新增入口为 `salad/train_salad.py` 和 `salad/evaluate_salad.py`，原有
`main.py` / `eval.py` 保持不变。新入口使用原仓库 SALAD 聚合器及 DINOv2，
使用原生 PyTorch 训练、MultiSimilarityLoss 和 MultiSimilarityMiner。
损失参数与原仓库一致：alpha=1、beta=50、base=0、miner epsilon=0.1。
无需 Lightning、FAISS 或 pytorch-metric-learning。

以下命令从 `/home/admin123/github/WORKSPACE` 执行。

可选的 patch 可靠性分支通过 `train_salad.py --reliability-ot` 开启，默认关闭。
实现、弱监督定义、训练/单图推理命令和测试见 [RELIABILITY_OT.md](RELIABILITY_OT.md)。
该分支只修改 `salad/`，使用已有源图配对，不调用生成模型。
固定教师缓存、轻量上下文及已知局部几何负监督见
[RELIABILITY_V2.md](RELIABILITY_V2.md)，默认保持旧模式。

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

## 3.2 Qwen 困难原图挖掘、固定预算生成与来源替换训练

新流程位于 `AdaptVPR/experiments/qwen_curriculum/`，按阶段执行，避免生成器、
验证器和训练模型同时占用内存。先从真实 GSV 训练池提取困难视角，再为每张
不同原图分配一个天气任务。独立评估集不参与挖掘或生成计划。

```bash
python AdaptVPR/experiments/qwen_curriculum/mine_sources.py \
  --checkpoint salad/checkpoint/dino_salad.ckpt \
  --real-data dataset/gsv-cities \
  --cities Bangkok BuenosAires LosAngeles Medellin \
  --output-dir outputs/qwen_curriculum/mining_4city \
  --num-sources 1000 --batch-size 16 --num-workers 0 \
  --query-chunk-size 128 --reference-chunk-size 1024 \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --seed 42

python AdaptVPR/experiments/qwen_curriculum/plan.py \
  --sources outputs/qwen_curriculum/mining_4city/sources.jsonl \
  --output-dir outputs/qwen_curriculum/generation_1000 \
  --num-images 1000 --seed 42 \
  --domain-weights night=0.5,snow=0.2,fog=0.2,rain=0.1

python AdaptVPR/experiments/qwen_curriculum/run.py generate \
  --run-dir outputs/qwen_curriculum/generation_1000 --max-calls 1000

python AdaptVPR/experiments/qwen_curriculum/run.py status \
  --run-dir outputs/qwen_curriculum/generation_1000
```

挖掘器使用同地点真实视角的留一法相似度和其他地点的困难负样本，按城市配额
选择，并限制每地点来源数。描述子和地点中心保存为磁盘映射数组；不用把全部
描述子读进 RAM。计划中的 1,000 个任务为 night 500、snow 200、fog 200、rain
100；标签经过固定种子洗牌。这组比例是当前提出的先验，不是 GIFT 的精确
配比。允许用 `--domain-stats` 提供含 `domain_weights` 和 `provenance` 的 JSON，
或显式修改权重；不分配 overcast/sun 任务。

计划复用发布版 Global 天气模板，负提示词为空；每张原图只计划一个变体。
原图实际内容 SHA-256、尺寸、挖掘信息、提示词和种子写入 `plan.jsonl`，
`plan_config.json` 固定请求及代码哈希。相同命令可恢复，修改配置需换输出目录。
Qwen 保存原图对应的训练尺寸 PNG；如另存高分辨率原始输出，两种文件分开保留。
`--max-calls` 限制本次新生成调用数，实际验收图片数以 `status` 为准。
`attempts.jsonl` 记录调用预算，`generated.jsonl` 记录已保存的图片字节，
`results.jsonl` 保留全部验收结果，`training_manifest.jsonl` 只供训练消费。

原有 Qwen 诊断结果可单独导入和重新验收，不消耗新的生成调用：

```bash
python AdaptVPR/experiments/qwen_curriculum/run.py import-existing \
  --manifest outputs/gen_diagnosis/qwen_source_aspect_100src/results.jsonl \
  --run-dir outputs/qwen_curriculum/reused_pilot
```

默认导入发布版提示词结果并排除 overcast；`--include-positive` 可另行纳入
positive 提示词结果。导入输出与新生成批次分别记录来源和验收结果。

生成、验证阶段结束后，先用下一节的 `train_compare.py prepare` 核验输入并固定
人工复核后的训练快照。释放 Qwen 模型后，再从所选四城市的完整真实训练池
微调 SALAD。以下是与正式对照设置一致的单支 `replace` 命令示例；正式
REAL/REPLACE 配对实验使用下一节的 driver：

```bash
python salad/train_salad.py \
  --real-data dataset/gsv-cities \
  --synthetic-manifest outputs/qwen_curriculum/compare_4city/accepted_manifest.jsonl \
  --cities Bangkok BuenosAires LosAngeles Medellin \
  --init-checkpoint salad/checkpoint/dino_salad.ckpt \
  --output-dir outputs/qwen_curriculum/manual_replace \
  --synthetic-mode replace --synthetic-fraction 0.5 \
  --images-per-place 4 --min-images-per-place 4 \
  --epochs 4 --batch-size 8 --num-workers 0 \
  --num-trainable-blocks 0 --image-size 224 224 \
  --learning-rate 1e-6 --weight-decay 0 --no-augment \
  --precision 32 --device cuda --save-every 4 \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --seed 42
```

首次使用与挖掘相同的四城市真实池，保留没有增强图的地点。当前四城市约
45,661 张真实图；1,000 张增强图扩展到全部 23 城市约 53 万原图时，训练曝光会
明显稀释。先固定保留评估集比较，再扩大真实池；省略 `--cities` 即使用全部
训练城市。`replace` 先按原
采样器选 K 个不同真实来源，再以 0.5 的概率替换有验收变体的来源槽位。
替换图必须对应该槽位的**精确 `source_path`**；同一来源的真实图与增强图不会
同时出现。无变体的来源仍使用原图；若 K 个槽位都被替换，则均匀选一个恢复
为真实图，所以每地点至少保留一个真实视角。多个条件时先均匀选条件，再选
条件内变体。磁盘上的原图全部保留。

这里的 0.5 是可替换来源槽位的概率，不是整批图片有 50% 合成；实际比例写入
`training_log.jsonl`。约 32 GB RAM 的机器建议从 8 地点 × 4 图片、0 workers
开始；图片按需解码，worker 预取仍会占用 RAM。正式四城市对照固定使用
0 workers 和 4 epochs。资源冒烟可另用独立目录与
`--epochs 1 --max-batches-per-epoch 2`，其结果独立记录。

训练配置与 checkpoint 保存 `synthetic_mode`；恢复必须使用原来的 mode 和
概率。旧 checkpoint 缺少该字段时按 `mix` 恢复，旧混合采样和 summary 格式保持
兼容。切换到来源替换是新的微调实验，使用新的输出目录和 `--init-checkpoint`。
比较时使用相同真实池、初始化、训练预算及保留评估集，分别运行真实图 baseline、
`mix` 和 `replace`。域权重和阈值在训练/开发数据上确定；最终测试集用于一次
固定协议比较，不根据测试结果继续调整生成计划。

## 3.3 固定 REAL / REPLACE 四城市对照

`AdaptVPR/experiments/qwen_curriculum/train_compare.py` 按顺序训练两支模型。
两支都从 `salad/checkpoint/dino_salad.ckpt` 初始化，使用 Bangkok、BuenosAires、
LosAngeles、Medellin 的完整可用真实训练池，包含没有增强图的地点。REAL
使用原图；REPLACE 按精确来源以 0.5 概率替换合格变体。共同设置固定为
4 epochs、冻结全部 DINOv2、仅训练 SALAD 聚合器、学习率 `1e-6`、
weight decay=0、无图像增广、224×224、FP32、每批 8 地点 × 4 视角、
0 workers、seed=42。最终只使用第 4 epoch，不按测试分数选择 checkpoint。

生成完成后，可先核验数据和固定协议，不加载训练或评估模型：

```bash
PYTHON=/home/admin123/miniconda3/envs/AdaptVPR/bin/python

$PYTHON AdaptVPR/experiments/qwen_curriculum/train_compare.py prepare \
  --generation-run-dir outputs/qwen_curriculum/generation_1000 \
  --output-dir outputs/qwen_curriculum/compare_4city \
  --review-exclusions outputs/qwen_curriculum/review_exclusions.jsonl \
  --evaluate-svox

$PYTHON AdaptVPR/experiments/qwen_curriculum/train_compare.py check-data \
  --generation-run-dir outputs/qwen_curriculum/generation_1000 \
  --output-dir outputs/qwen_curriculum/compare_4city \
  --review-exclusions outputs/qwen_curriculum/review_exclusions.jsonl \
  --evaluate-svox
```

`prepare` 和 `check-data` 要求生成 `summary.json` 为 `complete`，且所有计划
任务都有唯一结果。它们校验计划、调用账本、验收记录、真实来源、原始输出和
训练尺寸图片的哈希，并确认 REAL/REPLACE 使用相同的可用真实地点。
当前 `review_exclusions.jsonl` 按 `sample_id`、`source_sha256`、
`output_sha256` 和复核理由排除两张已确认问题图；只改变
`compare_4city/accepted_manifest.jsonl` 的冻结快照，保留生成目录的机器验收
记录和全部图片。排除清单的内容及 SHA-256 写入 `comparison_config.json`；
冻结后变更清单或训练设置需使用新输出目录。

生成仍在进行时，可显式启动等待完整生成后执行的对照命令：

```bash
$PYTHON AdaptVPR/experiments/qwen_curriculum/train_compare.py run \
  --generation-run-dir outputs/qwen_curriculum/generation_1000 \
  --output-dir outputs/qwen_curriculum/compare_4city \
  --wait-for-generation \
  --generation-service qwen-curriculum-1000.service \
  --stop-qwen-service \
  --review-exclusions outputs/qwen_curriculum/review_exclusions.jsonl \
  --evaluate-svox
```

等待每 60 秒只读取摘要和 systemd worker 状态，不加载模型或打开图片。
完整摘要出现且生成 worker 退出后才继续；worker 停止但摘要不完整、停止状态
或预算耗尽都会使训练中止。通过全部数据哈希核验后，`--stop-qwen-service`
检查 `adaptvpr-lightx2v.service` 的 `ExecStart` 确实属于本工作区，再停止该
Qwen 服务以释放资源。训练与评估模型按阶段加载。

`--evaluate-svox` 是可选的固定终点评估。它使用原生 SVOX test 的完整
17,166 张 gallery、14,278 张日间查询和 823 张夜间查询，25 米正样本半径、
224×224、FP32。两支模型各提取自己的图库特征，仅评估第 4 epoch；测试集
不用于调整训练预算、域配额或挑选 checkpoint。省略该 flag 则只完成训练。
输出包含冻结配置与验收快照、`real/` 和 `replace/` checkpoint、分支训练日志、
可选的 `evaluation/` 结果，以及带 checkpoint 哈希的 `comparison.json`。

中断后重复相同 `run` 命令会在 epoch 边界恢复未完成的分支。已有最终
checkpoint 会按协议检查后跳过该分支；完整 `comparison.json` 及两支权重
校验通过时直接返回，不重新加载模型。输入、快照或代码指纹变化会拒绝复用。

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

当前生成计划由 3.2 节的 `mine_sources.py` 从真实训练元数据构建，来源始终是
精确的 GSV 文件和地点标签。使用新的学生 checkpoint 挖掘下一轮时，换一个
输出目录，保留原有计划、生成结果及训练配置，便于比较。

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
