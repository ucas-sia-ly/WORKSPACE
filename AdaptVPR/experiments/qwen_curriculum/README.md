# Qwen 困难样本增广

**当前 2000 张续跑使用 [fast_campaign.py](fast_campaign.py)**，后台服务为 `qwen-curriculum-2000-fast.service`。4090 48 GiB 配置保持原 BF16、4 步、guidance=1、源比例大画布，将文本编码器/VAE 和前32个 DiT block 常驻 GPU，其余 block 双缓冲分块加载。单张同图同种子对照实测 63.53s → 26.99s，原始 PNG 内容 SHA-256 相同；首次请求需要加载显存缓存，不代表稳定速度。完整配置、恢复方法和 **2000 张训练终端指令**见 [GENERATION_2000_FAST.md](GENERATION_2000_FAST.md)。

原阶段已完成 998 张，另外两张中断时已落盘的输出转入独立零调用恢复阶段，原始错误账本保留。再生成原扩展计划的1000张，累计目标仍为2000张。累计进度在 `outputs/qwen_curriculum/generation_2000/summary.json`，生成日志为 `fast_generation.log`；原 `qwen-curriculum-2000.service` 已失败停止，不能再使用下文的旧协调器恢复命令。当前700张训练已经结束，2000张训练由用户在生成完成后启动。

新增模块实验使用 `--reliability-ot`，沿用已完成原版实验的700张固定图片池，重新训练两组配对共四个模型，自动输出原版/新版 SVOX 比较表。训练指令和辅助 source 计数见 [RELIABILITY_700_EXPERIMENT.md](RELIABILITY_700_EXPERIMENT.md)。当前先继续生成，由用户执行训练命令，不自动启动训练。

当前正式训练任务已改为 **700 张生成图、8:1 和 4:1 的两组配对对照，共四次训练、六个 SVOX 子集评估**。四组保留 DINOv2 预训练、SALAD 随机初始化，从新的 VPR 训练开始（50轮，学习率6e-5，训练最后4个 backbone block 和聚合器），不加载预训练 VPR checkpoint。使用 [test_700_ratio_experiment.py](test_700_ratio_experiment.py)，配置和查看方式见 [RATIO_700_EXPERIMENT.md](RATIO_700_EXPERIMENT.md)。此前的微调和全城市池任务已停止。下文保留生成管线说明和历史全池训练方案。

以下为此前排队方案的历史说明：生成目标扩展为累计2000张，在700张训练结束后由 `campaign.py` 续跑。该旧任务随后因冻结配置类型不一致退出，目前由上述独立性能配置续跑阶段接替。

累计计划使用原有困难度缓存，四城市各 500 个源图，覆盖 1491 个地点，每地点最多 2 个源图，源路径和内容均不重复。域配额为 night=1000、snow=400、fog=400、rain=200。先完成原 `generation_1000`，再完成独立的 `generation_2000/additional_1000`；原计划、图片、结果校验和与调用账本保持有效，汇总清单引用原图片路径，不复制图片。

排队时已有 757 条生成记录，另有 1 张已落盘的中断输出可零调用恢复；因此预计还需要 1242 次新调用。2000 是生成图片总数，自动验收通过数另行统计。续跑沿用 Qwen-Image-Edit-2511、4 步、guidance=1 和 `source_aspect_v1`，不重新生成已完成的图片，不重复发起结果不明的调用，也不自动启动另一轮训练。

旧排队任务的历史日志：

```bash
systemctl --user status qwen-curriculum-2000.service
cat outputs/qwen_curriculum/generation_2000/summary.json
tail -f outputs/qwen_curriculum/generation_2000/worker.log
```

旧日志的 `waiting_for_training` 表示当时正在等待完整训练。当前恢复使用新的协调器，并保留相同累计目录：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/fast_campaign.py run \
  --output-dir outputs/qwen_curriculum/generation_2000
```

当前推荐入口。流程固定为 **真实训练图难度挖掘 → Qwen 单次天气编辑 → 结构/天气筛选 → 同一源图概率替换训练**。不生成 overcast，不运行生成器 LoRA 训练，不在每张生成图上重载 SALAD，不根据验收失败连续加码编辑。

初始冻结阶段使用 Bangkok、LosAngeles、Medellin、BuenosAires 四个训练城市，预算为 1000 次生成调用，分配如下；扩展阶段沿用相同比例。

| 域 | 张数 | 比例 |
|---|---:|---:|
| night | 500 | 50% |
| snow | 200 | 20% |
| fog | 200 | 20% |
| rain | 100 | 10% |

这是初始实验先验。[GIFT 官方仓库](https://github.com/kuzhengyu/GIFT)公布了约 14 万张、三个城市、六类天气的合成数据，并指向 BoQ/VLAD-BuFF 的训练入口；没有公布可核实的域采样比例或源图难度挖掘实现。这里不宣称精确复现 GIFT，也不宣称未经对照实验的最优比例。以后可通过独立训练/验证划分上的域退化统计，以 `--domain-stats` 覆盖比例；不能用最终测试集调比例。

## 选择哪些图

用已发布的 `salad/checkpoint/dino_salad.ckpt` 作为固定教师，仅提取真实 GSV 训练图特征。每张图的难度为：

`最近其他地点质心相似度 - 同地点其余视角质心相似度`

正样本质心排除当前源图；负样本排除同地点和 25 米内的相邻地点，减少相邻街景的假负样本。每城市保留可靠正样本相似度至少 0.15、难度处于 70%–97.5% 分位的候选，优先选择较难图，每地点最多两张，每城市默认 250 张。上端分位截断是避免异常样本的启发式，并不能保证识别全部错标。配额不足时明确报错；不偷偷降低标准。

特征、地点累加及质心保存在磁盘 memmap；提取批次 16、查询块 128、参考块 1024、数据加载 workers=0。缓存绑定模型、CSV、图像路径/大小/mtime、预处理及代码；选定源图另验证内容 SHA-256。全库缓存检查不读取全部图片内容，移动/修改数据前需清理或重新生成缓存。

## 生成与验收

沿用 `outputs/gen_diagnosis/qwen_source_aspect_100src/images` 的 Qwen-Image-Edit-2511 + Lightning 4steps、guidance=1、`source_aspect_v1` 大画布。保持 released prompt 和空 negative prompt，保留原始大图，训练图仅 LANCZOS 缩回源图尺寸。每个不同源图生成一个随机域，域标签使用精确配额独立洗牌，避免源图排序与天气相关。

结构验收检查内点率、匹配数、空间覆盖和 homography 位移；天气验收检查 CLIP 的目标域相对晴天对比及相对源图的正向变化。`s_div` 只记录，不再以差异超过 0.15 作为通过条件。CLIP 是弱语义信号；首轮门限属于实验设置，需要人工复核。通过标记不等于天气或地点身份的绝对保证，生成质量也不直接证明 Recall 提升。

每次 POST 前原子保存预算预留；失败和中断未知结果也计入预算，不自动重试。图像保存后验收失败可以继续验收，不消耗新调用。并发锁防止同目录双 worker。计划、运行配置、结果、图像都有内容校验；配置变更使用新目录。验收合格图持续写入 `training_manifest.jsonl`。

## 在工作区执行

以下命令从 `/home/admin123/github/WORKSPACE` 执行，使用已安装的 AdaptVPR Python 环境。

```bash
PYTHON=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
systemctl --user stop adaptvpr-iclight.service

$PYTHON AdaptVPR/experiments/qwen_curriculum/mine_sources.py \
  --checkpoint salad/checkpoint/dino_salad.ckpt \
  --real-data dataset/gsv-cities \
  --cities Bangkok LosAngeles Medellin BuenosAires \
  --output-dir outputs/qwen_curriculum/mining_4city \
  --num-sources 1000 --batch-size 16 --num-workers 0 \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main

$PYTHON AdaptVPR/experiments/qwen_curriculum/plan.py \
  --sources outputs/qwen_curriculum/mining_4city/sources.jsonl \
  --output-dir outputs/qwen_curriculum/generation_1000 --num-images 1000

$PYTHON AdaptVPR/experiments/qwen_curriculum/run.py generate \
  --run-dir outputs/qwen_curriculum/generation_1000 --max-calls 1000

$PYTHON AdaptVPR/experiments/qwen_curriculum/run.py status \
  --run-dir outputs/qwen_curriculum/generation_1000
```

CPU CLIP 与几何匹配共用特征缓存。生成阶段不加载 SALAD。生成完毕并退出 worker 后，停止 Qwen 服务释放主存，再开始 SALAD 训练。

```bash
systemctl --user stop adaptvpr-lightx2v.service
$PYTHON salad/train_salad.py \
  --real-data dataset/gsv-cities \
  --cities Bangkok LosAngeles Medellin BuenosAires \
  --synthetic-manifest outputs/qwen_curriculum/generation_1000/training_manifest.jsonl \
  --synthetic-mode replace --synthetic-fraction 0.5 \
  --init-checkpoint salad/checkpoint/dino_salad.ckpt \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --epochs 4 --learning-rate 1e-6 --num-trainable-blocks 0 --save-every 4 \
  --batch-size 8 --num-workers 0 --precision 32 --no-augment --weight-decay 0 \
  --output-dir outputs/qwen_curriculum/salad_replace
```

先加 `--check-data` 检查映射。首轮为相同初始化、四轮、小学习率、冻结 backbone 的保守微调，纯真实图 baseline 使用相同训练预算；这组设置不是已证明的最优超参数。每批抽取不同真实源视角，拥有合格变体的源图以 0.5 概率替换为该源图变体，每地点样本至少保留一个真实视角。原图文件保留，其他真实训练图照常参与。不同时采样同一源图的真实和合成版本来凑正样本数量。

1000 张相对于全 23 城市较稀疏，首轮对照优先用上述四城市的完整真实池，记录实际合成曝光率；不能把 `synthetic-fraction=0.5` 误读成全部训练图片的一半是合成图。验证有效后再扩大全库或评估困难地点重采样。对照至少包含相同初始化/城市/训练预算的真实图 baseline 与 replace；最终评估使用独立域/地点协议。

现有 Qwen 图片可零生成调用复用，默认只纳入 released prompt 的四个目标域，原路径和原始图片保持不动：

```bash
$PYTHON AdaptVPR/experiments/qwen_curriculum/run.py import-existing \
  --manifest outputs/gen_diagnosis/qwen_source_aspect_100src/results.jsonl \
  --run-dir outputs/qwen_curriculum/reused_pilot
```

可用 `--include-positive` 将另一套 prompt 一并验收。历史复用图片来自随机源图，不算作新增的 1000 张困难源图；该 pilot 主要用于新验收规则诊断。

## 历史全池对照方案（本次不执行）

`train_compare.py` 串行完成两个匹配的训练组：REAL 使用四城市完整真实池，REPLACE 在相同数据池中按源图替换。两组均使用同一个已发布 checkpoint、seed=42、4 epochs、冻结 backbone、学习率 1e-6、无额外图像增强、FP32、batch=8 地点 × 4 图、workers=0。仅比较最终第四轮 checkpoint，不用测试结果筛选超参数或 checkpoint。

```bash
$PYTHON AdaptVPR/experiments/qwen_curriculum/train_compare.py run \
  --generation-run-dir outputs/qwen_curriculum/generation_1000 \
  --output-dir outputs/qwen_curriculum/compare_4city \
  --wait-for-generation --generation-service qwen-curriculum-1000.service \
  --stop-qwen-service --evaluate-svox \
  --review-exclusions outputs/qwen_curriculum/review_exclusions.jsonl
```

等待阶段仅读取 JSON 和服务状态。生成完成且 worker 退出后，核验全部计划、记录、图片、预算及源图标签，固定训练快照，然后停止本工作区的 Qwen 服务再训练。生成异常停止时不会转入部分数据训练。`prepare` 可以只导出固定配置；`check-data` 只检查两组数据。训练可从各组最近 epoch 恢复，已完成报告复用时校验其配置和 checkpoint。

机器接受清单与训练快照分开。`--review-exclusions` 要求精确 sample_id、源图 SHA、输出 SHA 和理由，排除有局部立面替换或在隐私模糊区域编造细节的已知失败，保留原始生成记录。当前的两条排除来自 `outputs/qwen_curriculum/review_357` 的 32 张分层抽查；该抽样不能估计全体误判率，也没有用于放宽自动门限。

本地完整 SVOX 测试协议可用：gallery 17,166，日间 queries 14,278，夜间 queries 823；25m UTM 正例、FP32 224×224、同一 gallery 顺序与精确检索。每个 checkpoint 只提取一次共用 gallery。`comparison.json` 保存 REAL、REPLACE 的 Recall@1/5/10 和百分点差值。未出结果前不宣称 Recall 改善。

历史后台任务及查看方式（当前生成入口见本文开头）：

```bash
systemctl --user status qwen-curriculum-1000.service qwen-curriculum-compare.service
tail -f outputs/qwen_curriculum/generation_1000/worker.log
tail -f outputs/qwen_curriculum/generation_1000/compare_worker.log
```

关闭聊天不会终止这些 systemd 用户任务；机器关机或用户服务管理器退出后，需要依据保存记录恢复。完成后的比较报告位于 `outputs/qwen_curriculum/compare_4city/comparison.json`。需要下一轮生成时手动重新启动 `adaptvpr-lightx2v.service`。
