# 从零负监督的实验继续改进 Reliability OT

这次只修改 `salad/`。原实验目录、AdaptVPR 和 Qwen 的已有文件保留。
`--reliability-ot` 仍默认关闭；v1 checkpoint、原模型推理接口和
MultiSimilarityLoss 保持兼容。新增设置均需明确开启。

## 已核实的问题

原版及 v1 实验使用相同的 700 张生成图、源图调度和原模型初值。
六个 SVOX 域的等权 R@1 如下；它不是按查询数加权的官方总 Recall。

| 生成组 | 原 SALAD | Reliability v1 | 差值 |
|---|---:|---:|---:|
| 8:1 | 79.18% | 75.66% | −3.52 pp |
| 4:1 | 64.39% | 67.78% | +3.39 pp |

两种比例还改变了图片池和更新预算：50 轮分别为 9,850 和 5,500 次更新，
不能把两个比例之间的差异全部归因于生成比例。

v1 两个生成组 50 轮的负 patch 标签均为 **0**，平均监督覆盖率仅
1.016% / 1.439%，最后的 `r≈0.9498`，空间标准差约 0.002。
正锚点和真实图先验都指向 0.95，因此统一高可信度是允许的解；均值下限
从未激活。原来的 teacher 虽然不回传梯度，但共享正在更新的 DINO，
并不是固定教师。辅助损失的数值大于 metric loss 不直接等于梯度更大，
软目标 BCE 本身还有非零熵下界。

SALAD 的 dustbin 边际目标质量仍为 N−M；均匀的 `lambda*(1-r)` 会作为行
偏置被抵消。必须让 head 学到空间差异，而不是单纯把 lambda 加大。
完整数据核验记录在 `outputs/reliability_v2/experiment_audit.json`。

## 本次实施

1. **固定教师目标缓存。** 离线用同一个冻结 DINO 对源图和生成图提取特征，
   沿用保守的结构对应规则，只保存 target/confidence。训练不保留额外教师，
   不再执行源图前向。源图、生成图、manifest、教师 checkpoint 和 tensor
   payload 都记录哈希；缓存不匹配或训练图片改变会报错。初版缓存要求
   `--no-augment`，避免标签网格与图像增广错位。
2. **有已知位置的几何负监督。** 只对批中的真实图临时复制一个远处区域，
   最多覆盖四分之一 patch 网格。只把像素实际改变的位置用于低可信度监督，
   平坦图或相同区域的无效复制不产生负标签。受损图只进入辅助前向，原检索
   图、标签和采样曝光保持不变。受损区域外对齐 detached 的干净预测，避免
   通过降低整张受损图的可信度来满足局部负样本。
3. **轻量邻域上下文。** 在现有 head 隐层加入残差 depthwise 3×3 Conv。
   hidden=64 时只增加 **640 个参数**，head 总计 49,921 个参数。
   原 head 的权重键名保留；初始输出仍为 0.9，不改变初始 OT 分配。
4. **隔离辅助梯度和随机数。** `--reliability-detach-features` 阻断 predictor
   到 DINO 的梯度，DINO 的原 cluster/score 检索分支仍正常接收 VPR 梯度。
   新 head 初始化不消耗主训练的 RNG，几何破坏使用独立的每步 generator，
   避免额外随机操作改变主分支的 dropout 序列。
5. **明确消融。** 固定 `r=0.9`、只用 VPR 学 head、完整辅助监督可以分别运行。
   固定模式的 checkpoint 会保存该模式，raw/native 加载均能保持单图行为。

未知生成 patch 仍不打负标签，自动拒绝或天气验证失败也不直接变成错误 mask。
几何破坏是可控的训练代理任务，不是 Qwen 幻觉的真值；其泛化收益必须独立验证。
目前没有增加新的生成调用、大型模型或 GPU 服务。

## 已准备的缓存与数据边界

已将旧实验的 700 行 manifest 固定到：

```text
salad/outputs/reliability_v2/fixed_700.jsonl
salad/outputs/reliability_v2/frozen_targets/index.json
salad/outputs/reliability_v2/frozen_targets/targets.pt
```

现有标准训练入口严格过滤验收结果，因此实际使用 **455 张通过图**，忽略
245 张拒绝图。冻结缓存包含 3,787 个正锚点、0 个结构负锚点及 112,693 个
未知 patch，覆盖率 3.251%；tensor 文件为 933,829 字节。这再次说明固定
teacher 无法自动创造负标签，几何负监督仍须独立提供。

**这个新 cohort 不等于原来包含全部 700 张的固定比例实验。** 以下命令也从
已训练的原版模型微调，而不是重新随机初始化 SALAD。判断模块收益时，应在
同一新 cohort、同一初始 checkpoint、同一更新预算上重跑相应消融，不能直接
把新结果与旧的 50 轮 scratch 结果相减。

需要为另一份固定 manifest 生成缓存时使用下面命令，并换新的输出目录：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/cache_reliability_targets.py \
  --real-data dataset/gsv-cities \
  --synthetic-manifest salad/outputs/reliability_v2/fixed_700.jsonl \
  --checkpoint salad/checkpoint/dino_salad.ckpt \
  --output-dir salad/outputs/reliability_v2/new_frozen_targets \
  --cities Bangkok BuenosAires LosAngeles Medellin \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --image-size 224 224 --batch-size 8 --device cuda
```

缓存建好以后，教师进程结束。已完成的缓存拒绝覆盖；恢复训练无需原教师文件。

## 建议的短程训练

以下命令从 WORKSPACE 执行，使用已准备好的缓存。先冻结 DINO、以较低学习率
微调原聚合器，以独立学习率训练新 head。这是初始验证设置，尚未用 Recall
选优。每批 32 张检索图，最多另加 4 张辅助受损图。

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/train_salad.py \
  --real-data dataset/gsv-cities \
  --synthetic-manifest salad/outputs/reliability_v2/fixed_700.jsonl \
  --cities Bangkok BuenosAires LosAngeles Medellin \
  --init-checkpoint outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1/generated_8to1/checkpoint.pt \
  --output-dir salad/outputs/reliability_v2/train \
  --synthetic-places-only --synthetic-mode mix --synthetic-fraction 0.5 \
  --epochs 3 --batch-size 8 --images-per-place 4 --min-images-per-place 4 \
  --num-trainable-blocks 0 --learning-rate 1e-6 --weight-decay 0 \
  --reliability-ot --reliability-context --reliability-detach-features \
  --reliability-target-cache salad/outputs/reliability_v2/frozen_targets \
  --reliability-head-lr 1e-4 --reliability-corruption-weight 0.05 \
  --reliability-corruption-max-images 4 \
  --no-augment --precision 16 --num-workers 0 --device cuda \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --seed 42
```

`synthetic-fraction` 是名额上限语义；不少地点只有一张生成图，实际曝光会小于
0.5。正式比较使用各组相同的曝光和采样设置，并查看日志，而不是假定比例。
需要检查输入时增加 `--check-data`；小试跑时换输出目录并添加
`--max-batches-per-epoch 2`。

新结构必须从不带 reliability head 的原 SALAD checkpoint 初始化。已有 v1
head 的 checkpoint 可以按 v1 结构恢复，但不会静默为其插入新 context 参数。
断点恢复时去掉 `--init-checkpoint`，改用 `--resume`，并保持该阶段原配置。

## 三个必要消融

每组沿用同一 manifest、初始模型、采样参数、seed 和总更新预算，使用新输出目录。

- **固定可信度、无辅助损失**：使用 `--reliability-ot --reliability-mode fixed`，
  将 `--reliability-loss-weight`、`--reliability-real-prior-weight` 和
  `--reliability-coverage-weight` 都设为 0；不传 context、cache 或 corruption。
  用于隔离新版 FP32/einsum 聚合路径与原路径的影响。
- **只用 VPR 学 head**：开启 context 和 detach，将上述三种权重设为 0，
  不传 cache/corruption。它不会执行源图 teacher 前向。
- **完整辅助监督**：使用上面的训练命令。日志分别记录结构负标签
  `negative_patches` 和已知破坏负标签 `corruption_negative_patches`，避免把
  代理任务产生的负标签误报为 Qwen 错误检出。

也需保留不带 `--reliability-ot` 的原版对照。训练/独立验证地点分离后检查
`r` 热图、内外可信度差及 OT 保留质量的空间变化，再投入多 seed 完整训练。
SVOX 用于最终报告，不用它反复调局部标签阈值或选择中间 checkpoint。

## 验证与局限

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -m unittest discover -s salad/tests -q
```

新增测试覆盖缓存身份/哈希、未知标签、context/detach/fixed 模式、几何破坏
mask、无效复制、独立随机流、masked loss 和 AMP。实际 CPU 训练/恢复测试
验证模型、优化器及 RNG 完全一致，且 context 参数有非零更新。原版及 v1
单元测试继续运行。

完整单元测试为 **147 项通过**。

最终 GPU 检查位于 `salad/outputs/reliability_v2/gpu_smoke_final/`；从旧实验的
原 SALAD 8:1 生成组 checkpoint 初始化，使用现有固定生成图、缓存目标和
FP16，每批 8 地点 × 4 张图，最多另加 4 张受损真实图。两轮、每轮两批，
进程 peak allocated 为 **589.37 MiB**、reserved 为 **732 MiB**；累计提供
432 个已知破坏负 patch。head/context 和优化器状态有限，context 的优化器
更新非零；加载后单图描述子为 `[1, 8448]`，L2 范数为 1。
记录在 `gpu_smoke_final.json`；这检查执行正确性，不用于评估 Recall。

另做了按地点分离的定位能力检查：四城市共 32 个不同源图地点，16 个训练、
16 个留出；冻结 DINO，一次缓存真实图和受损图特征，每个 head 仅训练 100 步。
两个 head 使用相同的图、mask、初始化及训练批次，lr=1e-3、均衡干净/受损
loss。这是单独的 head 容量诊断，不等同于上面的 VPR 微调配置。

| 留出 patch 指标 | 无 context | 有 context |
|---|---:|---:|
| 已知局部破坏 AUROC | 0.9072 | 0.9209 |
| mask 内平均 r | 0.4030 | 0.4075 |
| mask 外平均 r | 0.8196 | 0.8177 |

context 的 dustbin 概率相对固定 r：mask 内提高约 6.35 个百分点，外部下降
约 0.64 个百分点。说明补充明确负监督后，head 和 OT 确实能形成空间筛选，
不再仅靠统一高可信度满足目标。无 context 也达到较高 AUROC，表明负监督
缺失比 head 大小更关键；这一次小划分不足以证明 context 更好。

脚本、完整源图/随机种子 provenance、逐图指标及已检查热图位于
`salad/outputs/reliability_v2/localization_pilot/`。可复现命令为：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  salad/outputs/reliability_v2/localization_pilot/run_pilot.py
```

图中保留了较弱定位和 mask 外误报的案例。上述 AUROC 仅衡量主动制造的
copy/paste 几何破坏，不代表自然 Qwen 错误检出率或检索 Recall。

仍存在的限制：控制复制不覆盖全部生成错误；未知幻觉仍可能没有监督；冻结
标签只解决漂移，不保证标签正确或充分；均值下限与外部一致性均为软惩罚。
目前没有完整 v2 Recall 结果，因此没有跨域提升的结论。

## 本次代码文件

- `models/aggregators/salad.py`：context、detach、固定模式、初始化 RNG 隔离。
- `workflow/model.py`：新增结构的严格加载与 raw checkpoint 识别。
- `workflow/reliability.py`：支持 detached 的缓存目标。
- `workflow/training_data.py`：静态目标读取，避免源图解码和前向。
- `workflow/reliability_cache.py`、`cache_reliability_targets.py`：缓存及离线构建入口。
- `workflow/reliability_corruption.py`：真实图几何负监督和区域外一致性。
- `train_salad.py`：新开关、辅助项、独立随机流、优化器分组和恢复校验。
- `tests/test_reliability_cache.py`、`test_reliability_corruption.py`、
  `test_reliability_v2.py`：新增测试。
- `RELIABILITY_V2.md`、`RELIABILITY_OT.md`、`WORKFLOW.md`：使用说明。
