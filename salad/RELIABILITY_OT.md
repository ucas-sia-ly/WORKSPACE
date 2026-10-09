# SALAD 的可选 patch 可靠性分支

针对实际训练中“零负标签、统一高可信度”的后续改进，见
[RELIABILITY_V2.md](RELIABILITY_V2.md)。下面保留 v1 接口与实验使用说明。

全部实现和测试位于 `salad/`。DINOv2、原 SALAD 的 token/cluster/score
分支、描述子维度和 MultiSimilarityLoss 保留。训练入口是 `train_salad.py`；
原来的 `main.py` / `eval.py` 不变。`--reliability-ot` 默认关闭，关闭时不增加
head 参数、checkpoint key、数据返回值、优化器分组或可靠性训练配置。

## 机制与约束

开启后，在 DINOv2 局部特征上增加 `1×1 Conv → ReLU → 1×1 Conv → Sigmoid`，
得到每个 patch 的 `r`。ViT-B、hidden=64 时增加 49,281 个可训练参数。
原来的 dustbin 标量改成逐 patch 分数：

```text
dustbin_score[b, i] = original_dustbin + lambda * (1 - r[b, i])
loss = MultiSimilarityLoss
       + 0.1 * structural_weak_BCE
       + 0.01 * real_view_prior
       + 0.1 * mean_reliability_floor_penalty
```

默认 `lambda=2`，真实图先验是软目标 `r=0.95`；每张图的可靠性均值低于
0.5 时受到平方惩罚。辅助权重、均值下限和 head 学习率均可通过 CLI 修改。
head 最后一层初始化为零权重、`logit(0.9)` 偏置，因此初始 `r=0.9`。
统一增加 dustbin 行分数是 Sinkhorn 的行偏置对称性，初始分配不会改变；
加载旧模型时仅初始化这四个新参数及一个 lambda buffer，全部原始权重严格加载。

**SALAD 的 OT 边际质量固定。** 有 N 个 patch、M 个 cluster 时，收敛的
dustbin 行质量是 N−M。可靠性偏置改变的是哪些 patch 更倾向于 dustbin，
不能通过它任意增加全局丢弃比例。原版三次 Sinkhorn 迭代保留；有限迭代时
行边际存在近似误差。因此防塌缩项约束的是 `r`，而不是把固定 dustbin 质量
当作可学习的丢弃率。开启模式的 OT 和聚合累加使用 FP32，训练其余部分可用
FP16/BF16；新分支用矩阵乘法聚合，避免原版四维重复张量。

## 源图配对和弱监督

使用 manifest 的精确 `source_path → output_path` 关系，源图必须属于同一
GSV metadata 地点。增强图及源图共享 resize、水平翻转和颜色增广参数，避免
增广本身产生伪错位。源图只是训练时的 companion，不进入检索损失的额外图像槽位，
也不会改变原有采样器的地点标签、替换比例或至少一个真实视角的约束。

训练时使用同一个 DINOv2，无梯度、eval 模式提取被选中的源图特征。局部特征
先在每张图内中心化，再计算邻域自相似结构。结构签名在全局通道正交变换、
统一缩放及统一特征偏移下不变，但这不保证对所有真实天气变化都不变。

- **可信正样本**：结构对应唯一、双向一致，而且仍位于允许的一 patch 坐标容差内。
- **低可信度负样本**：有唯一、双向一致的结构对应，但偏移超过容差；原坐标
  结构更不一致，且至少三个邻居支持相同偏移。
- **未知区域**：平坦、模糊、没有可靠对应或只有低跨图特征相似度时，监督权重为零。

不使用天气类别作为错误标签，不设负样本配额，不因夜晚、雨雪或低 cosine
直接扣分。正负类 BCE 分别归一化，避免多量正样本淹没少量结构负样本。
目标和源特征全部 detached；不加载额外匹配器、生成模型或 GPU 服务。

这是保守的结构弱监督，**不是生成错误的真值检测器**。新造且没有对应的立面
可能仍是未知；严重天气遮挡也可能干扰结构证据。日志记录 positive/negative/
unknown patch 数、supervised_fraction、可靠性均值及空间标准差，必须结合样图和
独立检索评估判断。经过筛选的生成图可能几乎没有结构负锚点；均匀高 `r` 的
新增偏置会抵消，此时分支可能作用很小。均值下限是软惩罚，不是严格保证。
没有足够结构证据时允许全可信，不能为了训练 head 人为制造错误标签。

## 训练命令

以下命令在 WORKSPACE 执行。先把当前可用 manifest 固定在 `salad/` 中，
避免生成过程中持续追加 manifest 导致训练输入和恢复校验变化。
此命令只是准备快照，不启动或修改 Qwen。

```bash
mkdir -p salad/outputs/reliability_ot
cp outputs/qwen_curriculum/generation_1000/training_manifest.jsonl \
   salad/outputs/reliability_ot/accepted_manifest.jsonl
```

先从完整预训练 SALAD 开始，在有增强图的地点学习新分支。冻结 DINOv2，
给原聚合器较低学习率、给新 head 单独的学习率；每批 8 地点 × 4 视角。

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/train_salad.py \
  --real-data dataset/gsv-cities \
  --synthetic-manifest salad/outputs/reliability_ot/accepted_manifest.jsonl \
  --cities Bangkok BuenosAires LosAngeles Medellin \
  --init-checkpoint salad/checkpoint/dino_salad.ckpt \
  --output-dir salad/outputs/reliability_ot/warmup \
  --synthetic-places-only --synthetic-mode mix --synthetic-fraction 0.5 \
  --epochs 3 --batch-size 8 --images-per-place 4 --min-images-per-place 4 \
  --num-trainable-blocks 0 --image-size 224 224 \
  --learning-rate 1e-6 --weight-decay 0 \
  --reliability-ot --reliability-lambda 2 --reliability-head-lr 1e-4 \
  --no-augment --precision 16 --num-workers 0 --device cuda \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --seed 42
```

需要 metadata 检查时添加 `--check-data`；需要两批试跑时使用新输出目录，
并改为 `--epochs 1 --max-batches-per-epoch 2`。
辅助 loss 默认值已列在上面，无需每次指定。`--reliability-hidden-dim` 默认 64。

随后可用 warmup checkpoint 初始化，在完整四城市真实池做来源替换训练：
沿用上面命令，将 `--init-checkpoint` 改为
`salad/outputs/reliability_ot/warmup/checkpoint.pt`，换新输出目录，去掉
`--synthetic-places-only`，改为 `--synthetic-mode replace`。
完整池中源图变体稀疏时，实际 synthetic 和配对曝光会减少；先用有增强图的
地点学习分支可以减轻纯真实先验主导的问题。三轮 warmup 是起始实验设置，
尚未通过 Recall 实验选优；是否进行后续阶段应由固定验证集决定。

断点续训：重复该阶段原命令，去掉 `--init-checkpoint`，增加
`--resume <该阶段的 checkpoint_epoch_001.pt>`。总 epochs、数据和损失参数必须
一致。已训练 reliability checkpoint 必须显式带 `--reliability-ot`；不能默默
删掉 learned head。lambda 和 hidden dim 从 checkpoint 恢复，若显式指定则必须
与保存值一致。旧 checkpoint 初始化新 head 是明确 opt-in 的例外，恢复和推理
均严格要求所有参数完整。

为判断收益，使用同一固定 manifest、初始 checkpoint、随机种子和采样配置
比较原 SALAD 与开启分支的 SALAD，再在独立 day/night/weather 检索集评估。
本次实现测试并不证明 Recall 提升，也没有运行完整微调实验。

## 单图推理与评估

推理只需要图像和新 checkpoint。新入口会从 checkpoint 自动识别分支及
lambda；无需源图、manifest、Qwen 或匹配器，也不用额外的推理开关。
原版 checkpoint 自动按原版加载。

```python
import sys
sys.path.insert(0, "salad")
import torch
from workflow.model import load_checkpoint_model

model = load_checkpoint_model(
    "salad/outputs/reliability_ot/warmup/checkpoint.pt", "cuda",
    backbone_repo="/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main",
)
# image: ImageNet 标准化后的 [1, 3, 224, 224] 单张图像
with torch.no_grad():
    descriptor = model(image.cuda())
    # 可选诊断，不影响普通推理接口：
    descriptor, diagnostics = model(image.cuda(), return_aux=True)
    reliability_map = diagnostics["reliability"]  # [1, 1, 16, 16]
```

评估直接沿用 `evaluate_salad.py`，例如把 `--checkpoint` 换为新权重，并把
`--output` 指向 `salad/outputs/.../recall.json`。原版描述子维度不变。

## 测试

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -m unittest discover -s salad/tests -q
```

新增四组测试覆盖 OT 数值稳定、FP16/BF16、梯度检查、逐 patch 分配、旧模式
参数和数值兼容、共享配对增广、天气特征变化不直接产生负标签、局部结构错位、
未知区域、防塌缩、完整/原始 checkpoint 单图推理、旧权重严格初始化和实际
CPU 训练/断点恢复。恢复后的参数和优化器状态与连续训练逐项一致。
本次完整测试结果为 **103 项通过**。

GPU 检查使用真实 DINOv2 ViT-B、现有 Qwen 的 rain/night 配对图、FP16，
冻结 backbone；2 地点 × 2 视角另加 2 张 source companion，只做一批。
输入快照、checkpoint、训练日志和显存记录位于
`salad/outputs/reliability_ot_smoke/`。该检查衡量执行正确性，不是检索指标验证。
该小批量进程峰值 allocated 显存为 **390.82 MiB**、reserved 为 **434 MiB**，
head 参数及优化器状态有限且发生更新；加载新 checkpoint 后，单图描述子为
`[1, 8448]`、L2 范数为 1。两张图得到 9 个结构正锚点、0 个负锚点、503 个
未知 patch；这是保守弱监督覆盖率的实际检查，不据此声称错误检出率或 Recall 收益。

## 修改文件

- `models/aggregators/salad.py`：可选 head、逐 patch dustbin、FP32 OT、新分支节省内存聚合。
- `workflow/model.py`：辅助输出、严格初始化/加载、raw checkpoint 分支识别。
- `workflow/training_data.py`：精确 source companion、共享配对增广；默认接口不变。
- `workflow/reliability.py`：保守结构弱监督、真实先验和均值下限约束。
- `train_salad.py`：开关、辅助 loss、head 优化器分组、日志和断点配置校验。
- `tests/test_reliability_{ot,pairs,supervision,workflow}.py`：对应单元和集成测试。
- `RELIABILITY_OT.md`、`WORKFLOW.md`：使用说明。
