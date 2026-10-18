# CCR-VPR Stage 2A：SALAD 特征接口与冻结教师

本阶段没有新增 Feature/OT 蒸馏损失。未设置 `--teacher-checkpoint` 时，
训练与推理保持原行为。设置教师时，训练会在线产生真实主批次的教师目标，
测量并记录耗时与显存后释放这些目标；学生仍只使用已有训练损失。

## SALAD 辅助输出

`descriptor, aux = model(images, return_aux=True)` 新增以下字段：

| 字段 | 通用形状 | ViT-B/14、224×224 默认配置 |
|---|---|---|
| `backbone_local_features` | `[B,C,Hpatch,Wpatch]` | `[B,768,16,16]` |
| `content_features` | `[B,cluster_dim,N]` | `[B,128,256]` |
| `score_logits` | `[B,num_clusters,N]` | `[B,64,256]` |
| `log_assignment` | `[B,num_clusters+1,N]` | `[B,65,256]` |

`N = Hpatch × Wpatch`，由实际特征图决定，不固定为 256。
`content_features` 是 `cluster_features` 投影并展平的内容特征；
`score_logits` 是 Sinkhorn 前的簇 score；`log_assignment` 是完整对数分配，末行是 dustbin。

原有 `local_features`、`assignment_keep`、`assignment_dustbin`、`dustbin_mass`
以及开启可靠性分支时的 `reliability_logits` / `reliability` 都保留。
`backbone_local_features` 与 `local_features` 指向相同张量。
学生的这些张量没有 detach，可正常向 score/content 分支回传梯度。

没有改变 `score`、`cluster_features` 的模块顺序、参数键、dropout 调用顺序或
`return_aux=False` 的运算。不同 forward 的 descriptor 比较应在 eval 下进行，
或者在 train 下恢复同一 RNG 状态，以免把 dropout 差异误判为接口变化。

## OT 质量约束与有限迭代

令 `P = exp(log_assignment)`，簇数为 M，patch 数为 N：

- 每个 patch 的列和为 1；`assignment_keep + assignment_dustbin` 也应为 1。
- 总质量为 N。
- 收敛时，每个簇的行质量为 1，dustbin 的行质量为 N−M。

原实现保持 **3 次 Sinkhorn 迭代**。最后一步校正列边际，所以列和通常很准确，
但行边际对尖锐 score 可能仍有明显误差。测试分别检查有限迭代和较充分迭代的结果，
不会通过改变推理迭代次数或重新归一化输出来制造相等。

本次训练过的真实 checkpoint、两个 224×224 输入的 GPU 检查结果：

| 指标 | 实测 |
|---|---|
| 每 patch 质量和最大误差 | `5.36e-7` |
| 两张图各自总质量 | `256.0`、`256.0` |
| 3 次迭代后簇行质量范围 | `0.3590–1.0554` |
| 3 次迭代后 dustbin 行质量 | `199.8142`、`200.0173`，目标为 `192` |
| 3 次迭代后行质量最大绝对误差 | `8.01735` |
| 仅诊断用 100 次迭代后的行质量最大绝对误差 | `1.53e-5` |

这些有限迭代误差属于原有 OT 计算，本阶段没有改变它。

## 教师加载与兼容检查

`workflow.model.FrozenSALADTeacher` 接受显式 checkpoint 路径，并执行：

1. 要求完整 `state_dict` 和正数 `global_step` 的已训练 workflow/Lightning checkpoint。
2. 严格加载全部 backbone 和 SALAD 权重，不允许缺失聚合器后使用随机初始化参数补齐。
3. 检查 backbone 架构、backbone 归一化/返回 token 配置、簇数、输入通道、内容维度和 token 维度。
4. 检查输入尺寸、RGB、PIL bilinear resize、输入范围与 ImageNet mean/std。
5. `eval()`、`requires_grad_(False)` 和带 `torch.no_grad()` 的 forward；即使调用教师的 `train()` 或 `requires_grad_(True)`，仍保持冻结。

没有训练记录的 raw state_dict 无法与随机权重可靠区分，因此教师入口拒绝它。
旧 raw checkpoint 仍然可以严格加载为学生或用于普通推理。
训练元数据用于确认来源，不意味着代码能从任意权重数值证明其训练历史或质量。

旧 workflow/Lightning checkpoint 若未保存预处理字段，采用项目既有的
RGB/PIL bilinear/ImageNet 约定，并在 `preprocessing_source` 中记录这一推断。
显式保存但不一致的预处理信息会被拒绝。新的训练 checkpoint 写入 `image_preprocessing`。
本阶段教师的配置输入尺寸必须与学生一致；网络特征接口本身仍支持其他合法尺寸。

教师模块与学生分开构造，隔离临时初始化的 RNG，不进入学生 optimizer。
教师使用 FP32 在线前向，不受外层学生 AMP 影响。
主批次若包含旧 mix/replace 合成图，教师仅处理其中的真实图。
训练直接传入学生已经预处理的真实图 tensor，不额外读取或重新增强源图。

路径、SHA256、训练步数、模型配置和预处理约定记录在
`training_config.json` / checkpoint 的 `training_config.teacher` 中。
哈希和加载使用同一个打开的文件，避免其他进程原子替换 `checkpoint.pt` 后路径与已加载内容错配。
恢复时必须提供同一教师及分块配置；路径或内容哈希变化会被 resume 配置检查拒绝。
教师权重不重复写入学生 checkpoint，因此恢复时原教师文件仍需可用。

## 在线前向与分块

```python
from workflow.model import FrozenSALADTeacher

teacher = FrozenSALADTeacher(
    teacher_checkpoint, student.config,
    device="cuda", backbone_repo=backbone_repo,
)
descriptor, targets = teacher(real_images)  # no_grad + eval + FP32

# 分块时逐块消费，不要把所有 GPU 目标重新拼接起来。
for start_index, descriptor, targets in teacher.iter_targets(real_images, batch_size=2):
    # 在后续阶段消费目标；本阶段没有蒸馏 loss。
    del descriptor, targets
```

`--teacher-batch-size 0` 默认先尝试整批；正整数指定分块大小。
遇到 CUDA **前向** OOM 时自动减半重试，直到单张；单张仍失败会给出明确错误。
分块减少前向临时张量和目标的峰值，不减少学生/教师参数常驻显存。

`training_log.jsonl.teacher` 包含实际 `images`、`chunks`、`largest_chunk`、
`seconds`、`seconds_per_image`、`oom_retries`、逐图特征形状，以及 CUDA 的
baseline allocated、peak allocated、peak reserved 字节数。
计时在 CUDA 同步边界测量，包含前向、目标有限性检查和同步。
显存是教师前向区间的本进程 PyTorch allocator 峰值，包含当时常驻的学生、教师和输入；
不是显卡所有进程的占用，也不是整轮训练峰值。CPU 的 CUDA 显存字段为 null。

## 本次短程 GPU 实测

教师：已有训练 50 轮、`global_step=9850` 的完整 SALAD checkpoint：

```text
outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1/generated_8to1/checkpoint.pt
SHA256: f06772e6b22c5335efccdf0afd23a1afe18442703299281d23b0641286589513
```

ViT-B/14，224×224，8 张真实图，学生和教师均常驻 GPU。预热后每种分块测量 3 次：

| 教师分块 | 平均在线时间 | 实测 peak allocated |
|---|---|---|
| 整批 8 张 | 20.6 ms | 888.8 MiB |
| 每块 2 张 | 39.5 ms | 738.7 MiB |
| 每块 1 张 | 60.0 ms | 713.1 MiB |

测量时 GPU 与其他进程共享，因此时间仅代表本次环境，不是固定性能承诺。
此外完成了一步 FP16 学生 + FP32 教师 + Stage 1 辅助分支的训练 smoke；
教师未获得梯度，学生 score/content 分支均有非零有限梯度。
没有启动全城市训练，也没有添加 Feature/OT 蒸馏 loss。

日志、逐次测量、可复现检查脚本和 checkpoint 位于：
`salad/outputs/ccr_stage2a_smoke__o3ifbxx/`。

## 短程命令

在工作区根目录执行，使用新输出目录：

```bash
CCR_TEACHER=outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1/generated_8to1/checkpoint.pt
CCR_OUT=$(mktemp -d "$PWD/salad/outputs/ccr_stage2a_manual_XXXXXX")
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/train_salad.py \
  --real-data dataset/gsv-cities --cities Bangkok \
  --synthetic-manifest salad/outputs/reliability_ot_smoke/manifest.jsonl \
  --synthetic-places-only \
  --init-checkpoint "$CCR_TEACHER" --teacher-checkpoint "$CCR_TEACHER" \
  --teacher-batch-size 2 \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir "$CCR_OUT" --epochs 1 --max-batches-per-epoch 1 \
  --batch-size 2 --images-per-place 4 --min-images-per-place 4 \
  --image-size 224 224 --num-trainable-blocks 1 \
  --real-sampling coverage_pk --seed 13 --num-workers 0 --no-augment \
  --synthetic-aux --synthetic-exposure-prob 1 \
  --learning-rate 1e-6 --device cuda --precision 16
```

针对性回归：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=salad:salad/tests /home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  -m unittest test_frozen_teacher test_reliability_ot test_synthetic_aux \
  test_pretrained_initialization test_reliability_workflow -v
```
