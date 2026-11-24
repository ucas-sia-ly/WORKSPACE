# Stage 1：不更新权重的固定输入诊断

入口：`salad/diagnose_ccr_stage1.py`。只读取现有 checkpoint，使用 `autograd.grad`
读取各损失梯度，不创建 optimizer、不调用 step、不修改 `.grad`、不运行检索评估。
执行前后核验各 checkpoint SHA256、内存模型 state SHA256 和参数版本。

## 复现

```bash
cd /home/admin123/github/WORKSPACE
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python

# 固定 Resize 基线，输出目录必须为新目录。
"$PY" salad/diagnose_ccr_stage1.py \
  --output-dir salad/outputs/ccr_stage1_diagnostic_new \
  --batches 8 --full-batches 4 --controlled-batches 2 \
  --dropout-pairs 8 --repeats 4

# 原训练翻转/颜色增强只生成一次，随后各 checkpoint 复用相同输入。
"$PY" salad/diagnose_ccr_stage1.py \
  --output-dir salad/outputs/ccr_stage1_diagnostic_augmented_new \
  --batches 8 --full-batches 4 --controlled-batches 2 \
  --dropout-pairs 8 --repeats 4 --frozen-augmentation

# 仅用现有 JSON 生成完整分布报告，不重新调用模型。
"$PY" salad/summarize_ccr_diagnostics.py \
  --input-dir salad/outputs/ccr_stage1_diagnostic_augmented_new \
  --reference-dir salad/outputs/ccr_stage1_diagnostic_new
```

默认读取 `salad/outputs/ccr_accepted_13710_svox_categories/plan.json`，
使用初始 epoch_050 和 A/B0/B4/B5 的 seed42 最终 checkpoint。
不加载 M，也不修改原实验配置或结果。40GiB CUDA allocator / 20GiB RSS
资源保护与可行性实验一致，系统可用内存低于 3GiB 时停止。

## 固定输入与比较范围

| 面板 | 输入 | 用途 |
|---|---|---|
| training_subset | 8 个 batch，每个32地点×4真图，来自原10,818地点池 | 嵌套 P8/16/32 mining；P8 原始辅助触发语义梯度 |
| full_real | 4 个 batch，每个32地点×4真图，来自完整真实池，排除原校准和验证地点 | 比较不同真实数据池的 mining |
| controlled_source_included | 2个batch，每个8地点×4真图；每天气2地点 | 精确源图显式进入K，每地点1张生成图，排除辅助缺失因素 |
| local pairs | 8对，四天气各2对 | 同图、跨图、模式、精度与对应策略对照 |

自然面板 coverage_pk、epoch0、seed3042，不强制把生成源图塞入 K；p_exp=0.5。
受控面板显式加入源图、p_exp=1，属于诊断控制，不能用它估计实际训练触发率。
所有路径和输入张量摘要保存在 `fixed_batches.json`。

初始模型原本解冻4块，诊断时统一仅解冻最后1块和原可训练聚合器，与已训练
A/B0/B4/B5 保持参数集合一致。此诊断不评价额外解冻 block 的梯度。

## 输出解释

- `mined_pairs.csv`：eval FP32 和 train BF16 下，P=8/16/32 的候选/保留正负
  pair数、保留比例、有效anchor、MS loss及共同前8地点anchor统计。
- `gradient_metrics.csv`、`gradient_distributions.json`：逐分支原始/加权范数、
  两损失之间余弦/角度、总梯度和假设裁剪量。Score权重为0，仍报告其原始梯度。
- `local_consistency.csv`：Feature/Score/OT、cosine、支持比例、目标keep等逐对数据。
- `initial.json`、`A.json`、`B0.json`、`B4.json`、`B5.json`：完整原始数据、
  每anchor计数、每组梯度Gram矩阵、所有随机重复和局部统计分布。
- `protocol.json`：文件/参数未修改核验、样本数、输入协议和实测显存。
- `diagnostic.log`：资源监控和逐checkpoint进度。
- `report.md`：mining概览；若已执行汇总工具，另有 `detailed_report.md`。

局部配对目标来自原始冻结教师，使用原有 radius1 校准和几何证据。
B5 的梯度使用 verified q，其他 checkpoint 使用相同有效支持集的 uniform q。
这意味着不同 q 模式的梯度不能当作完全相同的目标进行逐数值比较。
局部一致性前向则同时报告 uniform/verified，便于独立比较。

自然梯度面板以 BF16 train 模式模拟原训练。受控梯度面板另有 FP32 train 和
FP32 local-eval 对照；local-eval 仅关闭局部学生前向随机性，真实/全局分支仍为
train 模式。所有损失均在FP32归约，不使用GradScaler。

同一梯度Gram矩阵给出 real_only、global_001/003/010、feature_only、feature_ot
组合。它们是假设的加权梯度，不是重新训练结果，不代表 AdamW 最终更新。
`groups.all` 的裁剪比例才对应全参数总梯度；其他组的独立裁剪量只是组内参考。

当真实 MS 梯度为零时，夹角没有定义，保存为 null，不能当作0度、90度或无冲突。
输出分布必须同时看有效数量。只对少量固定batch测量，不能替代全epoch裁剪统计。
Train BF16 mining先用microbatch16提取P32描述子再切片；梯度使用整批P8的32张图。
两者的Dropout随机实现可能不同，因此mining计数和梯度为零的batch数不要求逐批一致。

## Dropout / 精度对照

每个checkpoint都先保存它自己FP32 eval输出作为同权重目标：

1. 自身eval对自身eval：数值基线。
2. 自身FP32 train对自身FP32 eval：每对4次固定随机重复。
3. 自身BF16 eval/train对自身FP32 eval：精度与随机扰动对照。
4. 学生对初始教师同图：模型漂移。
5. 生成学生对真实教师：同坐标、verified uniform、verified q 三种对应归约。

教师FP32输出和对应支持集固定。不能把同图随机扰动值直接从跨图Loss中减去，
也不能用不同支持集或不同平均方式的数值相除来声称“多少比例属于Dropout”。
Feature来自Backbone；SALAD score/content的Dropout不会直接扰动它。
未知区域、无有效positive的零Loss和有效监督覆盖率均保留，避免将无监督误报为一致。

本次已经完成的结果：

- `salad/outputs/ccr_stage1_zero_update_20261019/`：固定Resize。
- `salad/outputs/ccr_stage1_zero_update_20261019_augmented/`：固定一次原策略增强；
  `findings.md`包含实测要点和图，`detailed_report.md`包含全部分支分布及两面板对照。
