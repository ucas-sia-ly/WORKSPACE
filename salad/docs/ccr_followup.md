# 饱和扫描后的审核、实现与下一轮命令

后续读取持续增加的Qwen图片，推荐[通用入口](qwen_finetuning.md)。下文保留已有
固定数据实验的配置和分析。未启动的旧B系列准备目录已清理，手动运行原配置会
重新准备；已有训练结果和当前正在运行的A/M实验不变。

本次只执行了无更新扫描、真实保持项的无更新梯度探针和测试，没有启动正式微调。
旧 checkpoint、原始实验结果和 Reliability V2 记录均不覆盖。

## 200-batch 扫描修正了什么

结果：`salad/outputs/ccr_mining_scan_200_20261019/`。
扫描初始 checkpoint，四个面板各200个固定P32×K4超级batch，共800次前向，
再从同一描述子取嵌套P8/16/32。使用固定一次原策略增强、Train BF16。
没有创建optimizer，没有backward，没有更新参数；文件与内存状态哈希核验一致。

| 地点池 / 主batch | P8空mining | P16空mining | P32空mining | P32有效anchor |
|---|---:|---:|---:|---:|
| 原子集 / 纯真实 | 152/200 | 78/200 | 8/200 | 5.13% |
| 原子集 / mix | 0/200 | 0/200 | 0/200 | 52.95% |
| 全真实池 / 纯真实 | 180/200 | 130/200 | 54/200 | 1.58% |
| 全真实池 / mix | 87/200 | 17/200 | 0/200 | 9.39% |

“约95% anchor没有训练对”不等于“95% batch没有真实梯度”。原子集P32已有
96%的batch保留训练对。空mask可以证明当前MS数据梯度为零；有训练对时，
前向扫描不声称已测量网络梯度大小，更不等同于历史Adam更新。
P32的空batch显著少于P8，因此扩大P仍是有直接依据的候选。

原子集M的P32 mined正pair中96.9%至少包含一张生成图；全部200个batch有
mined负pair指向真实reference。支持“生成图激活难正样本”的解释，但不保证
这种激活一定带来SVOX收益。全真实池M的实际生成比例约5.5%，原子集约31.1%。
全池用于覆盖更完整真实分布，不保证随机batch更难。

## 对Claude总结的审核

- MS训练信号稀疏和M激活mining有依据，但不能用少量样本证明整个GSV已被学透。
- 几个接近90度的有效夹角不足以证明长期漂移机制，更不能把未定义角度当正交。
- Adam对统一缩放整个梯度历史近似不敏感，需要状态匹配、eps可忽略等条件。
  这里只调一个损失，且有混合项和历史动量，不能断言降低权重无效。
- 当前batch真实数据梯度为0，不表示历史真实约束或Adam动量完全不存在。
- 同图OT随机差异0.324与跨图可信支持Loss不能加性分解，不能称三分之一是噪声。
- Feature只直接更新Backbone；OT还更新score。全参数范数小不等于Feature没作用，
  也不足以决定把权重放大10倍。
- 0/8或1/8的裁剪估计不能排除历史训练裁剪影响，现已加入实际训练监控。
- 几何缓存命中少主要说明逐点缓存覆盖不足，不等于LightGlue匹配失败；manifest
  中的聚合内点/H指标不能直接伪造逐Patch正标签。暂未自动扩充几何缓存。

## 已实现的可关闭改进

### 真实全局描述子保持

新增 `--real-global-distill-weight`，默认0；启用时要求完整已训练教师。
对当前主batch真实图的**相同增强张量**，教师和学生另行执行FP32 eval前向：

```text
L_keep = mean(0.5 * ||normalize(d_student) - normalize(d_teacher.detach())||²)
L_total = L_previous + real_global_distill_weight * L_keep
```

学生eval仍保留梯度。相同权重、输入和确定性前向时，Loss及梯度精确为0；
它是漂移后的恢复约束，不保证每步都有非零任务梯度，也不保证检索一定改善。
用平方差形式避免相同描述子自余弦的数值残差产生假保持梯度。
`--real-global-distill-batch-size`默认2，按全真实batch分母归约，逐块释放图。
所有backward之后仍只执行一次unscale/clip/step；教师始终无梯度。
checkpoint记录权重、微批次、FP32/eval/相同输入协议，resume必须匹配。

真实保持探针结果在 `salad/outputs/ccr_real_retention_probe_20261019/results.json`：
8张真实图、解冻2块，初始模型Loss及梯度范数均为0；旧B0/B4的Loss约
0.1245/0.0836，权重1时梯度范数约1.113/0.990。均未更新参数。
这验证了恢复信号存在，不是权重1的最优性或疗效证明。

### 局部前向与随机性

新增 `--local-distill-eval`，默认关闭；只在局部学生监督前向使用eval行为，
保留autograd，随后恢复主训练模式。Feature/Score/OT公式、可信支持、teacher和
q的detach语义不变。不会从Loss里减常数基线。

新增 `--isolate-auxiliary-rng`，默认关闭；全局生成与局部辅助使用独立、按
seed/step/分支确定的Torch随机流，不推进主真实分支Dropout随机状态。
原独立生成采样器RNG继续保留。新选项会进入恢复配置检查。

### 地点排除和训练监控

新增 `--exclude-places`，接受JSON列表 `[["city", place_id], ...]`，同时移除
该地点所有真实/生成图，并记录文件SHA。不能仅删除synthetic-places-only而
把独立校准和验证地点重新纳入训练。

新增 `--log-mined-pairs`，默认关闭，记录每轮：

- 有向real→real、real→synthetic、synthetic→real、synthetic→synthetic正/负及mined计数；
- 两种anchor曝光及有效数量，空mining batch比例；
- 实际总梯度裁剪触发比例。

这是对已有描述子的无梯度统计，快速路径不逐anchor进行GPU标量同步。
逐损失梯度范数继续使用固定诊断工具；每步额外求多项梯度不属于廉价监控，
本次没有将其加入所有训练步骤。

## 下一轮配置和命令

### 使用原版 dino_salad.ckpt

当前 `ccr_followup_am_fullreal.json` 的学生初值、教师已指定为
`salad/checkpoint/dino_salad.ckpt`；它是原版纯 `state_dict`，不带 `global_step`。
教师加载按完整文件SHA256注册兼容这份已有预训练文件：
`6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada`。
出处见[原版发布说明](https://github.com/serizba/salad/releases/tag/v1.0.0)
和仓库 `hubconf.py`。该SHA来自本地既有文件及历史实验记录，并非声称发布者
提供了此摘要；任意重命名的随机权重不会因为文件名相同而放行。

完整backbone与聚合器仍严格加载；未注册raw文件、原生checkpoint的显式零步
或负步数仍拒绝。不会伪造训练步数；教师记录 `checkpoint_global_step=null`、
`registered_pretrained_artifact`及原版URL。教师依然eval、no_grad且不进入optimizer。
raw文件未记录输入尺寸，当前显式采用224×224/RGB/双线性Resize/ImageNet归一化，
不是断言其原始训练尺寸为224。

该配置改用新目录 `salad/outputs/ccr_followup_dino_am_fullreal/`，
`pair_calibration=null`，执行时使用该教师重新校准并重新提取SVOX初始基线。
旧epoch050的校准、缓存评估、训练结果与冻结plan全部保留，不能混用。
准备/检查命令不带 `--execute`；实际运行沿用下方A/M命令。

其余B0/保持/Feature配置仍使用epoch050初值。跨方法对照时必须统一初始checkpoint
与教师，并使用对应独立目录和校准；不能把不同初值的实验当成损失消融。

共同设置：全真实池排除128个验证地点和12个校准地点，共62,374个地点；
生成训练池仍为13,505张，精确源图约束不变。P32、K4、2个可训练block、
lr=1e-5、BF16、42/43/44三个seed、每组3个完整epoch（5850次更新）。
这些是新实验候选，不能把新旧实验差异单独归因于某一个改动。

每组独立初始化、独立目录，划分和教师见各自冻结plan；不覆盖旧结果。
只使用SVOX，val监控、各test类别仅作初始/固定最终预算描述性报告。
既往已经查看过test，后续参数选择应基于validation，不能再把test当盲测。

优先只运行A/M主线：

```bash
cd /home/admin123/github/WORKSPACE
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
"$PY" salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_followup_am_fullreal.json --execute
```

在统一新设置下比较B0有无保持项：两个配置都保留global=0.1，
唯一额外机制是real retention。不要只拿新B0与旧P8/lr1e-6结果比较。

```bash
"$PY" salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_followup_b0_fullreal.json --execute

"$PY" salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_followup_b0_retention_fullreal.json --execute
```

真实保持权重1是独立候选，尚未验证能稳定检索表现。如果主线和保持项检查后
需要继续局部路线，再运行Feature-only（全局0.1、保持1、Feature0.1、Score/OT=0）：

```bash
"$PY" salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_followup_feature_retention_fullreal.json --execute
```

不要一次自动提交上述全部实验。命令需要用户手动执行；中断后使用同一命令恢复。
每个目录的 `plan.json` 与 `first_epoch_commands.txt` 可检查完整参数。
不同目录的对应seed/domain最终Recall及逐query变化用于有无保持/Feature的对照。
此前准备的四个epoch050目录还复用了旧实验的初始模型SVOX结果：checkpoint SHA、
评估协议、图片清单和结果SHA全部匹配，来源记录在 `initial/cache_provenance.json`。
这是已有初始结果的验证复用，没有启动新的模型评估，可省去重复提取初始描述子。

## 复现无更新扫描

```bash
"$PY" salad/scan_mined_pairs.py \
  --output-dir salad/outputs/ccr_mining_scan_new \
  --checkpoints initial --batches 200 \
  --cohorts training_subset full_real --compositions real mix --modes train_bf16
```

扫描输出完整CSV、来源计数、逐batch JSON及输入哈希。纯前向扫描没有实际测量
非空mask时的参数梯度，因此命名为“空mining比例”，不称整轮实测零梯度率。
