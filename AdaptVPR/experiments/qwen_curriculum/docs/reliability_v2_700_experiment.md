# Reliability V2 的700张配对训练与 SVOX 比较

本轮使用 `--reliability-v2`，重新训练8:1、4:1生成替换组及各自的全真实对照，共四个模型，再评估六个完整 SVOX 子集。沿用原版实验的同一700张生成图、精确源图槽位和训练设置；DINOv2保留预训练，SALAD聚合器与可靠性head随机初始化，不继承任何已训练的VPR模型。

固定图片池、两份历史对照和全部700对教师缓存已准备完成；37项相关CPU测试与真实32张FP32图像的前向/反向检查通过。没有启动正式训练，也没有对正式实验执行优化器更新。

## 启动与恢复

从 `/home/admin123/github/WORKSPACE` 执行：

```bash
cd /home/admin123/github/WORKSPACE

/home/admin123/miniconda3/envs/AdaptVPR/bin/python -u \
  AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py run \
  --reliability-v2 \
  --stop-qwen-service
```

原1000张生成阶段已经结束，因此不需要 `--wait-for-generation`。`run`先检查生成锁并停止Qwen服务，构建或核验全部700对的固定教师缓存，再串行完成四组训练和24次SVOX评估。停止Qwen不会删除已有图片。

默认输出目录为：

```text
outputs/qwen_curriculum/ratio_700_reliability_v2_8to1_4to1/
```

中断后重新执行同一条 `run` 命令即可恢复。完成的缓存和训练组会复用，未完成训练从最近保存的epoch checkpoint继续；模型、优化器、学习率调度和随机数状态一并恢复。图片、参数、缓存或代码发生变化时会拒绝沿用该目录，应指定新的 `--output-dir`。

如需仅冻结并检查输入，不加载教师模型、不开始训练：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py prepare \
  --reliability-v2
```

`prepare`不会构建尚未存在的GPU教师缓存；缓存构建由正式 `run` 阶段执行。

## 固定图片池与训练预算

| 训练组 | 每轮主损失真实图 | 每轮主损失生成图 | 每轮总图片数 |
|---|---:|---:|---:|
| generated_8to1 | 5600 | 700 | 6300 |
| true_8to1 | 6300 | 0 | 6300 |
| generated_4to1 | 2800 | 700 | 3500 |
| true_4to1 | 3500 | 0 | 3500 |

700张生成图包含原来的455张自动验收通过图与245张拒绝图，全部保留；不会改写质量标记，不换成后来生成的图片。每对模型使用相同source槽位、地点标签、批次顺序和初始化，生成组仅将对应的700个source路径替换为其生成版本。4:1图片池是8:1图片池的子集。

四组均为50 epochs、主学习率 `6e-5`、head学习率 `6e-5`、DINO最后4个block可训练、FP32、batch=8袋×4张图、seed=42、workers=0、无增广、weight_decay=0。所有原SALAD参数和DINO初值都核对内容哈希；四组的新head也使用相同初值。

两个比例的图片池与更新预算不同：50轮分别为9850次和5500次更新。主要比较每个比例内“生成组减全真实对照”的收益；不能仅凭两种比例的绝对Recall判断比例优劣。

## V2行为与缓存

V2显式开启轻量depthwise邻域context、head输入特征detach及learned模式。detach只阻断可靠性预测分支向DINO传递辅助梯度，DINO的原检索分支仍接受VPR梯度。dustbin lambda=2、hidden=64；结构损失/真实图先验/覆盖惩罚权重为0.1/0.01/0.1，覆盖下限0.5。

离线教师直接加载与学生初始DINO相同的raw预训练权重，冻结全部DINO参数，使用norm和patch features构建target/confidence网格。它不加载旧SALAD或VPR checkpoint。缓存覆盖全部700个精确source/output对；教师结束后释放模型，训练只读取小型CPU目标网格，不执行source companion前向。

缓存保存在新目录的 `frozen_targets/index.json` 与 `frozen_targets/targets.pt`，并由 `cache_binding.json` 绑定本次实验fingerprint、原manifest、逐图source/output内容、raw教师权重、index和tensor哈希。已完成缓存不覆盖；缓存身份或图片内容不匹配会报错。旧 `salad/outputs/reliability_v2/frozen_targets` 只有455对，不用于本轮。

几何负监督设置为 `corruption_weight=0.05`、`max_images=4`。每步最多对4张真实图制造局部copy-paste受损副本，使用独立随机流；受损图只参与辅助监督，原检索图和地点标签保持不变。辅助副本不计入上表的主曝光，也不改变700次生成替换计数。全真实对照也开启同一V2模块及真实图辅助监督。

自动拒绝图不会直接变成负patch标签；没有结构证据的区域仍为未知。受控破坏提供的是已知位置的代理负监督，其实际跨域收益由本次配对Recall实验检验。

## 结果与历史对照

SVOX评估使用day、night、rain、snow、sun、overcast六个原生test查询目录，共用17166张gallery、25米UTM阳性，图像224×224。只评估最终第50轮，不用SVOX选择中间checkpoint或调整模块参数。推理从checkpoint自动加载V2结构，只需单张query/gallery图。

新目录的 `baseline/` 冻结原版SALAD配置、初始化和报告；`previous_baseline/` 冻结上一版Reliability OT对应资料。两份历史对照的固定图片池、训练预算、checkpoint和SVOX结果都核验；当前模块代码可以与历史实现不同。

- `comparison.json`：V2四组内的生成/全真实Recall@1、@5、@10及配对差值。
- `module_comparison.md`、`module_comparison.json`：V2相对原版SALAD的各组变化及生成配对收益变化。
- `version_comparison.md`、`version_comparison.json`：V2相对上一版Reliability OT的相同比较。
- `progress.json`：缓存、训练、评估或完成状态。
- 四个训练子目录：初始化记录、可靠性诊断、训练日志、最新checkpoint与六份SVOX评估JSON。

Markdown表格展示R@1百分比及百分点差值；JSON保留完整R@1/5/10。主初始化和预算一致可以支持配对比较，但V2包含多个辅助机制，若要归因到单个机制仍需独立消融。

## CPU验证

```bash
PYTHON=/home/admin123/miniconda3/envs/AdaptVPR/bin/python

CUDA_VISIBLE_DEVICES='' $PYTHON \
  AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py self-test

CUDA_VISIBLE_DEVICES='' $PYTHON -m unittest discover \
  -s AdaptVPR/experiments/qwen_curriculum/tests \
  -p 'test_*reliability*.py' -v
```

新测试包括 `test_fixed_reliability_cache.py`、`test_fixed_reliability_v2_experiment.py` 与 `test_reliability_version_comparison.py`，覆盖全部通过/拒绝图的缓存、精确source和内容哈希、缓存恢复与防覆盖、静态五项tensor schema、实际轻量CPU训练/恢复、主曝光不变、context更新、历史版本身份及R@1/5/10比较。
