# 持续增加 Qwen 图片时的微调入口

统一使用 `salad/run_qwen_finetune.py` 和 `salad/configs/ccr_qwen.json`。
不再为图片数量建立 `accepted_13710` 一类新配置或手工复制 manifest。
旧入口和已有实验继续可用；新的入口复用同一训练、损失、采样和 SVOX 评估实现。

## 目录与数据选择

```text
outputs/qwen_curriculum/
  images/train/                 生成的训练尺寸图片
  images/raw/                   原始高清图片
  images/accepted/              当前审核通过的图片链接
  state/training_manifest.jsonl 当前通过审核的记录与精确源图映射
  state/finetuning/round_01/     本轮微调
    inputs/                     冻结清单、完整审核压缩档、配置、图片校验信息
    plan.json                   固定训练预算及地点划分
    runs/{A,M}/{seed}/          各组 checkpoint、日志、逐类别评估
    report.md                  实验结果汇总
```

首次运行取活动 manifest 与 `images/accepted` 的交集，并要求审核标记合格。
没有审核记录的图片不能直接用于训练，不根据同地点或文件名补造源图。
生成图必须实际位于 `images/train`，真实源图仍由 GSV 元数据检查地点归属。
清单与 accepted 链接不同步的数量会记录到 `inputs/receipt.json`；如果快照过程
中发生发布变化则报错，等待本次发布完成后重试，不保存半份快照。

每个源图、生成图及可选 mask 在首次快照时校验 SHA256。
训练清单只保留精确配对、天气、审核标记、hash 和 mask 路径；完整审核记录
保存在 `inputs/audit_manifest.jsonl.gz`，不复制 PNG，也不建立教师特征缓存。
所有采样和局部损失使用的审核字段均保留，训练 Loss 不变。

同一个输出目录续训时只使用该目录的冻结快照，**不会加入后来生成的图片**。
恢复时核验清单和配置，并检查图片大小、mtime；有变化的图片重新校验 hash。
这不是每次续训都读取全部图片字节的完整重哈希。已引用图片应保持不变；
活动 manifest 增加记录或 accepted 链接变化不改变既有实验。

## 命令

从工作区根目录执行。仅准备最新输入及计划，不使用 GPU、不训练：

```bash
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
"$PY" salad/run_qwen_finetune.py
```

只建立小型元数据快照、跳过全真实池及 SVOX 目录扫描：

```bash
"$PY" salad/run_qwen_finetune.py --snapshot-only
```

手动启动本轮；中断后重复相同命令：

```bash
"$PY" salad/run_qwen_finetune.py --execute
```

有新审核图片后，指定新的输出目录，首次准备会读取届时最新图片：

```bash
"$PY" salad/run_qwen_finetune.py \
  --output-dir outputs/qwen_curriculum/state/finetuning/round_02

"$PY" salad/run_qwen_finetune.py \
  --output-dir outputs/qwen_curriculum/state/finetuning/round_02 --execute
```

同一配置换目录表示从配置指定的初始权重开始一个独立实验；不是自动接着上一轮
checkpoint 继续训练。需要改变初值或训练设置时，修改配置并使用新的目录。
对不同数据快照做比较时要检查各自 `plan.json` 的训练、校准和留出地点；不能
假设新增图片后仍是完全相同的消融数据划分。不要在旧实验目录刷新输入。

只汇总已有结果：

```bash
"$PY" salad/run_qwen_finetune.py --summarize-only
```

默认使用 `dino_salad.ckpt`，A/M 各3个 seed，P32、K4、BF16、2个可训练block、
lr=1e-5、5850步；全真实地点池排除独立校准和留出地点。显存预算40GiB，
host RSS限制20GiB，4 workers且prefetch=1。配置不固定生成图片总数。
SVOX val用于过程监控；default/night/rain/snow/sun/overcast分别输出结果，
test只在初始和固定最终预算评估，不用于挑checkpoint或调参。

## 清理范围

训练中的实验和正式checkpoint不自动删除。历史诊断报告、逐query结果、
生成图片、审核账本、几何缓存及Reliability V2记录保留。
旧Stage 1/2A仅1–2步Smoke Test的一次性checkpoint可删除，保留其JSON报告和日志。
只准备但从未训练的旧实验目录可重新生成，不能把已完成实验当作缓存删除。
本次清理明细保存在 `salad/outputs/cleanup_smoke_and_drafts.json`。

单元测试使用临时目录模拟新增图片、恢复、精确源图冲突、审核交集、mask和文件
篡改，不加载模型、不训练、不使用GPU。
