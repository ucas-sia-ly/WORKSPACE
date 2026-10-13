# 新 SALAD 模块的700张配对实验

新模块为 `reliability_ot`。本次在四个训练组中都显式启用它，使用原版已完成实验的固定图片池，重新训练50轮，再测试六个完整 SVOX 子集。保留 DINOv2 预训练，SALAD 聚合器和可靠性 head 从新的 VPR 训练初始化，不继承旧 VPR checkpoint。

| 训练组 | 每轮主损失真实图 | 每轮主损失生成图 | 辅助 source 配对数 |
|---|---:|---:|---:|
| generated_8to1 | 5600 | 700 | 700 |
| true_8to1 | 6300 | 0 | 0 |
| generated_4to1 | 2800 | 700 | 700 |
| true_4to1 | 3500 | 0 | 0 |

辅助 source 必须是生成图的精确原始视角，只用于无梯度的 backbone teacher 和结构弱监督，不进入 VPR 主损失或上表的真实图曝光数。真实对照也有同样的可靠性 head，并应用真实图可靠性先验和覆盖下限，不提供生成配对。

图片池直接复制 `outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1` 的四份固定输入，包含相同700张生成图（455自动验收通过、245未通过），不使用后来新增的图片。源图槽位、标签、顺序、seed=42、50 epochs、学习率6e-5、DINO最后4个block可训练、FP32、batch=8袋×4图、无随机图像增强、workers=0、weight_decay=0 均保持原实验设置。新 head 在原层之后初始化，代码会核对原 SALAD 和 DINO 初始参数内容与旧实验一致。

模块参数沿用用户新模块的默认值：dustbin lambda=2、hidden=64、head learning_rate=6e-5；结构损失、真实图先验、覆盖惩罚权重为0.1/0.01/0.1，覆盖下限0.5。这些参数在配置中冻结，没有通过 SVOX 结果调参。

原版报告、初始化和配置会以内容校验和冻结在新目录的 `baseline/` 中。历史代码允许与当前模块代码不同，但历史图片、训练预算、checkpoint、完整SVOX协议及评估结果均被核验。新运行绑定当前 SALAD 实现（包括 `workflow/reliability.py`）；以后改代码或参数需使用新输出目录。

准备和训练命令，从 `/home/admin123/github/WORKSPACE` 执行：

```bash
PYTHON=/home/admin123/miniconda3/envs/AdaptVPR/bin/python

# 仅读取和冻结数据，已执行；不加载 GPU 模型，不启动训练
$PYTHON AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py prepare \
  --reliability-ot

# 用户执行后：等待本轮1000张生成结束，再释放Qwen资源并开始四组训练
$PYTHON -u AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py run \
  --reliability-ot --wait-for-generation --stop-qwen-service
```

本次没有自动启动后台训练。上面 `run` 的等待检查使用生成器自身的 `.run.lock`，能够识别终端前台生成，不依赖旧 systemd 生成服务的状态。生成中断或失败时等待任务不会转入训练；需先用原生成命令 resume。训练期间持有同一锁，避免另一个生成 worker 同时占用 GPU。

每组保存最新 epoch checkpoint，可以用同一 `run` 命令恢复。必须继续传 `--reliability-ot`，新 head 和模块参数必须与配置相同。推理从 checkpoint 自动加载已训练的 head，SVOX 只需要单张 query/gallery 图，不提供辅助 source。评估最终第50轮，不按测试结果选择中间 checkpoint。

新产物目录：`outputs/qwen_curriculum/ratio_700_reliability_ot_8to1_4to1/`。

- `comparison.json`：新模块的两个比例中生成组与全真实组的 Recall@1/5/10 和百分点差值。
- `module_comparison.json`：每个训练组相对原版的 Recall 变化，以及“生成减全真”配对收益的旧→新变化；保存全部三个 Recall 指标。
- `module_comparison.md`：按比例和 SVOX 子集列出的 R@1 对比表。
- 四个子目录：初始化校验、训练和可靠性诊断日志、checkpoint、六份 SVOX 评估 JSON。

验证在 CPU 上执行，不占用生成任务的 GPU：

```bash
CUDA_VISIBLE_DEVICES='' $PYTHON AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py self-test
CUDA_VISIBLE_DEVICES='' $PYTHON -m unittest discover \
  -s AdaptVPR/experiments/qwen_curriculum/tests -p 'test_*reliability*.py' -v
```

测试覆盖精确 source companion、主曝光和辅助曝光、真实 SALAD/head/结构损失/训练器/优化器的四组小型训练、checkpoint单图推理、初始化一致、历史封印验证和比较计算，以及前台生成锁和训练等待。
