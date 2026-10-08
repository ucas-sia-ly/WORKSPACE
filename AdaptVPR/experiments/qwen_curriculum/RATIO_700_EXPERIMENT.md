# 700 张生成图的两组配对实验

入口是 `test_700_ratio_experiment.py`。固定本轮执行顺序中的前 700 张成功生成图，每轮将它们对应的 source 视角替换为生成图。全真实对照使用完全相同的 source 槽位、地点标签、批次顺序和训练预算，只将这 700 个槽位还原为 source 图。

| 训练组 | 每轮真实图 | 每轮生成图 | 每轮总图片数 |
|---|---:|---:|---:|
| generated_8to1 | 5600 | 700 | 6300 |
| true_8to1 | 6300 | 0 | 6300 |
| generated_4to1 | 2800 | 700 | 3500 |
| true_4to1 | 3500 | 0 | 3500 |

比例是实际曝光数量。每轮遍历固定图片池一次，不使用概率替换，不在同一轮额外加入被替换的原图。4:1 的 source 池是 8:1 池的子集，两种比例使用同一批 700 张生成图。每袋有同地点的四个不同源视角，并保留真实上下文；不同袋可属于同一地点，标签保持一致。

本次固定清单中，455 张自动验收通过，245 张未通过。按“使用 700 张已生成图片”的要求全部纳入，保留质量标签供分析，不使用仅含通过图的 `training_manifest.jsonl`。因此结果反映这批生成图的整体作用。这里的自动验收标签不代表人工质量结论。

四组从新的 VPR 训练开始：仅加载本地 `~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth` 的 DINOv2 预训练权重，SALAD 聚合器随机初始化，不加载 `dino_salad.ckpt`。这是用户确认的“保留 DINOv2 预训练，SALAD 从0开始”。四组均为 seed=42、50 epochs、训练 DINOv2 最后4个 block 和聚合器、学习率 6e-5、AdamW weight_decay=0、FP32、无随机图像增强、batch=8 地点袋×4图、workers=0。每对训练的更新次数相同；不同图片池大小对应不同更新次数，需分别与各自配对对照比较。

每组 `initialization.json` 记录训练前的 backbone 和聚合器参数内容校验，四组必须拥有相同的初始权重。每轮覆盖保存最新 `checkpoint.pt`，最终另存编号 checkpoint，避免保存50份大模型。中断后只能恢复新实验目录内的 checkpoint。之前的微调输出保留在旧目录，不参与本次训练。

只评估最终第50轮 checkpoint。每个模型在完整原生 SVOX test 的 day、night、rain、snow、sun、overcast 六个子集上计算 Recall@1/5/10，共享 17,166 张 gallery，使用 25m UTM 正例、224×224 FP32 描述子和精确检索。本地没有 SVOX fog 查询目录。日间查询 14,278 张，其余五个子集分别为 823、937、870、854、872 张；无正例查询不进入 Recall 分母。测试结果不用于调参或选择 checkpoint。

从工作区执行：

```bash
PYTHON=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
$PYTHON AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py self-test
$PYTHON AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py prepare
$PYTHON AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py run \
  --stop-qwen-service
```

`self-test` 在临时目录用小型 CPU 模型跑通实际 SALAD 训练器的四种曝光配置，不加载 DINO、不占用生成任务的 GPU。`prepare` 导出固定图片清单、两个嵌套 source 池、文件内容校验和及实验配置。之后代码、初始化权重、源图、生成图或训练参数变更会被拒绝，修改配置需另选输出目录。

生成任务已经停止，当前后台服务 `qwen-ratio-700-vpr-scratch.service` 立即串行训练四组，再评估四个模型，不等待生成1000张。原来的等待服务 `qwen-ratio-700.service`、两组全城市池任务 `qwen-curriculum-compare.service` 和前台微调任务均已停止。已有生成图不会被删除，只使用原先固定的700张图。后台服务运行时不用再次执行前台 `run`，避免重复启动。

```bash
systemctl --user status qwen-ratio-700-vpr-scratch.service
cat outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1/progress.json
tail -f outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1/worker.log
```

全部新产物位于 `outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1/`。每组子目录保留 checkpoint、每轮真实/生成曝光日志和六个域的评估 JSON。完成后 `comparison.json` 汇总两组配对的 Recall 和生成组减真实组的百分点差值。断点恢复使用相同命令；完成组核验后跳过，已有评估按 checkpoint 校验后复用。`--epochs`、`--learning-rate`、`--trainable-blocks` 可在 `prepare` 前指定；调整后需另选输出目录。
