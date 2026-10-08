# 本次完善与验证记录

日期：2026-10-08。运行目录为 workspace 根目录，Python 为
`/home/admin123/miniconda3/envs/AdaptVPR/bin/python`，实际环境为 Python 3.10、
PyTorch 2.8.0+cu128、RTX 4090。原始数据、下载模型和已发布 SALAD checkpoint 未覆盖。

## 完成的修改

- 保留 Claude 的候选生成、原验证器过滤、当前 SALAD 反馈筛选、条件去噪 LoRA 的路线。
- 评分按训练地点批次抽取多视图负样本，每次批次先挖掘再平均效用；同地点候选使用相同随机上下文。
- 候选生成检查配置、源码、提示、来源与图片校验值，支持完整跳过、断行修复和损坏图像重生成。
- LoRA 使用真实 IC-Light 的八通道输入与 offset；训练参数注册、推理精度、严格权重加载、完整训练状态恢复均已接通。
- `run_loop.py` 接通逐轮学生更新、累计合成池、LoRA 更新、fresh final 训练和评估。无有效正样本时记录跳过并保留当前生成器。
- 三个实验臂共享初始学生和配置。`--match-pools` 用共同通过验证的提示组训练最终模型，避免生成器通过率不同造成样本组成混淆。
- 更新运行说明；保留已有 hard-case 提取兼容接口。没有为外部 SVOX 查询推断不存在的 GSV 来源。

## 自动化检查

| 检查 | 结果 | 日志 |
|---|---:|---|
| guidance 单元与回归测试 | 90 项通过 | [guidance_tests_codex.log](../../../outputs/vpr_guidance_smoke/guidance_tests_codex.log) |
| SALAD workflow 测试 | 39 项通过 | [salad_tests_codex.log](../../../outputs/vpr_guidance_smoke/salad_tests_codex.log) |
| 生成服务启动测试 | 4 项通过 | [adapter_tests_codex.log](../../../outputs/vpr_guidance_smoke/adapter_tests_codex.log) |
| 旧 `test_implementation.py` 冒烟测试 | 全部通过 | [legacy_smoke_codex.log](../../../outputs/vpr_guidance_smoke/legacy_smoke_codex.log) |
| Python 语法、`git diff --check` | 通过 | — |

这些回归覆盖批次挖掘数学、随机上下文、数据契约、非法参数、检查点加载、断点后权重/优化器/RNG 一致性、阶段缓存、零正样本分支和三个臂的共同样本组成。

## 真实 GPU 检查

### 160 张候选的训练信号探测

复用 Claude 已生成的 Bangkok 40 个提示组 × 4 张候选，用发布的
`salad/checkpoint/dino_salad.ckpt` 评分。负批次为 31 个其它地点 × 4 个真实视图，
抽样 16 次，池含 3,012 个地点、12,048 个图像。

- 160 张中 37 张通过原验证器，覆盖 18 个提示组。
- 37 张中 15 张在至少一次抽样批次里出现被挖掘的正样本。
- 每组选择一张后，18 张中 8 张的效用大于零。
- 所有通过候选的平均挖掘概率为 0.2179；入选候选为 0.2292。

“挖掘概率”是抽样批次的频率，不能与“15/37 张候选曾被挖掘”的比例混用。
完整输入校验值、设置及分布见 [评分摘要](../../../outputs/vpr_guidance_smoke/score_probe_codex/summary.json)。

### LoRA 训练、恢复及生成

使用上述 8 张效用大于零的图像，在原阈值 `--min-utility 0` 下训练 4 步。
128 个投影层共 1,594,368 个 LoRA 参数，loss、梯度和参数更新均为有限值，
最终缩放后的 ΔW Frobenius 范数为 0.08978。

从 schema 2 完整状态的第 4 步重新发布后，256 个权重张量与原输出逐项完全一致。
生成适配器成功加载该 LoRA，生成 2 张图并调用原验证器；这 2 张均未通过验证，
不会进入训练池。相同命令随后在没有 CUDA 的沙箱中成功复用完整候选，未加载生成模型。

产物见 [训练记录](../../../outputs/vpr_guidance_smoke/lora_feedback_codex.json)、
[恢复记录](../../../outputs/vpr_guidance_smoke/lora_feedback_codex_republished.json) 和
[生成完成记录](../../../outputs/vpr_guidance_smoke/candidates_lora_codex/generation_complete.json)。
之前 `--min-utility -1` 的两个机制检查步单独保存在 `lora_codex_smoke*`，不作为有效反馈训练结果。

### 两轮编排与 final 阶段

创建独立 CSV 小样本，覆盖 6 个真实地点、52 个真实视图；图片通过只读符号链接复用原数据。
以发布的 SALAD 为起始学生，运行两轮 `full`，每轮 3 个提示 × 2 张候选、学生训练 1 个 batch。

- 两轮分别有 1、2 张候选通过验证，各选择 1 张，最终池含 2 张。
- 第一轮没有有效挖掘正样本，LoRA 明确跳过，第二轮继续使用发布生成器。
- 第二轮实际采样到 1 张合成图，日志为 7 个真实曝光、1 个合成曝光；学生 loss 为零，符合这些小批次没有有效挖掘对的情况。
- 模拟最终训练 checkpoint 已写入但完成标记丢失：重新运行成功补回标记，其余已完成阶段通过输入/输出校验后跳过。
- `final` 从预训练 DINOv2 和随机 SALAD 聚合器开始，训练配置确认 `init_checkpoint: null`，没有复用反馈学生的聚合器。
- 最终评估入口完成 SVOX 夜景 17,166 个图库图像和 2 条查询的精确检索；该单 batch 随机聚合器冒烟模型的 R@1/5/10 均为零。

见 [两轮日志](../../../outputs/vpr_guidance_smoke/loop_codex.log)、
[恢复日志](../../../outputs/vpr_guidance_smoke/loop_codex_resume.log) 和
[final 日志](../../../outputs/vpr_guidance_smoke/final_codex.log)。

## 结果边界

已验证实现和运行链路，尚未完成多随机种子、足量训练的 random/select/full 检索对照实验。
4 步 LoRA、单 batch 学生/final 模型和 2 条查询不能证明 Recall 改善。

效用仍是“仅真实视图”批次的近似，不模拟其它合成图、训练增强或较小尾批次。
阶段缓存对大型数据目录记录文件清单、大小和 mtime；不是逐张 JPEG 的完整内容哈希。
历史 Claude 候选没有当时的完整 source/model/code 哈希，兼容迁移会明确标注 legacy 来源，保留原验证分数。
正式实验应使用稳定、未修改的数据与模型，完整生成流程按 [README](README.md) 运行。

## 2026-10-08 追加：基于 probe 结果的改进

1. **hardness 选中了幻觉内容。** 160 张候选中 utility 最高的 4 张合格图像都改写了场景内容（新增建筑或重画高楼），`s_geo` 仍在 0.78–0.86。inlier 覆盖率（Spearman ρ=0.14）、DINOv2 patch 余弦（ρ=−0.26）都不能把它们与外观变化区分开。可行的判据是用同一学生在真实 leave-one-out 正样本上的 identity margin 做校准：300 个 Bangkok place、2,225 个真实 anchor 的 margin 中位数为 0.31，2.5% 分位为 0.11。这 4 张图的 margin 为 0.015–0.051。已实现为 `score_candidates.py` 的 plausibility gate，对两种选择方式都生效，并接入 LoRA 与 loop 的 eligibility。
2. **final 的实验敏感度。** 新增 `--final-scope`（默认只在有合成图的 place 上训练）和 `--final-control real_only`。
3. **回归测试**：93 项通过（新增门限数学、两种选择方式下的排除、端到端评分门限、旧端到端测试显式关闭门限）。`test_implementation.py` 通过。`final` 的两种 control 用 dry-run 核对了命令。

未完成：带门限的 LoRA 训练和完整三分支 + real-only final 尚未运行。

