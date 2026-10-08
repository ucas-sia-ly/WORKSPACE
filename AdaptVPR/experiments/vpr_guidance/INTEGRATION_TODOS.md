# SALAD 反馈闭环：集成状态与后续实验

更新日期：2026-10-08。当前路线已经从“外部检索失败找回 GSV 源图”改为“训练源图的 verified 候选，经当前学生的 mined-positive utility 选择，再进行学生训练和条件去噪 LoRA”。可运行命令统一见 [README](README.md)，SALAD 数据与评估协议见 [工作流](../../../salad/WORKFLOW.md)。

## 已连接的组件

- [x] `generate_candidates.py` 复用发布的 Global prompts、AdaptVPR 源图解析、IC-Light 两阶段 sampler、rain 策略和 Global 双指标 verifier。
- [x] 固定 seed 与过滤后连续切片；候选身份、路径、配置/代码/输入指纹校验；JPEG 原子保存、截断尾行恢复与损坏图像补生成。
- [x] `score_candidates.py` 使用当前 SALAD checkpoint、GSV 元数据 place 标签、按 place 分组的真实负池与同 place 的真实正 co-anchors，按每次抽样 mining 后平均 utility。
- [x] `random` 与 `hardness` 在相同的合格候选组中选一张；评分记录 expected utility、mining probability 和 unusable 原因。
- [x] `salad/train_salad.py` 支持 real-only、real + accepted pool、完整学生 checkpoint 初始化及 epoch 边界恢复。
- [x] `train_lora.py` 只使用 verified 且高于 utility 阈值的 selected，训练现有 8-channel IC-Light UNet 的 attention LoRA；SALAD 不参与梯度传播。
- [x] LoRA 层注册、fp32 参数、冻结基础权重、scheduler prediction type、完整批次循环与数据路径/分辨率检查。
- [x] LoRA `.training.pt` v2 保存 optimizer、schedule、sampler 和随机状态；安全加载拒绝旧 v1 状态；服务权重在全部步骤完成后发布。
- [x] IC-Light adapter 可加载服务 LoRA；`run_loop.py` 连接 `random`、`select`、`full`，支持 extra args、无 mined positives 时保留生成器、阶段指纹缓存及恢复。
- [x] `audit` 与 `final --match-pools` 校验三个分支共同的 prompt/source/condition 组；最终学生使用预训练 backbone 和新随机 aggregator。
- [x] `--dry-run`、SALAD/LoRA `--check-data` 和 CPU/mock 回归检查。
- [x] 保留 `hard_cases.py`、`extract_hard_cases.py` 的兼容接口：只接受明确 source_id，不从外部 SVOX query 反推 GSV 源图。

旧的 `finetune_generator.py`、`losses.py`、`run_iterative_pipeline.py` 以及旧 QUICKSTART/设计摘要已经被 Claude 移除；它们不是待补齐的入口。当前 LoRA 目标是筛选正样本上的条件去噪 MSE，没有旧文档里的 identity/diversity 联合损失。

## 当前证据

现有 Bangkok probe 共 160 个候选、37 个通过旧 verifier。修正后的评分保留 18 个 prompt 组，8 张 selected 的 utility 大于 0。合格候选和 selected 的平均 mining probability 分别为 `0.2179`、`0.2292`。`train_lora.py --check-data` 读到 18 行、8 个合格训练例，分辨率为 `400 × 296`。

这些数字来自 `outputs/vpr_guidance_smoke/cand_probe/` 与 `outputs/vpr_guidance_smoke/score_probe_codex/`。legacy 迁移只校验当前配置、身份和产物，保留原 verifier scores，并记录历史 input/code/model hashes 缺失。它不等同于重新运行 verifier。

## 尚需完成的实验

2026-10-08 续作已增加 `adaptive_candidates.py`、在线评分、复用 CLIP 的天气 signal，以及 `run_loop.py --generation-mode adaptive`。133 项 CPU 回归通过；均衡 6 组 GPU 检查生成 20/24 张，避免 4 次调用。天气分数已在真实图像哈希绑定下重算并作按源图留一验证。范围与限制见 [自适应验证记录](ADAPTIVE_VALIDATION.md)；下面正式效果实验仍待执行。

- [ ] 按同一参数和 seed 跑完三个分支的多轮实验，保存每轮 accepted groups、utility 分布、mining probability、学生数据曝光与 LoRA 更新/跳过记录。
- [ ] 在三个 `final_pool.jsonl` 完成后执行 `audit --match-pools`，报告每个分支被排除的组及共同组数；以共同组的 matched pools 做公平 final。
- [ ] 加入同配方的 real-only final 基线，并完成 SVOX 各 condition 的独立 Recall@1/5/10 评估。反馈学生或少量 query 的 smoke 结果不能代替最终比较。
- [ ] 使用多个 seed 验证选择与 LoRA 的差异，报告均值、波动和 verifier 接受率，避免把一次小规模 probe 当作效果结论。
- [ ] 扩展训练城市/conditions，并在需要时运行 Nordland、RobotCar-Seasons 等协议；RobotCar 使用明确 eval manifest，不猜测配对关系。
- [ ] 检查 negative-pool size、抽样次数与 synthetic exposure 对结果的影响。当前 utility 是 real-only batch 的 MS 正项代理，不覆盖 synthetic co-anchors、随机训练增强、完整负项或所有尾 batch。

完整闭环已有入口，以上是尚待运行和验证的实验，不是宣称已获得的 Recall 改善。

## 运行时约束

同一实验 root 的参数、输入和关键实现必须保持不变。wrapper 对 manifest/checkpoint/元数据文件按内容哈希，对大型目录按 inventory、size、mtime 签名；后者不是全部 JPEG 内容校验。改配方或代码后使用新 root。

离线 fresh/student-0/final 使用本地 `--backbone-repo` 和 `--backbone-weights`。完整 SALAD `--init-checkpoint`/resume 已含 backbone，wrapper 会移除额外 backbone weights；final 始终建立新 aggregator。LoRA `--resume` 仅恢复同一次 v2 训练，`--init-lora` 用于新轮次，两者互斥。

正式效果报告应结合实验 config、pool audit、训练产物和独立评估输出。已完成的真实 GPU smoke、恢复核验与回归测试见 [验证报告](VALIDATION.md)，其中的小型 fixture 和有限 query 评估只证明机制可运行。
