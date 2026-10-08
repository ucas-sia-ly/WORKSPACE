# SALAD 反馈驱动的 IC-Light 候选选择与 LoRA 闭环

本目录接续 Claude 的新路线：在 **GSV-Cities 训练集内部**，使用发布的 AdaptCities Global prompt 生成多个候选；经 AdaptVPR 双指标验证后，让当前 SALAD 学生选择可被 Multi-Similarity miner 挖到的正样本，再以选中的图像训练学生和生成器 LoRA。入口已经连接到工作区的 `salad/train_salad.py` 和 `salad/evaluate_salad.py`。

外部 SVOX 检索失败只用于评估。它的 `source_id: null` 不代表一个可还原的 GSV-Cities 源图，不能用 query 名称、最近邻或字符串替换反推训练源图。`hard_cases.py`、`extract_hard_cases.py` 保留为旧数据格式的兼容工具，不是新闭环的输入。

## 一轮实际做什么

```text
固定训练源图 + 发布的 Global prompt
    → IC-Light 生成 K 个候选
    → Global 双指标验证
    → 当前学生估计每个合格候选的 mined-positive utility
    → 每个 sample_id 选一张，加入累计 pool
    → SALAD 在 real + pool 上继续训练
    → full 分支用累计选中且 utility > 0 的图像训练 LoRA
    → 下一轮使用更新后的学生，以及 full 分支更新后的生成器
```

`sample_id` 表示源图、condition、prompt 组成的候选组。同一源图可以有多个不同的 prompt ID；`--num-sources` 实际计数的是 prompt 行。先按城市、condition、Global route 过滤，再以固定 seed 打乱；第 r 轮取 `[r × sources_per_round, (r + 1) × sources_per_round)`，各轮不会重复取 prompt。候选 seed 只依赖基础 seed、sample_id 和候选 index。

生成调用现有 IC-Light adapter 的两阶段采样、Global negative prompt 和 `DualTraitEvaluator`。rain 的 high-res denoise 为 `0.22`，其他 condition 为 `0.30`；Global 接受阈值为 `s_geo >= 0.78` 且 `s_div >= 0.15`。新图像验证的是最终保存、供 SALAD 使用的 JPEG。全部候选都留在 manifest 中，失败图像不会进入训练。

| 分支 | 生成器 | 每组合格候选的选择方式 | 生成器更新 |
| --- | --- | --- | --- |
| `random` | 发布版 IC-Light | 确定性随机选择 | 无 |
| `select` | 发布版 IC-Light | utility 最大，平局时正相似度更低 | 无 |
| `full` | 第 0 轮发布版，以后可加载 LoRA | 同 `select` | 对累计 selected 中超过 utility 阈值的正样本做条件去噪训练 |

`random`、`select` 和 `full` 的发布版轮次复用同一候选目录。学生会随分支更新。旧轮次的 utility 保留各自评分时的学生结果，LoRA 训练前不会用最新学生重评分全部历史数据。某轮没有 mined positives 时，`full` 记录 `skipped_no_mined_positives`，保留原生成器并继续学生训练与后续轮次。

## utility 的含义和边界

设归一化描述符的余弦相似度为 `s(a, p)`，一个训练 batch 有 B 个 place、每个 place 有 K 张图。评分器每次抽取 `K - 1` 张同 place 的真实正图作为候选 a 的 co-anchors，以及 `B - 1` 个其他真实 place；每个负 place 提供 K 张不同视角。

对第 d 次抽样，令：

$$
h_d(a)=\max_{n\in N_d}s(a,n),\qquad
M_d(a)=\{p\in P_d:s(a,p)-\epsilon<h_d(a)\}.
$$

$$
U_d(a)=\frac{1}{\alpha}\log\left(1+\sum_{p\in M_d(a)}
\exp[-\alpha(s(a,p)-b)]\right),\qquad
U(a)=\frac{1}{D}\sum_{d=1}^{D}U_d(a).
$$

默认 `alpha=1`、`base=0`、`miner-margin=0.1`、`negative-draws=16`。先在每次抽样中执行严格 mining 比较、计算正项，再平均 utility；不能先平均 hardest negative 再进行 mining。`mining_probability` 是抽样中至少挖到一个正对的比例，`mined_pairs` 是平均正对数。相同 place 的候选共享随机抽样上下文。

这是 Multi-Similarity 正项的抽样代理。负池按 place 均匀抽样，各 place 的负视角在构建池时固定；评分没有模拟 synthetic co-anchors、训练图像增强、末尾不足一个 batch 的情况，也没有计算完整 MS 负项或真实训练梯度。因此更大的 U 不能直接解释成更好的检索性能。需要最终独立评估 Recall。

LoRA 对 selected 中 `passed=True`、`eligible_for_training=True` 且 `utility > --min-utility` 的图像均匀采样，默认阈值为 0，不按 U 加权。冻结 VAE、文本编码器与 IC-Light 基础 UNet，只训练 attention 的 fp32 LoRA 参数。源图使用 VAE posterior mode 作为 8-channel UNet 的条件，目标图使用 sampled latent，按模型 scheduler 的 `epsilon` 或 `v_prediction` 目标做去噪 MSE。SALAD 不参与反向传播。这里没有旧设计中的 `L_identity`、`L_diverse`，也不声称 LoRA 的 MSE 等同于检索损失。

## plausibility gate：为什么 hardness 不能单独使用

Bangkok probe 的实测显示：utility 最高的 4 张合格候选（`adapt_074450` k0/k2、`adapt_074526` k2/k3，U≈1.1–1.25）全部是**内容被改写**的图像：右侧凭空出现一栋殖民式建筑，或高楼立面被重画。它们的 `s_geo` 都过了 0.78。`s_geo` 是 RANSAC inlier *比例*，被替换区域只是不产生匹配，不会拉低比例。对 SALAD 来说，"外观难"和"根本不是这个地方"无法区分，而 hardness 排序恰恰偏好后者。不加约束时，`select` 会系统性地把错误标签的正样本放进训练池，`full` 的 LoRA 也会去学习幻觉结构。

因此评分器先用**真实视图**校准一个下限。对 `--calibration-places` 个训练 place 的每张真实图做 leave-one-out：作为 anchor，与同 place 其他真实图计算 identity margin，

$$m(a)=\overline{s(a,P)}-\mathbb{E}_d[h_d(a)],$$

这里同一学生、同一 batch 模型。真实跨年份、跨朝向视图的标签由数据集保证正确，所以合成候选若满足 `m < quantile_q(m_real)`（默认 `q=0.025`），比 97.5% 的真实正样本还"难"，更可能是内容改变而不是困难外观。这些候选标为 `plausible=false`，**对 random 与 hardness 两个分支都排除**，使两者从同一候选集合中选择。`train_lora.py` 与 loop 的 eligibility 统计也拒绝 `plausible=false` 的行。

实测：真实 margin 中位数 0.308，2.5% 分位为 0.109。门限恰好去掉上述 4 张（margin 0.015–0.051），其余 33 张保留；有 2 个组因无合格候选而退出，最终 16 组。`summary.json` 的 `plausibility_calibration`、`implausible_verified` 和 `groups_where_gate_changed_hardest` 记录了这一效果。`--plausibility-quantile 0` 关闭门限，可用作消融。

门限只检查标签是否可信，不能保证"更难"更有用；这一点仍需 final Recall 回答。

## 自适应候选生成（实验入口）

`adaptive_candidates.py` 在同一进程中加载 IC-Light、双指标 verifier 和当前 SALAD 学生。真实正图、负池和真实视图 plausibility 校准只提取一次；每张最终 JPEG 先经过原 Global 验证，再检查 CLIP 天气变化，最后进行学生评分。默认使用诊断中的 `SDEdit 0.85` 和精简天气优先 prompt；这些配置仅用于该实验入口，生产 prompt、adapter API、路由与 Global 阈值保持原协议。

默认天气门槛是固定尺度 CLIP contrast 的 `weather_shift > 6`，不是天气概率。该值是小规模非盲目视诊断得到的探索参数；不能把弱天气过滤或联合通过当作结构真值。用 `--weather-min-shift` 调整，`--disable-weather-gate` 做消融。天气信号复用 verifier 的源图和生成图 embedding，不加载第二份 CLIP。

每个 prompt 最多生成 `--num-candidates` 张。达到 `--min-candidates` 后，只要已经找到一张同时通过 Global、天气与 plausibility，且 `utility > --stop-utility`、`mining_probability >= --stop-mining-probability` 的候选，就停止该组生成。默认分别为 1、0、0.25。达到预算但没有达标时，从已有合格候选中选择一张；没有合格候选则该组退出。停止目标控制搜索预算，最终 hardness/random 选择仍使用所有已看到的合格候选，因此选中图不一定达到 mining 停止目标。`--sampling-policy fixed` 使用完全相同的实验采样器、门槛和 seed，但总是生成 K 张，可用于资源对照。

闭环可直接启用新入口：

```bash
python AdaptVPR/experiments/vpr_guidance/run_loop.py loop \
  --root outputs/vpr_guidance_adaptive --arm select \
  --gsv-root dataset/gsv-cities --cities Bangkok \
  --prompts dataset/AdaptCities/prompts/adaptcities_160k_prompts.jsonl \
  --rounds 3 --sources-per-round 500 --candidates 4 \
  --generation-mode adaptive \
  --adaptive-args='--min-candidates 1 --weather-min-shift 6 --stop-mining-probability 0.25' \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --salad-args='--batch-size 32 --images-per-place 4 --num-workers 4 --backbone-weights /home/admin123/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth'
```

`random`、`select`、`full` 都可用该模式；默认仍是原固定 K 流程。新模式的候选目录属于各分支当前学生，不能跨学生复用；分支候选预算与分布可能不同。adaptive-vs-fixed 回答资源和接受质量问题；它不能代替共享固定候选池上的 random-vs-hardness 选择消融，matched pools 也不能消除这种搜索差异。正式效果仍需同起点、同训练曝光、real-only 对照、多 seed 与完整独立 Recall 评估。

在线和独立 scorer 现在使用训练数据集最大真实视图数固定正图抽样上下文，保证同一候选逐张评分与批量评分一致。历史评分的 Monte Carlo 正图抽样可能不同；修改代码或参数后须使用新实验 root。

新入口逐张保存 `candidates.jsonl`，包含原 Global verdict、天气 contrast、utility、plausibility、最终 eligibility 和图像/行哈希。完整重跑校验产物后跳过模型加载；中断恢复只补未完成前缀；损坏图像使该组从首个损坏候选开始重生成。`summary.json` 记录每组尝试数、停止原因和相对于 K 的调用节省量。`--plan-only` 打印冻结配置，不加载模型也不写目录。SALAD 可用 `--score-args='--device cpu'` 放在 CPU 上，IC-Light 仍需要 CUDA。

天气校准复现：

```bash
python AdaptVPR/experiments/generation_diagnosis/calibrate_weather_signal.py \
  --reviews outputs/gen_diagnosis/manual_reviews_iclight.jsonl \
  --out outputs/gen_diagnosis/weather_calibration_new
```

它核对源图和输出哈希，并按源图留一校准；同源图的所有天气/方法都在同一个 fold，避免候选泄漏。现有 non-blind agent 标签仍不是独立人工真值。实现与实测范围见 [自适应验证记录](ADAPTIVE_VALIDATION.md)。

## 环境与依赖

以下命令均从工作区根目录执行。项目当前可用的环境是 `conda AdaptVPR`，也可以直接使用其绝对解释器路径。

```bash
cd /home/admin123/github/WORKSPACE
source /home/admin123/miniconda3/etc/profile.d/conda.sh
conda activate AdaptVPR

VPR_PYTHON=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
VPR_GUIDANCE=AdaptVPR/experiments/vpr_guidance
VPR_GSV=dataset/gsv-cities
VPR_PROMPTS=dataset/AdaptCities/prompts/adaptcities_160k_prompts.jsonl
VPR_BACKBONE_REPO=/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main
VPR_BACKBONE_WEIGHTS=/home/admin123/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
```

安装时需要四组依赖；本目录的 requirements 只是新增部分：

```bash
"$VPR_PYTHON" -m pip install \
  -r AdaptVPR/requirements.txt \
  -r AdaptVPR/adapters/requirements.txt \
  -r salad/requirements-workflow.txt \
  -r "$VPR_GUIDANCE/requirements.txt"
```

生成和 LoRA 训练需要 CUDA。真实验证还需要可用的 vismatch matcher 与 CLIP 本地模型。`common.py` 自动加载 `AdaptVPR/.env`，模型路径、matcher 和服务要求见 [API 合约](../../docs/API_CONTRACTS.md)。数据目录需要 `GSV-Cities/Images/` 和 `Dataframes/`。

离线新训练同时传 `--backbone-repo` 和 `--backbone-weights`：前者提供 DINOv2 代码，后者提供预训练 backbone；SALAD aggregator 仍随机初始化。完整 SALAD checkpoint 已包含 backbone，使用 `--init-checkpoint` 时不要额外传 `--backbone-weights`。闭环 wrapper 会在学生初始化和恢复时移除该参数，在 real-only 第 0 个学生和 fresh final 训练时保留它。评分/评估完整 checkpoint 只需本地 backbone repo。

## 分阶段运行

下面以 Bangkok 的 40 条 Global prompt、每条 4 个候选说明接口。`VPR_DEMO` 使用新的输出目录；已有结果的恢复方式见下文。

```bash
VPR_DEMO=outputs/vpr_guidance_demo

"$VPR_PYTHON" "$VPR_GUIDANCE/generate_candidates.py" \
  --prompts "$VPR_PROMPTS" --image-root "$VPR_GSV/Images" \
  --output-dir "$VPR_DEMO/candidates" --cities Bangkok \
  --offset 3 --num-sources 40 --num-candidates 4 --seed 42

"$VPR_PYTHON" "$VPR_GUIDANCE/score_candidates.py" \
  --candidates "$VPR_DEMO/candidates/candidates.jsonl" \
  --checkpoint salad/checkpoint/dino_salad.ckpt \
  --real-data "$VPR_GSV" --cities Bangkok \
  --selection hardness --output-dir "$VPR_DEMO/scoring" \
  --train-batch-size 32 --images-per-place 4 \
  --negative-pool-size 4096 --negative-draws 16 --miner-margin 0.1 \
  --backbone-repo "$VPR_BACKBONE_REPO" --batch-size 16 --num-workers 0
```

这里的公开 `dino_salad.ckpt` 只用于快速探查；正式闭环默认从真实训练集训练共享初始学生。评分输出 `scored.jsonl`、`selected.jsonl` 和 `summary.json`，并使用 GSV 元数据映射训练 place 标签。源图不属于可用训练 place 的条目会计入 `unusable`。

先验证 LoRA 输入和 SALAD 混合训练数据：

```bash
"$VPR_PYTHON" "$VPR_GUIDANCE/train_lora.py" \
  --selected "$VPR_DEMO/scoring/selected.jsonl" --check-data

"$VPR_PYTHON" salad/train_salad.py \
  --real-data "$VPR_GSV" --cities Bangkok \
  --synthetic-manifest "$VPR_DEMO/scoring/selected.jsonl" \
  --output-dir "$VPR_DEMO/student_check" --check-data
```

有 eligible mined positives 后，可以训练并在下一批 prompt 上使用 LoRA：

```bash
"$VPR_PYTHON" "$VPR_GUIDANCE/train_lora.py" \
  --selected "$VPR_DEMO/scoring/selected.jsonl" \
  --output "$VPR_DEMO/lora.safetensors" \
  --steps 1000 --batch-size 4 --rank 8 --alpha 8 \
  --precision auto --save-every 100 --seed 42

"$VPR_PYTHON" "$VPR_GUIDANCE/generate_candidates.py" \
  --prompts "$VPR_PROMPTS" --image-root "$VPR_GSV/Images" \
  --output-dir "$VPR_DEMO/next_candidates" --cities Bangkok \
  --offset 43 --num-sources 40 --num-candidates 4 --seed 42 \
  --lora "$VPR_DEMO/lora.safetensors"
```

如果 `--check-data` 显示 0 个训练例，单独训练器会提示跳过；`run_loop.py` 自动处理这一分支。训练器检查源/目标图像不同、文件可读、重复目标无冲突及批次分辨率一致。

## 一键闭环与公平最终比较

同一实验 root 的所有分支使用相同参数，并按顺序运行。`--salad-args` 管理训练配方，`--score-args` 设置抽样/描述符提取，`--lora-args` 设置生成器训练。wrapper 自动把 SALAD 的 batch size、images per place、最少真实视角和 miner margin 传给评分器，禁止 extra args 覆盖这些受管理参数。

```bash
VPR_RUN=outputs/vpr_guidance_experiment
VPR_SALAD_ARGS="--batch-size 32 --images-per-place 4 --num-workers 4 --backbone-weights $VPR_BACKBONE_WEIGHTS"
VPR_SCORE_ARGS="--negative-pool-size 4096 --negative-draws 16 --batch-size 16 --num-workers 0"
VPR_LORA_ARGS="--batch-size 4 --rank 8 --alpha 8 --save-every 100 --precision auto"

for VPR_ARM in random select full; do
  "$VPR_PYTHON" "$VPR_GUIDANCE/run_loop.py" loop \
    --arm "$VPR_ARM" --root "$VPR_RUN" --python "$VPR_PYTHON" \
    --gsv-root "$VPR_GSV" --prompts "$VPR_PROMPTS" --cities Bangkok \
    --rounds 3 --sources-per-round 500 --candidates 4 --seed 42 \
    --student-init-epochs 10 --student-epochs 2 --lora-steps 1000 \
    --backbone-repo "$VPR_BACKBONE_REPO" \
    --salad-args="$VPR_SALAD_ARGS" \
    --score-args="$VPR_SCORE_ARGS" --lora-args="$VPR_LORA_ARGS"
done
```

`--student-checkpoint PATH` 可提供共享初始学生，省去 real-only 初始化训练；各分支必须使用相同 checkpoint。正式控制实验需要说明这种初始化，不能把它与默认 fresh student 的结果混合比较。

`full` 的 LoRA 可能改变 verifier 接受率，因此各分支不会天然具有相同的 accepted groups。先 audit，再在三个分支共同保留的 prompt 交集上做 final：

```bash
"$VPR_PYTHON" "$VPR_GUIDANCE/run_loop.py" audit \
  --root "$VPR_RUN" --match-pools

for VPR_ARM in random select full; do
  "$VPR_PYTHON" "$VPR_GUIDANCE/run_loop.py" final \
    --arm "$VPR_ARM" --root "$VPR_RUN" --python "$VPR_PYTHON" \
    --gsv-root "$VPR_GSV" --cities Bangkok --seed 42 \
    --final-epochs 10 --match-pools \
    --backbone-repo "$VPR_BACKBONE_REPO" --salad-args="$VPR_SALAD_ARGS" \
    --svox-root dataset/svox --eval queries queries_night queries_rain queries_snow
done
```

默认 `--final-scope synthetic_places`：final SALAD 只在 pool 中有合成图的 place 上训练（`--synthetic-places-only`）。几百张合成图分散在几千个 Bangkok place 上时，合成曝光占比不足 1%，random 与 select 之间即使存在差异也测不出来。需要与 AdaptVPR 原始设置一致时，用 `--final-scope all_places`。

`--final-control real_only` 在**同一批 place** 上训练 synthetic fraction 为 0 的对照模型，输出到 `final_salad_real_only/` 与 `final_eval_real_only/`。没有这一对照，就无法区分"合成数据有用"和"选择方法有用"。最小决策实验为：

| 比较 | 回答的问题 |
| --- | --- |
| `random` vs `real_only` | AdaptVPR 式合成正样本在这些 place 上是否有用 |
| `select` vs `random` | 当前学生的 hardness 反馈是否优于随机选择 |
| `full` vs `select` | 把反馈蒸馏进生成器是否再带来收益 |

audit 写出 `pool_comparison.json` 和各分支的 `matched_pool.jsonl`，检查重复身份、合格状态以及同组 source/condition/prompt 一致性，报告被排除的组。匹配要求三个分支全部完成且交集非空。每个 final 从预训练 DINOv2 和**新随机 SALAD aggregator** 开始，不沿用反馈学生；相同 seed 和配方使最终训练起点、预算可比。交集控制了最终 pool 组成，不消除前面反馈学生经历不同数据的影响。未设置 `--svox-root` 时只训练 final；其他评估协议见 [SALAD 工作流](../../../salad/WORKFLOW.md)。

只查看预计命令，可在上述 `loop`/`final` 命令末尾加 `--dry-run`。它不会创建输出目录或加载模型；实际数据可读性应通过相应 `--check-data` 另行核验。

## 恢复、指纹与产物

生成器启动时验证配置、prompt 文件、选中源图内容、LoRA 内容、关键实现代码和已保存 JPEG 的 SHA-256。同一目录改 seed、slice、候选数量、prompt、source、LoRA 或验证实现会拒绝复用，要求新的输出目录。重复 sample_id 和清洗后的文件名碰撞也会拒绝。重跑完全相同的生成命令只补缺失/损坏图像，完整时不加载模型；仅恢复 JSONL 最后未完成的一行，内部损坏不能静默忽略。

早期 Claude manifest 可在旧配置、每行 seed/prompt/source/output/阈值及图像校验通过后迁移。其行标记 `provenance=legacy_identity_validated`；`generation_complete.json` 明确记录原始 source/model/code hashes 未保存、旧 verifier scores 保留。迁移不会把历史证据变成新的 GPU 重验证。

`run_loop.py` 将实验 root 绑定到参数、输入与代码指纹。每阶段先保存 request，只有所需产物完整才写 completion；缓存命中还检查输出内容。manifest、checkpoint、元数据文件使用内容 SHA-256；大型数据/代码目录的输入签名使用文件清单、size 和 mtime，**不等于对全部训练 JPEG 逐一做内容哈希**。代码或输入变化后使用新的实验 root。

| 产物 | 用途 |
| --- | --- |
| `generation_config.json` / `generation_complete.json` | 请求 fingerprint、manifest hash、候选/接受/legacy 数量 |
| `scored.jsonl` / `selected.jsonl` / `summary.json` | utility、mining probability、每组选择及统计 |
| `round_r/pool.jsonl` / `final_pool.jsonl` | 累计 accepted 样本 |
| `round_r/feedback.json` | mined 数量、LoRA 更新或跳过原因 |
| `checkpoint.pt` | SALAD 完整训练状态；wrapper 支持 epoch 边界恢复 |
| `lora.safetensors` / `lora.json` | 完成全部步骤后发布的服务权重和训练统计 |
| `lora.training.pt` | optimizer、schedule、sampler、随机状态和中间 LoRA |

重跑相同 `run_loop.py loop` 命令会恢复未完成训练。单独 LoRA 恢复时使用原始 selected 和训练参数，总 steps 仍为原计划值：

```bash
"$VPR_PYTHON" "$VPR_GUIDANCE/train_lora.py" \
  --selected "$VPR_DEMO/scoring/selected.jsonl" \
  --output "$VPR_DEMO/lora.safetensors" \
  --steps 1000 --batch-size 4 --rank 8 --alpha 8 \
  --precision auto --save-every 100 --seed 42 \
  --resume "$VPR_DEMO/lora.training.pt"
```

`--resume` 恢复同一次训练；`--init-lora` 从上一轮服务权重开始一次新训练，两者不能同时使用。LoRA recovery state 采用 `format_version=2` 和 `weights_only=True`，拒绝旧 v1 状态；不要以关闭安全加载绕过拒绝。这与 SALAD 自己的完整 checkpoint 格式不同，SALAD 仍使用其已实现的 v1 checkpoint 合约。

## 目前的实测范围

2026-10-08 的现有 Bangkok probe：40 组 × 4 = 160 个候选，其中旧 Global verifier 通过 37 个；未加门限的 place-grouped 评分选择了 18 组，其中 8 张 selected 的 `U > 0`。加入 plausibility gate 后为 16 组，其中 6 张 `U > 0`，见 `outputs/vpr_guidance_smoke/score_probe_gated/`。按 condition 的通过数为 overcast 35/112、rain 2/8、snow 0/24、night 0/12、fog 0/4，失败原因全部是 `s_geo`。合格候选平均 mining probability 为 `0.2179`，selected 平均为 `0.2292`。LoRA `--check-data` 确认可训练样本为 `8/18`，统一分辨率为 `400 × 296`。

来源是工作区的 `outputs/vpr_guidance_smoke/cand_probe/` 和 `outputs/vpr_guidance_smoke/score_probe_codex/summary.json`。这说明数据链路、评分和 mined 输入可用；这些是小规模探查统计，没有据此声称 Recall 提升。完整三分支、多 seed、公平 final 与外部评估仍需要实际跑完，状态见 [集成检查表](INTEGRATION_TODOS.md)。

真实 GPU LoRA 训练、v2 恢复后的权重一致性、带 LoRA 的生成/恢复、小型两轮闭环及 final 评估机制已另行核验，详见 [验证报告](VALIDATION.md)。该报告区分机制检查与正式检索效果实验。

运行 CPU 回归检查：

```bash
"$VPR_PYTHON" -m unittest discover \
  -s "$VPR_GUIDANCE/tests" -p 'test_*.py' -v
```
