# 夜间生成 v4：内容约束、自动审核与训练相机增强

最初版本只交付代码与 CPU 测试。本次续接复用了已完成的机器对照和 v3 信号，并对**已经生成的 20 张 v4 图**运行了冻结 DINO 内容信号提取。没有继续调用 Qwen，没有启动训练或检索评估。已有图像、v3 Prompt、旧计划、审核结果和训练记录均保留。

## 本次续接结果与下一步

当前入口为 `salad/complete_night_v4.py`，正式续接记录使用新的
`outputs/night_v4_completion_v3/`。这里的目录版本号是续接工程版本，内容筛选器版本为 **v4.2**；v1/v2 是保留的集成记录，不作为后续入口。

| 项目 | 实际结果 |
|---|---|
| v4.1 原控制检验 | 不可用；留出插入对照没有被拒绝，旧训练清单为空 |
| v4.2 控制复核 | 131 个良性对照：124 pass / 0 reject / 7 unknown；131 个插入对照：0 pass / 126 reject / 5 unknown |
| 原 v3 3,858 张 | 2,046 pass / 696 reject / 1,116 unknown |
| 现有 v4 20 张 | 15 pass / 0 reject / 5 unknown |
| 合并夜间候选 | 2,050 张、2,050 个地点；优先 v4、每地点最多一张、排除校准/验证地点 |

v4 的 5 个 Unknown 中，4 个是原几何/天气检查未通过，1 个是内容证据不够明确。合并清单含 2,035 张旧 v3、15 张新 v4，主要用于复用现有数据验证训练链路；它不能代表纯 v4 Prompt 的效果比较。纯 v4 的候选清单位于 `v4_audit/export/training_manifest.jsonl`，15 个地点的规模只适合小样本链路检查。

旧规则要求超过 93.66% 的 DINO 异常面积，因而无法识别局部插入。新增独立的 v4.2 使用局部 DINO/结构联合差异，窗口与阈值仅由 381 个 FIT 训练地点的机器控制对拟合，没有通过观察 Qwen 通过率或检索 test 分数来调阈值。旧规则和旧输出不改写。

**这些是探索性的自动筛选结果。** 131 个控制地点此前已参与发现 v4.1 失败，本次重用不能当作新的独立验证；控制集上的检验也不代表实际生成图误放率。当前局部窗口可能漏过小招牌、细文字，缺少结构证据的区域也未得到完整验证。因此 pass 只表示满足当前筛选规则，不代表已证明语义保真或真实夜景等价；2,050 张是实验候选，不据此启动全量正式训练。旧缓存里缺少单应性的 109 对仍保持 Unknown，不用单位变换补造证据。

从项目根目录执行：

```bash
cd /home/admin123/github/WORKSPACE
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python

# 默认只核验冻结状态和生成报告；不会生图、训练或运行模型。
"$PY" -B salad/complete_night_v4.py --phase report

# 已完成阶段会验证 SHA 后复用；只校准/审核，不训练。
# 当前冻结计划会自动取回已绑定的 v4 信号缓存位置。
"$PY" -B salad/complete_night_v4.py --phase audit --execute

# 只运行 A/M/M_camera 的 --check-data；不会实例化模型或创建 optimizer。
"$PY" -B salad/complete_night_v4.py --phase data-check --execute
```

输入清单、排除清单、相机 profile、checkpoint、训练/数据/模型实现、DINO 实现及地点 CSV 均记录 SHA。已完成阶段原子提交并校验；中断的 attempt 保留，重试使用新 attempt。输入或实现变化后需新建输出目录，不绕过旧密封恢复。生成目录日后增加图片也不会静默混入这次冻结的输入快照。

`report.md` 是简明状态，`report.json`/`training_preflight.json` 包含逐阶段、数据身份、排除与命令，`commands.sh` 包含显式的预检查及 **A/M/M_camera 各两 batch** smoke 命令。该入口不会执行其中的训练命令。先查看：

```bash
cat outputs/night_v4_completion_v3/commands.sh
```

需要手动启动某一个 smoke 时，可按方法名执行预先生成的精确 argv，例如 M_camera（把 `METHOD` 改成 A 或 M 即可运行另两组）：

```bash
METHOD=M_camera "$PY" -c 'import json,os,subprocess; p=json.load(open("outputs/night_v4_completion_v3/training_preflight.json")); assert not p["errors"], p["errors"]; c=next(c for c in p["commands"] if c["method"]==os.environ["METHOD"]); subprocess.run(c["argv"],check=True)'
```

三组使用相同完整真实地点池、初始 checkpoint、seed、P/K、LR、block、RandAugment、source preference 和两步预算。只有合成替换与 camera 开关不同，不添加 B0、局部蒸馏或 Reliability。当前每地点只保留一张生成图，`synthetic_place_slots=8` 是每 batch 的合成可用**地点数**，不是图片比例；P=32/K=4 下最多约 8/128=6.25% 生成图，仍以实际曝光日志为准。这是链路 smoke，不是正式效应比较或已选定的最优超参数。

相机 profile 保持 `exploratory`。本地尚无可用且未参与既往评估的独立真实夜景校准语料；不使用 SVOX/RobotCar 或已检查过的测试图拟合参数。后续正式 A/M/M_camera 对照需要另行冻结预算与多 seed 协议，不能直接续训两 batch smoke checkpoint。

CPU 验证入口（含临时图片、模拟服务、微型 CPU 模型，不接触真实生成服务）：

```bash
# 本次新增的续接、预检查和 v4.2 审核：共 24 项通过。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=salad:salad/tests:AdaptVPR \
  /home/admin123/miniconda3/envs/AdaptVPR/bin/python -m unittest \
  test_complete_night_v4 test_night_training_preflight
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=AdaptVPR \
  /home/admin123/miniconda3/envs/AdaptVPR/bin/python -m unittest discover \
  -s AdaptVPR/experiments/qwen_curriculum/tests -p test_content_audit_v42.py

# 原有回归入口：
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=salad:salad/tests \
  /home/admin123/miniconda3/envs/AdaptVPR/bin/python -m unittest \
  test_training_data test_synthetic_aux test_local_distillation test_source_replacement \
  test_camera_augmentation test_camera_calibration
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=AdaptVPR \
  /home/admin123/miniconda3/envs/AdaptVPR/bin/python -m unittest discover \
  -s AdaptVPR/experiments/qwen_curriculum/tests -p 'test_*v[34].py'
```

## 改动与边界

- 新 `prompt_bank_v4.py` 停用 shopfronts；所有 night 变体明确干路面、自然低饱和，只允许源图中已有灯、窗、招牌、车辆发光。钠灯只影响灯附近，禁止全图橙色。蓝调时刻占 night 变体抽样权重的 10%，是预设起点，不是从测试分数选出的最优比例。Prompt 约束不保证模型服从。
- `night_first_v4.py` 默认使用 `outputs/weather_v4/`。几何/天气通过仅为候选；记录 `base_quality_passed`，训练资格仍为 false。只有独立自动内容审核可导出最终训练清单。继续运行旧 `resume_control.py` 仍执行其冻结的 v3 计划；不要把旧入口当成 v4。
- 内容审核使用冻结本地 DINO 的局部特征差异，以及对齐后局部标准化灰度结构差异的联合异常。单应性映射显式使用匹配器方形坐标，边界完整 Patch 超出有效重叠则不算证据。阈值来自训练地点的机器构造对照，按地点分离拟合/留出检验，不设固定余弦阈值。
- 机器对照的正类为已知同图的光度变换，负类为受控插入其他训练地点的内容。它只能检验对这些受控破坏的检测能力，不能证明对所有生成幻觉有效。无法区分受控插入与正常光度变化时，校准不可用，后续一律 Unknown；Unknown 不进训练。
- 旧 shopfronts 按明确策略排除。可选 OCR 使用绑定图像 SHA 的坐标框缓存，只记录新增检测文本；亮灯后原有文字变得可读，也会新增 OCR 检测，因此 OCR 单独不作为编造判据。
- 相机增强默认关闭，仅处理训练中的生成 night/rainy_night。包含 gain/gamma、已有高光 bloom、方向模糊、线性光泊松—高斯噪声、JPEG、降饱和与暗角。它是外观模拟，不能恢复已经压成纯黑的细节，也不能凭空恢复真实灯源。混合/替换、辅助 query、局部学生分支均有接线；教师与局部对应支持使用干净输入。
- 相机参数使用独立键控 RNG，不额外消费主采样、辅助采样或 Dropout RNG；profile 内容和 SHA 进入 checkpoint，恢复时变更参数将报错。`--no-augment` 只关闭原 RandAugment，相机开关由 profile 单独控制。推理/验证不应用相机增强。

## 先审核现存 v3 图，不需要重新生成

以下保留最初 v4.1 的分步命令，供理解已有缓存来源。你已完成的 run1 不要重跑或覆盖；本次后续使用上面的 v4.2 续接入口。

```bash
cd /home/admin123/github/WORKSPACE
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
AUDIT=AdaptVPR/experiments/qwen_curriculum/audit_content_v4.py
V3=outputs/weather_v3/state/results.jsonl
OUT=outputs/night_content_audit_v4_run1
REPO=/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main
WEIGHTS=salad/checkpoint/dino_salad.ckpt

# 512 个训练地点，每地点构造一组光度对照/明确插入破坏，不使用人工标签。
"$PY" -B "$AUDIT" controls --manifest "$V3" \
  --training-root dataset/gsv-cities --dataset gsv-cities \
  --exclude-places outputs/qwen_curriculum/state/finetuning/real_baseline_v1/excluded_places.json \
  --output-dir "$OUT/controls" --count 512

# 下两条 --execute 只运行冻结 DINO，不训练、不生图；逐对处理，不缓存教师特征库。
"$PY" -B "$AUDIT" signals --manifest "$OUT/controls/controls.jsonl" \
  --output "$OUT/control_signals.jsonl" --dino-repo "$REPO" \
  --dino-checkpoint "$WEIGHTS" --device cuda --execute

# 0.05 是显式声明的研究操作点，数值阈值由对照估计，不代表幻觉概率保证。
"$PY" -B "$AUDIT" calibrate --manifest "$OUT/controls/controls.jsonl" \
  --signals "$OUT/control_signals.jsonl" --output "$OUT/calibration.json" \
  --false-alarm-rate 0.05

"$PY" -B "$AUDIT" signals --manifest "$V3" --output "$OUT/pair_signals.jsonl" \
  --dino-repo "$REPO" --dino-checkpoint "$WEIGHTS" --device cuda --execute
"$PY" -B "$AUDIT" audit --manifest "$V3" --signals "$OUT/pair_signals.jsonl" \
  --calibration "$OUT/calibration.json" --output-dir "$OUT/audit"
```

原 v4.1 的 `$OUT/audit/training_manifest.jsonl` 为空且不可用于训练；当前候选入口是 `outputs/night_v4_completion_v3/combined/training_manifest.jsonl`。`audited_manifest.jsonl` 保留所有 pass/reject/unknown、理由与原记录身份。校准不可用或证据缺失时清单允许为空，不能通过降低阈值追求通过率。

## 新 Prompt 计划

```bash
# CPU 计划准备（会筛源图和校验归档可靠度记录，不调用生成服务）
"$PY" -B AdaptVPR/experiments/qwen_curriculum/night_first_v4.py prepare
"$PY" -B AdaptVPR/experiments/qwen_curriculum/night_first_v4.py check

# 如以后决定继续生图，先用显式小预算；本次没有执行。
"$PY" -B AdaptVPR/experiments/qwen_curriculum/night_first_v4.py run \
  --domain night --max-calls 20
```

## 相机参数与独立校准

`salad/configs/synthetic_camera_exploratory.json` 是可选的有界研究起点，未用任何测试图拟合。先比较同一已审核清单的 M 与 M+camera，不能把相机增强的收益归到生成本身。真实图 A 保留相同地点/预算对照；若要研究更强普通增强的贡献，再独立声明 A 的相应增强消融。

校准输入必须显式声明独立真实夜景 split，并提供此前评估的完整排除清单。Tokyo247/SF-XL 部分图已经被本项目评估，不能整目录直接作为新校准集。RobotCar/SVOX 直接禁止。路径、已知内容 SHA 以及地点/轨迹组重合会报错；组 ID 必须跨清单一致。代码只能核对已记录的使用，不能证明未记录的历史用途。

真实校准清单示例（路径相对各自 JSON 文件）：

```json
{
  "version": 1,
  "role": "camera_calibration",
  "dataset": "independent-night-corpus",
  "split": "calibration",
  "condition": "night",
  "prior_use": "never_evaluated_or_used_for_selection",
  "images": [{"path": "images/scene001.jpg", "group_id": "independent-night-corpus/scene001"}]
}
```

排除清单示例（应填写完整已评估库存，不是只填一张）：

```json
{
  "version": 1,
  "role": "evaluation_exclusion",
  "images": [{"path": "/path/to/evaluated/image.jpg", "group_id": "corpus/evaluated-scene", "sha256": "optional-actual-image-sha256"}]
}
```

```bash
"$PY" -B salad/calibrate_synthetic_camera.py \
  --real-manifest /path/to/independent_calibration.json \
  --evaluation-exclusion /path/to/evaluation_exclusion.json \
  --synthetic-manifest outputs/night_v4_completion_v3/combined/training_manifest.jsonl \
  --output-dir outputs/camera_tone_fit_v1 --max-images 128
```

该工具只匹配亮度、黑区/高光比例、饱和度统计，拟合 gain/gamma/saturation。它保留其他操作的未标定先验，输出整体状态仍为 `exploratory`。完整随机增强后的统计与 tone-only 报告不相同；无配对元数据时不能从这些全局统计识别真实噪声、快门、ISP 或相机。

## 后续训练调用

使用本文开头经预检查生成的 `training_preflight.json` 和 `commands.sh`，不再维护另一套手写参数。A/M/M_camera 输出目录互相独立，已有目录不会被覆盖。

Smoke 的两 batch checkpoint 不能接着当作正式预算运行。正式实验另建目录、移除两 batch 上限，并重新从相同初始 checkpoint 开始。恢复正式实验的完整 epoch checkpoint 时，用原配置和 profile，并把初始化参数替换成 `--resume <新运行目录>/checkpoint.pt`。训练文件已经新增 opt-in 接口，旧冻结运行器的源码哈希检查会检测到代码版本变化；不要绕过它声称旧计划逐位续训。旧 checkpoint 和历史结果不变，新实验使用新代码快照与新输出目录。
