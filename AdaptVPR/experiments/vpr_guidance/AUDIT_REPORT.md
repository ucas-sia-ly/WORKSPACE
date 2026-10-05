# Global VPR-aware generator correctness audit

审计日期：2026-10-06（Asia/Shanghai）。仓库：ucas-sia-ly/WORKSPACE；分支：
`test/vpr-loss-generator`；起始提交：`2843caedd4dde98ccc93105aacf23fa739b09fc5`。

先完成只读审计，再修改确认的问题。覆盖实验目录全部入口、tests、原 AdaptVPR
Global agent / HTTP IC-Light adapter / frozen prompt 输入 / Global verifier，
以及本机官方 SALAD checkout（`6aede13a`）和官方源码。已有数据、模型、outputs
和未跟踪的 `Bag-of-Queries/` 均保留。

状态含义：**PASS** 为实际执行；**STATICALLY VERIFIED** 为代码交叉核对；
**UNVERIFIED** 为本次没有实际验证。CPU 替身测试不代表真实模型整条流水线通过。

## 1. Critical bugs found

| 优先级 | 文件 / 函数 | 问题、原因及实验后果 | 修复 |
|---|---|---|---|
| P0 | `iclight.py / load_lora` | `strict=False` 只检查 unexpected 中的 LoRA 子集，遗漏 missing keys；空或缺损 adapter 能成功加载。payload 还能夹带并改写 base weights，使 C 实验被错误标为训练好的 LoRA，或同时改变基座。 | 检查完整 adapter key 集合、shape、finite、rank/alpha、target modules、base metadata；禁止非 adapter keys；合并原基座 state 后使用 `strict=True`。真实 PEFT round-trip 和损坏 checkpoint 回归测试通过。 |
| P0 | `AdaptVPR/verification/evaluator.py / _extract_clip_feature, _compute_s_div, evaluate` | CLIP NaN cosine 经 Python `min/max` 得到 `s_div=1.0`，高 geometry 时会错误 passed，使异常图成为训练 positive。 | embedding 必须 finite 且 norm 非零，cosine 与最终分数必须 finite；无效值明确报错。Global threshold 保持 0.78 / 0.15。NaN/Inf/零向量回归通过。 |
| P0 | `generate_dataset.py / main`, `mixed_salad.py / load_synthetic_manifest` | 生成只依赖 filename 拆分标签；下游完全相信 manifest 的 city/place，未核对 source identity；非空字符串 `"false"` 也被当作真，Local/Dual 记录未受 route 限制。错标或异常 manifest 可以将异地图片作为 positive。没有发现现有 500 sources 已错标，发现的是可复现的校验缺口。 | GSV Dataframes 作为 ground truth，核对完整 source identity、city、canonical integer place_id；仅实际 Boolean true、Global 记录可入池。错误 source 标签和非 Global 输入会拒绝，字符串 false 不会被加载。 |

## 2. Important bugs found

| 优先级 | 文件 / 函数 | 问题、原因及实验后果 | 修复 |
|---|---|---|---|
| P1 | `mixed_salad.py / load_synthetic_manifest, _plan_epoch_mix` | 重复 manifest 行被当成不同 synthetic 文件；同一个文件可占多个 K-item slots，容量和曝光统计虚高。 | resolved path 去重；同一文件的冲突标签报错；按唯一文件容量分配 quota。 |
| P1 | `train_salad.py / SetDatasetEpoch` | 日志只写完整 epoch 的计划；max_steps 截断、DDP padding、各 rank 同时写日志使声称的实际比例失真。复现：9 places、K=4、2 ranks 时，padding 会重复 place，实际 slots 与计划不同。 | loader 携带真实 slot metadata，训练 callback 在交给官方 training_step 前移除；仅已消费 batch 计数；跨 rank sum，rank0 写 journal；输出 actual ratio/fraction、world_size、epoch_complete。 |
| P1 | `train_salad.py / official_model_class` | 官方 VPRModel 自行在 optimizer_step 推进 scheduler，同时返回的 scheduler 被 Lightning 默认按 epoch 再推进。CPU toy 4 updates 得到 scheduler last_epoch=6。 | 本地 subclass 保留官方 loss/miner/training_step，只把 scheduler 设为 step interval、optimizer_step 只调用 optimizer；每个 optimizer update 推进一次。 |
| P1 | `teacher.py / from_pil` | source 先 PIL resize，generated 直接 tensor antialias resize，实际 preprocessing 不同；cosine teacher target 有系统差异。 | 两者统一 RGB [0,1] tensor → 322 bilinear/antialias → ImageNet normalization。实际官方 teacher forward 数值保持不变。 |
| P1 | `teacher.py / source descriptor cache`, `train_generator.py / read_manifest` | `.pt` 为无 metadata tensor；更换同维度教师仍会误用旧缓存，baseline/source 文件错换未被检测。 | 记录 actual teacher weights SHA256、source path/content SHA256、preprocess version、dimension；验证 shape/finite/unit norm。manifest 验证 source/baseline SHA、sample ID 和 sampling policy。旧缓存要求重新 prepare。 |
| P1 | `generate_dataset.py / main` | 默认加载 SALAD，离线生成依赖教师及其显存，违反优化后直接采样的设计。 | 默认无教师；只有显式 `--salad-audit` 才加载，且仅统计 cosine，无 latent optimization。 |
| P2 | `iclight.py / generate_released`, `prepare_data.py`, `train_generator.py` | 实验默认 negative prompt 与原 Global agent 不同；source 缺少原 HTTP wrapper 的 JPEG95 重编码。原始 B 不能声称与原 Global 条件完全一致。 | 使用共享 Global negative default / JPEG95 conditioning；训练 concat 使用同一处理；保持 released frozen prompt 原文。记录 CFG、scheduler、两 stage steps/strength 和实际 verifier policy；B/C 同输入同 policy。 |
| P2 | `train_salad.py / main, freeze_backbone_prefix` | defaults 为 16 places、322 input、wd=1e-3、全局 shuffle，与官方 main 不同；官方 frozen prefix 仅 forward detach，仍 requires_grad，DDP 有 unused parameters 风险。 | defaults 改为 60 places、224、wd=9.5e-9、city-local shuffle；显式冻结 backbone 非末4 blocks/非训练 prefix。架构、loss/miner、RandAugment 保持官方。 |
| P2 | `train_generator.py`, `train_salad.py` | 缺完整 resume 接口；旧目录覆盖、恢复旧 checkpoint 追加未来日志会污染统计和随机性。 | generator 保存 LoRA/AdamW/global_step/全部采样 RNG，校验 config 和 log/checkpoint；SALAD 保存 full Lightning checkpoint/RNG，支持单设备完整 epoch 恢复，明确拒绝 mid-epoch/DDP resume；已有输出冲突不静默覆盖。 |
| P2 | `data.py / read_global_prompts`, `prepare_data.py / main` | sample_id 可含路径分隔符造成 cache 写出目录或覆盖；准备中途失败会留下看似可用的截断 manifest。 | 安全且唯一的 basename ID，source/条件/image 提前校验；新空目录；全样本完成后才将 partial manifest 原子改名。 |
| P2 | `train_generator.py / validate_args` | timestep-window=0 会因 `[-0:]` 实际选择全 schedule；save-every=0 会取模异常，负 loss 权重等未被限制。 | 验证正数 window/interval/steps、finite 正 lr/clip、非负且至少一项非零的 loss weights、epsilon prediction type。 |

原 two-pass 梯度方法和 DDIM x0 公式**没有发现数学错误**。令
`a_bar = scheduler.alphas_cumprod[t]`，加噪为
`sqrt(a_bar)*z0 + sqrt(1-a_bar)*epsilon`，恢复为
`(zt-sqrt(1-a_bar)*epsilon_pred)/sqrt(a_bar)`。不是单步 alpha。
Pass A 在相同预测 x0 处计算 reduced guidance loss 的 cotangent，Pass B 使用
`sum(x0_train * grad_x0.detach())`，准确应用一阶链式法则；没有二阶 backward。
确定性 tiny diffusers UNet + PEFT 的全参数梯度与直接 backward 最大差为 0。
`torch.inference_mode` conditioning 经 UNet hook 的 cat 成为普通 tensor，未发现这会切断梯度。

普通 `persistent_workers=False` 本来没有确认的 stale epoch bug；本次增加共享 epoch
和 loader/sampler 提前更新，覆盖 persistent workers 与 resume 创建 iterator 的时序，
并补实际多 worker 回归，而不是把原代码静态推测为有 bug。

## 3. Files changed

| 文件 | 修改原因 |
|---|---|
| `data.py` | 安全 ID、规范 place ID、完整 GSV dataframe identity / label 验证、空输出保护。 |
| `prepare_data.py` | 数据预校验、source 标签、统一 policy、带 metadata cache、SHA 和完整 manifest 发布。 |
| `iclight.py` | 严格 LoRA 格式/加载、fp32 trainables、原 Global conditioning/negative/采样 policy。 |
| `teacher.py` | 统一可微 preprocessing、teacher/source/cache 身份校验；保留 frozen input-gradient forward。 |
| `train_generator.py` | manifest 校验、参数日志/freeze 断言、VJP helper、完整 resume、日志身份保护、loss/grad finite 检查。 |
| `generate_dataset.py` | 默认无需教师、dataframe labels、严格 Global acceptance、policy 记录、不覆盖。 |
| `mixed_salad.py` | 唯一 synthetic pool、source label/route/Boolean 校验、共享 epoch、slot metadata。 |
| `train_salad.py` | 官方 defaults/city sampling、scheduler 修正、freeze prefix、实际曝光/DDP 统计、完整 epoch resume。 |
| `AdaptVPR/verification/evaluator.py` | 原 verifier finite/norm/cosine fail-closed；不改变匹配方法和阈值。 |
| `tests/test_data_pipeline.py` | 实际曝光、标签、重复文件、persistent workers、DDP padding 回归。 |
| `tests/test_data_validation.py` | dataframe identity、leading zeros、city/pano underscores、collision、安全 ID。 |
| `tests/test_preparation.py` | CPU 替身验证 source/prompt/baseline/descriptor/label 对应及错换 baseline 检测。 |
| `tests/test_generation.py` | 真实 PEFT strict round-trip、损坏 checkpoint、B/C policy、默认无教师、verifier finite。 |
| `tests/test_gradients.py` | pixels / LoRA 梯度、two-pass 等价、scheduler inverse、cache、resume 数值一致。 |
| `tests/test_salad_training.py` | Lightning scheduler / 实际 batch accounting / resume 控制验证。 |
| `smoke_teacher_cpu.py` | 可复现的真实 pretrained VAE + SALAD CPU 梯度验证。 |
| `tests/smoke_salad_real.py` | 真实 cached DINOv2/fresh SALAD 的官方 loss、batch shape 和 optimizer smoke。 |
| `README.md`, `AUDIT_REPORT.md` | 实际接口、cache migration、曝光/恢复限制与验证证据。 |

## 4. Tests actually run

执行环境：AdaptVPR conda Python 3.10，torch 2.8.0+cu128、diffusers 0.36.0、
PEFT 0.21.2、transformers 4.57.6、Lightning 2.6.6、torchvision 0.23.0+cu128。
原 IC-Light requirements 的 diffusers 0.27.2 / transformers 4.36.2 仅作源码
交叉检查，本次没有切换环境实跑这些旧版本。默认 base Python 3.13
缺少 diffusers：`python -c 'import torch, diffusers'` 实际失败；这不是已通过的环境。
本次未改系统 Python 或安装新模型。

以下 `VPR_AUDIT_PY` 指实际使用的
`/home/admin123/miniconda3/envs/AdaptVPR/bin/python`；命令从仓库根执行。

```bash
VPR_AUDIT_PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
"$VPR_AUDIT_PY" -m compileall -q AdaptVPR/experiments/vpr_guidance
"$VPR_AUDIT_PY" -m unittest discover -s AdaptVPR/experiments/vpr_guidance/tests -v
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.prepare_data --help
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.train_generator --help
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.generate_dataset --help
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.train_salad --help
git diff --check
```

最终集中验证：上述 compileall、**43 tests / OK**、四个 help 和 diff check 全部
**PASS，exit 0**。完整 compile/tests/help 输出：`/tmp/vpr-audit-final-tests.log`。
多 worker 测试需要本地执行权限；沙箱内 Unix socket 被限制，获得执行审批后实跑通过。
审计前 compileall、原 5 tests、四个 help 同样 PASS。

真实 pretrained CPU 梯度命令（PASS，exit 0）：

```bash
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.smoke_teacher_cpu \
  --salad-repo /home/admin123/.cache/torch/hub/serizba_salad_main \
  --vae-base-model models/stable-diffusion-v1-5 \
  --source-image dataset/gsv-cities/Images/London/London_0006553_2008_10_263_51.51773366144955_-0.1279837045183434_cJvHsqb_EJPfIZ71qIXMPg.jpg
```

实测：CPU fp32，teacher descriptor `[1,8448]`、norm 1；patched 与官方 forward
descriptor 最大差 0；VAE decoded `[1,3,64,64]`、range `[0,1]`；
generated pixel gradient norm `0.0757807642`，predicted x0 latent gradient norm
`0.1679451019`；teacher/VAE trainable count 和参数梯度均为 0。
tiny UNet + PEFT VPR-only 梯度非零，two-pass 与单次完整 backward 一致；
generator checkpoint reload 后下一步 loss / LoRA weights 与连续训练逐位一致。

真实数据只读验证命令（PASS）：

```bash
"$VPR_AUDIT_PY" - <<'PY'
import json
from pathlib import Path
import pandas as pd
from AdaptVPR.experiments.vpr_guidance.data import GSVLabelIndex, gsv_image_name
idx = GSVLabelIndex(Path('dataset/gsv-cities/Dataframes'))
rows = [json.loads(x) for x in Path('outputs/generator_train/generator_train.jsonl').read_text().splitlines() if x.strip()]
for row in rows:
    key = idx.key_for_source(Path(row['source_path']))
    assert row.get('city') == key[0]
print('source identities:', len(rows), 'dataframe identities:', len(idx.identities))
root = Path('dataset/gsv-cities')
df = pd.read_csv(root / 'Dataframes/Bangkok.csv')
assert all((root / 'Images' / r['city_id'] / gsv_image_name(r)).is_file() for _, r in df.head(1000).iterrows())
print('1000 real dataframe image paths: PASS')
PY
```

实际 outputs：500 source identities 与 529506 个 GSV identity index 匹配；1000
Bangkok dataframe 路径全部存在。此检查不表示旧 descriptor/baseline cache 格式合格。

### 真实 GPU smoke：已执行

沙箱内 `nvidia-smi` 失败、CUDA False 是设备隔离；经 `require_escalated` 本地执行审批
后，实际 `nvidia-smi` / AdaptVPR-env torch CUDA 检测 **PASS**：RTX 4090，约 49 GiB，
driver 570.195.03 / CUDA 12.8。默认 base 环境 torch cu130 与 driver 不兼容，也不能用它
判断实际 AdaptVPR 环境是否可运行。没有把初次沙箱检测误写为整机无 GPU。

独立 `/tmp/vpr-audit-smoke-20261006`，最初 2 places × 4 real images，后来扩为
4 places × 4 real images，night/snow/rain/fog 各一个 source。使用本机真实模型权重。
以下变量只缩写实跑路径；各入口实跑参数如下：

```bash
SMOKE_ROOT=/tmp/vpr-audit-smoke-20261006
TEACHER_ROOT=/home/admin123/.cache/torch/hub/serizba_salad_main
SALAD_ROOT=/home/admin123/diffusion_vpr_test/reference_code/salad
export ICLIGHT_ROOT="$PWD/IC-Light"
export ICLIGHT_BASE_MODEL_PATH="$PWD/models/stable-diffusion-v1-5"
export ICLIGHT_MODEL_PATH="$PWD/models/iclight-ckpt/iclight_sd15_fc.safetensors"
export VISMATCH_ROOT="$PWD/vismatch"

"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.prepare_data \
  --prompts "$SMOKE_ROOT/prompts.jsonl" --image-root "$SMOKE_ROOT/gsv/Images" \
  --output-dir "$SMOKE_ROOT/prepared" --salad-repo "$TEACHER_ROOT"

"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.train_generator \
  --manifest "$SMOKE_ROOT/prepared/generator_train.jsonl" \
  --output-dir "$SMOKE_ROOT/generator_vpr_only" --max-steps 2 --save-every 1 \
  --lambda-diff 0 --lambda-vpr 0.1 --lambda-keep 0 --salad-repo "$TEACHER_ROOT"

"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.train_generator \
  --manifest "$SMOKE_ROOT/prepared/generator_train.jsonl" \
  --output-dir "$SMOKE_ROOT/generator_resume" --max-steps 3 --save-every 1 \
  --lambda-diff 0 --lambda-vpr 0.1 --lambda-keep 0 --salad-repo "$TEACHER_ROOT" \
  --resume "$SMOKE_ROOT/generator_vpr_only/checkpoints/step_000002.pt"

"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.train_generator \
  --manifest "$SMOKE_ROOT/prepared/generator_train.jsonl" \
  --output-dir "$SMOKE_ROOT/generator_default" --max-steps 1 --save-every 1 \
  --salad-repo "$TEACHER_ROOT"
```

全部 **PASS，exit 0**。VPR-only 专门隔离梯度，不改变研究默认 loss；另一个 default
run 验证完整三项组合。真实 LoRA trainable `1594368 / 861126852`，比例 0.185149%；
256 adapter state keys 全 fp32 finite，128 个 lora_B 都更新，AdamW moments fp32 finite。
VPR-only 两步 LoRA grad norm `0.01582645 / 0.02450804`，x0 guide norm
`0.01849715 / 0.03764749`；strict resume step2→3，grad norm `0.01428778`，
全部 optimizer step=3。默认三项：diff `0.07533060`、vpr `0.60629505`、
keep `0.05002526`，加权 total `0.13846137`，LoRA grad norm `0.03079395`。
这已实际证明真实 frozen SALAD → fp16 VAE → 完整 IC-Light UNet → fp32 LoRA 的梯度。

真实生成命令（以下三对入口每条 **PASS，exit 0**）：

```bash
# night：B / C
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.generate_dataset --prompts "$SMOKE_ROOT/prompts.jsonl" --image-root "$SMOKE_ROOT/gsv/Images" --output-dir "$SMOKE_ROOT/synthetic_B" --limit 1
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.generate_dataset --prompts "$SMOKE_ROOT/prompts.jsonl" --image-root "$SMOKE_ROOT/gsv/Images" --output-dir "$SMOKE_ROOT/synthetic_C" --limit 1 --lora "$SMOKE_ROOT/generator_vpr_only/checkpoints/step_000002.pt"
# snow：B / C
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.generate_dataset --prompts "$SMOKE_ROOT/prompts_snow.jsonl" --image-root "$SMOKE_ROOT/gsv/Images" --output-dir "$SMOKE_ROOT/synthetic_B_snow" --limit 1
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.generate_dataset --prompts "$SMOKE_ROOT/prompts_snow.jsonl" --image-root "$SMOKE_ROOT/gsv/Images" --output-dir "$SMOKE_ROOT/synthetic_C_snow" --limit 1 --lora "$SMOKE_ROOT/generator_vpr_only/checkpoints/step_000002.pt"
# rain / fog：B / C
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.generate_dataset --prompts "$SMOKE_ROOT/prompts_extra.jsonl" --image-root "$SMOKE_ROOT/gsv_extended/Images" --output-dir "$SMOKE_ROOT/synthetic_B_extra" --limit 2
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.generate_dataset --prompts "$SMOKE_ROOT/prompts_extra.jsonl" --image-root "$SMOKE_ROOT/gsv_extended/Images" --output-dir "$SMOKE_ROOT/synthetic_C_extra" --limit 2 --lora "$SMOKE_ROOT/generator_vpr_only/checkpoints/step_000002.pt"
```

真实 CLIP + SuperPoint/LightGlue 成功加载，无 fallback，无教师 audit、无 latent
optimization；各配对 source SHA / city-place / prompt / negative / seed / sampling /
actual verifier policy 全相同，仅 C 加 LoRA。以保存的 records 为准：

| 条件 | B geometry / diversity | C geometry / diversity | passed |
|---|---|---|---|
| night | 0.736089 / 0.226683 | 0.741835 / 0.235547 | B/C 均 false |
| snow | 0.723127 / 0.280235 | 0.735849 / 0.278121 | B/C 均 false |
| rain | 0.730700 / 0.309028 | 0.722222 / 0.305608 | B/C 均 false |
| fog | 0.782443 / 0.219359 | 0.743636 / 0.217616 | 仅 B true |

只有 B 雾景进入真实 synthetic manifest，严格继承 Bangkok / 0000649。C 的四张都
被拒绝，未改阈值或伪造合格样本。观察到雾气和可见 appearance change；night 样本
并未充分呈现请求的夜景语义。长 prompt 仍受原 SD CLIP 77-token 截断限制，两组一致，
不能声称 prompt 全部文本都参与了编码。

真实 fresh SALAD A，AMP16-mixed / 2 workers / 3 steps（**PASS，exit 0**）：

```bash
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.train_salad \
  --salad-root "$SALAD_ROOT" --gsv-root "$SMOKE_ROOT/gsv" \
  --output-dir "$SMOKE_ROOT/salad_A" --real-ratio 8 --synthetic-ratio 0 \
  --cities Bangkok --batch-size 2 --img-per-place 4 --workers 2 \
  --precision 16-mixed --seed 42 --max-steps 3 --accelerator gpu
```

实际 29.8M trainable / 58.2M frozen，global_step=scheduler_steps=3；每完整 epoch
8 real / 0 synthetic。第一步有实际非零 optimizer moments；极小数据后两步 miner 的
trivial pairs 导致 loss=0，正常记录，不能把短训练 loss 当效果证明。
从第一完整 epoch checkpoint、仅保留至该 checkpoint 的 config/journal 到独立
`salad_A_resume`，同命令加 `--resume <salad_A step1 checkpoint>` 到 step3：**PASS**。
所有最终 state_dict tensors 与不中断 A 逐位一致，scheduler/optimizer LR/AMP scaler
也一致；证据 `salad_A_resume/resume_evidence.json`。

该次恢复的实际 checkpoint 为
`salad_A/checkpoints/salad-epochepoch=000-stepstep=000001.ckpt`（之后 filename
关闭 auto_insert_metric_name，避免重复字段）。恢复命令：

```bash
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.train_salad \
  --salad-root "$SALAD_ROOT" --gsv-root "$SMOKE_ROOT/gsv" \
  --output-dir "$SMOKE_ROOT/salad_A_resume" --real-ratio 8 --synthetic-ratio 0 \
  --cities Bangkok --batch-size 2 --img-per-place 4 --workers 2 \
  --precision 16-mixed --seed 42 --max-steps 3 --accelerator gpu \
  --resume "$SMOKE_ROOT/salad_A/checkpoints/salad-epochepoch=000-stepstep=000001.ckpt"
```

另外真实官方 CPU fresh SALAD 1 step（**PASS，exit 0**，不是模型替身）：

```bash
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.tests.smoke_salad_real \
  --salad-root "$SALAD_ROOT" \
  --dino-root /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --gsv-root /home/admin123/github/WORKSPACE/Bag-of-Queries/data/train/gsv-cities \
  --output-dir /tmp/vpr_audit_salad_cpu_smoke_20261006
```

这里只读取 GSV-Cities 数据，不使用 BoQ model/loss。实测 batch `[2,2,3,224,224]`、
labels `[2,2]`、loss `1.20763636`、gradient norm `3.99913669`；aggregator 更新、
frozen prefix 无梯度，descriptor `[2,8448]` 且 normalized。

最后，在同一 4-place × 4-real 扩展数据上运行 A/B/C 各 **3 GPU steps**：

```bash
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.train_salad --salad-root "$SALAD_ROOT" --gsv-root "$SMOKE_ROOT/gsv_extended" --output-dir "$SMOKE_ROOT/salad_A_extended" --real-ratio 8 --synthetic-ratio 0 --cities Bangkok --batch-size 2 --img-per-place 4 --workers 2 --precision 16-mixed --seed 42 --max-steps 3 --accelerator gpu
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.train_salad --salad-root "$SALAD_ROOT" --gsv-root "$SMOKE_ROOT/gsv_extended" --output-dir "$SMOKE_ROOT/salad_B_extended" --real-ratio 8 --synthetic-ratio 1 --cities Bangkok --batch-size 2 --img-per-place 4 --workers 2 --precision 16-mixed --seed 42 --max-steps 3 --accelerator gpu --synthetic-manifest "$SMOKE_ROOT/synthetic_B_extra/synthetic_manifest.jsonl"
"$VPR_AUDIT_PY" -m AdaptVPR.experiments.vpr_guidance.train_salad --salad-root "$SALAD_ROOT" --gsv-root "$SMOKE_ROOT/gsv_extended" --output-dir "$SMOKE_ROOT/salad_C_extended" --real-ratio 8 --synthetic-ratio 1 --cities Bangkok --batch-size 2 --img-per-place 4 --workers 2 --precision 16-mixed --seed 42 --max-steps 3 --accelerator gpu --synthetic-manifest "$SMOKE_ROOT/synthetic_C_extra/synthetic_manifest.jsonl"
```

三个命令均 **PASS，exit 0**；checkpoints finite，global_step=scheduler_steps=3。
实际消费，而非文件数量或完整 epoch 计划：

| 组别 | 完整 epoch real:synthetic | 最后 partial epoch | 三步合计 | capacity / target | coverage_limited |
|---|---|---|---|---|---|
| A | 16:0 | 8:0 | 24:0 | 0 / 0 | false |
| B | 15:1 | 7:1 | 22:2 | 1 / 2 | true |
| C | 16:0 | 8:0 | 24:0 | 0 / 2 | true |

B 的唯一 accepted 雾景确实进入其 source place 的训练 slot，C 因无 accepted 图
安全退回 real-only exposure。两者均没有通过重复同一 synthetic 文件补足8:1。
最后 step3 checkpoint 是 partial epoch，不能 resume；step2 的完整 epoch checkpoint
可按支持范围恢复。证据 `salad_extended_evidence.json`、各组 `mix_stats.jsonl` 与
`salad_{A,B,C}_extended_stdout.log` 包含实际 command 和统计。

三组模型 hyperparameters、seed、real dataframe、batch/augmentation/AMP/workers 完全
一致。独立 A/C GPU runs 的最终 tensors 有微小差异（max abs `2.22e-4`）；同 seed
不代表 CUDA 全部运算逐位确定。前述 resume 与其自身 reference 逐位一致是另一个
已实测结论。正式效果对比仍需控制数值随机性与实际 exposure；这里不声称性能结论。

真实链路已覆盖 prepare → generator training/checkpoint reload → B/C sampling →
真实 verifier → accepted-only manifest → 含真实 accepted B 图的 mixed dataset →
fresh SALAD。C 的零 accepted 分支同样实际执行，没有假称 C 正样本分支通过。

## 5. GPU-dependent unverified items

- **UNVERIFIED**：长期默认组合 generator 训练的显存峰值、不同分辨率/批量/其他 GPU
  的稳定性。短 GPU 训练、strict reload、VPR-only 真梯度和默认三项组合已 PASS。
- **UNVERIFIED**：完整多 GPU training/all-reduce 和 distributed resume。单 GPU /
  2 workers / AMP / RandAugment 完整 epoch 恢复已实际验证；DDP padding 统计为 CPU
  sampler 回归，不冒充实际多 GPU 训练。
- **UNVERIFIED**：旧 pinned 0.27.2 栈实跑。当前环境 0.36 的真实 CPU/GPU 检查通过。
- **UNVERIFIED**：长期训练后的 C 是否提升接受率、满足每种目标 weather/time 语义。
  本次短 VPR-only C 没有合格样本，是实际观察，不应写成成功产出 C 正样本。
- **UNVERIFIED**：A/B/C held-out Recall@K / VPR performance；没有运行 benchmark，不能
  用 teacher cosine、smoke loss 或四张图判断 C>B>A。

旧 500-row cache 缺 metadata，且生成 policy 已对齐；正式实验应使用新目录重新
prepare 和生成 B/C，不应把旧结果与本次新 policy 混用。

## 6. Final pipeline

1. 从 AdaptVPR prompt JSONL 选 Global，按 condition 筛选，排除 Local/Dual。
2. 从 GSV CSV 核对 source 完整 identity，得到 ground-truth city/place_id。
3. 用原 Global negative default、frozen prompt、JPEG95 conditioning、固定 seed/policy 缓存 baseline。
4. frozen pretrained SALAD 对原 source 计算 descriptor，记录 teacher/source/preprocess 身份。
5. 训练只把 UNet attention LoRA fp32 参数交给 AdamW，base/VAE/text/teacher 全 frozen。
6. baseline latent 加 scheduler 定义的 noise，预测 epsilon，按 cumulative alpha 恢复 x0。
7. x0 → frozen VAE → RGB [0,1] → differentiable frozen SALAD → cosine VPR / L1 keep loss。
8. Pass A 求 guidance 的 x0 cotangent；Pass B 以正确 VJP + diffusion MSE 回传 LoRA。
9. 保存严格 LoRA checkpoint 及 optimizer/step/RNG；不保存可训练教师。
10. B 不加载 LoRA，C 加载 LoRA；共享生成 policy，默认不加载 SALAD，无 latent optimization。
11. 原 Global verifier 检查 geometry/diversity，异常报错，只有 passed=true 写 synthetic manifest。
12. synthetic 标签严格继承已校验 source；唯一文件只替换同 place slot。
13. epoch quota 按 8:1 target 与真实 capacity 分配；统计实际消费，包括 padding/partial epoch。
14. 重新初始化 fresh SALAD，采用官方 metric loss/miner，只由混合图像与 place labels 训练。
15. A/B/C 使用相同训练/评估条件，最终比较 held-out VPR performance，不能以 teacher cosine 替代。

## 7. Research validity check

| 问题 | 结论与边界 |
|---|---|
| teacher 完全 frozen？ | **PASS**：真实 CPU teacher requires_grad=0、参数 grads=0；启动有断言。 |
| SALAD gradient 到 generator LoRA？ | **PASS**：真实完整 IC-Light GPU VPR-only→LoRA 非零，base/teacher/VAE frozen；另有 tiny VJP 与 direct backward 相等。 |
| generator 仍产生 domain change？ | **PASS（短样本外观）**：B/C fog 有可见雾气，diversity 分数非零；目标语义质量/长训效果不能无条件答 yes，night 质量不充分。 |
| inference 不依赖 SALAD？ | **PASS**：真实 B/C 生成未加载教师、audit opt-in，无 latent optimization。 |
| verifier 只接受 Global 合格图？ | **PASS**：真实 CLIP/LightGlue，7 rejected / 1 accepted，records/manifest 一致；finite 回归通过。 |
| synthetic place label 严格正确？ | **PASS**：dataframe/source 校验、错标拒绝、canonical IDs、跨 place slot tests；现有500 source身份通过。 |
| 8:1 是实际 exposure？ | **PASS（计数正确）**：足够容量 unit epoch 实际32:4；真实GPU B完整epoch15:1、C16:0，均capacity-limited，partial也独立记录。当前真实小池没有达到8:1，不能回答无条件yes。 |
| fresh SALAD 没有 teacher loss？ | **PASS（真实模型短训练）/STATICALLY VERIFIED**：仅官方 MultiSimilarityLoss/Miner，metadata 在进入 training_step 前移除，fresh aggregator 不加载 teacher checkpoint。 |
| B/C 除 LoRA 同 generation/verifier？ | **PASS（真实8图）**：各配对 source SHA / labels / prompts / seed / sampling / actual verifier policy 完全相同。 |
| A/B/C 除数据保持训练条件？ | **PASS（配置核验和真实短训练）**：model hparams/budget/seed/real DF/batch/AMP/workers 一致；实际synthetic exposure不同且已记录，独立GPU runs不保证逐位确定，held-out performance 未测。 |

交叉核对来源：[官方 SALAD main](https://github.com/serizba/salad/blob/main/main.py)、
[VPRModel](https://github.com/serizba/salad/blob/main/vpr_model.py)、
[GSV Dataset](https://github.com/serizba/salad/blob/main/dataloaders/GSVCitiesDataset.py)、
[DINOv2 backbone](https://github.com/serizba/salad/blob/main/models/backbones/dinov2.py)。
未引入 InfoNCE/triplet/CLIP/DINO/perceptual generator loss、BoQ ensemble、Local/Dual 或额外生成模型。

## Final diff summary

修复集中在 checkpoint、verifier、标签、cache、曝光和训练控制；保留 generator 的
diffusion + cosine VPR + L1 keep 设计，保留 fresh SALAD 的官方 metric loss。
原 AdaptVPR 核心只增加 verifier 数值合法性保护，未重写其生成/匹配/阈值逻辑。
