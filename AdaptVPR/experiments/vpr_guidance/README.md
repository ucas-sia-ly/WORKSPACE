# VPR-aware Global-domain generation for AdaptVPR

This directory implements the AdaptVPR-style pipeline you asked for, restricted to **Global/domain changes only**:

1. Use AdaptVPR Global prompts (weather / illumination / time-of-day) with IC-Light.
2. Train a small IC-Light UNet LoRA with a **frozen pretrained SALAD teacher** so generated images retain more place-discriminative information.
3. Run the trained generator normally to build a synthetic hard-positive pool. There is **no SALAD and no latent optimization at inference time**.
4. Apply the original AdaptVPR Global verifier. Only geometry/diversity-passing images are eligible for VPR training.
5. Each accepted synthetic image inherits the parent GSV-Cities `(city, place_id)` label.
6. Mix verified synthetic images with real GSV-Cities images, default real:synthetic = **8:1**.
7. Train a **fresh SALAD** with its original DINOv2+SALAD architecture and metric-learning loss.

The pretrained SALAD in Stage 2 is only a **teacher for the generator**. The SALAD trained in Stage 4 is a new VPR model and does not reuse teacher descriptors or add a teacher loss.

This pipeline does not use AdaptVPR Local/Dual generation, LightX2V, Qwen planning, reflection control, inference-time VPR guidance, or BoQ training.

---

## 0. Environment

Run from the repository root in the existing AdaptVPR / pinned IC-Light environment:

```bash
python -m pip install -r AdaptVPR/experiments/vpr_guidance/requirements.txt

export ICLIGHT_ROOT="$PWD/IC-Light"
export ICLIGHT_BASE_MODEL_PATH="$PWD/models/stable-diffusion-v1-5"
export ICLIGHT_MODEL_PATH="$PWD/models/iclight-ckpt/iclight_sd15_fc.safetensors"
```

For training the new VPR model, clone the official SALAD repository separately, e.g.:

```bash
git clone https://github.com/serizba/salad.git external/salad
```

You also need GSV-Cities in the usual layout:

```text
dataset/gsv-cities/
  Dataframes/<City>.csv
  Images/<City>/*.jpg
```

---

## 1. Prepare generator-training targets

Do **not** train the domain generator against the original daytime image. That would encourage it to undo the requested snow/night/rain/fog appearance. Instead, cache the released IC-Light domain-shift result as the diffusion/preservation target, while the SALAD teacher descriptor comes from the original source image.

```bash
python -m AdaptVPR.experiments.vpr_guidance.prepare_data \
  --prompts /path/to/adaptcities_prompts.jsonl \
  --image-root dataset/gsv-cities/Images \
  --conditions snow night rain fog \
  --output-dir outputs/generator_train
```

Only prompt records with `route == "global"` are used.

GSV-Cities Dataframes are required as label ground truth. They default to
`<image-root>/../Dataframes`; use `--dataframe-dir` for a different layout. Every
source identity and any declared city/place label are checked against these CSVs
before models are loaded. Sample IDs must be safe, unique output basenames.
Preparation and generation require a new or empty output directory and refuse
to overwrite an existing experiment. Preparation publishes its final manifest
only after all rows have been written successfully.

Output:

```text
outputs/generator_train/
  baseline/*.png
  source_salad/*.pt
  generator_train.jsonl
```

Meaning:

- `baseline/*.png`: original released IC-Light domain-shift output;
- `source_salad/*.pt`: frozen pretrained SALAD descriptor of the source image;
- `generator_train.jsonl`: source/prompt/baseline/descriptor mapping.

Descriptor caches include the source path/content hash, actual teacher weight
hash, preprocessing version and descriptor dimension. The manifest also stores
source/baseline content hashes, dataframe labels and generation policy. Legacy
tensor-only caches must be rebuilt with `prepare_data` in a new directory.

---

## 2. Train the VPR-aware IC-Light generator

```bash
python -m AdaptVPR.experiments.vpr_guidance.train_generator \
  --manifest outputs/generator_train/generator_train.jsonl \
  --output-dir outputs/iclight_vpr_lora \
  --max-steps 1000 \
  --lambda-diff 1.0 \
  --lambda-vpr 0.1 \
  --lambda-keep 0.05
```

Only IC-Light UNet attention LoRA parameters are updated. IC-Light base weights, VAE, text encoder and SALAD teacher stay frozen.
Startup prints every trainable parameter name, trainable/total counts and the
ratio, and verifies that the optimizer owns exactly these LoRA parameters. The
targets are `to_q`, `to_k`, `to_v` and `to_out.0` in UNet attention modules; LoRA
weights and AdamW state use fp32 while the released base model uses fp16.

For cached baseline latent `z0`, training samples a low-noise timestep from the released 25-step DDIM schedule and optimizes:

```text
L_diff = MSE(epsilon_pred, epsilon)
L_vpr  = 1 - cos(SALAD(x0_pred), SALAD(source))
L_keep = L1(decode(x0_pred), released_ICLight_output)
```

`L_diff` preserves the original domain-generation behavior. `L_vpr` pushes the generator to preserve place-discriminative content. `L_keep` discourages unnecessary appearance/geometry drift.

The implementation uses a two-pass first-order gradient so the trainable UNet graph and full DINOv2-SALAD graph do not need to stay resident simultaneously.
Pass A computes the VPR/keep gradient with respect to a detached, differentiable
predicted latent. Pass B injects that cotangent using a **sum**, giving the same
first derivative as direct backpropagation for identical deterministic UNet
outputs. A CPU test compares all LoRA gradients against a single full backward.
The teacher runs in eval mode with all parameters frozen; its backbone forward
retains input gradients instead of the upstream training-only detach.

Generator checkpoints include optimizer/global-step and Python/torch/CUDA RNG
states. Use `--resume <checkpoint>` to continue with identical settings. Resuming
into an existing output requires its log to end exactly at that checkpoint; use
a new output directory when restoring an older checkpoint. Legacy weight-only
LoRA files support strict inference loading, but cannot resume training.

To resume to a total of 10000 steps with live TensorBoard curves, keep the
original training settings and add these options to the training command:

```bash
  --resume outputs/iclight_vpr_lora/checkpoints/step_003000.pt \
  --max-steps 10000 \
  --tensorboard-dir outputs/iclight_vpr_lora/tensorboard
```

```bash
tensorboard --logdir outputs/iclight_vpr_lora/tensorboard --host 127.0.0.1 --port 6006
```

Open http://127.0.0.1:6006. Scalars include the diffusion/VPR/keep losses,
their weighted contributions, total loss, SALAD cosine, gradient norms and
learning rate. `loss/guidance_proxy` is the signed first-order gradient proxy;
`loss/total` reports the weighted diffusion/VPR/keep objective. TensorBoard
history is rebuilt from the validated `train.jsonl` at startup, preserving the
global step and replacing stale events from an interrupted run. Use a dedicated
TensorBoard directory for each training run. Generation validation metrics are
computed separately by `generate_dataset`; these training curves do not measure
full-sampling quality.

Checkpoints are saved under:

```text
outputs/iclight_vpr_lora/checkpoints/
```

---

## 3. Build two synthetic pools with the **same** generation/verifier code

This is important for a fair downstream experiment. `generate_dataset.py` supports both the original released generator and the VPR-aware generator.

### B. Original AdaptVPR Global-generation control

Omit `--lora`:

```bash
python -m AdaptVPR.experiments.vpr_guidance.generate_dataset \
  --prompts /path/to/adaptcities_prompts.jsonl \
  --image-root dataset/gsv-cities/Images \
  --conditions snow night rain fog \
  --output-dir outputs/synthetic_original
```

This writes records with:

```text
generator_variant = released
```

### C. VPR-aware generator

Use the trained LoRA:

```bash
python -m AdaptVPR.experiments.vpr_guidance.generate_dataset \
  --prompts /path/to/adaptcities_prompts.jsonl \
  --image-root dataset/gsv-cities/Images \
  --lora outputs/iclight_vpr_lora/checkpoints/step_001000.pt \
  --conditions snow night rain fog \
  --output-dir outputs/synthetic_vpr_aware
```

This writes records with:

```text
generator_variant = vpr_lora
```

Both variants share the same two-stage IC-Light sampling and the original
AdaptVPR `DualTraitEvaluator` with `route="global"`. They preserve released frozen
prompts verbatim, use the original Global negative prompt by default, and match
the original HTTP generator's JPEG-95 source conditioning. An explicit input
`negative_prompt` is honored identically by both variants. DDIM, seed, CFG=7.5,
size, 25 stage-1 steps and 20 effective refinement steps are shared; refinement
strength is 0.22 for rain and 0.30 otherwise. Both stages share the same UNet.
LoRA loading requires the complete adapter key set and matching shapes, rank,
targets and base metadata; missing weights or base-model keys are errors.

Generation skips SALAD by default. `--salad-audit` explicitly enables the
optional preservation statistic. It performs no latent optimization.
The original Global thresholds remain geometry >= 0.78 and diversity >= 0.15;
non-finite/invalid verifier values abort the run. Records store sampling and
actual verifier policy, including any original matcher fallback. B/C comparisons
must use identical policies and inputs. After these policy/cache fixes, rebuild
both pools instead of comparing a newly generated C pool with an old B pool.

Each output directory contains:

```text
accepted/                  # verifier-passing images
rejected/                  # verifier-failing images
records.jsonl              # all candidates and verifier scores
synthetic_manifest.jsonl   # accepted candidates only; used by VPR training
```

Each accepted synthetic image inherits its source `(city, place_id)` label. The optional `salad_preservation_cosine` in `records.jsonl` is only an audit statistic; it is not used by the downstream VPR loss.

---

## 4. Train fresh SALAD models for the actual downstream experiment

The important experiment is not whether the teacher SALAD likes the generated images. The important experiment is whether **new VPR models trained with those images perform better**.

Use the same SALAD architecture, optimizer, loss, batch size, training steps and seed for all groups.

### A. Real-only baseline

```bash
python -m AdaptVPR.experiments.vpr_guidance.train_salad \
  --salad-root external/salad \
  --gsv-root dataset/gsv-cities \
  --output-dir outputs/salad_A_real_only \
  --real-ratio 1 \
  --synthetic-ratio 0 \
  --max-steps 4000 \
  --seed 42
```

No synthetic manifest is required when `--synthetic-ratio 0`.

### B. Real + original AdaptVPR Global synthetic

```bash
python -m AdaptVPR.experiments.vpr_guidance.train_salad \
  --salad-root external/salad \
  --gsv-root dataset/gsv-cities \
  --synthetic-manifest outputs/synthetic_original/synthetic_manifest.jsonl \
  --output-dir outputs/salad_B_original_synth \
  --real-ratio 8 \
  --synthetic-ratio 1 \
  --max-steps 4000 \
  --seed 42
```

### C. Real + VPR-aware-generator synthetic

```bash
python -m AdaptVPR.experiments.vpr_guidance.train_salad \
  --salad-root external/salad \
  --gsv-root dataset/gsv-cities \
  --synthetic-manifest outputs/synthetic_vpr_aware/synthetic_manifest.jsonl \
  --output-dir outputs/salad_C_vpr_aware_synth \
  --real-ratio 8 \
  --synthetic-ratio 1 \
  --max-steps 4000 \
  --seed 42
```

The fresh VPR model keeps the official SALAD setup:

- DINOv2 ViT-B/14 backbone;
- last four trainable backbone blocks;
- SALAD aggregator: 64 clusters, cluster dim 128, token dim 256;
- MultiSimilarityLoss;
- MultiSimilarityMiner with margin 0.1.

Defaults also follow the official `main.py`: 60 places per batch, four images per
place, 224x224 input, AdamW at 6e-5 with weight decay 9.5e-9, and a linear schedule
from 1.0 to 0.2. Places shuffle within cities by default; `--shuffle-all` changes
that for all comparison groups. ImageNet normalization and RandAugment are
retained. A local wrapper fixes the inherited scheduler's double advance so it
steps once per optimizer update; the official metric loss/miner/training step
are unchanged. Frozen backbone-prefix parameters are marked non-trainable for
DDP compatibility.

Full downstream checkpoints store optimizer, scheduler, step/epoch and RNG
state. `--resume` currently supports a completed epoch on one device, with
unchanged experiment settings. Mid-epoch or distributed resume is rejected
explicitly. `--accelerator cpu --precision 32-true` can run CPU smoke tests;
production defaults remain GPU with mixed precision.

Synthetic images receive **no special downstream loss**. They are simply same-place hard positives with the parent geographical label.

---

## 5. How the 8:1 real:synthetic exposure works

SALAD trains on places, with `K` images for each place. `mixed_salad.py` keeps this layout unchanged.

At the beginning of every epoch it computes the target number of synthetic image slots from the requested ratio. For 8:1:

```text
synthetic_fraction = 1 / 9
```

It then assigns synthetic slots only to places that have verified generated images. Allocation is deterministic for a given seed/epoch and never uses the same synthetic file twice inside one K-image place item.

Every epoch writes its actual exposure to:

```text
<output-dir>/mix_stats.jsonl
```

Important fields:

```text
target_synthetic_slots
synthetic_capacity_slots
planned_synthetic_slots
actual_synthetic_slots
actual_real_slots
actual_ratio
actual_synthetic_fraction
coverage_limited
epoch_complete
```

If `coverage_limited=true`, the accepted synthetic pool is too sparse to reach the requested 8:1 exposure without duplicating images. The code intentionally uses the lower achievable fraction instead of silently oversampling the same synthetic image.

The loader attaches slot counts which a callback removes before the official
SALAD training step. Only batches actually consumed by training count toward
the journal. DDP counters are summed across ranks and only rank zero writes the
file; sampler padding and partial epochs can make actual exposure differ from
the full dataset plan. `epoch_complete=false` marks a partial epoch. Shared epoch
state prevents stale quota plans even with persistent workers. Repeated paths
are deduplicated, conflicting labels are rejected, and only actual Boolean
`passed=true`/`eligible_for_training=true` Global records with verified source
labels enter the pool. Capacity-limited and padded runs must be reported with
their measured fraction; an 8:1 request is not a universal guarantee.

---

## 6. What to compare

The core comparison is:

```text
A. fresh SALAD trained on real GSV-Cities only
B. fresh SALAD trained on real + original Global synthetic images
C. fresh SALAD trained on real + VPR-aware Global synthetic images
```

Keep everything other than the synthetic dataset fixed.

Evaluate A/B/C on the same VPR benchmarks, especially adverse-domain queries such as night, snow/season and rain/fog where available.

The desired result is not simply:

```text
teacher SALAD(source, generated) increases
```

The desired result is:

```text
new VPR trained with C > new VPR trained with B
```

while B versus A tells you how much ordinary AdaptVPR-style Global augmentation already helps.

---

## 7. Recommended first run

Do not begin with the entire dataset. First validate the full loop with a small Global subset:

```text
1. prepare ~100-500 Global prompts
2. train generator LoRA for a short run
3. generate both original and VPR-aware pools from the same prompts/seeds
4. inspect verifier pass rate and SALAD audit distribution
5. train A/B/C with identical short SALAD budgets
6. only then scale generator data and VPR training steps
```

The decisive downstream metric is the held-out VPR benchmark performance of A/B/C, not the generator-training loss by itself.

See [AUDIT_REPORT.md](AUDIT_REPORT.md) for confirmed bugs, executed commands,
CPU evidence and the GPU-dependent checks that remain unverified.
