# VPR-aware Global-domain generation for AdaptVPR

This directory implements the intended AdaptVPR-style training loop for **Global/domain changes only**:

1. Start from AdaptVPR Global prompts (weather / illumination / time-of-day) and the released IC-Light generator.
2. Train a small IC-Light UNet LoRA with a **frozen pretrained SALAD teacher**. The generated image is encouraged to keep the source place descriptor while the original diffusion objective and a baseline-image preservation term keep the requested domain edit.
3. Run the trained generator normally, with **no SALAD and no inference-time latent guidance**, to create a large synthetic hard-positive pool.
4. Apply the original AdaptVPR Global verifier. Only candidates passing the geometry/diversity thresholds are eligible for VPR training.
5. Every accepted synthetic image inherits its parent GSV-Cities `(city, place_id)` label.
6. Train a **new SALAD** with the original SALAD architecture/loss on real GSV-Cities plus verified synthetic images. The default exposure ratio is real:synthetic = **8:1**, matching AdaptVPR's reported best setting.

There is no Local/Dual route, LightX2V, Qwen planner, reflection controller, inference-time SALAD guidance, or BoQ training in this pipeline.

## Environment

Run from the repository root in the existing AdaptVPR / pinned IC-Light environment:

```bash
python -m pip install -r AdaptVPR/experiments/vpr_guidance/requirements.txt
export ICLIGHT_ROOT="$PWD/IC-Light"
export ICLIGHT_BASE_MODEL_PATH="$PWD/models/stable-diffusion-v1-5"
export ICLIGHT_MODEL_PATH="$PWD/models/iclight-ckpt/iclight_sd15_fc.safetensors"
```

You also need a local checkout of the official `serizba/salad` repository when training the new VPR model.

## Stage 1 — cache the released generator targets

The generator is not trained to copy the original daytime source directly. That would encourage it to remove the requested snow/night/rain/fog appearance. Instead we first cache the released IC-Light output as a pseudo-target and cache the source SALAD descriptor.

```bash
python -m AdaptVPR.experiments.vpr_guidance.prepare_data \
  --prompts /path/to/adaptcities_prompts.jsonl \
  --image-root dataset/gsv-cities/Images \
  --conditions snow night rain fog \
  --output-dir outputs/generator_train
```

Output:

```text
outputs/generator_train/
  baseline/*.png
  source_salad/*.pt
  generator_train.jsonl
```

Only prompt rows with `route == "global"` are used.

## Stage 2 — train the VPR-aware domain generator

```bash
python -m AdaptVPR.experiments.vpr_guidance.train_generator \
  --manifest outputs/generator_train/generator_train.jsonl \
  --output-dir outputs/iclight_vpr_lora \
  --max-steps 1000 \
  --lambda-diff 1.0 \
  --lambda-vpr 0.1 \
  --lambda-keep 0.05
```

Only UNet attention LoRA parameters are updated. IC-Light base weights, VAE, text encoder and the SALAD teacher remain frozen.

For a baseline target latent `z0`, a low-noise released-DDIM timestep is sampled. The training objective contains:

```text
L_diff = MSE(epsilon_pred, epsilon)
L_vpr  = 1 - cos(SALAD(x0_pred), SALAD(source))
L_keep = L1(decode(x0_pred), released_ICLight_output)
```

The implementation uses a two-pass first-order gradient so the UNet graph and full DINOv2-SALAD graph do not need to remain in memory simultaneously. `L_diff` preserves IC-Light's original domain-generation behavior; `L_vpr` preserves place-discriminative content; `L_keep` discourages unnecessary visual drift.

Checkpoints are written to `outputs/iclight_vpr_lora/checkpoints/`.

## Stage 3 — generate and verify the synthetic VPR pool

Use a trained LoRA checkpoint to run the **normal released IC-Light sampling path**. SALAD is not part of inference; it is optional only as an audit score.

```bash
python -m AdaptVPR.experiments.vpr_guidance.generate_dataset \
  --prompts /path/to/adaptcities_prompts.jsonl \
  --image-root dataset/gsv-cities/Images \
  --lora outputs/iclight_vpr_lora/checkpoints/step_001000.pt \
  --conditions snow night rain fog \
  --output-dir outputs/vpr_synthetic
```

The original AdaptVPR `DualTraitEvaluator` is called with `route="global"`. The repository's Global thresholds are therefore used unchanged. Generated images are written to `accepted/` or `rejected/`.

Important outputs:

```text
outputs/vpr_synthetic/records.jsonl             # all generated candidates
outputs/vpr_synthetic/synthetic_manifest.jsonl  # accepted candidates only
```

Each accepted row contains the source ID/path, generated path, condition, prompt, `s_geo`, `s_div`, and inherited `city` / `place_id`. Only `passed == true` and `eligible_for_training == true` images enter the VPR training pool.

## Stage 4 — train a new SALAD from real + synthetic data

Clone the official SALAD repository separately, then run:

```bash
python -m AdaptVPR.experiments.vpr_guidance.train_salad \
  --salad-root /path/to/serizba-salad \
  --gsv-root dataset/gsv-cities \
  --synthetic-manifest outputs/vpr_synthetic/synthetic_manifest.jsonl \
  --output-dir outputs/salad_real8_synth1 \
  --real-ratio 8 \
  --synthetic-ratio 1 \
  --max-steps 4000
```

The new VPR model keeps SALAD's original setup:

- DINOv2 ViT-B/14 backbone
- last four trainable backbone blocks
- SALAD aggregator (64 clusters, cluster dim 128, token dim 256)
- MultiSimilarityLoss
- MultiSimilarityMiner, margin 0.1

Synthetic images are not given a special loss. They are simply extra same-place hard positives and inherit the parent geographical label, as in AdaptVPR. The model still sees GSV-Cities places in the normal `[batch_places, images_per_place, C, H, W]` format.

The default real:synthetic exposure is 8:1. `mixed_salad.py` samples synthetic images only from verified candidates belonging to the same `(city, place_id)` as the current place; places without verified generated images remain fully real.

## What to compare

The central downstream experiment is not the teacher SALAD score. Train fresh VPR models under equal optimization budgets and compare them on the same VPR benchmarks:

```text
A. real GSV-Cities only
B. real + original AdaptVPR Global synthetic images
C. real + VPR-aware-generator synthetic images   <-- ours
```

Keep the VPR architecture, loss, steps, batch size and real:synthetic ratio identical. The important question is whether C improves the newly trained SALAD, especially under night/snow/rain/fog queries.

The pretrained SALAD used during generator training is a **teacher only**. The SALAD trained in Stage 4 is a new model and must not reuse teacher descriptors or teacher weights as an additional objective.
