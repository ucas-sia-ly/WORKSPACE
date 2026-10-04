# Phase 1: frozen VPR feedback for Global IC-Light

This isolated experiment tests whether SALAD image gradients can preserve place
information during released AdaptVPR Global-route generation. No existing
AdaptVPR files are changed. There is no HTTP request, pipeline `__call__`, planner,
reflection, Qwen3-VL, LightX2V, Local/Dual route, optimizer, or parameter training.

Run modules **from the repository root**, in the existing AdaptVPR/IC-Light
Python environment. Install the small additional dependency list:

```bash
python -m pip install -r AdaptVPR/experiments/vpr_guidance/requirements.txt
export ICLIGHT_BASE_MODEL_PATH="$PWD/models/stable-diffusion-v1-5"
export ICLIGHT_MODEL_PATH="$PWD/models/iclight-ckpt/iclight_sd15_fc.safetensors"
export ICLIGHT_ROOT="$PWD/IC-Light"
```

The adapter loader requires CUDA and reads the existing downloaded weights.
It constructs the exact released SD1.5 pipelines, additive IC-Light FC UNet,
concat-conditioning hook and DDIM configuration; this experiment only calls
individual model, prompt encoding, latent preparation and scheduler methods.
Use the released model snapshots identified in the adapter. Torch Hub downloads
SALAD/DINOv2 code and pretrained weights on first use; internet access and a
writable Torch Hub cache are required. `--salad-repo serizba/salad:<commit>` and
`--boq-repo amaralibey/Bag-of-Queries:<commit>` can pin upstream code. Their refs,
model revisions and runtime versions are recorded. A local cloned Hub repository
path is also accepted for offline loading. The nested DINOv2 Hub load is
controlled by upstream SALAD/BoQ; archive the Torch Hub cache for exact reruns.

## Two-sample snow debug run

```bash
python -m AdaptVPR.experiments.vpr_guidance.generate \
  --conditions snow --output-dir /tmp/vpr-snow --dry-run
python -m AdaptVPR.experiments.vpr_guidance.generate \
  --conditions snow --output-dir /tmp/vpr-snow \
  --seed 42 --guidance-scale 0.01 0.03 0.1 --guidance-every 5 --geometry
```

Defaults select only `adapt_000066` and `adapt_000071` from
`AdaptVPR/tests/demo_10_prompts.jsonl`; Local/Dual rows are always excluded.
Without `--conditions`, all Global rows in the input JSONL are used.
`--prompts /path/to/released.jsonl --image-root /path/to/Images` supports arbitrary
released prompt files with `sample_id`, `source_id`, `route`, `condition`, `prompt`
and optional `negative_prompt`. Prompts are used verbatim. Default negatives
come from the released Global-route `global_negative_prompt()`; sampling follows
`adapters/iclight_sd15_fc.py`. Conditions are exact case-insensitive names,
e.g. `--conditions snow night rain fog`. `--limit 2` limits debug workload.

Each sample generates one baseline, then regenerates from the **same seed and
prompt** independently for every scale. Baseline and guided runs use 25 DDIM
text-to-image steps, CFG 7.5, eta 0, then the released PIL/8-bit image round trip
and stochastic VAE encoding plus DDIM img2img refinement with the same generator
stream. Refinement strength is 0.30 (0.22 for rain), highres scale 1.0, and nominal
highres steps 20. Preserve the released truncation: `int(20 / strength)` scheduler
steps and `int(steps * strength)` active steps gives 19 refinement steps, hence
44 actual steps in total. Resolution follows the source rounded down to multiples
of eight. These sampling settings are deliberately not CLI-tunable.

`--guidance-every 5` selects actual steps 5, 10, ... across both stages.
`--guidance-last-n 10` restricts guidance to the final 10 of the 44 actual steps
(steps 35 and 40 with the default cadence). Zero means unrestricted. A zero
scale is a useful baseline-equivalence control. Scale is the RMS magnitude of
an update to the diffusion latent, rather than classifier-free prompt guidance:

```text
x_t = detach(x_t).float().requires_grad_(True)
epsilon = frozen_UNet(x_t, t, prompt, concat_source)
x0 = DDIM.step(epsilon, t, x_t).pred_original_sample
image = clamp(frozen_VAE.decode(x0 / vae_scale) / 2 + 0.5, 0, 1)
desc = normalize(frozen_SALAD(preprocess(image)))
loss = mean(1 - dot(desc, cached_normalized_source_descriptor))
grad = autograd.grad(loss, x_t)
x_t = detach(x_t - guidance_scale * grad / RMS(grad))
epsilon = frozen_UNet(x_t, t, prompt, concat_source)  # recomputed
x_next = detach(DDIM.step(epsilon, t, x_t).prev_sample)
```

All model parameters have `requires_grad=False`. Selected steps retain the
current UNet, VAE and SALAD graph; previous steps and the inter-stage round trip
are detached. SALAD runs in float32; IC-Light retains the released float16 model
stack, with a float32 latent leaf for gradient calculation. Nonfinite/zero
gradients fail explicitly. No model parameter receives a gradient.

**SALAD's upstream backbone detaches its input features even in eval mode.**
`vpr.py` replaces only that backbone forward with the identical token, block,
norm and reshape computation, removing `no_grad` and `detach`. Simply freezing
parameters or changing `num_trainable_blocks` would not restore input gradients.
The [official SALAD backbone](https://github.com/serizba/salad/blob/main/models/backbones/dinov2.py)
and [evaluation transform](https://github.com/serizba/salad/blob/main/eval.py)
are the references: 322x322 bilinear resize and ImageNet mean/std. Generated
predictions use differentiable antialiased tensor resize; stored-image evaluation
and source targets use the official PIL-resize-before-ToTensor transform.
The unavoidable PIL quantization is omitted during differentiable guidance.

Outputs include `source.png`, `baseline.png`, `guided_<scale>.png`, labeled
three-image `comparison_<scale>.png`, and cached `source_salad.pt`. `records.jsonl`
contains IDs, condition, seed, scale/cadence, source-to-baseline/guided SALAD
cosines, optional geometry scores, paths, sampling configuration and per-step
loss/gradient diagnostics. Output records are created exclusively: use a fresh
output directory for each run. Source similarity is a diagnostic; it is not the
retrieval success metric. `--geometry` requires OpenCV SIFT and measures
homography RANSAC inliers divided by source keypoints (2048 features, ratio 0.75,
3-pixel threshold, at least 8 tentative matches). This independent diagnostic
scores failed verification as zero; without it scores are null.

## Retrieval against other captures

```bash
python -m AdaptVPR.experiments.vpr_guidance.evaluate_retrieval \
  --records /tmp/vpr-snow/records.jsonl \
  --database-root dataset/gsv-cities/Images/Bangkok \
  --with-boq --output /tmp/vpr-snow/retrieval.json
```

Choose a database with **alternate captures and distractors**, and keep it fixed
between comparisons. Database filenames are parsed by `gsv_pairs.py`. Positives
share `(city, place_id)`; e.g. Bangkok place 0000002 captured in 2017 and 2020.
Panorama IDs may contain underscores. The exact source capture (entire filename
stem, independent of path/extension) is excluded from **both** ranking and
positives for each query. Places in different cities are distinct. Queries with
no alternate capture are reported and skipped identically for both variants;
an entirely ineligible evaluation fails. The two snow samples are a debugging
set, not enough evidence of general improvement.

The evaluator reports Recall@1/5 as fractions, median first-positive rank
(1-based), and mean cosine over all eligible positive captures per query,
then averaged over queries. Runs are grouped by scale, cadence, late-step
window and seed. Stable ranking breaks cosine ties by sorted database order.
Descriptors are computed one image at a time and stored on CPU during the run.
For CPU evaluation in environments with xFormers installed, set
`XFORMERS_DISABLED=1` before starting Python so DINOv2 uses native attention.

SALAD is the primary metric. `--with-boq` loads **DINOv2-BoQ** (12288 dimensions)
only inside the evaluator. It uses the [official BoQ transform](https://github.com/amaralibey/Bag-of-Queries):
tensor bicubic antialiased 322x322 resize and ImageNet normalization. BoQ never
participates in guidance. Full `success_test=pass` requires strictly improved
SALAD R@1, no degradation in any of BoQ's four retrieval metrics, and no decrease
in mean geometry score on the same eligible queries. Missing BoQ or geometry
produces `insufficient_evidence`; inspect the images and individual geometry
scores as well. Improvements confined to SALAD may indicate model-specific
optimization. Weather/prompt fidelity should also be inspected in comparisons.

## Checks and current validation limits

```bash
python -m unittest discover -s AdaptVPR/experiments/vpr_guidance/tests -v
```

Nine CPU checks exercise real DDIM scheduler steps with small frozen networks,
nonzero input gradients, loss reduction, zero-scale equality, graph detachment,
parameter freezing, both generation stages with real small Diffusers UNet/VAE
components and helper methods, SALAD backbone-forward value equivalence, filename parsing,
source-capture exclusion, and success gating. At implementation time the local
AdaptVPR environment has torch 2.8.0+cu128 and diffusers 0.36.0; CUDA is
unavailable. The snow input dry run resolves both source files (each place also
has alternate captures in the Bangkok database). A separate CPU smoke check in
the local `countermine-vpr` environment loaded cached pretrained DINOv2-SALAD
through Torch Hub from an existing local SALAD checkout. A darkened snow source
produced an 8448-dimensional descriptor, cosine loss 0.001430 and finite nonzero
image-gradient RMS 8.54e-5; all parameters remained frozen. This verifies the
real SALAD image gradient, not generation quality. The full snow run reaches the
explicit CUDA-unavailable error. Real IC-Light snow generation and retrieval
results remain unverified until CUDA works and SALAD's extra dependencies are
installed in the AdaptVPR environment. No empirical improvement is claimed.

## Phase 2 TODO — do not start automatically

Only after Phase 1 demonstrates steering and passes the held-out/geometry
checks: investigate LoRA fine-tuning of the IC-Light UNet using a **single-timestep
predicted-x0 VPR loss**. Freeze SALAD and VAE, retain image gradients, and train
only LoRA parameters. This directory intentionally contains no fine-tuning code.
