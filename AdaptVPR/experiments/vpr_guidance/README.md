# Phase 1: frozen VPR feedback for Global IC-Light

Phase 1 tests whether image gradients from frozen DINOv2-SALAD can steer the
released AdaptVPR Global-route IC-Light generator to retain place-recognition
information. It does not train parameters. The guidance formula, preprocessing,
DDIM timing, gradient normalization, prompts, VAE and weights remain unchanged.
There is no planner, reflection, Qwen3-VL, LightX2V, Local/Dual route or new loss.
BoQ remains evaluation-only. Phase 2 is a TODO, not part of this experiment.

Interpret three query sets against the same database:

- **Source:** the original GSV-Cities capture, saved losslessly as RGB `source.png`.
- **Baseline:** the existing explicit IC-Light sampler with guidance scale zero.
- **Guided:** that sampler with the current frozen-SALAD latent updates enabled.

The eventual hypothesis is `R_source > R_guided > R_baseline`. Report what the
metrics actually show, including cases where ordinary generation does not lose
R@1, guidance makes results worse, or guidance exceeds the source performance.

## Setup

Run commands **from the repository root**, in the existing AdaptVPR/IC-Light
Python environment. Install the experiment dependencies without replacing the
existing diffusion stack:

```bash
conda activate AdaptVPR
python -m pip install -r AdaptVPR/experiments/vpr_guidance/requirements.txt
export ICLIGHT_BASE_MODEL_PATH="$PWD/models/stable-diffusion-v1-5"
export ICLIGHT_MODEL_PATH="$PWD/models/iclight-ckpt/iclight_sd15_fc.safetensors"
export ICLIGHT_ROOT="$PWD/IC-Light"
```

The adapter loader requires CUDA and the existing downloaded checkpoints.
It constructs the released SD1.5 model, additive IC-Light FC checkpoint, source
concat hook and DDIM configuration. Use the snapshots identified in
`adapters/iclight_sd15_fc.py`. SALAD/DINOv2 Torch Hub and LPIPS pretrained AlexNet
may download weights on first use; internet access and writable caches are
needed. `--geometry` requires OpenCV SIFT. Optional equivalence `--ssim` requires
`scikit-image`; neither option silently substitutes another metric.

`--salad-repo serizba/salad:<commit>` and
`--boq-repo amaralibey/Bag-of-Queries:<commit>` can pin upstream repositories.
Local cloned Hub repository paths are also accepted. Upstream controls its
nested DINOv2 Hub load; archive the Hub cache for exact reruns. For CPU retrieval
with xFormers installed, set `XFORMERS_DISABLED=1` before starting Python.
Generation and baseline equivalence still require CUDA.

## Cheap smoke-test workflow

Use fresh output directories for Steps B/C; generation records are created
exclusively. Both default prompt files contain exactly two Global snow samples,
`adapt_000066` and `adapt_000071`. Local and Dual rows are always excluded.
The city image root below avoids indexing the entire GSV-Cities dataset.
Do not automatically run a larger experiment after these steps.

### Step A — Unit tests

```bash
python -m unittest discover -s AdaptVPR/experiments/vpr_guidance/tests -v
```

### Step B — Released vs explicit baseline equivalence

Before interpreting guidance, check that changing the sampler implementation
alone does not materially change the baseline. A difference here could otherwise
be mistaken for a VPR-guidance effect.

```bash
python -m AdaptVPR.experiments.vpr_guidance.check_baseline_equivalence \
  --conditions snow --limit 2 --seed 42 \
  --image-root dataset/gsv-cities/Images/Bangkok \
  --output-dir /tmp/vpr-equivalence-smoke --dry-run
python -m AdaptVPR.experiments.vpr_guidance.check_baseline_equivalence \
  --conditions snow --limit 2 --seed 42 \
  --image-root dataset/gsv-cities/Images/Bangkok \
  --output-dir /tmp/vpr-equivalence-smoke
```

Path A calls the **untouched existing `adapter.generate(GenerateRequest(...))`**
function directly, including its original Diffusers pipeline calls. No HTTP
service or client-side JPEG re-encoding is involved. Path B calls the current
`ICLightExperiment.generate(...)` with `Guidance(scale=0)` and no descriptor
or target. Its trace must be empty. The checker shares the loaded weights, uses
identical source pixels, prompts, negative prompt, seed and sampling settings,
and restores temporary adapter bindings after each released call. The normal
VPR-guided generator continues to use only individual model/scheduler methods.

For each sample it saves `source.png`, `released_baseline.png`,
`explicit_baseline.png`, and labeled `comparison.png` (source | released |
explicit). `records.jsonl` contains IDs, original source path, prompt/negative,
seed, output paths and metrics:

- `pixel_mae` and `pixel_rmse`: raw RGB uint8 levels, range 0–255.
- `psnr`: dB with data range 255. Exact matches use JSON `null` and
  `psnr_is_infinite=true`; the console prints `inf`, preserving valid JSON.
- `lpips`: pretrained AlexNet LPIPS v0.1 on RGB tensors in [-1,1], following the
  [official LPIPS interface](https://github.com/richzhang/PerceptualSimilarity).
- `salad_cosine_between_outputs`: cosine between normalized frozen-SALAD
  descriptors of the two stored outputs.
- Optional `ssim` with `--ssim`.

The console prints per-sample values and their means; `summary.json` stores the
means and warning count. Default diagnostics warn on SALAD cosine below **0.98**
or LPIPS above **0.10**. Override `--warn-salad-cosine`, `--warn-lpips`, or add
`--warn-pixel-mae` / `--warn-psnr`. These thresholds do **not** abort a run and
are not hard scientific acceptance criteria. Bitwise equality is not required.
If a warning occurs, the console prints both configurations, including actual
scheduler configs, full schedules, used timesteps, seed, CFG and refinement
settings. Configurations are always recorded, and configuration differences
also warn. Investigate unexpected differences before interpreting guidance.

For other released files, supply `--prompts /path/to/prompts.jsonl`, appropriate
`--conditions snow night rain fog`, and `--image-root /path/to/Images`.

### Step C — Two-image guided smoke test

```bash
python -m AdaptVPR.experiments.vpr_guidance.generate \
  --conditions snow --limit 2 --seed 42 \
  --image-root dataset/gsv-cities/Images/Bangkok \
  --output-dir /tmp/vpr-guided-smoke \
  --guidance-scale 0.003 0.01 0.03 --guidance-every 5 --guidance-last-n 10 \
  --geometry --dry-run
python -m AdaptVPR.experiments.vpr_guidance.generate \
  --conditions snow --limit 2 --seed 42 \
  --image-root dataset/gsv-cities/Images/Bangkok \
  --output-dir /tmp/vpr-guided-smoke \
  --guidance-scale 0.003 0.01 0.03 --guidance-every 5 --guidance-last-n 10 \
  --geometry
```

Each sample generates one baseline and independently restarts from the same
seed/prompt for each guided scale. Outputs include `source.png`, `baseline.png`,
`guided_<scale>.png`, labeled `comparison_<scale>.png`, cached `source_salad.pt`,
and `records.jsonl`. The records retain sampling settings, source/output SALAD
cosines, optional geometry scores and per-step loss/gradient diagnostics.
Prompts are verbatim. Default negatives come from released
`global_negative_prompt()`; optional input `negative_prompt` is honored.
Input rows need `sample_id`, `source_id`, `route`, `condition`, and `prompt`.

### Step D — Small source/baseline/guided retrieval evaluation

Once Step C produces real images, use other captures of each demo place and a
small set of distractors. This CPU-only setup snippet creates a symlink database
with every capture of the two demo places plus 32 distractor captures. Use a
fresh directory; it verifies alternate positives before evaluation.

```bash
python - <<'PY'
from pathlib import Path
from AdaptVPR.experiments.vpr_guidance.generate import ROOT, read_prompts
from AdaptVPR.experiments.vpr_guidance.gsv_pairs import image_index, parse_filename, positive_indices

images = list(image_index(ROOT / 'dataset/gsv-cities/Images/Bangkok').values())
rows = read_prompts(ROOT / 'AdaptVPR/tests/demo_10_prompts.jsonl', ['snow'], 2)
places = {parse_filename(row['source_id']).place_key for row in rows}
selected = [path for path in images if parse_filename(path).place_key in places]
selected += [path for path in images if parse_filename(path).place_key not in places][:32]
root = Path('/tmp/vpr-smoke-database')
root.mkdir(exist_ok=False)
for path in selected:
    (root / path.name).symlink_to(path)
for row in rows:
    positives = positive_indices(row['source_id'], selected)
    assert positives, f"No alternate captures for {row['sample_id']}"
    print(row['sample_id'], 'alternate positives:', len(positives))
print('Database captures:', len(selected))
PY
python -m AdaptVPR.experiments.vpr_guidance.evaluate_retrieval \
  --records /tmp/vpr-guided-smoke/records.jsonl \
  --database-root /tmp/vpr-smoke-database \
  --with-boq --output /tmp/vpr-guided-smoke/retrieval.json
```

Omit `--with-boq` for a SALAD-only infrastructure check. This 50-capture local
subset and two queries are for debugging; their recalls are too coarse to
establish general improvement. Sorted distractors are not a representative
benchmark. Later evaluations must keep a larger, fixed database including both
positives and distractors. If alternate captures are absent, only verify parsing
and exclusion; do not report retrieval success. Entirely ineligible evaluation
fails explicitly, and partial skips are listed for every query set.

## Retrieval definitions, summaries and anti-leakage

The evaluator computes source, baseline and guided descriptors against the
**same database and same alternate positives**. A place identity is
`(city, place_id)`: Bangkok place 0000002 captured in 2017 and 2020 is one place;
the same numeric ID in another city is different. Panorama IDs may contain
underscores. The exact source capture is excluded from **both ranking and
positives** for all three query sets, identified by its full filename stem,
independent of path or extension. Exact resolved query paths are excluded too,
with explicit assertions that none can be a valid positive. Thus using the
original source as a query never lets it retrieve itself.

For each model, each scale/cadence/late-step window/seed group reports an overall
aggregate and the conditions actually present (including snow/night/rain/fog
when available). Missing conditions are not invented. A condition containing
only skipped queries has zero eligible queries and null metrics. Metrics are:

- Recall@1 and Recall@5, stored as fractions and displayed as percentages.
- Median first-positive rank, 1-based; lower is better.
- Mean cosine over all eligible positive captures for each query, then averaged
  over queries. Higher is better. Stable ties use sorted database order.

Define the dataset-level generation loss as `R1_source - R1_baseline` and
guidance gain as `R1_guided - R1_baseline`. When generation loss is positive:

```text
recovery_ratio = (R1_guided - R1_baseline) / (R1_source - R1_baseline)
recovery_ratio_percent = 100 * recovery_ratio
```

For source 0.82, baseline 0.61, guided 0.73, recovery is 0.5714 = 57.14%.
When `R1_source <= R1_baseline`, both ratio values are JSON `null`, with an
explanation that generation did not reduce R@1 relative to source. Ratios are
not clamped: negative recovery indicates regression; above 100% means the gain
exceeds the measured source-to-baseline loss.

Paired query ranks use the first positive place capture. Count guided rank
**better**, **equal**, or **worse** relative to baseline. Over eligible pairs:

```text
guided_win_rate = better / num_pairs
guided_non_worse_rate = (better + equal) / num_pairs
guided_cosine_better_fraction = count(positive_cosine_guided > positive_cosine_baseline) / num_pairs
```

Cosine comparisons are strict; ties are not wins. These paired statistics can
show improvement even when a small dataset's Recall@1 does not change.

`--output retrieval.json` writes the machine-readable summary. The console
prints source/baseline/guided tables plus recovery, rank counts and win rates,
for overall and each condition. `retrieval.per_query.jsonl` is written alongside
it (override with `--per-query-output`). Each record contains sample/source IDs,
condition, settings, source/baseline/guided paths, positive database capture IDs,
and flat `salad_source_rank`, `salad_baseline_rank`, `salad_guided_rank` and
corresponding `*_positive_cosine` fields; BoQ adds matching `boq_*` fields.
Skipped records retain null ranks/cosines and a reason. This supports inspecting
individual failures without recomputing descriptors.

SALAD is the primary metric. Held-out DINOv2-BoQ (12288 dimensions) is loaded
only by the evaluator, never by guidance. If SALAD improves but BoQ strongly
degrades, guidance may be exploiting SALAD-specific features rather than
generally preserving place information. If SALAD and BoQ both improve, that is
stronger evidence of preservation of general VPR-relevant information. Inspect
weather fidelity and geometry as well. The existing `success_test` requires
strict SALAD R@1 improvement, no degradation of BoQ's four metrics, and no
decrease in mean geometry score on the eligible pairs. Missing BoQ/geometry is
insufficient evidence. This flag does not replace baseline-equivalence review.

## Unchanged sampling and guidance method

Sampling remains 25 base DDIM steps, CFG 7.5, eta 0, highres scale 1.0, then the
released PIL/8-bit round trip and stochastic VAE encoding with the same generator
stream. Strength is 0.30, or 0.22 for rain; nominal highres steps are 20. Released
truncation uses `int(20 / strength)` scheduler steps and
`int(steps * strength)` active steps, producing 19 refinement steps and 44
actual steps in total. Source dimensions round down to multiples of eight.
These settings are not CLI-tunable. `--guidance-every 5 --guidance-last-n 10`
selects steps 35 and 40 across both stages. A zero late window means all steps.

```text
x_t = detach(x_t).float().requires_grad_(True)
epsilon = frozen_UNet(x_t, t, prompt, concat_source)
x0 = DDIM.step(epsilon, t, x_t).pred_original_sample
image = clamp(frozen_VAE.decode(x0 / vae_scale) / 2 + 0.5, 0, 1)
desc = normalize(frozen_SALAD(preprocess(image)))
loss = mean(1 - dot(desc, cached_normalized_source_descriptor))
grad = autograd.grad(loss, x_t)
x_t = detach(x_t - guidance_scale * grad / RMS(grad))
epsilon = frozen_UNet(x_t, t, prompt, concat_source)
x_next = detach(DDIM.step(epsilon, t, x_t).prev_sample)
```

All model parameters have `requires_grad=False`. The graph covers only the
current selected step, with frozen UNet/VAE/SALAD forwards; previous steps are
detached. SALAD is float32; IC-Light retains float16, with a float32 gradient
leaf. Nonfinite/zero gradients fail explicitly. SALAD's upstream backbone
explicitly detaches features, so the existing experiment replaces that forward
with the same token/block/norm/reshape operations while retaining image grads.
See the [official backbone](https://github.com/serizba/salad/blob/main/models/backbones/dinov2.py)
and [evaluation transform](https://github.com/serizba/salad/blob/main/eval.py):
SALAD uses 322x322 bilinear resize and ImageNet normalization. Guidance uses
differentiable antialiased tensor resize; stored images use PIL resize before
ToTensor. Guidance omits PIL quantization. BoQ follows its
[official tensor bicubic transform](https://github.com/amaralibey/Bag-of-Queries).

Optional geometry remains SIFT homography RANSAC inliers/source keypoints:
2048 features, ratio 0.75, 3-pixel threshold, at least 8 tentative matches.
Failed verification scores zero; omitted geometry scores are null.

## Validation and remaining risks

The 16 CPU tests cover the original nine guidance checks plus equivalence
metrics, warning behavior, the untouched adapter vs explicit sampler with small
real Diffusers pipelines at both strengths, anti-leakage, recovery edge cases,
rank ties/skips, per-condition output and full JSON/table reporting. The tiny
pipeline test checks configuration/timestep agreement; it cannot validate real
IC-Light checkpoint outputs on CUDA. The [validation report](VALIDATION.md)
lists exact commands and results.

Both real snow dry runs resolve source files. The local 50-capture smoke database
has 3 and 13 alternate positives for the two demo queries. CUDA is unavailable
in this environment, so real baseline-equivalence metrics, guided images and
retrieval results have not been produced. LPIPS is also absent from the current
AdaptVPR environment until the updated requirements are installed. No empirical
VPR improvement is claimed.

Remaining uncertainty is whether full pretrained CUDA pipeline calls and the
explicit loop agree closely under this installed Diffusers version and its
floating-point behavior. Defaults may change across library versions; recorded
schedules/configurations help diagnose that. The released HTTP client performs
additional JPEG re-encoding; equivalence deliberately checks the adapter with
identical original source pixels, not that transport artifact. No guidance bug
was found or changed in this evaluation task.

## Phase 2 TODO — do not start automatically

Only after trustworthy Phase 1 evidence and held-out/geometry checks, investigate
LoRA fine-tuning of the IC-Light UNet with a **single-timestep predicted-x0 VPR
loss**. Freeze SALAD/VAE and train only LoRA parameters. This task adds no
training, new loss, hard negatives, ranking loss, VPR model or generation route.
