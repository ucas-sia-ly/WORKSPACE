# Independent trained-inpainting backend and dev pilot

`generation/inpainting_editor.py` adds `TrainedInpaintingEditor` without changing
`generation/targeted_editor.py`, its public interface, its smoke runner/tests, or
the LightX2V audit. The old 4-channel masked latent-blending baseline remains
available under its original name. It is not used as a fallback here.

## Local trained checkpoint requirement

Explicitly export `INPAINTING_MODEL_PATH` before loading the new backend; it is
the only accepted model-path source. There is no default path, environment-file
auto-loading, model discovery, hub ID, download, or fallback to the old variable.
See [environment template](../configs/inpainting_editor.env.example).

The backend supports standard Diffusers `StableDiffusionXLInpaintPipeline`
(SDXL) and `StableDiffusionInpaintPipeline` (SD1.x/SD2.x) checkpoints,
with a **9-channel input / 4-channel output UNet** and
4-channel latent VAE. Nine inputs are noisy latents (4), the mask (1), and
masked-image latents (4). Model-index component types, configs and local
safetensors are checked before loading, then loaded channel counts are checked
again. Actual pipeline class and loaded UNet weight shape are also checked.
SDXL requires both text encoders/tokenizers and the expected text/time
conditioning config (`addition_embed_type=text_time`, cross-attention 2048,
projection input 2816). Base SD/SDXL, legacy 4-channel blending, arbitrary/custom pipelines,
Flux and Qwen are rejected. Only standard local components are accepted.

The user must supply weights actually trained for inpainting. The architecture
check is necessary, but cannot establish a checkpoint's training history from
its channel count alone. The checkpoint's files, including any local model card,
are hashed into the run audit. Random tiny components are used only for unit
tests and never presented as trained-model pilot evidence.

Loading hard-codes `local_files_only=True`, `use_safetensors=True` and offline hub
settings. There is no pickle or automatic-download fallback. Existing component
safety-checker configuration is preserved. SDXL retains its checkpoint scheduler
(EulerDiscreteScheduler for the supplied model); SD1/2 uses DDIM eta=0. Strength=1,
fresh per-call seeded CPU RNG, deterministic math attention and TF32 disabled.
Defaults: 20 steps, guidance 7.5, CUDA/float16 (CPU requires float32), one image
per call. SDXL float16 loading explicitly selects the local `fp16` safetensors
variant; missing variant files fail without downloading. Model configs, consumed
weight-file hashes, scheduler, library versions and pipeline source hashes are
recorded. Optional SDXL watermarking is disabled and recorded; it is not involved
in mask semantics. Tests use the installed Diffusers 0.36.0.

The user-supplied SDXL checkpoint is locally located at
`workspace/models/stable-diffusion-xl-1.0-inpainting-0.1`, with declared origin
`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`. Its model index retains a base
SDXL `_name_or_path` value, while its inpainting UNet config has nine channels
and an inpainting-specific origin. Acceptance uses structure and actual loaded
weights, not the inherited name. The local VAE config records
`madebyollin/sdxl-vae-fp16-fix`; the precise supplied component configs/bytes are
preserved in the audit, without replacing or downloading components.

```python
from generation.inpainting_editor import TrainedInpaintingConfig, TrainedInpaintingEditor
editor = TrainedInpaintingEditor(TrainedInpaintingConfig.from_env())
final = editor.edit(source_image, render_mask, prompt, seed)
raw = editor.last_raw_output
```

## Core and Render semantics

The backend accepts **Render Mask only**. Core is saved as the scientific
targeting footprint and never passed as the diffusion mask. The pilot validates
Core ⊆ Render, rechecks Core metrics against original continuous token weights,
and reconstructs Render with the frozen bounded-dilation settings to verify it.
Neither mask is resized, softened or redrawn by a VLM.

For original dimensions not divisible by 8 (including 400×300 SOURCE images),
the backend replicates the RGB edge on the right/bottom and zero-pads Render to
the next multiple of 8. No existing source or mask pixel changes position.
Padding and sampling dimensions are logged. SD1/2 uses the unchanged audited
9-channel path in the old editor; SDXL has its own instrumented path. Each actual UNet forward must receive the
Render latent mask plus masked-source latents. A dropped/changed mask fails.
SDXL checks finite denoising latents and VAE decode values. SDXL micro-conditioning
uses the padded native height/width as original and target size, with crop offset
(0,0); images are not upscaled to 1024. This preserves the candidate geometry but
does not guarantee generation quality at these smaller resolutions.

The padded raw output is preserved. Outputs are cropped to the original size,
then **every RGB pixel outside native Render is restored exactly from SOURCE**.
Core does not bound this cleanup: the Render-minus-Core allowance remains
available for edge/shadow/blending. Saving the pre-composite raw output ensures
the generation can be audited independently of final restoration. No-op/failed
generation is not accepted as evidence of realism or hardness.

## Pilot selection and execution

Default input is the existing 20-query dev scene/family candidate audit. No final
eval is inspected. No new VLM planning, vulnerability mining, retrieval scoring,
hardness filtering, rejection sampling, output ranking, or training occurs.
One query/place, one candidate, one fixed family prompt and one seed are used.
Every generated result or error is retained; failures are never replaced with
new queries or seeds.

The default `--selection-mode frozen` uses the existing frozen weighted threshold
0.3 and all geometry/precision gates. Candidate ranking matches the scientific
evaluation: max weighted coverage, max target precision, min area, min centroid
distance, then candidate ID. Query order remains the frozen dev order. Defaults
request 10 images, bounded above by 20. An insufficient cohort is reported as
`INSUFFICIENT_ELIGIBLE_QUERIES`; it never silently relaxes gates or pads repeats.

Current frozen inputs have only **2 eligible queries**, so a 10-image scientific
pilot cannot be created from them. A separately authorized rendering-only pilot
can explicitly use `--selection-mode render_semantics`: retain geometry,
connectivity, family area and precision requirements, but do not require weighted
coverage ≥0.3. There are **19** eligible queries under that explicit mode. Each
record retains all threshold failures. This option does not modify the frozen
scientific configuration or make these candidates scientifically accepted.

Prepare paired inputs without loading any model:

```bash
python scripts/stage3_pilot_inpainting.py --prepare-only --count 10
```

For an explicitly authorized separate rendering pilot and a supplied trained
checkpoint, use a fresh output directory:

```bash
export INPAINTING_MODEL_PATH=/your/local/stable-diffusion-xl-1.0-inpainting-0.1
python scripts/stage3_pilot_inpainting.py --selection-mode render_semantics \
  --count 10 --prepare-only --output outputs/stage3_dev/sdxl_inpainting_pilot
python scripts/stage3_pilot_inpainting.py \
  --run-prepared outputs/stage3_dev/sdxl_inpainting_pilot
```

Without `--prepare-only`, the CLI prepares then runs. The model path remains an
environment variable, never a CLI fallback. `--run-prepared` verifies all saved
input/plan/code hashes. A missing/invalid model produces
`BLOCKED_MODEL_CONFIGURATION`, with zero generation; that untouched prepared
plan can be run after setting the correct environment. Existing generated
evidence is refused, including partial failures. After any code/input change,
prepare a fresh output directory.

Before any pilot image is generated, `generation/inpainting_validation.py` runs
four gates against the actual local checkpoint: loaded pipeline/weights/dtype,
zero-mask identity without sampling, same-seed raw/final/latent byte determinism,
and raw-output/latent sensitivity to an exact legal translation of Render with
source/prompt/seed held fixed. All four must pass. Full controls and audits are
saved in `backend_validation/`. These involve three nonzero generation calls,
separate from the ten pilot images, and no BoQ computation. A failure stops the
pilot as `BACKEND_VALIDATION_FAILED` and retains its evidence.

## Saved evidence and tests

The run root contains `protocol.json`, `planned_edits.json`, `sampling_config.json`
(once a valid model is available) and `summary.json`. Every sample directory has:

- `source.png`, `core_mask.png`, `render_mask.png`, and `prompt.txt`;
- `raw_output_padded.png` (full sampling output), `raw_output.png` (native crop
  before restoration), `final_output.png` and `difference.png`;
- `audit.json`: family, seed, prompt, source/place identity, success or error,
  sampling config/checkpoint hashes, mask participation at every step and exact
  outside-Render RGB verification. Core metrics/thresholds remain in the plan.

`COMPLETE` means all planned samples generated and passed the pixel-restoration
contract. It is not a realism, family correctness, geometry or VPR-hardness label.
`COMPLETE_WITH_ERRORS` retains every failed attempt without hardness filtering.

```bash
python -m unittest discover -s tests -p 'test_inpainting*.py' -v
python -m unittest discover -s tests -p test_targeted_editor.py -v
```

Tests cover fail-closed model configuration, forced local loading, 4-channel
rejection, checkpoint mutation, real tiny 9-channel Diffusers sampling, Render
participation, same-seed repeatability, sensitivity of raw output to mask moves,
non-multiple-of-8 padding, exact restoration, closed family prompts, explicit
gate relaxation, preserving failures, and preventing output overwrite. Tiny
random-model and recording-spy tests do not count toward the requested dev pilot.
