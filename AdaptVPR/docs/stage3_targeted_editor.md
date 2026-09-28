# Independent TargetedEditor

The LightX2V audit found no native spatial mask input in its Qwen image-edit
backend. This implementation adds a separate, **local Diffusers masked editor**.
It does not change LightX2V, `run.py`, plan/prompt behavior, the dry-run reader,
VLM planning, reflection, geometry verification, VPR filtering, or training.

## Interface and configuration

```python
from generation.targeted_editor import DiffusersMaskedEditor, MaskedEditorConfig

editor = DiffusersMaskedEditor(MaskedEditorConfig.from_env())
output = editor.edit(source_image, target_mask, prompt, seed)  # PIL.Image -> PIL.Image
```

`TargetedEditor` is the abstract interface. `DiffusersMaskedEditor` implements it
using the actual `StableDiffusionInpaintPipeline` with DDIM (`eta=0`). Configure
via [targeted_editor.env.example](../configs/targeted_editor.env.example) or an
explicit `MaskedEditorConfig`. No weight directory or service endpoint is
hardcoded. This backend uses local weights, not HTTP; no endpoint is needed.
There is no automatic download and no fallback to ordinary image editing/mock.

Required environment variable: `TARGETED_EDITOR_MODEL_PATH`, an existing local
Diffusers-format Stable Diffusion directory. Optional variables:
`TARGETED_EDITOR_DEVICE` (cuda), `TARGETED_EDITOR_DTYPE` (float16 on CUDA,
float32 on CPU), `TARGETED_EDITOR_STEPS` (20), `TARGETED_EDITOR_GUIDANCE` (7.5),
and `TARGETED_EDITOR_NEGATIVE_PROMPT` (empty). The example is a template, not a
file silently loaded from `.env`. Explicitly export the settings before use.

This implementation was validated with Diffusers 0.36.0, Torch 2.8.0+cu128,
Pillow 12.3.0, and the existing local SD1.5 base weights. It supports SD pipelines
with 4- or 9-channel UNets; it is not an SDXL/Flux/Qwen universal adapter.
The production loader retains the model's configured safety checker. Flagged
outputs and unavailable/invalid backends raise errors rather than return a mock.

## Where the mask participates

Both branches run actual mask-dependent generation:

- **4-channel SD (the real smoke run):** Diffusers encodes the source image, then
  at **every denoising step** computes
  `latents = (1 - mask) * source_latents_at_next_noise_level + mask * sampled_latents`.
  This is masked latent sampling using a base SD model; it is **not** a claim
  that the SD1.5 base checkpoint is a dedicated inpainting-trained 9-channel model.
- **9-channel inpainting SD:** the actual UNet input concatenates noisy latents,
  the mask channel, and masked-source latents at **every forward call**.
  This path was tested with a tiny randomly initialized UNet/VAE; no trained
  9-channel checkpoint was used in the five-image smoke run.

The backend instruments the real pipeline instance without modifying third-party
source files. It verifies the prepared latent mask against the independently
computed nearest-coordinate input mask. For 4 channels, it numerically checks
that each step's actual latents equal the masked blending formula. For 9
channels, a UNet forward pre-hook checks the actual mask and masked-image input
channels. Missing/changed masks, incomplete traces, or mismatched operations fail.
Hooks are restored on success and failure. Each step records its latent digest
and mask-operation verification in `last_audit`.

After this genuine masked sampling, `Image.composite(raw, source, mask)` restores
exact RGB pixels outside the requested region, eliminating VAE reconstruction
leakage. This **additional cleanup is not the source of the claimed sampling
capability**. `last_raw_output` preserves the pre-composite result, and the smoke
test compares raw outputs/latent traces when moving the mask. A method that only
edits a whole image and composites afterward would fail these sampling checks.

## Input policies and reproducibility

- Source must be PIL; output is RGB at the source's decoded size. There is no
  EXIF transpose, hidden spatial resize or crop. Nonzero edits require width and
  height divisible by 8. Input objects are not mutated.
- Mask size must match source size. Accept mode `1`, mode `L` values `{0,1}`
  (BoQ export), or mode `L` values `{0,255}`. These normalize losslessly to
  `{0,255}`, with white meaning editable. Soft grayscale, RGB, and mixed 1/255
  foreground values are rejected; there is no arbitrary threshold or dilation.
- **Zero mask:** return an exact RGB source copy without loading a model or
  sampling; audit says `zero_mask_identity`, `generated=false`. This is a
  deliberate no-op, not a pretend generation result.
- A nonzero mask disappearing at VAE latent resolution is rejected. The latent
  footprint is nearest-sampled and reported; the final pixel support remains
  the original full-resolution mask.
- `load_task_pair` reads and hashes the same bytes it decodes, checks the task's
  `image_key`, source/mask hashes and dimensions, and rejects a swapped/corrupt
  source or mask. Pairing is relative to the trusted normalized target record;
  the PIL-only interface cannot infer semantic identity from an anonymous mask.
- Each call receives a fresh CPU `torch.Generator` seeded explicitly. DDIM is
  configured with eta=0; sampling uses deterministic kernels, math SDPA, TF32
  disabled and cuDNN benchmarking disabled. Global torch flags are restored
  afterward, and targeted-editor sampling calls share a lock. Arbitrary unrelated
  torch callers in the same process should not run concurrently with that scope.
- `CUBLAS_WORKSPACE_CONFIG` defaults to `:4096:8` before the loader initializes
  CUDA. If another caller already initialized CUDA, configure it before launching
  Python. Unsupported deterministic operations fail explicitly.
- Byte-repeatability is checked on the same software/device setup; identical
  bytes across different GPU models or library versions are not promised.

## Five-image smoke test

```bash
export TARGETED_EDITOR_MODEL_PATH=/your/local/diffusers-model
export TARGETED_EDITOR_STEPS=20
python scripts/run_targeted_smoke.py /path/to/targets.jsonl \
  --prompt 'A bright orange construction barrier, realistic street photograph, natural lighting.' \
  --seed 0 --check-only
python scripts/run_targeted_smoke.py /path/to/targets.jsonl \
  --prompt 'A bright orange construction barrier, realistic street photograph, natural lighting.' \
  --seed 0
```

The independent smoke entrypoint selects the first five target records and uses
seeds `base_seed + index`. Existing `run_targeted.py` remains dry-run only.
`--check-only` validates config and paired input files without importing/loading
models. Default output: `outputs/stage3_targeted/smoke/`; use a fresh `--output`
for another run. Existing evidence is never overwritten.

For each primary image it saves:

- `source.png`, original binary `mask.png`, and `mask_view.png` (0/255 display).
- `raw_output.png` (pre-composite masked sampling) and `output.png`.
- `difference.png`: max absolute RGB-channel difference from source, 0–255;
  `difference_x4.png`: display gain ×4; `raw_difference.png`: raw-output difference.
- `audit.json`: source/mask hashes, exact seed/prompt/config, sampling mechanism,
  every step's operation check and latent digest, pixel-change metrics and PNG hashes.

Additional controls reuse the first source:

1. **repeat:** identical prompt/mask/seed; raw output, final output and all latent
   digests must match.
2. **zero:** exact source copy, no sampling.
3. **moved:** translate the mask with wrap-around while holding source/prompt/seed
   fixed; the raw output and denoising latent digest must change. This is a
   diagnostic, **not** a scientific shape-matched random occlusion control.

`contact_sheet.png` shows the five primary source/mask/raw/output/difference rows.
`smoke_summary.json` records PASS/FAIL. A PASS checks actual sampling participation,
nonzero target-region changes, exact final outside-region preservation, and the
three controls. It does **not** establish realism, object correctness, geometry
preservation, VPR hardness or suitability for a final paper experiment.

## Tests

```bash
python -m unittest discover -s tests -p test_targeted_editor.py -v
python -m unittest discover -s tests -p 'test_*.py'
```

Contract spies check the PIL interface, zero policy, binary normalization,
wrong-size rejection, source/mask binding, seed and prompt plumbing, and failure
cleanup. An isolated subprocess uses real Diffusers with tiny random UNet/VAE
components to verify both 4- and 9-channel sampling, repeatability, raw sensitivity
to moved masks, and rejection of a mask dropped at the lower level. Only the text
encoder is replaced by fixed embeddings in that unit test. These synthetic
components are never used by the smoke CLI or production loader.
