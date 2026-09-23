# External components

AdaptVPR publishes the orchestration, routing, prompt construction, reflection,
and verification policy. Third-party implementations and model weights are not
vendored in this repository.

| Role | Component used by AdaptVPR | Integration | Public default |
|---|---|---|---|
| Planner and prompt refiner | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | OpenAI-compatible HTTP API | model alias `qwen3-vl-4b-instruct-remote`, API base `http://127.0.0.1:23002/v1` |
| Global generator | [IC-Light](https://github.com/lllyasviel/IC-Light) | Included HTTP adapter | `http://127.0.0.1:8002/generate` |
| Local/Dual generator | `Qwen/Qwen-Image-Edit-2511` + `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`, served by [LightX2V](https://github.com/ModelTC/LightX2V) | Included HTTP adapter | `http://127.0.0.1:8001/generate` |
| Appearance verifier | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | Loaded in the AdaptVPR process with Transformers | `openai/clip-vit-base-patch32` |
| Geometry verifier | [VisMatch (SuperPoint + LightGlue)](https://github.com/gmberton/vismatch) | Imported as a local Python package | `superpoint-lightglue` |

The Qwen model value is the **served model alias**, not necessarily the model's
download path. Set `ADAPTVPR_PLANNER_MODEL` to the exact model ID returned by
your OpenAI-compatible server.

## Expected installation layout

One convenient local layout is:

```text
workspace/
├── AdaptVPR/
├── IC-Light/
├── LightX2V/
└── vismatch/
```

Paths and endpoints are configured through `.env`; no personal absolute paths
should be committed. Copy the public template first:

```bash
cp configs/default.env.example .env
```

Use the single `workspace/LightX2V/` directory for the pinned revision and its
local compatibility patches. `scripts/prepare_demo_environment.sh` installs
from this directory and creates it only when missing; it does not create an
additional worktree. For this layout, set `LIGHTX2V_ROOT=../LightX2V` in `.env`.
When copying the source without its Git metadata, retain
`LightX2V/.adaptvpr-source-revision` so the adapter can validate its base revision.

## Planner

Serve Qwen3-VL-4B-Instruct through an OpenAI-compatible endpoint, then
configure:

```env
ADAPTVPR_PLANNER_API_KEY=local-placeholder
ADAPTVPR_PLANNER_API_BASE=http://127.0.0.1:23002/v1
ADAPTVPR_PLANNER_MODEL=qwen3-vl-4b-instruct-remote
```

The planner and the prompt refiner share this endpoint. The server implementation
and Qwen weights are not included.

## Generators

AdaptVPR talks to IC-Light and LightX2V through the small HTTP adapters under
`adapters/`. The upstream projects do not expose this API by default. Clone the
pinned upstream revisions, download the named checkpoints, set their paths in
`.env`, and start both included adapters with
`scripts/start_generation_services.sh`.

The IC-Light adapter pins GitHub commit
`bcf3f29ca85be8a4686215f477b546f5030be8b7`, checkpoint
`lllyasviel/ic-light@9cad1878695f546a7fb9eaca14e2a89131ba5ffe` file
`iclight_sd15_fc.safetensors`, and Stable Diffusion v1.5 revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
The adapters accept either a Git checkout at the pinned commit or the official
GitHub commit tarball with its commit written to `.adaptvpr-source-revision` in
the extracted source root. LightX2V reports tracked source modifications in
`/health` so compatibility patches remain visible rather than being mistaken
for an unmodified upstream checkout.

### Reproducible Qwen-LightX2V adapter

“Qwen-LightX2V” in this release means the following concrete image-to-image
stack, rather than a standalone checkpoint of that name:

- base model: `Qwen/Qwen-Image-Edit-2511`;
- acceleration LoRA repository: `lightx2v/Qwen-Image-Edit-2511-Lightning`;
- LoRA file: `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`;
- LightX2V model class/task: `qwen-image-edit-2511` / `i2i`;
- sampling: 4 inference steps, guidance scale 1.0, adaptive resize, PyTorch SDPA.
- LightX2V GitHub commit: `522609ecc121b49c20d201b3f00c3dc052821bce`.

Download both checkpoints, install LightX2V and the adapter dependencies, then
run the adapter shipped in this repository:

```bash
pip install -v -e /path/to/LightX2V
pip install -r adapters/requirements.txt
export ICLIGHT_ROOT=/path/to/IC-Light
export ICLIGHT_BASE_MODEL_PATH=/models/stable-diffusion-v1-5
export ICLIGHT_MODEL_PATH=/models/iclight_sd15_fc.safetensors
export LIGHTX2V_ROOT=/path/to/LightX2V
export LIGHTX2V_MODEL_PATH=/models/Qwen-Image-Edit-2511
export LIGHTX2V_LORA_PATH=/models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
scripts/start_generation_services.sh
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8001/health
```

The adapter exposes the exact `/health` and `/generate` contract consumed by
`generation/lightx2v.py`. Model loading errors remain visible through `/health`
instead of allowing the orchestration layer to mistake an unloaded service for
a ready generator.

`scripts/start_generation_services.sh` launches the two adapters from this
repository while adding each configured upstream checkout to its Python import
path. It does not expect or execute a third-party `app.py`. If services are
managed outside AdaptVPR, set `ICLIGHT_AUTO_START=0` and
`LIGHTX2V_AUTO_START=0` and provide only their URLs.

### BF16 generation with limited host RAM

The Qwen transformer alone contains approximately 38 GiB of BF16 weights.
CPU block offload ordinarily keeps the entire transformer in host memory, so
it is insufficient on a machine with 32 GB RAM. Disk offload keeps two block
buffers in host memory and two on the GPU, reading the next block from disk.

Prepare the existing base model and Lightning LoRA once:

```bash
python scripts/prepare_qwen_disk_weights.py --plan
python scripts/prepare_qwen_disk_weights.py
bash scripts/start_generation_services.sh --wait
```

Preparation preserves BF16 and uses the pinned upstream LoRA merge operation
order, one block at a time. It writes approximately 38 GiB to a separate
directory (allow at least 40 GiB free space), keeps the original checkpoints,
and verifies each output with SHA-256. Rerunning resumes completed blocks and
rebuilds missing or corrupt ones. A complete manifest is required at startup.
The preparation script updates `.env` only after validation:

```env
LIGHTX2V_DISK_OFFLOAD=1
LIGHTX2V_DISK_MODEL_PATH=/path/to/Qwen-Image-Edit-2511-Lightning-BF16-blocks
```

The adapter uses the original model directory for the text encoder, tokenizer,
and VAE, and applies no additional LoRA at startup because it is already merged
into the blocks. The sampling settings remain 4 steps and guidance scale 1.0.
The local LightX2V compatibility patch streams text encoder shards directly to
the target device using `low_cpu_mem_usage=True` and an explicit `device_map`.
This avoids retaining the complete encoder state in host RAM before copying it
to the GPU; its weights remain BF16. The adapter enables this only for disk
offload. `/health` reports that the upstream checkout is modified.
The text encoder still needs about 15.5 GiB of GPU memory, in addition to block
buffers, activations and other services. Disk reads can slow inference; complete
generation must be validated on the target hardware.

Small numerical and CUDA buffer checks (no full model loading):

```bash
python tests/check_qwen_disk_weights.py
```

## Verifiers

CLIP and the geometry matcher run in the AdaptVPR process rather than through
network ports. CLIP is loaded by Transformers. The geometry package must expose:

```python
from vismatch import get_matcher
matcher = get_matcher(name, device=device, max_num_keypoints=n_kpts)
```

The returned matcher must provide `load_image(...)` and return `matched_kpts0`
and `num_inliers` when called on a pair of images.
