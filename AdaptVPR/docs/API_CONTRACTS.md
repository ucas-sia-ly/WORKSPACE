# Generator service API contracts

Qwen-LightX2V is the default local HTTP generator for Global, Local and Dual
routes, and for the [staged Qwen curriculum](../experiments/qwen_curriculum/README.md).
IC-Light remains available for historical diagnosis. Clients and the service
they call must share readable filesystem paths. The FastAPI adapters are
distributed in this repository; upstream checkouts and model weights are not.

`bash scripts/start_generation_services.sh --wait` starts and checks Qwen only.
Add `--with-iclight` when an independent historical IC-Light run needs that service.

## Health checks

### IC-Light (optional historical service)

```http
GET /health
```

Expected response includes `status: "ok"`, `model_loaded: true`, and
`generator_ready: true`. A `200 OK` response from an unloaded adapter is not
treated as ready.

### LightX2V

```http
GET /health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "generator_ready": true
}
```

If `generator_ready` is omitted, it is treated as equal to `model_loaded`.

## IC-Light generation (historical)

```http
POST /generate
Content-Type: application/json
```

Request:

```json
{
  "image_path": "/shared/path/input.jpg",
  "prompt": "generation prompt",
  "negative_prompt": "negative prompt",
  "seed": 42,
  "highres_scale": 1.0,
  "highres_denoise": 0.3,
  "num_inference_steps": 20,
  "highres_steps": 20
}
```

The four high-resolution fields are optional.

For historical IC-Light LoRA inference, set `ADAPTVPR_LORA_CHECKPOINT` in `.env` or the
service environment to a checkpoint saved by
`adapters/iclight_lora.py`. The adapter reads rank and alpha from
the checkpoint metadata and applies it to the shared UNet used by both sampling
stages. An unset or empty value uses vanilla IC-Light. Relative paths in `.env`
are resolved against the AdaptVPR root by the startup script. Invalid checkpoints
fail initialization and are reported by `/health`; they do not silently fall back.
Restart the IC-Light service after changing the checkpoint. For the bundled
launcher, from the AdaptVPR directory:

```bash
export ADAPTVPR_LORA_CHECKPOINT=/absolute/path/to/lora_final.safetensors
ADAPTVPR_FORCE_RESTART=1 bash scripts/start_generation_services.sh --with-iclight --wait
```

With `--with-iclight`, the launcher restarts both requested services. This option loads the
project's custom LoRA format; external Diffusers/PEFT checkpoints need conversion.

## LightX2V generation

Global, Local and Dual routes use the same endpoint and are distinguished by
the prompt constructed by AdaptVPR. The staged curriculum calls this endpoint
directly with its frozen weather prompt and an empty negative prompt.

```http
POST /generate
Content-Type: application/json
```

Request:

```json
{
  "image_path": "/shared/path/input.jpg",
  "prompt": "generation prompt",
  "negative_prompt": "negative prompt",
  "seed": 42,
  "infer_steps": 4,
  "guidance_scale": 1.0
}
```

The released adapter fixes the public reference implementation to
`Qwen/Qwen-Image-Edit-2511` plus
`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`. It is configured
for four denoising steps and guidance scale 1.
Requests whose `infer_steps` or `guidance_scale` differ from the loaded adapter
configuration are rejected instead of silently changing the reported setup.

The adapter selects the output canvas from each source image's aspect ratio
(`canvas_policy: "source_aspect_v1"`), and passes an explicit `target_shape`
in **[height, width]** order to LightX2V. The target pixel budget is 1472×1104;
each side is rounded to a multiple of 16 within the pinned runner's 256..1664
range. A 400×300 source therefore generates at 1472×1104, rather than the
upstream default 1664×928. Nonstandard ratios have a small alignment rounding
error; ratios beyond 6.5:1 in either orientation return HTTP 422, since the
runner would otherwise clamp and change them. No adapter crop or source padding
is applied. This canvas policy does not guarantee that the model preserves all
scene contents or the camera viewpoint.

`GET /health` reports `canvas_policy` and a `canvas` object with the pixel
budget, alignment and side limits. After changing the adapter, restart only
the Qwen service to apply the code (`systemctl --user restart adaptvpr-lightx2v`).

## Generation response

Both services return:

```json
{
  "result_path": "/shared/path/generated.jpg"
}
```

`result_path` must exist and be readable by the AdaptVPR process. The route-agent
client resizes LightX2V output to the reference resolution if necessary. The
staged curriculum preserves the raw PNG and saves a separate LANCZOS-resized
training image at the source dimensions, with hashes for both artifacts.

LightX2V additionally returns `canvas_policy`, `source_dimensions` and
`raw_dimensions` in **[width, height]** order, `target_shape` in **[height,
width]** order, and `metadata_path`. The metadata JSON is saved beside the raw
PNG and includes the source path and seed. An output whose actual dimensions
do not match the requested canvas returns HTTP 500, rather than allowing a
client resize to conceal the mismatch. Existing clients can continue reading
only `result_path`.

Route-agent clients use a default request timeout of 300 seconds. Configure it
with `ICLIGHT_API_TIMEOUT` or `LIGHTX2V_API_TIMEOUT`; configure the route-agent
LightX2V retry count with `LIGHTX2V_API_RETRIES`. The staged curriculum uses
`--request-timeout` (default 600 seconds), reserves its call budget before each
POST, and makes no automatic generation retry. Its experimental structure and
weather gates are separate from the route agent's `DualTraitEvaluator` thresholds;
the curriculum records `s_div` without imposing a minimum of 0.15.
