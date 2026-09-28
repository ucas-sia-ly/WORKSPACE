"""Independent masked editing contract; no planner, LightX2V or verifier imports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import inspect
import math
import os
from pathlib import Path, PurePosixPath
import threading

from PIL import Image


_SAMPLING_LOCK = threading.Lock()


@contextmanager
def _deterministic_sampling():
    """Scope process-global torch flags and deterministic math attention."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with _SAMPLING_LOCK:
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        benchmark, cudnn_det = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
        matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with sdpa_kernel(SDPBackend.MATH):
                yield
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, cudnn_det
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul_tf32, cudnn_tf32


class TargetedEditor(ABC):
    @abstractmethod
    def edit(self, source_image: Image.Image, target_mask: Image.Image,
             prompt: str, seed: int) -> Image.Image:
        """Edit white/selected pixels; nonzero masks MUST enter masked sampling."""


def pixel_sha256(image: Image.Image) -> str:
    header = f"{image.mode}:{image.width}:{image.height}:".encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Accept mode 1, L {0,1}, or L {0,255}; reject soft/mixed/RGB masks.

    Return L {0,255} for Diffusers, explicitly lifting BoQ's 0/1 PNG foreground.
    No resizing, thresholding, dilation, EXIF transpose, or mutation of input.
    """
    if not isinstance(mask, Image.Image) or mask.mode not in ("1", "L"):
        raise ValueError("target_mask must be a single-channel PIL image (1 or L)")
    if mask.size != size:
        raise ValueError("target_mask size must equal source_image size")
    mask = mask.convert("L")
    values = {i for i, count in enumerate(mask.histogram()) if count}
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError("non-binary mask: accept {0,1} or {0,255}; no implicit thresholding")
    return mask.point([0] + [255] * 255)


def load_task_pair(task: dict) -> tuple[Image.Image, Image.Image]:
    """Revalidate the source/mask binding of a normalized targeted task."""
    for field in ("sample_id", "image_key", "source_path", "mask_original_path", "source_sha256",
                  "mask_original_sha256", "source_width", "source_height"):
        if field not in task:
            raise ValueError(f"task missing {field}")
    key = PurePosixPath(task["image_key"])
    if key.is_absolute() or ".." in key.parts or not key.parts:
        raise ValueError("invalid task image_key")
    source_path = Path(task["source_path"])
    if source_path.parts[-len(key.parts):] != key.parts:
        raise ValueError("source/image_key pairing mismatch")
    images = []
    for path_key, hash_key in (("source_path", "source_sha256"), ("mask_original_path", "mask_original_sha256")):
        path = Path(task[path_key])
        # Decode the same bytes that were hashed, avoiding a path re-open race.
        import io
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != task[hash_key]:
            raise ValueError(f"source/mask pairing: {hash_key} mismatch for {task['sample_id']}")
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            images.append(image.copy())
    source, mask = images
    if source.size != (task["source_width"], task["source_height"]):
        raise ValueError("task source dimensions mismatch")
    canonical_mask(mask, source.size)
    return source.convert("RGB"), mask


@dataclass(frozen=True)
class MaskedEditorConfig:
    model_path: str
    device: str = "cuda"
    dtype: str = "float16"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    local_files_only: bool = True

    def __post_init__(self):
        if not self.model_path or not isinstance(self.model_path, str):
            raise ValueError("model_path must be explicitly configured")
        if self.device not in ("cpu", "cuda") and not self.device.startswith("cuda:"):
            raise ValueError("device must be cpu or cuda[:index]")
        if self.dtype not in ("float16", "float32") or (self.device == "cpu" and self.dtype != "float32"):
            raise ValueError("dtype must be float16/float32; CPU requires float32")
        if type(self.num_inference_steps) is not int or self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 1:
            raise ValueError("guidance_scale must be finite and >= 1")

    @classmethod
    def from_env(cls):
        path = os.getenv("TARGETED_EDITOR_MODEL_PATH", "").strip()
        if not path:
            raise ValueError("Set TARGETED_EDITOR_MODEL_PATH; no model path is hardcoded")
        device = os.getenv("TARGETED_EDITOR_DEVICE", "cuda")
        return cls(
            model_path=path, device=device,
            dtype=os.getenv("TARGETED_EDITOR_DTYPE", "float32" if device == "cpu" else "float16"),
            num_inference_steps=int(os.getenv("TARGETED_EDITOR_STEPS", "20")),
            guidance_scale=float(os.getenv("TARGETED_EDITOR_GUIDANCE", "7.5")),
            negative_prompt=os.getenv("TARGETED_EDITOR_NEGATIVE_PROMPT", ""),
        )


class DiffusersMaskedEditor(TargetedEditor):
    """StableDiffusionInpaintPipeline with audited mask use inside every step.

    4-channel SD: mask blends source/noisy source latents at every step.
    9-channel inpainting SD: mask and masked-image latents enter every UNet call.
    Outside-mask RGB pixels are restored AFTER true masked sampling to remove VAE
    leakage. Raw output and per-step checks are retained separately for audit.
    This class does not support arbitrary pipelines, remote endpoints or mocks.
    """

    def __init__(self, config: MaskedEditorConfig):
        self.config = config
        self._pipe = None
        self._lock = threading.Lock()
        self.last_audit = None
        self.last_raw_output = None

    def _load_pipeline(self):
        if self._pipe is None:
            # Must precede CUDA/cuBLAS initialization in this process.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            import torch
            from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
            model_path = Path(self.config.model_path).expanduser().resolve()
            if not model_path.is_dir():
                raise ValueError(f"TARGETED_EDITOR_MODEL_PATH must be a local Diffusers directory: {model_path}")
            if self.config.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA device is unavailable")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(model_path), torch_dtype=getattr(torch, self.config.dtype),
                local_files_only=self.config.local_files_only,
            )
            # DDIM eta=0 and one per-call generator give explicit deterministic plumbing.
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(self.config.device)
            pipe.set_progress_bar_config(disable=True)
            self._pipe = pipe
        return self._pipe

    def edit(self, source_image, target_mask, prompt, seed) -> Image.Image:
        with self._lock:
            self.last_audit = None
            self.last_raw_output = None
            if not isinstance(source_image, Image.Image):
                raise ValueError("source_image must be a PIL image")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a nonempty string")
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("seed must be an integer in [0, 2**63)")
            source = source_image.convert("RGB")
            mask = canonical_mask(target_mask, source.size)
            area = mask.histogram()[255]
            audit = dict(
                schema_version=1, backend="diffusers_stable_diffusion_inpaint", seed=seed,
                prompt=prompt, source_pixel_sha256=pixel_sha256(source), mask_pixel_sha256=pixel_sha256(mask),
                source_size=list(source.size), mask_pixels=area, mask_policy="strict_binary_0_1_or_0_255",
                config=asdict(self.config), final_unmasked_rgb_restoration=True, sampling_steps=[],
            )
            if area == 0:
                audit.update(status="zero_mask_identity", generated=False, mask_participated_in_sampling=False)
                self.last_raw_output = source.copy()
                self.last_audit = audit
                return source.copy()
            # No hidden resizing; smoke inputs (640x480) already satisfy this.
            if source.width % 8 or source.height % 8:
                raise ValueError("nonzero-mask source dimensions must be multiples of 8; no implicit resize")
            pipe = self._load_pipeline()
            audit["determinism"] = "DDIM eta=0; CPU per-call RNG; deterministic kernels; math SDPA; TF32 off"
            with _deterministic_sampling():
                raw = self._sample(pipe, source, mask, prompt, seed, audit)
            if not isinstance(raw, Image.Image) or raw.size != source.size:
                raise RuntimeError("Masked pipeline returned an invalid image or wrong dimensions")
            raw = raw.convert("RGB")
            output = Image.composite(raw, source, mask)
            audit.update(status="edited", generated=True, mask_participated_in_sampling=True,
                         raw_output_pixel_sha256=pixel_sha256(raw), output_pixel_sha256=pixel_sha256(output))
            self.last_raw_output = raw.copy()
            self.last_audit = audit
            return output

    def _sample(self, pipe, source, mask_image, prompt, seed, audit):
        import torch
        import diffusers
        from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
        if not isinstance(pipe, StableDiffusionInpaintPipeline) or not isinstance(pipe.scheduler, DDIMScheduler):
            raise TypeError("Require the actual StableDiffusionInpaintPipeline with DDIMScheduler")
        channels = pipe.unet.config.in_channels
        if channels not in (4, 9):
            raise ValueError("masked backend requires a 4- or 9-channel SD UNet")
        audit.update(
            unet_in_channels=channels,
            sampling_mechanism="per_step_latent_mask_blend" if channels == 4 else "mask_and_masked_image_unet_channels",
            diffusers_version=diffusers.__version__, torch_version=torch.__version__,
            pipeline_source_sha256=file_sha256(inspect.getfile(StableDiffusionInpaintPipeline)),
            scheduler_source_sha256=file_sha256(inspect.getfile(DDIMScheduler)),
        )
        latent_h, latent_w = source.height // pipe.vae_scale_factor, source.width // pipe.vae_scale_factor
        # Independently compute the exact floor-coordinate nearest latent mask.
        full = torch.tensor(bytearray(mask_image.tobytes()), dtype=torch.float32).reshape(source.height, source.width) / 255
        expected = full[(torch.arange(latent_h) * source.height // latent_h)[:, None],
                        (torch.arange(latent_w) * source.width // latent_w)[None, :]]
        if not expected.any():
            raise ValueError("nonzero target disappears at latent resolution; refusing ineffective mask")
        trace = {}
        prepare_latents = pipe.prepare_latents
        prepare_masks = pipe.prepare_mask_latents
        scheduler_step = pipe.scheduler.step

        def capture_latents(*args, **kwargs):
            result = prepare_latents(*args, **kwargs)
            if channels == 4:
                trace["noise"], trace["image_latents"] = result[1], result[2]
            return result

        def capture_masks(*args, **kwargs):
            actual_mask, masked_latents = prepare_masks(*args, **kwargs)
            wanted = expected.to(actual_mask.device, actual_mask.dtype)[None, None].expand_as(actual_mask)
            if not torch.equal(actual_mask, wanted):
                raise RuntimeError("Mask reaching sampling differs from the paired input's nearest latent mask")
            trace["mask"], trace["masked_image_latents"] = actual_mask, masked_latents
            trace["mask_verified"] = True
            return actual_mask, masked_latents

        def capture_step(*args, **kwargs):
            result = scheduler_step(*args, **kwargs)
            trace["raw_step"] = result[0].clone()
            return result

        def verify_unet(module, args):
            if channels == 9:
                sample = args[0]
                if not (torch.equal(sample[:, 4:5], trace["mask"]) and
                        torch.equal(sample[:, 5:9], trace["masked_image_latents"])):
                    raise RuntimeError("UNet did not receive mask + paired masked-image latents")
            trace["unet_calls"] = trace.get("unet_calls", 0) + 1

        def step_end(pipeline, index, timestep, values):
            actual_mask = values["mask"]
            if not trace.get("mask_verified") or not torch.equal(actual_mask, trace["mask"]):
                raise RuntimeError("Sampling mask was dropped or changed")
            if trace.get("unet_calls") != index + 1:
                raise RuntimeError("Expected exactly one audited UNet call per step")
            if channels == 4:
                init = trace["image_latents"]
                if index < len(pipe.scheduler.timesteps) - 1:
                    next_t = pipe.scheduler.timesteps[index + 1]
                    init = pipe.scheduler.add_noise(init, trace["noise"], torch.tensor([next_t]))
                init_mask = actual_mask[:1]
                wanted = (1 - init_mask) * init + init_mask * trace["raw_step"]
                if not torch.equal(values["latents"], wanted):
                    raise RuntimeError("Mask did not control the actual per-step latent blend")
            audit["sampling_steps"].append(dict(
                step=index, timestep=int(timestep), latent_mask_pixels=int(actual_mask[0].sum().item()),
                latent_mask_shape=list(actual_mask.shape[-2:]), mask_operation_verified=True,
                latent_sha256=hashlib.sha256(values["latents"].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            ))
            return values

        # Instrument only this pipeline instance; restore even if generation fails.
        hook = pipe.unet.register_forward_pre_hook(verify_unet)
        pipe.prepare_latents, pipe.prepare_mask_latents = capture_latents, capture_masks
        pipe.scheduler.step = capture_step
        try:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt, image=source.copy(), mask_image=mask_image.copy(),
                    width=source.width, height=source.height, strength=1.0,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale, negative_prompt=self.config.negative_prompt,
                    generator=generator, eta=0.0, output_type="pil",
                    callback_on_step_end=step_end, callback_on_step_end_tensor_inputs=["latents", "mask"],
                )
            if len(audit["sampling_steps"]) != self.config.num_inference_steps:
                raise RuntimeError("Incomplete mask sampling trace")
            if any(getattr(result, "nsfw_content_detected", None) or []):
                raise RuntimeError("Pipeline safety checker flagged the output; not accepting a replacement image")
            if len(result.images) != 1:
                raise RuntimeError("Expected one generated image")
            return result.images[0]
        finally:
            pipe.prepare_latents, pipe.prepare_mask_latents = prepare_latents, prepare_masks
            pipe.scheduler.step = scheduler_step
            hook.remove()
