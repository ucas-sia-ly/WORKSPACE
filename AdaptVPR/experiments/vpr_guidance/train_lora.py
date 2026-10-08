"""Fine-tune IC-Light LoRA by denoising verified positives mined by SALAD.

No gradient passes through SALAD. Training uses the serving adapter's actual
8-channel IC-Light UNet and source VAE mode. --init-lora starts a new round;
--resume recovers this run's optimizer, scheduler, sampler and random state.
The final serving checkpoint is published only after every requested step.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
from pathlib import Path

from common import file_sha256, read_jsonl, use_adaptvpr, write_json

use_adaptvpr()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lora_utils import (  # noqa: E402
    freeze_non_lora_parameters, get_lora_parameters, inject_lora_into_unet,
    load_lora_checkpoint, load_lora_state_dict, lora_state_dict,
    report_trainable_parameters, save_lora_checkpoint, unfreeze_lora_parameters,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, help="Final serving LoRA .safetensors")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-lora", type=Path, help="Previous round's serving LoRA")
    initialization.add_argument("--resume", type=Path, help="This run's full .training.pt state")
    parser.add_argument("--min-utility", type=float, default=0.0,
                        help="Require utility > this threshold (0 means actually mined by SALAD)")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save full recovery state every N steps, and at completion")
    parser.add_argument("--precision", choices=("auto", "bf16", "32"), default="auto")
    parser.add_argument("--check-data", action="store_true",
                        help="Validate selected files and report eligible examples without loading models")
    args = parser.parse_args(argv)
    if not args.check_data and args.output is None:
        parser.error("--output is required for training")
    for name in ("rank", "steps", "batch_size", "log_every", "save_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("alpha", "learning_rate", "grad_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.warmup_steps < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--warmup-steps and --weight-decay must be non-negative")
    if not math.isfinite(args.min_utility):
        parser.error("--min-utility must be finite")
    return args


def filter_training_rows(rows, min_utility=0.0):
    """Shared with orchestration: keep only verified rows above the utility threshold."""
    if not math.isfinite(min_utility):
        raise ValueError("min_utility must be finite")
    usable = []
    for index, row in enumerate(rows):
        if (row.get("passed") is not True or row.get("eligible_for_training", True) is not True
                or row.get("plausible", True) is not True):
            continue
        utility = row.get("utility", 0.0)
        if isinstance(utility, bool) or not isinstance(utility, (int, float)) or not math.isfinite(utility):
            raise ValueError(f"Selected row {index + 1} has invalid utility {utility!r}")
        if utility > min_utility:
            usable.append(row)
    return usable


def load_training_rows(paths, min_utility, *, allow_empty=False):
    rows = [row for path in paths for row in read_jsonl(path)]
    usable = filter_training_rows(rows, min_utility)
    if not usable:
        if allow_empty:
            return rows, usable, None
        raise SystemExit(f"No verified selected positives have utility > {min_utility}. "
                         "Skip generator fine-tuning and retain the previous generator for this round.")
    sizes, validated, seen_outputs = set(), [], {}
    source_paths = set()
    for index, row in enumerate(usable):
        for field in ("source_path", "output_path", "prompt"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Training row {index + 1} requires non-empty {field}")
        source_path, output_path = Path(row["source_path"]).resolve(), Path(row["output_path"]).resolve()
        for field, path in (("source_path", source_path), ("output_path", output_path)):
            if not path.is_file():
                raise ValueError(f"Training row {index + 1}: {field} is not a readable image file: {path}")
        if source_path == output_path or os.path.samefile(source_path, output_path):
            raise ValueError(f"Training row {index + 1}: output_path must be a generated image, not the source")
        # Canonical paths also make duplicate checks catch different symlink spellings.
        row = {**row, "source_path": str(source_path), "output_path": str(output_path)}
        if output_path in seen_outputs:
            if seen_outputs[output_path] != row:
                raise ValueError(f"Training row {index + 1}: conflicting duplicate output_path {output_path}")
            continue
        seen_outputs[output_path] = row
        source_paths.add(source_path)
        size = _valid_size_of(source_path)
        target_size = _valid_size_of(output_path)
        if target_size != size:
            raise ValueError(f"Training row {index + 1}: source size {size} differs from target {target_size}")
        sizes.add(size)
        validated.append(row)
    if set(seen_outputs) & source_paths:
        raise ValueError("Training output_path must not point to another selected row's real source image")
    if len(sizes) != 1:
        raise SystemExit(f"Mixed source resolutions {sorted(sizes)}; batching needs one size")
    return rows, validated, sizes.pop()


def _valid_size_of(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        width, height = image.size
    return max(8, width // 8 * 8), max(8, height // 8 * 8)


def to_pixels(path: Path, size) -> torch.Tensor:
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize(size)).astype("float32") / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


class CyclingBatchSampler:
    """Shuffle full epochs and refill until a batch is full, even for one example."""

    def __init__(self, size, batch_size, seed):
        if size < 1 or batch_size < 1:
            raise ValueError("Sampler size and batch size must be positive")
        self.size, self.batch_size = size, batch_size
        self.random = random.Random(seed)
        self.order = []

    def next_batch(self):
        while len(self.order) < self.batch_size:
            self.order.extend(self.random.sample(range(self.size), self.size))
        return [self.order.pop() for _ in range(self.batch_size)]

    def state_dict(self):
        return {"size": self.size, "batch_size": self.batch_size,
                "order": list(self.order), "random_state": self.random.getstate()}

    def load_state_dict(self, state):
        if (state["size"], state["batch_size"]) != (self.size, self.batch_size):
            raise ValueError("Resume sampler size/batch size differ from this run")
        if any(not isinstance(i, int) or not 0 <= i < self.size for i in state["order"]):
            raise ValueError("Resume sampler contains invalid indices")
        self.order = list(state["order"])
        self.random.setstate(state["random_state"])


@torch.no_grad()
def encode_prompts(prompts, tokenizer, text_encoder, device):
    """Match StableDiffusionPipeline's CLIP truncation and attention-mask policy."""
    tokens = tokenizer(prompts, padding="max_length", truncation=True,
                       max_length=tokenizer.model_max_length, return_tensors="pt")
    attention_mask = (tokens.attention_mask.to(device)
                      if getattr(text_encoder.config, "use_attention_mask", False) else None)
    return text_encoder(tokens.input_ids.to(device), attention_mask=attention_mask)[0]


def diffusion_target(scheduler, latent, noise, timesteps):
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "v_prediction":
        return scheduler.get_velocity(latent.float(), noise.float(), timesteps)
    raise ValueError(f"Unsupported diffusion prediction_type: {prediction_type}")


def denoising_loss(unet, scheduler, latent, condition, text, *, noise=None, timesteps=None):
    if noise is None:
        noise = torch.randn(latent.shape, device=latent.device, dtype=torch.float32)
    if timesteps is None:
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                  (latent.shape[0],), device=latent.device)
    # Compute the noise schedule in fp32; bf16 can round early alpha values to 1.
    noisy = scheduler.add_noise(latent.float(), noise.float(), timesteps).to(latent.dtype)
    target = diffusion_target(scheduler, latent, noise, timesteps)
    prediction = unet(noisy, timesteps, encoder_hidden_states=text,
                      cross_attention_kwargs={"concat_conds": condition}).sample
    return F.mse_loss(prediction.float(), target.float())


def make_lr_scheduler(optimizer, args):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps))
        * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps))))


def training_state_path(output):
    return Path(output).with_suffix(".training.pt")


@torch.no_grad()
def lora_delta_norm(layers):
    # ||BA||_F^2 = sum((B^T B) * (A A^T)); materialize only rank x rank matrices.
    squares = [((layer.lora_B.float().T @ layer.lora_B.float())
                * (layer.lora_A.float() @ layer.lora_A.float().T)).sum() * layer.scaling ** 2
               for layer in layers.values()]
    return math.sqrt(max(0.0, float(torch.stack(squares).sum())))


def save_training_state(path, layers, optimizer, schedule, sampler, config, step, history, running):
    first = next(iter(layers.values()))
    numpy_rng = np.random.get_state()
    payload = {
        "format_version": 2, "config": config, "step": step,
        "lora": lora_state_dict(layers),
        "lora_metadata": {"lora_rank": str(first.rank), "lora_alpha": str(first.alpha),
                          "num_layers": str(len(layers))},
        "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
        "sampler": sampler.state_dict(), "history": history, "running": running,
        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "numpy_rng": {"algorithm": numpy_rng[0], "state": numpy_rng[1].tolist(),
                      "position": int(numpy_rng[2]), "has_gauss": int(numpy_rng[3]),
                      "cached_gaussian": float(numpy_rng[4])},
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_state(path, layers, optimizer, schedule, sampler, config):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError("Legacy or unsafe LoRA training state is unsupported; restart this training run") from exc
    if state.get("format_version") != 2:
        raise ValueError("Unsupported LoRA training checkpoint format; restart this training run")
    if state.get("config") != config:
        old = state.get("config", {})
        changed = sorted(key for key in set(old) | set(config) if old.get(key) != config.get(key))
        raise ValueError(f"Resume configuration differs: {', '.join(changed)}")
    if not isinstance(state.get("step"), int) or not 0 <= state["step"] <= config["steps"]:
        raise ValueError("Invalid LoRA resume step")
    load_lora_state_dict(layers, state["lora"], state["lora_metadata"])
    optimizer.load_state_dict(state["optimizer"])
    schedule.load_state_dict(state["schedule"])
    sampler.load_state_dict(state["sampler"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    numpy_rng = state["numpy_rng"]
    np.random.set_state((numpy_rng["algorithm"], np.asarray(numpy_rng["state"], dtype=np.uint32),
                         numpy_rng["position"], numpy_rng["has_gauss"], numpy_rng["cached_gaussian"]))
    if state["cuda_rng"]:
        if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA RNG devices differ from this run")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"], list(state["history"]), list(state["running"])


def train_steps(layers, optimizer, schedule, sampler, loss_for_indices, args,
                *, start_step=0, history=None, running=None, save_callback=None):
    """Run optimizer steps; a saved boundary contains every state needed for recovery."""
    params = get_lora_parameters(layers)
    history, running = list(history or []), list(running or [])
    for step in range(start_step, args.steps):
        loss = loss_for_indices(sampler.next_batch())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        running.append(loss.item())
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            update = lora_delta_norm(layers)
            entry = {"step": step + 1, "loss": float(np.mean(running)), "grad_norm": float(grad_norm),
                     "lr": schedule.get_last_lr()[0], "delta_w_fro": update}
            history.append(entry)
            running = []
            print(f"[lora] step {entry['step']}: loss={entry['loss']:.4f} "
                  f"grad={entry['grad_norm']:.3e} |dW|={update:.3e}", flush=True)
        if save_callback and ((step + 1) % args.save_every == 0 or step + 1 == args.steps):
            save_callback(step + 1, history, running)
    return history


def main(argv=None):
    args = parse_args(argv)
    rows, usable, size = load_training_rows(args.selected, args.min_utility, allow_empty=args.check_data)
    if args.check_data:
        import json

        print(json.dumps({"selected_rows": len(rows), "training_examples": len(usable),
                          "min_utility": args.min_utility, "resolution": size}))
        return
    width, height = size
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("--precision bf16 requires CUDA bfloat16 support")
    base_dtype = (torch.bfloat16 if args.precision != "32" and torch.cuda.is_bf16_supported()
                  else torch.float32)
    device = "cuda"
    print(f"[lora] {len(usable)}/{len(rows)} selected positives used, size {width}x{height}")

    import adapters.iclight_sd15_fc as adapter
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    base = adapter._required_path("ICLIGHT_BASE_MODEL_PATH")
    offset = adapter._required_path("ICLIGHT_MODEL_PATH")
    if offset.name != adapter.CHECKPOINT_FILENAME:
        raise SystemExit(f"ICLIGHT_MODEL_PATH must select {adapter.CHECKPOINT_FILENAME}")
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae").to(device, base_dtype).eval().requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder").to(device, base_dtype)
    text_encoder.eval().requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
    adapter._configure_unet(unet, offset)
    unet.to(device, base_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    if noise_scheduler.config.prediction_type not in {"epsilon", "v_prediction"}:
        raise SystemExit(f"Unsupported prediction_type: {noise_scheduler.config.prediction_type}")

    freeze_non_lora_parameters(unet)
    layers = inject_lora_into_unet(unet, rank=args.rank, alpha=args.alpha, dtype=torch.float32)
    if args.init_lora:
        load_lora_checkpoint(layers, args.init_lora)
    unfreeze_lora_parameters(layers)
    report_trainable_parameters(unet, layers)
    unet.enable_gradient_checkpointing()
    unet.train()

    optimizer = torch.optim.AdamW(get_lora_parameters(layers), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    schedule = make_lr_scheduler(optimizer, args)
    sampler = CyclingBatchSampler(len(usable), args.batch_size, args.seed)
    selected_sha256 = {str(path.resolve()): file_sha256(path) for path in args.selected}
    config = {name: getattr(args, name) for name in (
        "rank", "alpha", "steps", "batch_size", "learning_rate", "weight_decay", "grad_clip",
        "warmup_steps", "seed", "log_every", "min_utility")}
    config.update({"selected_sha256": selected_sha256, "resolution": [width, height],
                   "base_model": str(base), "iclight_offset_sha256": file_sha256(offset),
                   "base_dtype": str(base_dtype),
                   # Diffusers' _use_default_values list can have a different order per process.
                   "noise_scheduler": {key: value for key, value in noise_scheduler.config.items()
                                       if not key.startswith("_")},
                   "lora_layers": list(layers)})
    start_step, history, running = 0, [], []
    if args.resume:
        start_step, history, running = load_training_state(
            args.resume, layers, optimizer, schedule, sampler, config)
        print(f"[lora] resumed completed step {start_step}/{args.steps}", flush=True)

    @torch.no_grad()
    def encode_batch(batch):
        source = torch.stack([to_pixels(Path(row["source_path"]), size) for row in batch]).to(device, base_dtype)
        target = torch.stack([to_pixels(Path(row["output_path"]), size) for row in batch]).to(device, base_dtype)
        condition = vae.encode(source).latent_dist.mode() * vae.config.scaling_factor
        latent = vae.encode(target).latent_dist.sample() * vae.config.scaling_factor
        text = encode_prompts([row["prompt"] for row in batch], tokenizer, text_encoder, device)
        return latent, condition, text

    def loss_for_indices(indices):
        return denoising_loss(unet, noise_scheduler, *encode_batch([usable[index] for index in indices]))

    def save_callback(step, log, pending):
        save_training_state(training_state_path(args.output), layers, optimizer, schedule,
                            sampler, config, step, log, pending)

    history = train_steps(layers, optimizer, schedule, sampler, loss_for_indices, args,
                          start_step=start_step, history=history, running=running, save_callback=save_callback)
    if start_step == args.steps:
        # A completed recovery state can republish missing final artifacts to a new output path.
        save_callback(args.steps, history, running)
    mined_examples = sum(row.get("utility", 0.0) > 0 for row in usable)
    save_lora_checkpoint(layers, args.output, metadata={
        "base_model": str(base), "iclight_offset": offset.name,
        "iclight_offset_sha256": config["iclight_offset_sha256"],
        "training_examples": len(usable), "steps": args.steps,
        "selection_min_utility": args.min_utility, "mined_training_examples": mined_examples,
        "prediction_type": noise_scheduler.config.prediction_type,
        "objective": "iclight_denoising_on_salad_mined_verified_positives",
    })
    write_json(args.output.with_suffix(".json"), {
        "args": {key: str(value) if isinstance(value, Path) else
                 ([str(x) for x in value] if isinstance(value, list) else value)
                 for key, value in vars(args).items()},
        "selected_sha256": selected_sha256, "training_examples": len(usable),
        "selected_rows": len(rows), "resolution": [width, height], "history": history,
        "mined_training_examples": mined_examples,
        "completed_steps": args.steps, "resumed_from_step": start_step,
        "training_state": str(training_state_path(args.output)),
        "prediction_type": noise_scheduler.config.prediction_type,
    })
    print(f"[lora] saved {args.output}")


if __name__ == "__main__":
    main()
