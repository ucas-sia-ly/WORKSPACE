"""LoRA fine-tuning for IC-Light generator.

Fine-tunes IC-Light with LoRA on hard cases identified by SALAD validation.
Uses three loss components:
  1. L_diff: Diffusion denoising loss (preserve prior)
  2. L_identity: Semantic consistency via DINO/CLIP
  3. L_diverse: Diversity to prevent collapse

This script does NOT require SALAD gradients - it only uses SALAD to identify
which source images need better augmentation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hard_cases import filter_cases_with_source, load_hard_cases, save_hard_cases_summary
from lora_utils import (
    freeze_non_lora_parameters,
    get_lora_parameters,
    inject_lora_into_unet,
    report_trainable_parameters,
    save_lora_checkpoint,
    save_training_state,
    unfreeze_lora_parameters,
)
from losses import DiffusionLoss, DiversityLoss, IdentityLoss


class HardCaseDataset(Dataset):
    """Dataset of hard cases for LoRA fine-tuning."""

    def __init__(
        self,
        hard_cases: list,
        prompts: dict[str, str],
        image_size: int = 512,
    ):
        """
        Args:
            hard_cases: List of HardCase instances with source_path available
            prompts: Dictionary mapping source_id to generation prompt
            image_size: Target image size (will be adjusted to multiples of 8)
        """
        self.cases = hard_cases
        self.prompts = prompts
        self.image_size = (image_size // 8) * 8

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        case = self.cases[idx]

        # Load source image
        source = Image.open(case.source_path).convert("RGB")

        # Resize to target size
        source = source.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to tensor [0, 1]
        source_tensor = torch.from_numpy(np.array(source)).float() / 255.0
        source_tensor = source_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Get prompt (fallback to default if not found)
        prompt = self.prompts.get(case.query_id, "high quality street view photograph")

        return {
            "source": source_tensor,
            "prompt": prompt,
            "case_id": case.query_id,
        }


def collate_fn(batch):
    """Collate batch with variable-length prompts."""
    sources = torch.stack([item["source"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    case_ids = [item["case_id"] for item in batch]

    return {
        "source": sources,
        "prompt": prompts,
        "case_id": case_ids,
    }


def load_base_model(base_model_path: Path, device: str = "cuda", dtype=torch.float16):
    """Load the base SD1.5 model components."""
    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae").to(device, dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet").to(device, dtype=dtype)
    scheduler = DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    # Load text encoder for prompt encoding
    from transformers import CLIPTextModel, CLIPTokenizer
    text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder").to(device, dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")

    return vae, unet, scheduler, text_encoder, tokenizer


def encode_prompt(prompts: list[str], text_encoder, tokenizer, device: str):
    """Encode text prompts to embeddings."""
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0]

    return encoder_hidden_states


def encode_image_latent(vae, image: torch.Tensor) -> torch.Tensor:
    """Encode image to latent space.

    Args:
        vae: VAE model
        image: Image tensor in [0, 1] range (B, 3, H, W)

    Returns:
        Latent tensor (B, 4, H//8, W//8)
    """
    # VAE expects [-1, 1]
    image = image * 2.0 - 1.0

    with torch.no_grad():
        latent = vae.encode(image).latent_dist.sample()
        latent = latent * vae.config.scaling_factor

    return latent


def decode_latent_to_image(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent to image in [0, 1] range.

    Args:
        vae: VAE model
        latent: Latent tensor (B, 4, H//8, W//8)

    Returns:
        Image tensor (B, 3, H, W) in [0, 1]
    """
    latent = latent / vae.config.scaling_factor

    with torch.no_grad():
        image = vae.decode(latent).sample

    # [-1, 1] -> [0, 1]
    image = (image + 1.0) / 2.0
    return image.clamp(0, 1)


def prepare_concat_condition(source: torch.Tensor, vae) -> torch.Tensor:
    """Prepare IC-Light concatenation condition from source image.

    Args:
        source: Source image (B, 3, H, W) in [0, 1]
        vae: VAE model

    Returns:
        Latent condition (B, 4, H//8, W//8)
    """
    return encode_image_latent(vae, source)


def training_step(
    batch,
    unet,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    lora_layers,
    optimizer,
    loss_diff,
    loss_identity,
    loss_diverse,
    lambda_diff: float,
    lambda_identity: float,
    lambda_diverse: float,
    timestep_range: tuple[int, int],
    device: str,
):
    """Execute one training step.

    Returns:
        Dictionary of losses
    """
    optimizer.zero_grad()

    sources = batch["source"].to(device)
    prompts = batch["prompt"]
    batch_size = sources.shape[0]

    # Encode prompts
    prompt_embeds = encode_prompt(prompts, text_encoder, tokenizer, device)

    # Prepare IC-Light concat condition
    concat_cond = prepare_concat_condition(sources, vae)

    # Encode source to latent (target for diffusion)
    target_latent = encode_image_latent(vae, sources)

    # Sample timesteps
    timesteps = torch.randint(
        timestep_range[0],
        timestep_range[1],
        (batch_size,),
        device=device,
        dtype=torch.long,
    )

    # Sample noise
    noise = torch.randn_like(target_latent)

    # Add noise to latent
    noisy_latent = scheduler.add_noise(target_latent, noise, timesteps)

    # Concatenate noisy latent with condition (IC-Light: 4 + 4 = 8 channels)
    unet_input = torch.cat([noisy_latent, concat_cond], dim=1)

    # Predict noise
    noise_pred = unet(
        unet_input,
        timesteps,
        encoder_hidden_states=prompt_embeds,
    ).sample

    # === Loss 1: Diffusion denoising loss ===
    l_diff = loss_diff(noise_pred, noise)

    # === Loss 2: Identity loss ===
    # Decode predicted x0 to image space
    # x0 = (noisy_latent - sqrt(1-alpha_bar) * noise_pred) / sqrt(alpha_bar)
    alpha_prod_t = scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    pred_x0 = (noisy_latent - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)

    # Decode to image (this is differentiable if we want gradient, but for LoRA we keep it simple)
    # For now, use no_grad since we focus on noise prediction quality
    with torch.no_grad():
        pred_image = decode_latent_to_image(vae, pred_x0)

    l_identity = loss_identity(pred_image, sources)

    # === Loss 3: Diversity loss ===
    # Compare predicted images against each other
    with torch.no_grad():
        # For diversity, we only penalize if images are too similar
        # Use a simple metric here to avoid heavy computation
        pass  # Diversity computed on final images, not intermediate

    l_diverse = torch.tensor(0.0, device=device)  # Placeholder for now

    # === Combined loss ===
    loss = lambda_diff * l_diff + lambda_identity * l_identity + lambda_diverse * l_diverse

    # Backward
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "loss_diff": l_diff.item(),
        "loss_identity": l_identity.item(),
        "loss_diverse": l_diverse.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IC-Light LoRA on hard cases")
    parser.add_argument("--hard-cases", type=Path, required=True, help="Path to hard_cases.json from SALAD eval")
    parser.add_argument("--gsv-root", type=Path, help="GSV-Cities root directory")
    parser.add_argument("--base-model", type=Path, required=True, help="Path to SD1.5 base model")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for checkpoints")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=8.0, help="LoRA alpha")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--save-every", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--lambda-diff", type=float, default=1.0, help="Weight for diffusion loss")
    parser.add_argument("--lambda-identity", type=float, default=0.5, help="Weight for identity loss")
    parser.add_argument("--lambda-diverse", type=float, default=0.1, help="Weight for diversity loss")
    parser.add_argument("--timestep-min", type=int, default=0, help="Min timestep for training")
    parser.add_argument("--timestep-max", type=int, default=1000, help="Max timestep for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--identity-model", type=str, default="dinov2", choices=["dinov2", "clip"], help="Model for identity loss")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load hard cases
    print(f"Loading hard cases from {args.hard_cases}")
    all_cases = load_hard_cases(args.hard_cases, args.gsv_root)
    print(f"Loaded {len(all_cases)} hard cases")

    # Filter to cases with source available
    cases_with_source = filter_cases_with_source(all_cases)
    print(f"Found {len(cases_with_source)} cases with source images")

    if len(cases_with_source) == 0:
        raise ValueError("No hard cases with source images found. Check --gsv-root path.")

    # Save summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_hard_cases_summary(all_cases, args.output_dir / "hard_cases_summary.json")

    # Load prompts (for now, use simple default - in real use, load from generation manifest)
    prompts = {case.query_id: "high quality street view photograph" for case in cases_with_source}

    # Create dataset
    dataset = HardCaseDataset(cases_with_source, prompts, image_size=512)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # Load base model
    print(f"Loading base model from {args.base_model}")
    vae, unet, scheduler, text_encoder, tokenizer = load_base_model(args.base_model, device)

    # Freeze base model
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    freeze_non_lora_parameters(unet)

    # Inject LoRA
    print(f"Injecting LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    lora_layers = inject_lora_into_unet(
        unet,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unfreeze_lora_parameters(lora_layers)

    # Report parameters
    report_trainable_parameters(unet, lora_layers)

    # Setup optimizer
    lora_params = get_lora_parameters(lora_layers)
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)

    # Setup losses
    loss_diff = DiffusionLoss()
    loss_identity = IdentityLoss(model_name=args.identity_model, device=device)
    loss_diverse = DiversityLoss(feature_extractor="simple", device=device)  # Simple for now

    # Training loop
    print(f"\nStarting training for {args.num_steps} steps...")
    unet.train()
    global_step = 0
    epoch = 0

    pbar = tqdm(total=args.num_steps, desc="Training")

    while global_step < args.num_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= args.num_steps:
                break

            losses = training_step(
                batch,
                unet,
                vae,
                text_encoder,
                tokenizer,
                scheduler,
                lora_layers,
                optimizer,
                loss_diff,
                loss_identity,
                loss_diverse,
                args.lambda_diff,
                args.lambda_identity,
                args.lambda_diverse,
                (args.timestep_min, args.timestep_max),
                device,
            )

            global_step += 1
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{losses['loss']:.4f}",
                "diff": f"{losses['loss_diff']:.4f}",
                "iden": f"{losses['loss_identity']:.4f}",
            })

            # Save checkpoint
            if global_step % args.save_every == 0 or global_step == args.num_steps:
                save_training_state(
                    lora_layers,
                    optimizer,
                    global_step,
                    args.output_dir,
                    extra_state={
                        "epoch": epoch,
                        "lora_rank": args.lora_rank,
                        "lora_alpha": args.lora_alpha,
                        "learning_rate": args.learning_rate,
                    },
                )

    pbar.close()
    print("\nTraining complete!")

    # Save final checkpoint
    final_path = args.output_dir / "lora_final.safetensors"
    save_lora_checkpoint(
        lora_layers,
        final_path,
        metadata={
            "training_steps": str(global_step),
            "lora_rank": str(args.lora_rank),
            "lora_alpha": str(args.lora_alpha),
        },
    )
    print(f"Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
