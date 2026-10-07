"""Test script for LoRA fine-tuning implementation.

Runs sanity checks on:
- Hard case loading
- Loss computation
- LoRA injection
- Training step
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def test_hard_cases():
    """Test hard case loading."""
    print("\n=== Testing Hard Case Loading ===")

    from hard_cases import HardCase, save_hard_cases_summary

    # Create dummy hard cases
    cases = [
        HardCase(
            query_id=f"test_{i}",
            query_path=f"/tmp/query_{i}.jpg",
            source_path=f"/tmp/source_{i}.jpg" if i % 2 == 0 else None,
            correct_match_id=f"correct_{i}",
            wrong_match_id=f"wrong_{i}",
            retrieval_rank=i + 5,
            distance_to_wrong=0.1 + i * 0.05,
            distance_to_correct=0.4 + i * 0.05,
        )
        for i in range(10)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "summary.json"
        save_hard_cases_summary(cases, output_path)
        assert output_path.exists()
        print(f"✓ Hard case summary saved to {output_path}")

        # Check content
        import json
        data = json.loads(output_path.read_text())
        assert data["total_cases"] == 10
        assert data["cases_with_source"] == 5
        print(f"✓ Summary contains {data['total_cases']} cases, {data['cases_with_source']} with source")

    print("✓ Hard case loading test passed")


def test_lora_utils():
    """Test LoRA utilities."""
    print("\n=== Testing LoRA Utils ===")

    from lora_utils import (
        LoRALayer,
        freeze_non_lora_parameters,
        get_lora_parameters,
        inject_lora_into_unet,
        save_lora_checkpoint,
        unfreeze_lora_parameters,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test LoRALayer
    print("\n[1/4] Testing LoRALayer...")
    lora = LoRALayer(in_features=512, out_features=512, rank=8, alpha=8.0)
    x = torch.randn(2, 10, 512)
    y = lora(x)
    assert y.shape == (2, 10, 512)
    print(f"  LoRA input shape: {x.shape}, output shape: {y.shape}")
    print("  ✓ LoRALayer forward pass works")

    # Test injection into a simple UNet-like module
    print("\n[2/4] Testing LoRA injection...")

    class SimpleUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = torch.nn.Linear(512, 512)
            self.to_k = torch.nn.Linear(512, 512)
            self.to_v = torch.nn.Linear(512, 512)
            self.to_out = torch.nn.Sequential(
                torch.nn.Linear(512, 512),
            )

    unet = SimpleUNet().to(device)
    original_param_count = sum(p.numel() for p in unet.parameters())

    lora_layers = inject_lora_into_unet(unet, rank=8, alpha=8.0)

    print(f"  Injected LoRA into {len(lora_layers)} layers")
    assert len(lora_layers) == 4  # to_q, to_k, to_v, to_out.0
    print("  ✓ LoRA injection works")

    # Test freezing/unfreezing
    print("\n[3/4] Testing freeze/unfreeze...")
    freeze_non_lora_parameters(unet)
    trainable_before = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    assert trainable_before == 0  # All params frozen

    unfreeze_lora_parameters(lora_layers)
    lora_params = get_lora_parameters(lora_layers)
    trainable_after = sum(p.numel() for p in lora_params)

    print(f"  Original params: {original_param_count:,}")
    print(f"  LoRA params: {trainable_after:,}")
    print(f"  LoRA percentage: {100 * trainable_after / original_param_count:.2f}%")
    assert trainable_after > 0
    print("  ✓ Freeze/unfreeze works")

    # Test save/load
    print("\n[4/4] Testing checkpoint save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_lora.safetensors"
        save_lora_checkpoint(lora_layers, checkpoint_path)
        assert checkpoint_path.exists()
        print(f"  ✓ Checkpoint saved to {checkpoint_path}")

        # Create new LoRA layers and load
        new_unet = SimpleUNet().to(device)
        new_lora_layers = inject_lora_into_unet(new_unet, rank=8, alpha=8.0)

        from lora_utils import load_lora_checkpoint
        load_lora_checkpoint(new_lora_layers, checkpoint_path)
        print("  ✓ Checkpoint loaded successfully")

    print("\n✓ All LoRA utils tests passed")


def test_training_step_smoke():
    """Check a minimal diffusion step updates only the injected LoRA weights."""
    print("\n=== Testing Training Step (Smoke Test) ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Creating minimal components...")

    # Create minimal fake models for smoke test
    from types import SimpleNamespace

    from diffusers import DDIMScheduler

    class FakeVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(scaling_factor=0.18215)
            self.projection = torch.nn.Conv2d(3, 4, 1)

        def encode(self, x):
            # RGB slicing cannot create four channels; project explicitly.
            x = torch.nn.functional.interpolate(
                x, scale_factor=0.125, mode="bilinear", align_corners=False
            )
            latent = self.projection(x)
            return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latent))

    class FakeUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = torch.nn.Conv2d(8, 16, 3, padding=1)
            # Match the Linear attention targets supported by LoRA injection.
            self.to_q = torch.nn.Linear(16, 16)
            self.to_k = torch.nn.Linear(16, 16)
            self.to_v = torch.nn.Linear(16, 16)
            self.to_out = torch.nn.Sequential(torch.nn.Linear(16, 16))
            self.conv_out = torch.nn.Conv2d(16, 4, 3, padding=1)

        def forward(self, x, timestep, encoder_hidden_states):
            features = self.conv_in(x)
            batch_size, channels, height, width = features.shape
            tokens = features.flatten(2).transpose(1, 2)
            attended = torch.nn.functional.scaled_dot_product_attention(
                self.to_q(tokens), self.to_k(tokens), self.to_v(tokens)
            )
            tokens = tokens + self.to_out(attended)
            features = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
            return SimpleNamespace(sample=self.conv_out(features))

    vae = FakeVAE().to(device)
    unet = FakeUNet().to(device)
    scheduler = DDIMScheduler(num_train_timesteps=1000)

    # Inject LoRA
    from lora_utils import inject_lora_into_unet, get_lora_parameters, freeze_non_lora_parameters, unfreeze_lora_parameters
    vae.requires_grad_(False)
    freeze_non_lora_parameters(unet)
    lora_layers = inject_lora_into_unet(unet, rank=4, alpha=4.0)
    assert set(lora_layers) == {"to_q", "to_k", "to_v", "to_out.0"}, \
        "Expected LoRA injection into all four Linear attention projections"
    unfreeze_lora_parameters(lora_layers)

    # Setup optimizer
    lora_params = get_lora_parameters(lora_layers)
    assert lora_params, "LoRA injection produced no optimizer parameters"
    lora_before = [param.detach().clone() for param in lora_params]
    base_before = {name: param.detach().clone() for name, param in unet.named_parameters()}
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

    # Create fake batch
    batch_size = 2
    sources = torch.rand(batch_size, 3, 64, 64, device=device)

    print("Running one forward/backward pass...")

    optimizer.zero_grad()

    # Use the fake VAE to obtain a genuine four-channel latent at 1/8 resolution.
    with torch.no_grad():
        latent = vae.encode(sources * 2.0 - 1.0).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    assert latent.shape == (batch_size, 4, 8, 8)
    noise = torch.randn_like(latent)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    noisy = scheduler.add_noise(latent, noise, timesteps)

    concat = latent
    unet_input = torch.cat([noisy, concat], dim=1)
    assert unet_input.shape == (batch_size, 8, 8, 8)

    fake_text_embeds = torch.randn(batch_size, 77, 768, device=device)
    noise_pred = unet(unet_input, timesteps, fake_text_embeds).sample
    assert noise_pred.shape == noise.shape

    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    assert torch.isfinite(loss).item(), "Diffusion loss must be finite"

    print(f"Loss: {loss.item():.4f}")

    loss.backward()
    assert all(param.grad is not None for param in lora_params), \
        "All injected LoRA parameters must participate in backpropagation"
    assert all(torch.isfinite(param.grad).all().item() for param in lora_params), \
        "LoRA gradients must be finite"
    # A can have zero gradient on step one because B starts at zero.
    assert any(torch.count_nonzero(param.grad).item() > 0 for param in lora_params), \
        "At least one LoRA gradient must be nonzero"
    optimizer.step()

    assert any(not torch.equal(before, after.detach())
               for before, after in zip(lora_before, lora_params)), \
        "The optimizer must update LoRA weights"
    # LoRA layers are registered children, so exclude them from the base check.
    lora_ids = {id(param) for param in lora_params}
    assert all(param.grad is None and torch.equal(base_before[name], param.detach())
               for name, param in unet.named_parameters() if id(param) not in lora_ids), \
        "Frozen base UNet parameters must remain unchanged"
    assert all(param.grad is None for param in vae.parameters()), \
        "The frozen VAE must not receive gradients"

    print("✓ Training step smoke test passed")


def main():
    """Run all tests."""
    print("="*60)
    print("Running LoRA Fine-tuning Implementation Tests")
    print("="*60)

    try:
        test_hard_cases()
        test_lora_utils()
        test_training_step_smoke()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
