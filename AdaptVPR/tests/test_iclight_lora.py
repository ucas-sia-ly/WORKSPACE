"""Inference/checkpoint compatibility after relocating the historical IC helper."""

import tempfile
import unittest
from pathlib import Path

import torch

from adapters.iclight_lora import (
    get_lora_parameters, inject_lora_into_unet, load_lora_checkpoint,
    load_lora_state_dict, lora_state_dict, read_lora_metadata, save_lora_checkpoint,
)


def attention():
    return torch.nn.ModuleDict({"to_q": torch.nn.Linear(4, 4), "to_k": torch.nn.Linear(4, 4)})


class IcLightLoraCompatibilityTests(unittest.TestCase):
    def test_existing_checkpoint_format_round_trip_preserves_inference(self):
        torch.manual_seed(7)
        source = attention()
        target = attention()
        target.load_state_dict(source.state_dict())
        source_layers = inject_lora_into_unet(source, rank=2, alpha=4)
        with torch.no_grad():
            for parameter in get_lora_parameters(source_layers):
                parameter.normal_()
        inputs = torch.randn(3, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "historical.safetensors"
            save_lora_checkpoint(source_layers, checkpoint)
            metadata = read_lora_metadata(checkpoint)
            self.assertEqual((metadata["lora_rank"], metadata["lora_alpha"]), ("2", "4"))
            target_layers = inject_lora_into_unet(target, rank=2, alpha=4)
            load_lora_checkpoint(target_layers, checkpoint)
        self.assertTrue(torch.equal(source["to_q"](inputs), target["to_q"](inputs)))

    def test_invalid_checkpoint_load_keeps_existing_inference_weights(self):
        layers = inject_lora_into_unet(attention(), rank=2, alpha=4)
        before = lora_state_dict(layers)
        invalid = {key: value.clone() for key, value in before.items()}
        invalid["to_k.lora_B"] = torch.full((4, 2), float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_lora_state_dict(layers, invalid, {"lora_rank": "2", "lora_alpha": "4", "num_layers": "2"})
        self.assertTrue(all(torch.equal(value, lora_state_dict(layers)[key]) for key, value in before.items()))

    def test_mixed_dtype_inference_preserves_base_output_dtype(self):
        model = attention().to(torch.bfloat16)
        inject_lora_into_unet(model, rank=2, alpha=4, dtype=torch.float32)
        output = model["to_q"](torch.randn(3, 4, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_double_adapter_injection_rejected(self):
        model = attention()
        inject_lora_into_unet(model, rank=2)
        with self.assertRaisesRegex(RuntimeError, "already injected"):
            inject_lora_into_unet(model, rank=2)


if __name__ == "__main__":
    unittest.main()
