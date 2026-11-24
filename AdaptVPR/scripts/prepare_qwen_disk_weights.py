#!/usr/bin/env python3
"""Prepare resumable BF16 blocks; no inference, quantization, or model download."""

import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation.preflight import load_environment
from generation.qwen_disk_weights import build_plan, prepare_weights


def main():
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true", help="Read headers and report sizes; do not convert weights.")
    args = parser.parse_args()
    model = Path(os.environ["LIGHTX2V_MODEL_PATH"])
    lora = Path(os.environ["LIGHTX2V_LORA_PATH"])
    output = (args.output or Path(os.getenv(
        "LIGHTX2V_DISK_MODEL_PATH", str(model.parent / (model.name + "-Lightning-BF16-blocks"))
    ))).resolve()
    identity, tensors, _, groups, targets = build_plan(model, lora)
    sizes = {name: sum(tensors[key]["data_offsets"][1] - tensors[key]["data_offsets"][0] for key in keys)
             for name, keys in groups.items()}
    print(f"Output: {output}", flush=True)
    print(f"{identity['num_layers']} transformer blocks + non-block weights; {len(targets)} LoRA targets", flush=True)
    print(f"BF16 weights: {sum(sizes.values()) / 2**30:.2f} GiB; largest block: {max(sizes.values()) / 2**30:.2f} GiB", flush=True)
    if args.plan:
        return
    existing = sum(p.stat().st_size for p in output.glob("*.safetensors")) if output.is_dir() else 0
    parent = output
    while not parent.exists():
        parent = parent.parent
    required = max(0, sum(sizes.values()) - existing) + 2 * max(sizes.values())
    if shutil.disk_usage(parent).free < required:
        raise SystemExit(f"Insufficient disk space: need approximately {required / 2**30:.1f} GiB free")
    import torch
    from dotenv import set_key

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    prepare_weights(model, lora, output)
    set_key(str(ROOT / ".env"), "LIGHTX2V_DISK_MODEL_PATH", str(output))
    set_key(str(ROOT / ".env"), "LIGHTX2V_DISK_OFFLOAD", "1")
    print("BF16 blocks verified; .env configured for disk offload. Original weights were not modified.", flush=True)
    print("Start services: bash scripts/start_generation_services.sh", flush=True)


if __name__ == "__main__":
    main()
