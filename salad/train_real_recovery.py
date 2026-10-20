#!/usr/bin/env python3
"""Isolated real-only augmentation/retention experiment; legacy trainer stays frozen."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import itertools
import random
from unittest.mock import patch

import torch

import train_salad
from workflow.training_data import MixedGSVCitiesDataset, CoveragePlaceBatchSampler


class WindowSampler:
    """Checkpoint windows slice one continuous coverage stream, never restart its prefix."""
    def __init__(self, num_places, batch_size, seed, window_steps):
        self.base = CoveragePlaceBatchSampler(num_places, batch_size, seed)
        self.window_steps, self.epoch = window_steps, 0

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Negative window")
        self.epoch = epoch

    def __len__(self):
        return self.window_steps

    def __iter__(self):
        offset = self.epoch * self.window_steps
        remaining = self.window_steps
        while remaining:
            cycle, start = divmod(offset, len(self.base))
            self.base.set_epoch(cycle)
            count = min(remaining, len(self.base) - start)
            yield from itertools.islice(iter(self.base), start, start + count)
            offset += count
            remaining -= count


class RecoveryDataset(MixedGSVCitiesDataset):
    """Real-only data with keyed augmentation RNG shared across retention controls."""
    def __init__(self, *args, augmentation_policy, **kwargs):
        super().__init__(*args, **kwargs)
        if self.synthetic_fraction or self.reliability_pairs or self.summary["num_synthetic_images"]:
            raise ValueError("RecoveryDataset supports real-only data")
        from torchvision import transforms as T
        self.policy = augmentation_policy
        self.transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, magnitude=9, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.summary["recovery_augmentation"] = dict(
            policy=self.policy, rng="keyed_cpu_v1", randaugment_ops=3, randaugment_magnitude=9)

    def __getitem__(self, key):
        from PIL import Image
        cycle, index = key if isinstance(key, tuple) else (self.epoch, key)
        place = self.places[index]
        seed = int.from_bytes(hashlib.sha256(
            f"recovery:{self.seed}:{cycle}:{place.city}:{place.place_id}".encode()
        ).digest()[:8], "big") % (2**63)
        # num_workers=0 must not let augmentation consume main-model Dropout RNG.
        python_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                random.seed(seed)
                if self.policy == "weak":
                    return super().__getitem__((cycle, index))
                selected = self.sample_paths(index, epoch=cycle)
                tensors = []
                for path, synthetic in selected:
                    if synthetic:
                        raise ValueError("Unexpected synthetic image")
                    with Image.open(path) as image:
                        tensors.append(self.transform(image.convert("RGB")))
                result = (torch.stack(tensors),
                          torch.full((len(selected),), place.label, dtype=torch.long),
                          torch.zeros(len(selected), dtype=torch.bool))
                if self.return_sampling_metadata:
                    lookup = {path: i for i, path in enumerate(place.real_paths)}
                    result += (dict(place_index=index, real_indices=torch.tensor(
                        [lookup[path] for path, _ in selected], dtype=torch.long)),)
                return result
        finally:
            random.setstate(python_state)


def validate_args(args):
    if (args.synthetic_manifest or args.synthetic_aux or args.synthetic_fraction
            or args.reliability_ot or train_salad.pair_supervision_enabled(args)
            or args.synthetic_places_only or args.no_augment):
        raise ValueError("Recovery uses augmented real MS, with optional real retention only")
    if args.real_sampling != "coverage_pk" or args.precision not in ("32", "bf16"):
        raise ValueError("Use coverage_pk and explicit BF16/FP32")
    if args.window_steps < 1 or args.max_batches_per_epoch or args.max_steps != args.window_steps * args.epochs:
        raise ValueError("max-steps must equal window-steps * epochs; no batch truncation")
    if args.teacher_checkpoint and not args.real_global_distill_weight:
        raise ValueError("Teacher is only instantiated for positive real retention weight")
    if args.init_checkpoint and args.teacher_checkpoint:
        if args.init_checkpoint.resolve() != args.teacher_checkpoint.resolve():
            raise ValueError("Retention teacher must be the initial checkpoint")


def train(args):
    validate_args(args)
    import torchvision
    import workflow.training_data as data
    import workflow.model as model
    original_config = train_salad.resolved_training_config
    original_teacher = model.FrozenSALADTeacher

    def configuration(*a, **kw):
        config = original_config(*a, **kw)
        config["recovery"] = dict(
            version=1, augmentation=args.augmentation_policy, window_steps=args.window_steps,
            sampler="continuous_coverage_windows_v1", augmentation_rng="keyed_cpu_v1",
            teacher_init_rng="fork_cpu_cuda", num_workers=args.num_workers,
            torch_version=str(torch.__version__), torchvision_version=torchvision.__version__,
            scheduler="cosine stepped once per checkpoint window; fixed declared horizon",
            randaugment=dict(num_ops=3, magnitude=9, interpolation="bilinear"),
        )
        if args.teacher_checkpoint and config["teacher"]["checkpoint_sha256"] != config["init_checkpoint_sha256"]:
            raise ValueError("Teacher and initial checkpoint SHA256 differ")
        return config

    def teacher(*a, **kw):
        # A and A+retention must construct the student from the same RNG state.
        device = torch.device(kw["device"])
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            return original_teacher(*a, **kw)

    with ExitStack() as stack:
        stack.enter_context(patch.object(data, "MixedGSVCitiesDataset",
            lambda *a, **kw: RecoveryDataset(*a, augmentation_policy=args.augmentation_policy, **kw)))
        stack.enter_context(patch.object(data, "CoveragePlaceBatchSampler",
            lambda *a, **kw: WindowSampler(*a, window_steps=args.window_steps, **kw)))
        stack.enter_context(patch.object(model, "FrozenSALADTeacher", teacher))
        stack.enter_context(patch.object(train_salad, "resolved_training_config", configuration))
        return train_salad.train(args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--augmentation-policy", choices=("weak", "randaugment"), default="weak")
    parser.add_argument("--window-steps", type=int, default=100)
    extra, rest = parser.parse_known_args(argv)
    args = train_salad.parse_args(rest)
    vars(args).update(vars(extra))
    return args


if __name__ == "__main__":
    train(parse_args())
