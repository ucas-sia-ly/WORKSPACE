"""Train a fresh SALAD on real GSV-Cities and verified synthetic hard positives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .data import require_empty_output
from .mixed_salad import MixedGSVCitiesDataset

DEFAULT_CITIES = [
    "Bangkok", "BuenosAires", "LosAngeles", "MexicoCity", "OSL", "Rome", "Barcelona", "Chicago",
    "Madrid", "Miami", "Phoenix", "TRT", "Boston", "Lisbon", "Medellin", "Minneapolis", "PRG",
    "WashingtonDC", "Brussels", "London", "Melbourne", "Osaka", "PRS",
]


class CityPlaceSampler(DistributedSampler):
    """Keep the official city-local batches and explicitly account for DDP padding."""
    def __init__(self, dataset, num_replicas=1, rank=0, shuffle_all=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=True, seed=dataset.seed, drop_last=False)
        self.shuffle_all = shuffle_all

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.dataset.set_epoch(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        if self.shuffle_all:
            indices = list(range(len(self.dataset)))
            rng.shuffle(indices)
        else:
            for city in self.dataset.cities:
                city_indices = [i for i, key in enumerate(self.dataset.keys) if key[0] == city]
                rng.shuffle(city_indices)
                indices.extend(city_indices)
        padding = self.total_size - len(indices)
        if padding:
            indices += (indices * ((padding + len(indices) - 1) // len(indices)))[:padding]
        return iter(indices[self.rank:self.total_size:self.num_replicas])


class MixedDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=60, workers=8, shuffle_all=False):
        super().__init__()
        self.dataset, self.batch_size, self.workers = dataset, batch_size, workers
        self.shuffle_all = shuffle_all
        self._restored_epoch = None

    def state_dict(self):
        # This runner resumes only completed epochs. Lightning 2.x can expose
        # the saved epoch until FitLoop.reset advances it, after iterator setup.
        return {"next_epoch": self.trainer.current_epoch + 1}

    def load_state_dict(self, state_dict):
        self._restored_epoch = state_dict["next_epoch"]
        self.dataset.set_epoch(self._restored_epoch)

    def train_dataloader(self):
        # Lightning can create its iterator before on_train_epoch_start when
        # restoring a checkpoint. Plan the epoch before workers are launched.
        epoch = self.trainer.current_epoch if self._restored_epoch is None else self._restored_epoch
        self._restored_epoch = None
        self.dataset.set_epoch(epoch)
        sampler = CityPlaceSampler(self.dataset, self.trainer.world_size,
                                   self.trainer.global_rank, self.shuffle_all)
        sampler.set_epoch(epoch)
        return DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=sampler,
            num_workers=self.workers, pin_memory=True, drop_last=False,
            persistent_workers=False,
        )


class SetDatasetEpoch(pl.Callback):
    """Count image slots from batches actually consumed by the official model."""
    def __init__(self, output_dir: Path, training_budget=None):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.history_path = self.output_dir / "mix_stats.jsonl"
        self.actual_synthetic = self.actual_real = self.actual_batches = 0
        self.epoch_complete = False
        self._batch_counts = None
        self.training_budget = training_budget

    def on_train_epoch_start(self, trainer, pl_module):
        trainer.datamodule.dataset.set_epoch(trainer.current_epoch)
        self.actual_synthetic = self.actual_real = self.actual_batches = 0
        self.epoch_complete = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not isinstance(batch, list) or len(batch) != 3:
            raise ValueError("SALAD mix accounting requires image, label, metadata batches")
        metadata = batch.pop()
        if not torch.all(metadata["epoch"] == trainer.current_epoch):
            raise RuntimeError(f"DataLoader worker used stale epochs {metadata['epoch'].tolist()}; expected {trainer.current_epoch}")
        self._batch_counts = (int(metadata["synthetic_slots"].sum().item()),
                              int(metadata["real_slots"].sum().item()))
        # The official VPRModel receives exactly (places, labels), and its loss
        # and miner never see synthetic accounting metadata.

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._batch_counts is None:
            raise RuntimeError("missing mix metadata for the consumed batch")
        synthetic, real = self._batch_counts
        self.actual_synthetic += synthetic
        self.actual_real += real
        self.actual_batches += 1
        self._batch_counts = None
        self.epoch_complete = self.actual_batches == trainer.num_training_batches
        if self.training_budget is not None and trainer.global_step >= self.training_budget:
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer, pl_module):
        counts = torch.tensor([self.actual_synthetic, self.actual_real, self.actual_batches],
                              device=pl_module.device, dtype=torch.long)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        synthetic, real, batches = map(int, counts.tolist())
        stats = dict(trainer.datamodule.dataset.mix_stats)
        stats.update(
            actual_synthetic_slots=synthetic, actual_real_slots=real,
            actual_total_slots=synthetic + real, actual_batches=batches,
            actual_ratio=[real, synthetic],
            actual_real_to_synthetic=(real / synthetic) if synthetic else None,
            actual_synthetic_fraction=synthetic / (real + synthetic) if real + synthetic else 0.0,
            epoch_complete=self.epoch_complete, world_size=trainer.world_size,
            global_step=trainer.global_step,
        )
        if trainer.is_global_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stats) + "\n")
            print(
                f"mix epoch={stats['epoch']}: target_synthetic_slots={stats['target_synthetic_slots']} "
                f"synthetic_capacity_slots={stats['synthetic_capacity_slots']} "
                f"planned_synthetic_slots={stats['planned_synthetic_slots']} "
                f"actual_synthetic_slots={synthetic} actual_real_slots={real} "
                f"actual_ratio={real}:{synthetic} coverage_limited={stats['coverage_limited']} "
                f"epoch_complete={self.epoch_complete}", flush=True,
            )

    def state_dict(self):
        numpy_rng = list(np.random.get_state())
        numpy_rng[1] = numpy_rng[1].tolist()
        return {"epoch_complete": self.epoch_complete,
                "python_rng": random.getstate(), "numpy_rng": numpy_rng,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    def load_state_dict(self, state_dict):
        self.epoch_complete = state_dict["epoch_complete"]
        random.setstate(state_dict["python_rng"])
        numpy_rng = list(state_dict["numpy_rng"])
        numpy_rng[1] = np.asarray(numpy_rng[1], dtype=np.uint32)
        np.random.set_state(tuple(numpy_rng))
        torch.set_rng_state(state_dict["torch_rng"].cpu())
        if state_dict["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in state_dict["cuda_rng"]])


def official_model_class(base_class):
    """Keep the official training loss and step its scheduler exactly once."""
    class FreshSALAD(base_class):
        def configure_optimizers(self):
            optimizers, schedulers = super().configure_optimizers()
            if len(optimizers) != 1 or len(schedulers) != 1:
                raise ValueError("expected the official single optimizer/scheduler SALAD interface")
            return {"optimizer": optimizers[0],
                    "lr_scheduler": {"scheduler": schedulers[0], "interval": "step"}}

        def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
            optimizer.step(closure=optimizer_closure)

    return FreshSALAD


def freeze_backbone_prefix(model):
    """Express official forward freezing in parameters, also safe for DDP."""
    backbone = model.backbone
    dino = backbone.model
    n = backbone.num_trainable_blocks
    if not 0 < n <= len(dino.blocks):
        raise ValueError("invalid number of trainable DINOv2 blocks")
    dino.requires_grad_(False)
    for block in dino.blocks[-n:]:
        block.requires_grad_(True)
    if backbone.norm_layer:
        dino.norm.requires_grad_(True)


def validate_resume(path: Path, output_dir: Path, config: dict, devices: int):
    if devices != 1:
        raise ValueError("resume currently requires one device to restore the saved RNG state")
    previous = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if previous != config:
        raise ValueError("resume configuration differs from the original experiment")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("callbacks", {}).get("SetDatasetEpoch")
    if state is None or not state.get("epoch_complete"):
        raise ValueError("resume is supported only at a completed epoch; checkpoint ended mid-epoch")
    if not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers"):
        raise ValueError("resume requires a full checkpoint with optimizer and scheduler states")
    if checkpoint["global_step"] >= config["max_steps"]:
        raise ValueError("checkpoint already reached the configured training budget")
    journal = output_dir / "mix_stats.jsonl"
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["global_step"] > checkpoint["global_step"] or row["epoch"] > checkpoint["epoch"]:
                    raise ValueError("output journal is newer than this checkpoint; cannot append a stale resume")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--salad-root", type=Path, required=True, help="local official serizba/salad checkout")
    p.add_argument("--gsv-root", type=Path, required=True)
    p.add_argument("--synthetic-manifest", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path, help="full checkpoint saved at a completed epoch")
    p.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    p.add_argument("--batch-size", type=int, default=60)
    p.add_argument("--img-per-place", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shuffle-all", action="store_true", help="shuffle across cities instead of official city-local batches")
    p.add_argument("--real-ratio", type=int, default=8)
    p.add_argument("--synthetic-ratio", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", choices=["gpu", "cpu", "auto"], default="gpu")
    args = p.parse_args()
    if args.synthetic_ratio > 0 and args.synthetic_manifest is None:
        p.error("--synthetic-manifest is required when --synthetic-ratio > 0")
    if args.synthetic_ratio == 0:
        args.synthetic_manifest = None
    if args.batch_size < 2 or args.img_per_place < 2:
        p.error("metric training requires at least two places per batch and two images per place")
    if args.max_steps < 1 or args.devices < 1 or args.workers < 0:
        p.error("max-steps/devices must be positive and workers must be non-negative")
    if args.image_size < 126 or args.image_size % 14:
        p.error("image-size must be divisible by 14 with more than 64 patch tokens")

    salad_root = args.salad_root.resolve()
    if not (salad_root / "vpr_model.py").is_file():
        raise FileNotFoundError(f"official SALAD checkout not found: {salad_root}")
    sys.path.insert(0, str(salad_root))
    from vpr_model import VPRModel
    if Path(sys.modules[VPRModel.__module__].__file__).resolve().parent != salad_root:
        raise RuntimeError("another vpr_model module shadowed the requested SALAD checkout")

    out = args.output_dir.resolve()
    rank_zero = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    if args.resume is None and rank_zero:
        require_empty_output(out)
    pl.seed_everything(args.seed, workers=True)
    dataset = MixedGSVCitiesDataset(
        args.gsv_root, args.synthetic_manifest, args.cities,
        img_per_place=args.img_per_place, min_img_per_place=args.img_per_place,
        real_to_synth=(args.real_ratio, args.synthetic_ratio),
        image_size=(args.image_size, args.image_size), seed=args.seed,
        return_mix_metadata=True,
    )
    if len(dataset) < 2:
        raise ValueError("metric training requires at least two distinct GSV places")
    dm = MixedDataModule(dataset, args.batch_size, args.workers, args.shuffle_all)
    model = official_model_class(VPRModel)(
        backbone_arch="dinov2_vitb14",
        backbone_config={"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
        agg_arch="SALAD",
        agg_config={"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
        lr=6e-5, optimizer="adamw", weight_decay=9.5e-9,
        lr_sched="linear",
        lr_sched_args={"start_factor": 1.0, "end_factor": 0.2, "total_iters": args.max_steps},
        loss_name="MultiSimilarityLoss", miner_name="MultiSimilarityMiner", miner_margin=0.1,
    )
    freeze_backbone_prefix(model)
    config = {
        "experiment_kind": "real_only" if args.synthetic_ratio == 0 else "real_plus_synthetic",
        "gsv_root": str(args.gsv_root.resolve()), "salad_root": str(salad_root),
        "salad_code_sha256": {
            name: hashlib.sha256((salad_root / name).read_bytes()).hexdigest()
            for name in ("vpr_model.py", "models/backbones/dinov2.py",
                         "models/aggregators/salad.py", "utils/losses.py")
        },
        "gsv_dataframe_sha256": {
            city: hashlib.sha256((args.gsv_root / "Dataframes" / f"{city}.csv").read_bytes()).hexdigest()
            for city in args.cities
        },
        "synthetic_manifest": str(args.synthetic_manifest.resolve()) if args.synthetic_manifest else None,
        "synthetic_manifest_sha256": hashlib.sha256(args.synthetic_manifest.read_bytes()).hexdigest()
            if args.synthetic_manifest else None,
        "cities": args.cities, "batch_size_places": args.batch_size,
        "img_per_place": args.img_per_place, "image_size": args.image_size,
        "real_ratio": args.real_ratio, "synthetic_ratio": args.synthetic_ratio,
        "max_steps": args.max_steps, "seed": args.seed, "workers": args.workers,
        "shuffle_all": args.shuffle_all,
        "precision": args.precision, "devices": args.devices, "accelerator": args.accelerator,
        "learning_rate": 6e-5, "weight_decay": 9.5e-9,
        "initial_mix_stats": dataset.mix_stats,
    }
    if args.resume is not None:
        validate_resume(args.resume.resolve(), out, config, args.devices)
    elif rank_zero:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=out / "checkpoints", filename="salad-epoch{epoch:03d}-step{step:06d}",
        save_top_k=-1, every_n_epochs=1, save_last=True, save_weights_only=False,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        # Lightning 2.x's max_steps loop can resume a boundary checkpoint as the
        # previous epoch. Use its epoch loop and stop at the identical step budget.
        accelerator=args.accelerator, devices=args.devices, max_epochs=args.max_steps,
        precision=args.precision, default_root_dir=out,
        callbacks=[SetDatasetEpoch(out, args.max_steps), checkpoint], logger=True,
        log_every_n_steps=10, num_sanity_val_steps=0, use_distributed_sampler=False,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=str(args.resume.resolve()) if args.resume else None)


if __name__ == "__main__":
    main()
