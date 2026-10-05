"""Train a fresh SALAD model on real GSV-Cities, optionally mixed with verified synthetic hard positives."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .mixed_salad import MixedGSVCitiesDataset

DEFAULT_CITIES = [
    "Bangkok", "BuenosAires", "LosAngeles", "MexicoCity", "OSL", "Rome", "Barcelona", "Chicago",
    "Madrid", "Miami", "Phoenix", "TRT", "Boston", "Lisbon", "Medellin", "Minneapolis", "PRG",
    "WashingtonDC", "Brussels", "London", "Melbourne", "Osaka", "PRS",
]


class MixedDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=16, workers=8):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        # No persistent workers: each epoch receives a fresh deterministic mix plan.
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )


class SetDatasetEpoch(pl.Callback):
    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.history_path = self.output_dir / "mix_stats.jsonl"

    def on_train_epoch_start(self, trainer, pl_module):
        ds = trainer.datamodule.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(trainer.current_epoch)
        stats = getattr(ds, "mix_stats", None)
        if stats:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stats) + "\n")
            print(
                "mix epoch={epoch}: real={real} synth={synth} target_synth={target} "
                "capacity={capacity} eligible_places={eligible} coverage_limited={limited}".format(
                    epoch=stats["epoch"],
                    real=stats["planned_real_slots"],
                    synth=stats["planned_synthetic_slots"],
                    target=stats["target_synthetic_slots"],
                    capacity=stats.get("synthetic_capacity_slots", 0),
                    eligible=stats["eligible_places"],
                    limited=stats["coverage_limited"],
                ),
                flush=True,
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--salad-root", type=Path, required=True, help="local official serizba/salad checkout")
    p.add_argument("--gsv-root", type=Path, required=True)
    p.add_argument(
        "--synthetic-manifest",
        type=Path,
        default=None,
        help="verified synthetic manifest. Omit for the real-only control; use --synthetic-ratio 0",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cities", nargs="*", default=DEFAULT_CITIES)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-per-place", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--real-ratio", type=int, default=8)
    p.add_argument("--synthetic-ratio", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--devices", type=int, default=1)
    args = p.parse_args()

    if args.synthetic_ratio > 0 and args.synthetic_manifest is None:
        p.error("--synthetic-manifest is required when --synthetic-ratio > 0")
    if args.synthetic_ratio == 0:
        args.synthetic_manifest = None

    salad_root = args.salad_root.resolve()
    if not (salad_root / "vpr_model.py").is_file():
        raise FileNotFoundError(f"official SALAD checkout not found: {salad_root}")
    sys.path.insert(0, str(salad_root))
    from vpr_model import VPRModel

    pl.seed_everything(args.seed, workers=True)
    dataset = MixedGSVCitiesDataset(
        gsv_root=args.gsv_root,
        synthetic_manifest=args.synthetic_manifest,
        cities=args.cities,
        img_per_place=args.img_per_place,
        min_img_per_place=args.img_per_place,
        real_to_synth=(args.real_ratio, args.synthetic_ratio),
        image_size=(322, 322),
        seed=args.seed,
    )
    dm = MixedDataModule(dataset, args.batch_size, args.workers)

    # Keep the official SALAD architecture and metric-learning objective fixed.
    model = VPRModel(
        backbone_arch="dinov2_vitb14",
        backbone_config={"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
        agg_arch="SALAD",
        agg_config={"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
        lr=6e-5,
        optimizer="adamw",
        weight_decay=1e-3,
        lr_sched="linear",
        lr_sched_args={"start_factor": 1.0, "end_factor": 0.2, "total_iters": args.max_steps},
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.1,
    )

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    experiment_kind = "real_only" if args.synthetic_ratio == 0 else "real_plus_synthetic"
    (out / "config.json").write_text(
        json.dumps(
            {
                "experiment_kind": experiment_kind,
                "synthetic_manifest": str(args.synthetic_manifest.resolve()) if args.synthetic_manifest else None,
                "cities": args.cities,
                "batch_size_places": args.batch_size,
                "img_per_place": args.img_per_place,
                "real_ratio": args.real_ratio,
                "synthetic_ratio": args.synthetic_ratio,
                "max_steps": args.max_steps,
                "seed": args.seed,
                "initial_mix_stats": dataset.mix_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=out / "checkpoints",
        filename="salad-{step:06d}",
        save_top_k=-1,
        every_n_train_steps=max(1, args.max_steps // 4),
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_steps=args.max_steps,
        precision=args.precision,
        default_root_dir=out,
        callbacks=[checkpoint, SetDatasetEpoch(out)],
        logger=True,
        log_every_n_steps=10,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
