"""Train a fresh SALAD model on real GSV-Cities + verified synthetic hard positives."""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .mixed_salad import MixedGSVCitiesDataset

DEFAULT_CITIES = [
    "Bangkok","BuenosAires","LosAngeles","MexicoCity","OSL","Rome","Barcelona","Chicago",
    "Madrid","Miami","Phoenix","TRT","Boston","Lisbon","Medellin","Minneapolis","PRG",
    "WashingtonDC","Brussels","London","Melbourne","Osaka","PRS",
]


class MixedDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=16, workers=8):
        super().__init__(); self.dataset=dataset; self.batch_size=batch_size; self.workers=workers
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.workers, pin_memory=True, drop_last=False,
                          persistent_workers=self.workers > 0)


class SetDatasetEpoch(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        ds = trainer.datamodule.dataset
        if hasattr(ds, "set_epoch"): ds.set_epoch(trainer.current_epoch)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--salad-root", type=Path, required=True, help="local official serizba/salad checkout")
    p.add_argument("--gsv-root", type=Path, required=True)
    p.add_argument("--synthetic-manifest", type=Path, required=True)
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
    args=p.parse_args()

    salad_root=args.salad_root.resolve()
    if not (salad_root/"vpr_model.py").is_file():
        raise FileNotFoundError(f"official SALAD checkout not found: {salad_root}")
    sys.path.insert(0, str(salad_root))
    from vpr_model import VPRModel

    pl.seed_everything(args.seed, workers=True)
    dataset=MixedGSVCitiesDataset(
        gsv_root=args.gsv_root,
        synthetic_manifest=args.synthetic_manifest,
        cities=args.cities,
        img_per_place=args.img_per_place,
        min_img_per_place=args.img_per_place,
        real_to_synth=(args.real_ratio,args.synthetic_ratio),
        image_size=(322,322), seed=args.seed,
    )
    dm=MixedDataModule(dataset,args.batch_size,args.workers)

    # Same architecture and metric-learning objective as official SALAD.
    model=VPRModel(
        backbone_arch="dinov2_vitb14",
        backbone_config={"num_trainable_blocks":4,"return_token":True,"norm_layer":True},
        agg_arch="SALAD",
        agg_config={"num_channels":768,"num_clusters":64,"cluster_dim":128,"token_dim":256},
        lr=6e-5,
        optimizer="adamw",
        weight_decay=1e-3,
        lr_sched="linear",
        lr_sched_args={"start_factor":1.0,"end_factor":0.2,"total_iters":args.max_steps},
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.1,
    )

    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    checkpoint=pl.callbacks.ModelCheckpoint(
        dirpath=out/"checkpoints", filename="salad-{step:06d}", save_top_k=-1,
        every_n_train_steps=max(1,args.max_steps//4), save_last=True,
    )
    trainer=pl.Trainer(
        accelerator="gpu", devices=args.devices, max_steps=args.max_steps,
        precision=args.precision, default_root_dir=out,
        callbacks=[checkpoint,SetDatasetEpoch()], logger=True,
        log_every_n_steps=10, enable_checkpointing=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model,datamodule=dm)


if __name__ == "__main__":
    main()
