"""One real official SALAD training step using cached DINOv2 and four GSV images.

Run this separately from unit tests; it requires official model code/weights/data.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytorch_lightning as pl
import torch

from AdaptVPR.experiments.vpr_guidance.data import gsv_image_name, require_empty_output
from AdaptVPR.experiments.vpr_guidance.mixed_salad import MixedGSVCitiesDataset
from AdaptVPR.experiments.vpr_guidance.train_salad import (
    MixedDataModule, SetDatasetEpoch, freeze_backbone_prefix, official_model_class,
)


class Probe(pl.Callback):
    def __init__(self):
        self.evidence = {}

    def on_after_backward(self, trainer, model):
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        norm = torch.stack([g.float().norm().square() for g in gradients]).sum().sqrt()
        assert norm > 0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        self.evidence["gradient_norm"] = float(norm)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.evidence.update(loss=float(outputs["loss"]), batch_shape=list(batch[0].shape),
                             label_shape=list(batch[1].shape), dtype=str(batch[0].dtype),
                             device=str(batch[0].device))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--salad-root", type=Path, required=True)
    p.add_argument("--dino-root", type=Path, required=True, help="cached facebookresearch/dinov2 source")
    p.add_argument("--gsv-root", type=Path, required=True)
    p.add_argument("--city", default="Bangkok")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu")
    args = p.parse_args()
    out = args.output_dir.resolve()
    require_empty_output(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    pl.seed_everything(42, workers=True)
    sys.path.insert(0, str(args.salad_root.resolve()))
    from vpr_model import VPRModel
    original_load = torch.hub.load

    def cached_load(repo, name, *a, **kw):
        assert repo == "facebookresearch/dinov2", f"unexpected hub repository {repo}"
        return original_load(str(args.dino_root.resolve()), name, *a, source="local", **kw)

    with patch.object(torch.hub, "load", cached_load):
        model = official_model_class(VPRModel)(
            backbone_arch="dinov2_vitb14",
            backbone_config={"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
            agg_arch="SALAD", agg_config={"num_channels": 768, "num_clusters": 64,
                                           "cluster_dim": 128, "token_dim": 256},
            lr=6e-5, optimizer="adamw", weight_decay=9.5e-9,
            lr_sched="linear", lr_sched_args={"start_factor": 1., "end_factor": .2, "total_iters": 4},
            loss_name="MultiSimilarityLoss", miner_name="MultiSimilarityMiner", miner_margin=.1,
        )
    freeze_backbone_prefix(model)
    original_weight = model.aggregator.token_features[0].weight.detach().clone()
    source_paths, selected = [], []
    frame = pd.read_csv(args.gsv_root / "Dataframes" / f"{args.city}.csv")
    for _, group in frame.groupby("place_id"):
        available = []
        for _, row in group.iterrows():
            source = args.gsv_root / "Images" / args.city / gsv_image_name(row)
            if source.is_file():
                available.append((row, source))
        if len(available) >= 2:
            selected.extend(row for row, _ in available[:2])
            source_paths.extend(path for _, path in available[:2])
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError("smoke requires two GSV places each with two existing real images")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "Dataframes").mkdir()
        image_dir = root / "Images" / args.city
        image_dir.mkdir(parents=True)
        pd.DataFrame(selected).to_csv(root / "Dataframes" / f"{args.city}.csv", index=False)
        for path in source_paths:
            shutil.copyfile(path, image_dir / path.name)
        ds = MixedGSVCitiesDataset(root, None, [args.city], img_per_place=2,
                                  min_img_per_place=2, real_to_synth=(1, 0), return_mix_metadata=True)
        probe = Probe()
        trainer = pl.Trainer(accelerator=args.accelerator, devices=1, max_epochs=1, precision="32-true",
                             callbacks=[SetDatasetEpoch(out, training_budget=1), probe],
                             logger=False, enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, num_sanity_val_steps=0,
                             use_distributed_sampler=False)
        trainer.fit(model, datamodule=MixedDataModule(ds, batch_size=2, workers=0))
        assert trainer.global_step == 1 and model.lr_schedulers().last_epoch == 1
        assert not torch.equal(original_weight.to(model.device), model.aggregator.token_features[0].weight)
        model.eval()
        images, _, _ = ds[0]
        with torch.no_grad():
            descriptors = model(images.to(model.device))
        assert descriptors.shape == (2, 8448)
        torch.testing.assert_close(descriptors.norm(dim=-1), torch.ones(2, device=model.device))
        probe.evidence.update(global_step=trainer.global_step, scheduler_steps=model.lr_schedulers().last_epoch,
                              descriptor_shape=list(descriptors.shape),
                              source_paths=[str(path.resolve()) for path in source_paths],
                              aggregator_updated=True, frozen_prefix_has_no_grad=True)
        (out / "smoke.json").write_text(json.dumps(probe.evidence, indent=2), encoding="utf-8")
        print(json.dumps(probe.evidence, indent=2))


if __name__ == "__main__":
    main()
