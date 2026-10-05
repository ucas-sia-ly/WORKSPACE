"""CPU tests for downstream trainer controls; these are not a pretrained-model smoke."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from torchvision import transforms as T

from AdaptVPR.experiments.vpr_guidance.mixed_salad import MixedGSVCitiesDataset
from AdaptVPR.experiments.vpr_guidance.train_salad import (
    CityPlaceSampler, MixedDataModule, SetDatasetEpoch, freeze_backbone_prefix,
    official_model_class, validate_resume,
)
from AdaptVPR.experiments.vpr_guidance.tests.test_data_pipeline import make_mixed_fixture


class TinyOfficialModel(pl.LightningModule):
    """Match official scheduler hooks while keeping CPU control tests small."""
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.1))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=6e-5, weight_decay=9.5e-9)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=.2,
                                                       total_iters=4)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()

    def training_step(self, batch, batch_idx):
        places, labels = batch
        b, k, c, h, w = places.shape
        self.asserted_shape = (b, k, c, h, w)
        assert labels.view(-1).numel() == b * k
        descriptor = self.projection(places.view(b * k, c, h, w).mean(dim=(-1, -2)))
        loss = descriptor.square().mean()
        assert torch.isfinite(loss)
        return loss


def cpu_trainer(callbacks, max_steps=4, max_epochs=-1):
    for callback in callbacks:
        if isinstance(callback, SetDatasetEpoch):
            callback.training_budget = max_steps
    return pl.Trainer(accelerator="cpu", devices=1, max_epochs=max_steps if max_epochs == -1 else max_epochs,
                      logger=False, callbacks=callbacks, enable_checkpointing=any(
                          isinstance(callback, pl.callbacks.ModelCheckpoint) for callback in callbacks),
                      enable_model_summary=False, enable_progress_bar=False,
                      num_sanity_val_steps=0, use_distributed_sampler=False)


class SALADTrainingTests(unittest.TestCase):
    def test_scheduler_is_stepped_once_and_partial_epoch_is_measured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest, _ = make_mixed_fixture(tmp, places=4)
            ds = MixedGSVCitiesDataset(root, manifest, ["Bangkok"], return_mix_metadata=True)
            ds.transform = T.ToTensor()
            model = official_model_class(TinyOfficialModel)()
            trainer = cpu_trainer([SetDatasetEpoch(Path(tmp) / "complete")])
            trainer.fit(model, datamodule=MixedDataModule(ds, batch_size=2, workers=0))
            self.assertEqual(model.lr_schedulers().last_epoch, trainer.global_step)
            self.assertEqual(trainer.global_step, 4)
            records = [json.loads(line) for line in (Path(tmp) / "complete" / "mix_stats.jsonl").read_text().splitlines()]
            self.assertEqual([r["actual_total_slots"] for r in records], [16, 16])
            self.assertTrue(all(r["epoch_complete"] for r in records))
            partial_model = official_model_class(TinyOfficialModel)()
            partial = cpu_trainer([SetDatasetEpoch(Path(tmp) / "partial")], max_steps=1)
            partial.fit(partial_model, datamodule=MixedDataModule(ds, batch_size=2, workers=0))
            record = json.loads((Path(tmp) / "partial" / "mix_stats.jsonl").read_text())
            self.assertEqual(record["actual_total_slots"], 8)
            self.assertFalse(record["epoch_complete"])

    def test_epoch_boundary_resume_restores_optimizer_scheduler_rng_and_worker_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest, _ = make_mixed_fixture(tmp, places=4)
            out = Path(tmp) / "resume"
            def dm():
                ds = MixedGSVCitiesDataset(root, manifest, ["Bangkok"], return_mix_metadata=True)
                ds.transform = T.ToTensor()
                return MixedDataModule(ds, batch_size=2, workers=2)
            pl.seed_everything(42, workers=True)
            reference = official_model_class(TinyOfficialModel)()
            cpu_trainer([SetDatasetEpoch(Path(tmp) / "reference")]).fit(reference, datamodule=dm())
            pl.seed_everything(42, workers=True)
            first = official_model_class(TinyOfficialModel)()
            checkpoint = pl.callbacks.ModelCheckpoint(dirpath=out / "checkpoints", save_last=True,
                                                       save_top_k=-1, every_n_epochs=1)
            initial_trainer = cpu_trainer([SetDatasetEpoch(out), checkpoint], max_epochs=1)
            initial_trainer.fit(first, datamodule=dm())
            config = {"max_steps": 4}
            (out / "config.json").write_text(json.dumps(config))
            validate_resume(Path(checkpoint.last_model_path), out, config, devices=1)
            saved = torch.load(checkpoint.last_model_path, map_location="cpu", weights_only=False)
            self.assertTrue(saved["optimizer_states"][0]["state"])
            self.assertEqual(saved["global_step"], 2)
            resumed = official_model_class(TinyOfficialModel)()
            resumed_trainer = cpu_trainer([SetDatasetEpoch(out)])
            resumed_trainer.fit(resumed, datamodule=dm(), ckpt_path=checkpoint.last_model_path)
            self.assertEqual(resumed_trainer.global_step, 4)
            self.assertEqual(resumed.lr_schedulers().last_epoch, 4)
            self.assertEqual(resumed_trainer.current_epoch, 2)
            for expected, actual in zip(reference.parameters(), resumed.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "journal is newer"):
                validate_resume(Path(checkpoint.last_model_path), out, config, devices=1)
            saved["callbacks"]["SetDatasetEpoch"]["epoch_complete"] = False
            mid = out / "mid.ckpt"
            torch.save(saved, mid)
            with self.assertRaisesRegex(ValueError, "mid-epoch"):
                validate_resume(mid, out, config, devices=1)

    def test_freeze_matches_official_last_blocks_and_norm(self):
        dino = torch.nn.Module()
        dino.patch_embed = torch.nn.Linear(3, 4)
        dino.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(6)])
        dino.norm = torch.nn.LayerNorm(4)
        model = SimpleNamespace(backbone=SimpleNamespace(model=dino, num_trainable_blocks=2,
                                                         norm_layer=True))
        freeze_backbone_prefix(model)
        self.assertFalse(any(p.requires_grad for p in dino.patch_embed.parameters()))
        self.assertFalse(any(p.requires_grad for block in dino.blocks[:-2] for p in block.parameters()))
        self.assertTrue(all(p.requires_grad for block in dino.blocks[-2:] for p in block.parameters()))
        self.assertTrue(all(p.requires_grad for p in dino.norm.parameters()))

    def test_city_sampler_is_repeatable_and_set_epoch_updates_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest, _ = make_mixed_fixture(tmp)
            ds = MixedGSVCitiesDataset(root, manifest, ["Bangkok"])
            a, b = CityPlaceSampler(ds, num_replicas=2, rank=0), CityPlaceSampler(ds, num_replicas=2, rank=1)
            a.set_epoch(7)
            b.set_epoch(7)
            self.assertEqual(ds.epoch, 7)
            self.assertEqual(list(a), list(a))
            self.assertEqual(len(list(a)) + len(list(b)), 10)


if __name__ == "__main__":
    unittest.main()
