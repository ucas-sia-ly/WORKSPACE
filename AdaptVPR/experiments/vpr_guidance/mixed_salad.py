"""GSV-Cities + verified synthetic place dataset for SALAD-style training.

Each item is one geographical place with K images. Synthetic slots are allocated
at epoch level so the requested real:synthetic exposure is reproducible and
measurable. Every synthetic image comes from the same verified (city, place_id)
and inherits the parent place label, following AdaptVPR-style hard-positive
augmentation.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def build_transform(image_size=(322, 322)):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
    ])


def _gsv_name(row, place_id):
    city = row["city_id"]
    pid = str(int(place_id)).zfill(7)
    pano = row["panoid"]
    year = str(row["year"]).zfill(4)
    month = str(row["month"]).zfill(2)
    north = str(row["northdeg"]).zfill(3)
    return f"{city}_{pid}_{year}_{month}_{north}_{row['lat']}_{row['lon']}_{pano}.jpg"


def load_synthetic_manifest(path: Path | None):
    if path is None:
        return {}
    by_place = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("passed") or not row.get("eligible_for_training"):
            continue
        key = (str(row["city"]), str(row["place_id"]).zfill(7))
        gen = Path(row["generated_path"])
        if not gen.is_file():
            raise FileNotFoundError(gen)
        by_place.setdefault(key, []).append(gen)
    return by_place


class MixedGSVCitiesDataset(Dataset):
    def __init__(
        self,
        gsv_root: Path,
        synthetic_manifest: Path | None,
        cities,
        img_per_place=4,
        min_img_per_place=4,
        real_to_synth=(8, 1),
        image_size=(322, 322),
        seed=42,
    ):
        self.root = Path(gsv_root)
        self.cities = list(cities)
        self.k = int(img_per_place)
        self.min_k = int(min_img_per_place)
        self.real_ratio, self.synth_ratio = map(int, real_to_synth)
        if self.real_ratio < 0 or self.synth_ratio < 0 or self.real_ratio + self.synth_ratio <= 0:
            raise ValueError("invalid real_to_synth ratio")
        if self.k < 1 or self.k > self.min_k:
            raise ValueError("img_per_place must be in [1, min_img_per_place]")
        if self.synth_ratio > 0 and synthetic_manifest is None:
            raise ValueError("synthetic_manifest is required when synthetic_ratio > 0")

        self.synthetic = load_synthetic_manifest(synthetic_manifest)
        self.transform = build_transform(image_size)
        self.seed = int(seed)
        self.epoch = 0

        frames = []
        for city in self.cities:
            df = pd.read_csv(self.root / "Dataframes" / f"{city}.csv")
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        df = df[df.groupby(["city_id", "place_id"])["place_id"].transform("size") >= self.min_k]
        self.groups = {
            (str(city), str(int(pid)).zfill(7)): group.copy()
            for (city, pid), group in df.groupby(["city_id", "place_id"])
        }
        self.keys = sorted(self.groups)
        self.label_map = {key: i for i, key in enumerate(self.keys)}
        self._synthetic_quota = {}
        self._mix_stats = {}
        self.set_epoch(0)

    def _plan_epoch_mix(self):
        total_slots = len(self.keys) * self.k
        target_synth = round(
            total_slots * self.synth_ratio / (self.real_ratio + self.synth_ratio)
        )
        capacities = {
            key: min(self.k, len(self.synthetic.get(key, [])))
            for key in self.keys
        }
        eligible = [key for key, capacity in capacities.items() if capacity > 0]
        total_capacity = sum(capacities.values())
        planned_synth = min(target_synth, total_capacity)

        quota = {key: 0 for key in self.keys}
        if planned_synth and eligible:
            rng = random.Random(self.seed * 1_000_003 + self.epoch * 10_000_019)
            order = list(eligible)
            rng.shuffle(order)
            remaining = planned_synth
            # Deterministic round-robin allocation up to each place's *unique*
            # synthetic capacity. This prevents reusing the same generated image
            # twice inside one K-image place sample just to hit a nominal ratio.
            while remaining > 0:
                progressed = False
                for key in order:
                    if remaining <= 0:
                        break
                    if quota[key] < capacities[key]:
                        quota[key] += 1
                        remaining -= 1
                        progressed = True
                if not progressed:
                    break

        actual_synth = sum(quota.values())
        self._synthetic_quota = quota
        self._mix_stats = {
            "epoch": self.epoch,
            "total_places": len(self.keys),
            "eligible_places": len(eligible),
            "slots_per_place": self.k,
            "total_slots": total_slots,
            "synthetic_capacity_slots": total_capacity,
            "target_synthetic_slots": target_synth,
            "planned_synthetic_slots": actual_synth,
            "planned_real_slots": total_slots - actual_synth,
            "target_real_to_synthetic": [self.real_ratio, self.synth_ratio],
            "achieved_synthetic_fraction": (actual_synth / total_slots) if total_slots else 0.0,
            "coverage_limited": actual_synth < target_synth,
        }

    @property
    def mix_stats(self):
        return dict(self._mix_stats)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if hasattr(self, "keys"):
            self._plan_epoch_mix()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        group = self.groups[key]
        rng = random.Random((self.seed + 1) * 1_000_003 + self.epoch * 10_000_019 + index)
        rows = [row for _, row in group.iterrows()]
        rng.shuffle(rows)
        selected = rows[: self.k]
        paths = [
            self.root / "Images" / row["city_id"] / _gsv_name(row, int(row["place_id"]))
            for row in selected
        ]

        synth_pool = list(self.synthetic.get(key, []))
        synth_slots = min(
            self._synthetic_quota.get(key, 0),
            self.k,
            len(paths),
            len(synth_pool),
        )
        if synth_slots:
            slot_ids = list(range(self.k))
            rng.shuffle(slot_ids)
            rng.shuffle(synth_pool)
            for slot, synthetic_path in zip(slot_ids[:synth_slots], synth_pool[:synth_slots]):
                paths[slot] = synthetic_path

        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))

        label = self.label_map[key]
        return torch.stack(images), torch.full((self.k,), label, dtype=torch.long)
