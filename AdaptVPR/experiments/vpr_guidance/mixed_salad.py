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

from .data import GSVLabelIndex, canonical_place_id, gsv_image_name, validate_source_label

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def build_transform(image_size=(224, 224)):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
    ])


def _gsv_name(row, place_id):
    return gsv_image_name(row, place_id)


def load_synthetic_manifest(path: Path | None, labels: GSVLabelIndex | None = None):
    if path is None:
        return {}
    path = Path(path).resolve()
    by_place, file_labels = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("passed") is not True or row.get("eligible_for_training") is not True:
            continue
        if row.get("route") != "global":
            raise ValueError("only verified Global synthetic images may enter SALAD training")
        key = (str(row["city"]), canonical_place_id(row["place_id"]))
        source = Path(row["source_path"])
        source = (path.parent / source).resolve() if not source.is_absolute() else source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if labels is not None:
            key = validate_source_label(row, source, labels)
        gen = Path(row["generated_path"])
        gen = (path.parent / gen).resolve() if not gen.is_absolute() else gen.resolve()
        if not gen.is_file():
            raise FileNotFoundError(gen)
        if gen in file_labels:
            if file_labels[gen] != key:
                raise ValueError(f"synthetic file has conflicting place labels: {gen}")
            continue
        file_labels[gen] = key
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
        image_size=(224, 224),
        seed=42,
        return_mix_metadata=False,
        shared_mix_plan=None,
    ):
        self.root = Path(gsv_root)
        self.cities = list(cities)
        if not self.cities or len(set(self.cities)) != len(self.cities):
            raise ValueError("cities must be a non-empty list without duplicates")
        self.k = int(img_per_place)
        self.min_k = int(min_img_per_place)
        self.real_ratio, self.synth_ratio = map(int, real_to_synth)
        if self.real_ratio < 0 or self.synth_ratio < 0 or self.real_ratio + self.synth_ratio <= 0:
            raise ValueError("invalid real_to_synth ratio")
        if self.k < 1 or self.k > self.min_k:
            raise ValueError("img_per_place must be in [1, min_img_per_place]")
        if self.synth_ratio > 0 and synthetic_manifest is None:
            raise ValueError("synthetic_manifest is required when synthetic_ratio > 0")
        if self.synth_ratio == 0:
            synthetic_manifest = None

        self.synthetic = load_synthetic_manifest(
            synthetic_manifest,
            GSVLabelIndex(self.root / "Dataframes") if synthetic_manifest is not None else None,
        )
        self.transform = build_transform(image_size)
        self.seed = int(seed)
        self.return_mix_metadata = bool(return_mix_metadata)
        # Workers share the epoch number, including under spawn and persistent
        # workers. Each worker reconstructs the same quota before reading a place.
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._planned_epoch = None

        frames = []
        for city in self.cities:
            df = pd.read_csv(self.root / "Dataframes" / f"{city}.csv", dtype={"place_id": str})
            if not (df["city_id"] == city).all():
                raise ValueError(f"GSV dataframe {city}.csv has inconsistent city_id values")
            df["place_id"] = df["place_id"].map(canonical_place_id)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        df = df[df.groupby(["city_id", "place_id"])["place_id"].transform("size") >= self.min_k]
        self.groups = {
            (str(city), canonical_place_id(pid)): group.copy()
            for (city, pid), group in df.groupby(["city_id", "place_id"])
        }
        self.keys = sorted(self.groups)
        if not self.keys:
            raise ValueError("no GSV places have the requested number of real images")
        self.label_map = {key: i for i, key in enumerate(self.keys)}
        self.shared_capacities = None
        if shared_mix_plan is not None:
            plan = json.loads(Path(shared_mix_plan).read_text())
            self.shared_capacities = {(r["city"], canonical_place_id(r["place_id"])): r["capacity"]
                                      for r in plan["capacities"]}
            if plan["total_places"] != len(self.keys) or plan["seed"] != self.seed:
                raise ValueError("shared B/C exposure plan does not match real places or seed")
            if plan["requested_ratio"] != [self.real_ratio, self.synth_ratio]:
                raise ValueError("shared B/C exposure plan ratio mismatch")
            for key, capacity in self.shared_capacities.items():
                if (key not in self.groups or not isinstance(capacity, int) or capacity < 0
                        or capacity > min(self.k, len(self.synthetic.get(key, [])))):
                    raise ValueError("shared synthetic capacity exceeds verified pool")
        self._synthetic_quota = {}
        self._mix_stats = {}
        self.set_epoch(0)

    def _plan_epoch_mix(self):
        total_slots = len(self.keys) * self.k
        target_synth = round(
            total_slots * self.synth_ratio / (self.real_ratio + self.synth_ratio)
        )
        capacities = {
            key: (self.shared_capacities.get(key, 0) if self.shared_capacities is not None
                  else min(self.k, len(self.synthetic.get(key, []))))
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
        self._planned_epoch = self.epoch
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
        self._ensure_epoch_plan()
        return dict(self._mix_stats)

    @property
    def epoch(self):
        return int(self._shared_epoch.item())

    def _ensure_epoch_plan(self):
        if self._planned_epoch != self.epoch:
            self._plan_epoch_mix()

    def set_epoch(self, epoch: int):
        self._shared_epoch.fill_(int(epoch))
        if hasattr(self, "keys"):
            self._ensure_epoch_plan()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self._ensure_epoch_plan()
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
        item = (torch.stack(images), torch.full((self.k,), label, dtype=torch.long))
        if self.return_mix_metadata:
            return (*item, {"synthetic_slots": synth_slots, "real_slots": self.k - synth_slots,
                           "epoch": self.epoch})
        return item
