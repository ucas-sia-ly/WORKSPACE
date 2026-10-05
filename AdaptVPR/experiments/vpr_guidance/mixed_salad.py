"""GSV-Cities + verified synthetic place dataset for SALAD-style training.

Each item is one geographical place with K images.  A controlled fraction of
image slots is replaced by verified generated images from the same place.  This
preserves SALAD's original metric-learning labels while exposing it to domain
hard positives.
"""
from __future__ import annotations

import json, random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN_STD = {"mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]}


def build_transform(image_size=(322,322)):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN_STD["mean"], std=IMAGENET_MEAN_STD["std"]),
    ])


def _gsv_name(row, place_id):
    city=row["city_id"]; pid=str(int(place_id)).zfill(7); pano=row["panoid"]
    year=str(row["year"]).zfill(4); month=str(row["month"]).zfill(2); north=str(row["northdeg"]).zfill(3)
    return f"{city}_{pid}_{year}_{month}_{north}_{row['lat']}_{row['lon']}_{pano}.jpg"


def load_synthetic_manifest(path: Path):
    by_place={}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        if not row.get("passed") or not row.get("eligible_for_training"):
            continue
        key=(str(row["city"]), str(row["place_id"]).zfill(7))
        gen=Path(row["generated_path"])
        if not gen.is_file(): raise FileNotFoundError(gen)
        by_place.setdefault(key, []).append(gen)
    return by_place


class MixedGSVCitiesDataset(Dataset):
    def __init__(self, gsv_root: Path, synthetic_manifest: Path, cities, img_per_place=4,
                 min_img_per_place=4, real_to_synth=(8,1), image_size=(322,322), seed=42):
        self.root=Path(gsv_root); self.cities=list(cities); self.k=img_per_place; self.min_k=min_img_per_place
        self.real_ratio, self.synth_ratio = real_to_synth
        if self.real_ratio < 0 or self.synth_ratio < 0 or self.real_ratio+self.synth_ratio <= 0:
            raise ValueError("invalid real_to_synth ratio")
        self.synthetic=load_synthetic_manifest(synthetic_manifest)
        self.transform=build_transform(image_size); self.seed=seed; self.epoch=0
        frames=[]
        for city in self.cities:
            df=pd.read_csv(self.root/"Dataframes"/f"{city}.csv")
            df["_city"] = city; frames.append(df)
        df=pd.concat(frames, ignore_index=True)
        df=df[df.groupby(["city_id","place_id"])["place_id"].transform("size") >= self.min_k]
        self.groups={(str(city), str(int(pid)).zfill(7)): group.copy()
                     for (city,pid),group in df.groupby(["city_id","place_id"])}
        self.keys=sorted(self.groups)

    def set_epoch(self, epoch:int): self.epoch=int(epoch)
    def __len__(self): return len(self.keys)

    def __getitem__(self, index):
        key=self.keys[index]; group=self.groups[key]
        rng=random.Random((self.seed+1)*1000003 + self.epoch*10000019 + index)
        rows=[r for _,r in group.iterrows()]; rng.shuffle(rows)
        selected=rows[:self.k]
        paths=[self.root/"Images"/r["city_id"]/_gsv_name(r, int(r["place_id"])) for r in selected]
        synth_pool=self.synthetic.get(key, [])
        # Per-place stochastic replacement yields the requested dataset-level ratio
        # while keeping the PxK structure expected by SALAD.
        p_synth=self.synth_ratio/(self.real_ratio+self.synth_ratio)
        for i in range(self.k):
            if synth_pool and rng.random() < p_synth:
                paths[i]=synth_pool[rng.randrange(len(synth_pool))]
        images=[]
        for path in paths:
            with Image.open(path) as im: images.append(self.transform(im.convert("RGB")))
        # Stable city-aware integer label; only equality inside a batch matters for SALAD loss.
        label=self.keys.index(key)
        return torch.stack(images), torch.full((self.k,), label, dtype=torch.long)
