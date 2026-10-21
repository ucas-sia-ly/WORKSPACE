#!/usr/bin/env python3
"""Read-only probe: how much do generated vs real weather changes hurt descriptors?

For a generated image G of real source S, ``drop = cos(S, R) - cos(G, R)`` where R
is another real GSV view of the same place. For an SVOX-train weather query W,
``drop = cos(D, P) - cos(W, P)`` where D is the nearest daytime query (<= 5 m) and
P is W's nearest gallery positive. Both measure how far the weather change alone
moves the descriptor away from a fixed reference of the same place. The
``day`` control replaces W by a different daytime query 2-10 m from D, so it
measures capture-to-capture variation without any weather change.

Cosines are on L2-normalized descriptors of one frozen checkpoint. This is a
diagnostic of edit strength, not a training signal; it never reads SVOX test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]


def quantiles(values):
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1)))]
    return dict(n=len(values), mean=statistics.fmean(values), p10=pick(.1), median=pick(.5), p75=pick(.75),
                p90=pick(.9), above_0_1=sum(v > .1 for v in values) / len(values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "salad/checkpoint/dino_salad.ckpt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/qwen_curriculum/state/training_manifest.jsonl")
    parser.add_argument("--svox-root", type=Path, default=ROOT / "dataset/svox")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/qwen_curriculum/state/analysis/difficulty_probe")
    parser.add_argument("--backbone-repo", type=Path, default=Path("/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main"))
    parser.add_argument("--per-condition", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from workflow.evaluation import EvaluationImage, _images_in, _utm, extract_descriptors
    from workflow.model import load_checkpoint_model

    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("passed") is True and r.get("eligible_for_training") is True]
    # v3 rows carry their prompt variant; calibrate variants, not only domains.
    group = lambda row: f"{row['condition']}/{row['prompt_variant']}" if row.get("prompt_variant") else row["condition"]
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[group(row)].append(row)
    chosen = []
    for condition, items in sorted(by_condition.items()):
        rng.shuffle(items)
        chosen += items[:args.per_condition]
    images_dir = ROOT / "dataset/gsv-cities/Images"
    triples = []
    for row in chosen:
        source = Path(row["source_path"])
        prefix = f"{row['city']}_{int(row['place_id']):07d}_"
        others = sorted(p for p in (images_dir / row["city"]).glob(prefix + "*") if p != source)
        if others:
            triples.append((row, source, rng.choice(others)))

    model = load_checkpoint_model(args.checkpoint, device="cuda", backbone_repo=args.backbone_repo)
    image_size = tuple(model.image_size)

    def embed(paths):
        unique = sorted({str(p) for p in paths})
        descriptors = extract_descriptors(model, [EvaluationImage(p, Path(p)) for p in unique], image_size,
                                          "cuda", batch_size=64, num_workers=4)
        descriptors = torch.nn.functional.normalize(descriptors, dim=1)
        return {p: descriptors[i] for i, p in enumerate(unique)}

    vectors = embed([p for row, s, r in triples for p in (s, r, Path(row["output_path"]))])
    cos = lambda a, b: float(vectors[str(a)] @ vectors[str(b)])
    generated = defaultdict(lambda: defaultdict(list))
    for row, source, reference in triples:
        output = Path(row["output_path"])
        for key in {group(row), row["condition"] + "/all"}:
            generated[key]["drop"].append(cos(source, reference) - cos(output, reference))
            generated[key]["cos_to_source"].append(cos(output, source))

    root = args.svox_root.resolve()
    gallery = _images_in(root / "images/train/gallery", root)
    day = _images_in(root / "images/train/queries", root)
    gallery_tree, day_tree = cKDTree([_utm(i) for i in gallery]), cKDTree([_utm(i) for i in day])
    pairs = defaultdict(list)
    for domain in ("night", "rain", "snow", "sun", "overcast"):
        for query in _images_in(root / f"images/train/queries_{domain}", root):
            xy = _utm(query)
            dg, gi = gallery_tree.query(xy)
            dd, di = day_tree.query(xy)
            if dg <= 25 and dd <= 5:
                pairs[domain].append((query, day[di], gallery[gi]))
    for query in rng.sample(day, min(1500, len(day))):
        xy = _utm(query)
        dg, gi = gallery_tree.query(xy)
        near = [day[i] for i in day_tree.query_ball_point(xy, 10)
                if 2 <= math.dist(_utm(day[i]), xy)]
        if dg <= 25 and near:
            pairs["day"].append((rng.choice(near), query, gallery[gi]))
    vectors.update(embed([i.path for items in pairs.values() for t in items for i in t]))
    real = {}
    for domain, items in pairs.items():
        drops = [cos(d.path, p.path) - cos(w.path, p.path) for w, d, p in items]
        real[domain] = dict(drop=quantiles(drops))

    report = dict(checkpoint=str(args.checkpoint), definition=__doc__.split("\n\n")[1].strip(),
                  generated={c: {k: quantiles(v) for k, v in m.items()} for c, m in generated.items()},
                  real_svox_train=real)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = ["# Weather-induced descriptor drop", "", report["definition"], "",
             "| Source | Condition | n | median drop | p75 | p90 | drop > 0.1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for condition, metrics in sorted(report["generated"].items()):
        q = metrics["drop"]
        lines.append(f"| generated | {condition} | {q['n']} | {q['median']:.3f} | {q['p75']:.3f} | {q['p90']:.3f} | {100*q['above_0_1']:.1f}% |")
    for domain, metrics in sorted(real.items()):
        q = metrics["drop"]
        lines.append(f"| SVOX real | {domain} | {q['n']} | {q['median']:.3f} | {q['p75']:.3f} | {q['p90']:.3f} | {100*q['above_0_1']:.1f}% |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
