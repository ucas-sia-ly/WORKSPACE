"""Run A/B/C x SVOX, RobotCar-Seasons, Nordland and write comparable summaries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from .data import atomic_json
from .real_data import load_real_split


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("a-checkpoint", "b-checkpoint", "c-checkpoint", "salad-root", "svox-root",
                 "robotcar-root", "nordland-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/full_run/eval"))
    for name in ("svox-metadata", "robotcar-metadata", "nordland-metadata"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--image-size", type=int, default=322)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--faiss-threads", type=int, default=8)
    p.add_argument("--positive-radius", type=float, default=25.)
    p.add_argument("--frame-tolerance", type=int, default=10)
    return p


def summary_row(variant, results):
    labels = {"A": "A Real", "B": "B Original synthetic", "C": "C VPR-aware synthetic"}
    row = {"Train": labels[variant]}
    for dataset, title in (("svox", "SVOX"), ("robotcar-seasons", "RobotCar")):
        conditions = results[dataset]["recall_metrics"]["conditions"]
        for condition, metrics in conditions.items():
            for k in (1, 5, 10):
                row[f"{title} {condition.title()} R@{k}"] = metrics[f"R@{k}"]
        if dataset == "robotcar-seasons":
            others = [m for c, m in conditions.items() if not c.startswith("night")]
            for k in (1, 5, 10):
                row[f"RobotCar Other R@{k}"] = sum(m[f"R@{k}"] for m in others) / len(others) if others else None
        for k in (1, 5, 10):
            row[f"{title} Macro R@{k}"] = results[dataset]["recall_metrics"]["macro_average"][f"R@{k}"]
    for k in (1, 5, 10):
        row[f"Nordland R@{k}"] = results["nordland"]["recall_metrics"]["overall"][f"R@{k}"]
    return row


def main():
    args = args_parser().parse_args()
    datasets = {"svox": (args.svox_root, args.svox_metadata),
                "robotcar-seasons": (args.robotcar_root, args.robotcar_metadata),
                "nordland": (args.nordland_root, args.nordland_metadata)}
    # Check every dataset before expensive descriptor extraction starts.
    for dataset, (root, metadata) in datasets.items():
        load_real_split(SimpleNamespace(dataset=dataset, dataset_root=root, salad_root=args.salad_root,
                                        metadata=metadata, reference_dir=None, query_dirs=None,
                                        positive_radius=args.positive_radius, frame_tolerance=args.frame_tolerance))
    checkpoints = {"A": args.a_checkpoint, "B": args.b_checkpoint, "C": args.c_checkpoint}
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant, checkpoint in checkpoints.items():
        results[variant] = {}
        for dataset, (root, metadata) in datasets.items():
            output = out / f"{variant}_{dataset}.json"
            command = [sys.executable, "-u", "-m", "AdaptVPR.experiments.vpr_guidance.evaluate_real",
                       "--checkpoint", str(checkpoint), "--salad-root", str(args.salad_root),
                       "--dataset", dataset, "--dataset-root", str(root), "--output", str(output),
                       "--image-size", str(args.image_size), "--batch-size", str(args.batch_size),
                       "--workers", str(args.workers), "--faiss-threads", str(args.faiss_threads),
                       "--positive-radius", str(args.positive_radius), "--frame-tolerance", str(args.frame_tolerance)]
            if metadata:
                command += ["--metadata", str(metadata)]
            subprocess.run(command, check=True)
            results[variant][dataset] = json.loads(output.read_text())
    for dataset in datasets:
        baseline = results["A"][dataset]
        for variant in ("B", "C"):
            other = results[variant][dataset]
            for key in ("protocol", "split_sha256", "preprocessing", "retrieval", "architecture"):
                if baseline[key] != other[key]:
                    raise ValueError(f"A/B/C {dataset} {key} mismatch")
    rows = [summary_row(variant, results[variant]) for variant in checkpoints]
    differences = {}
    for condition, b_metrics in results["B"].items():
        differences[condition] = {
            compare: {k: results[right][condition]["recall_metrics"]["macro_average"][k] -
                         results[left][condition]["recall_metrics"]["macro_average"][k]
                      for k in ("R@1", "R@5", "R@10")}
            for compare, left, right in (("B_minus_A", "A", "B"), ("C_minus_B", "B", "C"))}
    atomic_json(out / "evaluation_summary.json", {"results": results, "table": rows,
                                                   "macro_deltas_percentage_points": differences,
                                                   "robotcar_other_definition": "condition macro-average excluding night and night-rain"})
    required = ["Train", "SVOX Night R@1", "SVOX Rain R@1", "SVOX Snow R@1",
                "RobotCar Night R@1", "RobotCar Other R@1", "Nordland R@1"]
    fields = required + sorted(set().union(*(row.keys() for row in rows)) - set(required))
    with (out / "evaluation_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evaluation summaries: {out}")


if __name__ == "__main__":
    main()
