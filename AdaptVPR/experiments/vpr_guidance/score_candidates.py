"""Score verified candidates with the current SALAD student and select per group.

The manifest is directly consumable by SALAD's MixedGSVCitiesDataset. Scoring
uses the same metadata place labels and path resolver as training. Utility is
an expectation over real-only, place-grouped batch draws; synthetic co-anchors
and stochastic training image augmentation are not simulated.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from common import file_sha256, use_salad, write_json, write_jsonl

use_salad()

import torch  # noqa: E402

from feedback import identity_margin, plausibility_floor, score_candidates, select_per_group  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True,
                        help="candidates.jsonl from generate_candidates.py")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Current SALAD student checkpoint")
    parser.add_argument("--real-data", type=Path, required=True, help="GSV-Cities root (Images/, Dataframes/)")
    parser.add_argument("--cities", nargs="+", help="Must match the cities used for SALAD training")
    parser.add_argument("--min-images-per-place", type=int, default=4)
    parser.add_argument("--selection", choices=["hardness", "random"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-batch-size", type=int, default=32, help="Places per SALAD training batch")
    parser.add_argument("--images-per-place", type=int, default=4)
    parser.add_argument("--negative-pool-size", type=int, default=4096,
                        help="Number of real training places; each supplies images-per-place distinct views")
    parser.add_argument("--negative-draws", type=int, default=16)
    parser.add_argument("--plausibility-quantile", type=float, default=0.025,
                        help="Exclude candidates whose identity margin is below this quantile of "
                             "leave-one-out real-view margins (0 disables the gate)")
    parser.add_argument("--calibration-places", type=int, default=300,
                        help="Real training places scored leave-one-out to calibrate the gate")
    parser.add_argument("--miner-margin", type=float, default=0.1, help="Same as train_salad --miner-margin")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--base", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone-repo", type=Path)
    args = parser.parse_args(argv)
    for name in ("negative_pool_size", "negative_draws", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.train_batch_size < 2 or args.images_per_place < 2:
        parser.error("--train-batch-size and --images-per-place must be at least 2")
    if args.min_images_per_place < args.images_per_place:
        parser.error("--min-images-per-place must be at least --images-per-place")
    if not 0 <= args.plausibility_quantile < 1 or args.calibration_places <= 0:
        parser.error("--plausibility-quantile must be in [0, 1) and --calibration-places positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and positive")
    if not math.isfinite(args.base) or not math.isfinite(args.miner_margin) or args.miner_margin < 0:
        parser.error("--base must be finite and --miner-margin must be finite and nonnegative")
    return args


def _distribution(values):
    if not values:
        return None
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def load_verified_candidates(manifests, real_data, place_of, *, real_paths=None):
    """Resolve files once, reject conflicting identities, and deduplicate outputs."""
    from workflow.training_data import _resolve_manifest_file

    verified, unusable = [], defaultdict(int)
    outputs, identities, groups = {}, {}, {}
    total = 0
    for input_path in manifests:
        manifest = Path(input_path).expanduser().resolve()
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                total += 1
                location = f"{manifest}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: malformed JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{location}: candidate entry must be an object")
                if row.get("passed") is not True:
                    unusable["rejected_by_verifier"] += 1
                    continue
                sample_id, index = row.get("sample_id"), row.get("candidate_index")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(f"{location}: verified candidate requires a nonempty sample_id")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError(f"{location}: candidate_index must be a nonnegative integer")
                if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                    raise ValueError(f"{location}: verified candidate requires a nonempty prompt")
                if row.get("condition") is not None and not isinstance(row["condition"], str):
                    raise ValueError(f"{location}: condition must be a string when provided")
                if "s_geo" in row and (isinstance(row["s_geo"], bool) or
                        not isinstance(row["s_geo"], (int, float)) or not math.isfinite(row["s_geo"])):
                    raise ValueError(f"{location}: s_geo must be a finite number when provided")
                paths = {
                    name: _resolve_manifest_file(row.get(name), manifest=manifest,
                                                  line_number=line_number, field_name=name,
                                                  real_data=Path(real_data).expanduser().resolve())
                    for name in ("source_path", "output_path")
                }
                key = place_of.get(paths["source_path"])
                if key is None:
                    unusable["source_not_a_training_place"] += 1
                    continue
                if (paths["output_path"] in (real_paths if real_paths is not None else place_of) or
                        paths["output_path"] == paths["source_path"]):
                    raise ValueError(f"{location}: output_path must be a generated image, not a real GSV image")
                row = {**row, **{name: str(path) for name, path in paths.items()}, "_place": key}
                group = (row["source_path"], row.get("condition"), row["prompt"])
                if sample_id in groups and groups[sample_id] != group:
                    raise ValueError(f"{location}: sample_id has conflicting source, condition, or prompt")
                groups[sample_id] = group
                identity = (sample_id, index)
                if identity in identities and identities[identity] != paths["output_path"]:
                    raise ValueError(f"{location}: candidate identity refers to multiple output files")
                identities[identity] = paths["output_path"]
                if paths["output_path"] in outputs:
                    if outputs[paths["output_path"]] != row:
                        raise ValueError(f"{location}: duplicate output has conflicting candidate metadata")
                    unusable["duplicate_candidates"] += 1
                    continue
                outputs[paths["output_path"]] = row
                verified.append(row)
    verified.sort(key=lambda row: (row["sample_id"], row["candidate_index"]))
    return verified, dict(unusable), total


def sample_negative_pool(real_views, size, images_per_place, seed):
    """Sample places uniformly, then K distinct real views per sampled place."""
    rng = random.Random(seed)
    keys = rng.sample(sorted(real_views), min(size, len(real_views)))
    return [(key, rng.sample(real_views[key], images_per_place)) for key in keys]


def main(argv=None):
    args = parse_args(argv)
    from workflow.evaluation import extract_descriptors
    from workflow.model import load_checkpoint_model
    from workflow.training_data import MixedGSVCitiesDataset

    dataset = MixedGSVCitiesDataset(
        args.real_data, None, cities=args.cities, images_per_place=args.images_per_place,
        min_images_per_place=args.min_images_per_place, augment=False)
    if len(dataset) < 2:
        raise ValueError("Need at least two eligible training places to define negatives")
    place_of = {path: (p.city, p.place_id) for p in dataset.places for path in p.real_paths}
    real_views = {(p.city, p.place_id): p.real_paths for p in dataset.places}
    verified, unusable, num_candidates = load_verified_candidates(
        args.candidates, args.real_data, place_of,
        real_paths=dataset.source_index)
    effective_batch_size = min(args.train_batch_size, len(dataset))
    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "candidate_manifests": [{"path": str(path.expanduser().resolve()), "sha256": file_sha256(path)}
                                for path in args.candidates],
        "selection": args.selection,
        "candidates": num_candidates,
        "verified_training_candidates": len(verified),
        "unusable": unusable,
        "score_settings": {
            "real_data": str(args.real_data.expanduser().resolve()), "cities": dataset.cities,
            "train_batch_size": args.train_batch_size, "images_per_place": args.images_per_place,
            "effective_train_batch_size": effective_batch_size,
            "min_images_per_place": args.min_images_per_place,
            "negative_pool_size_requested": args.negative_pool_size,
            "negative_draws": args.negative_draws, "miner_margin": args.miner_margin,
            "alpha": args.alpha, "base": args.base, "seed": args.seed,
            "utility": "expected_mined_ms_positive_term",
            "plausibility_quantile": args.plausibility_quantile,
            "calibration_places": args.calibration_places,
            "batch_context": "real_only_fixed_negative_views_per_place",
            "positive_views_per_draw": args.images_per_place - 1,
            "training_augmentation_simulated": False,
            "partial_tail_batches_simulated": False,
        },
    }
    negatives_per_batch = (effective_batch_size - 1) * args.images_per_place
    pool = sample_negative_pool(real_views, args.negative_pool_size, args.images_per_place, args.seed)
    pool_keys = [key for key, _ in pool]
    # Validate the same-place exclusion before loading a potentially large model.
    if verified and any(sum(key != row["_place"] for key in pool_keys) < effective_batch_size - 1
                        for row in verified):
        raise ValueError("Negative pool lacks enough other places; increase --negative-pool-size")
    gate = args.plausibility_quantile > 0
    calibration_keys = (random.Random(args.seed + 1).sample(sorted(real_views),
                                                            min(args.calibration_places, len(real_views)))
                        if gate else [])
    scored, floor, calibration = [], None, None
    if verified:
        images = {}
        for row in verified:
            images.setdefault(Path(row["output_path"]), None)
            for path in real_views[row["_place"]]:
                images.setdefault(path, None)
        for key in calibration_keys:
            for path in real_views[key]:
                images.setdefault(path, None)
        for _, views in pool:
            for path in views:
                images.setdefault(path, None)
        paths = list(images)
        model = load_checkpoint_model(args.checkpoint, args.device, backbone_repo=args.backbone_repo)
        descriptors = extract_descriptors(model, [SimpleNamespace(path=p) for p in paths], model.image_size,
                                          args.device, args.batch_size, args.num_workers)
        index = {path: i for i, path in enumerate(paths)}
        pool_desc = torch.stack([descriptors[[index[p] for p in views]] for _, views in pool])

        def score(anchors, positive_sets, places):
            same = torch.tensor([[place == key for key in pool_keys] for place in places])
            return score_candidates(anchors, positive_sets, pool_desc, same, negatives_per_batch,
                                    args.negative_draws, args.alpha, args.base, args.miner_margin, args.seed,
                                    positives_per_batch=args.images_per_place - 1)

        scores = score(descriptors[[index[Path(r["output_path"])] for r in verified]],
                       [descriptors[[index[p] for p in real_views[r["_place"]]]] for r in verified],
                       [r["_place"] for r in verified])
        if gate:
            # Leave-one-out: each real view is an anchor against its place's other real views.
            anchors, positive_sets, places = [], [], []
            for key in calibration_keys:
                views = real_views[key]
                for i, view in enumerate(views):
                    anchors.append(index[view])
                    positive_sets.append(descriptors[[index[p] for j, p in enumerate(views) if j != i]])
                    places.append(key)
            real_scores = score(descriptors[anchors], positive_sets, places)
            floor = plausibility_floor(real_scores, args.plausibility_quantile)
            calibration = {
                "places": len(calibration_keys), "real_anchors": len(real_scores),
                "quantile": args.plausibility_quantile, "margin_floor": floor,
                "real_mined_rate": statistics.fmean(s["mining_probability"] for s in real_scores),
                "real_margin": _distribution([identity_margin(s) for s in real_scores]),
            }
        scored = [{**{k: v for k, v in row.items() if k != "_place"}, **s,
                   "identity_margin": identity_margin(s),
                   "plausible": floor is None or identity_margin(s) >= floor,
                   "eligible_for_training": True} for row, s in zip(verified, scores)]
    selected = select_per_group(scored, args.selection, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "scored.jsonl", scored)
    write_jsonl(args.output_dir / "selected.jsonl", selected)
    all_u = [r["utility"] for r in scored]
    sel_u = [r["utility"] for r in selected]
    by_condition = defaultdict(list)
    for row in scored:
        by_condition[row.get("condition")].append(row["mining_probability"])
    summary.update({
        "status": "complete" if scored else "no_verified_training_candidates",
        "plausibility_calibration": calibration,
        "implausible_verified": sum(not r["plausible"] for r in scored),
        # Would the ungated hardness rule have picked an implausible image in that group?
        "groups_where_gate_changed_hardest": sum(
            1 for group in {r["sample_id"] for r in scored}
            if not max((r for r in scored if r["sample_id"] == group),
                       key=lambda r: (r["utility"], -r["mean_positive_similarity"]))["plausible"]),
        "groups_selected": len(selected),
        "negatives_per_batch": negatives_per_batch,
        "negative_places_per_batch": effective_batch_size - 1,
        "negative_pool_size": len(pool),
        "negative_pool_images": len(pool) * args.images_per_place,
        "miner_margin": args.miner_margin,
        # Expected probability of mining a positive in one sampled batch.
        "mined_rate_all_verified": statistics.fmean(r["mining_probability"] for r in scored) if scored else 0.0,
        "mined_rate_selected": statistics.fmean(r["mining_probability"] for r in selected) if selected else 0.0,
        "candidates_mined_in_any_draw": sum(r["mining_probability"] > 0 for r in scored),
        "mined_rate_by_condition": {str(k): statistics.fmean(v) for k, v in by_condition.items()},
        "utility_all_verified": _distribution(all_u), "utility_selected": _distribution(sel_u),
        "mean_positive_similarity_selected": _distribution([r["mean_positive_similarity"] for r in selected]),
        "s_geo_selected": _distribution([r["s_geo"] for r in selected if "s_geo" in r]),
    })
    write_json(args.output_dir / "summary.json", summary)
    print(f"[score] verified={len(scored)} groups={len(selected)} "
          f"mined_probability(all)={summary['mined_rate_all_verified']:.3f} "
          f"mean_U(all)={statistics.fmean(all_u) if all_u else 0:.4f} "
          f"mean_U(selected)={statistics.fmean(sel_u) if sel_u else 0:.4f}")


if __name__ == "__main__":
    main()
