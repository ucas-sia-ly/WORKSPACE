"""Validate an experimental weather threshold against existing visual reviews.

Example, requiring no CLIP/GPU inference::

    python AdaptVPR/experiments/generation_diagnosis/calibrate_weather_signal.py \
      --reviews outputs/gen_diagnosis/manual_reviews_iclight.jsonl \
      --legacy-probe outputs/gen_diagnosis/weather_clip_probe.json \
      --out outputs/gen_diagnosis/weather_calibration

Omit --legacy-probe/--scores to recompute with the existing evaluator's CLIP.
Every review is joined to its frozen generator row and checked against actual
source/output bytes. The old probe has only ordered scores, so importing it
cannot establish that its embeddings came from those bytes; this limitation is
carried into every imported row and the report. Labels are exploratory visual
reviews, not independent weather or geometry ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ADAPTVPR_ROOT = Path(__file__).resolve().parents[2]
if str(ADAPTVPR_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTVPR_ROOT))
from experiments.qwen_curriculum.common import file_sha256, read_jsonl, use_adaptvpr, write_json, write_jsonl
from experiments.qwen_curriculum.weather_signal import (
    WEATHER_TEXT, WeatherSignal,
    WeatherSignalEvaluator, weather_definition,
)

# Historical IC-Light diagnosis only; QwenQualityVerifier has its own explicitly
# experimental per-condition settings and does not consume this threshold.
EXPLORATORY_MIN_WEATHER_SHIFT = 6.0


def fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _image_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def validate_reviews(path: Path) -> tuple[list[dict], dict]:
    """Require immutable generator provenance before considering any label."""
    reviews, joined, seen, runs = read_jsonl(path), [], set(), {}
    image_hashes = {}
    for review in reviews:
        condition, label = review.get("condition"), review.get("weather_present")
        if condition not in WEATHER_TEXT or label not in {"yes", "weak", "no", "uncertain"}:
            raise ValueError("Review contains unsupported condition or weather_present label")
        run_dir = _image_path(review["run"], path.parent)
        if run_dir not in runs:
            config_path, manifest_path = run_dir / "generation_config.json", run_dir / "results.jsonl"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("fingerprint") != fingerprint({k: v for k, v in config.items() if k != "fingerprint"}):
                raise ValueError(f"Invalid generation config fingerprint: {config_path}")
            records = {}
            for row in read_jsonl(manifest_path):
                # A checkpoint can contain a retried row; the latest is the frozen
                # record used by the report and visual-review producer.
                records[_image_path(row["output_path"], run_dir)] = row
            runs[run_dir] = (config, records, {
                "config_path": str(config_path), "config_sha256": file_sha256(config_path),
                "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
            })
        config, records, _ = runs[run_dir]
        source = _image_path(review["source_path"], path.parent)
        output = _image_path(review["output_path"], run_dir)
        if output not in records:
            raise ValueError(f"Review output is absent from generator checkpoint: {output}")
        row = records[output]
        sources = {int(item["src"]): item for item in config["sources"]}
        frozen_source = sources[int(review["source_index"])]
        if (_image_path(frozen_source["source_path"], run_dir) != source
                or _image_path(row["source_path"], run_dir) != source
                or row.get("src") != review["source_index"]
                or row.get("cond") != condition or row.get("method") != review.get("method")
                or row.get("seed") != review.get("seed")
                or row.get("status") != "ok"
                or row.get("config_fingerprint") != config["fingerprint"]
                or review.get("config_fingerprint") != config["fingerprint"]
                or type(review.get("metric_passed")) is not bool
                or row.get("passed") is not review["metric_passed"]):
            raise ValueError(f"Review/generator identity mismatch: {output}")
        for image_path in (source, output):
            if image_path not in image_hashes:
                with Image.open(image_path) as image:
                    image.verify()
                image_hashes[image_path] = file_sha256(image_path)
        source_hash, output_hash = image_hashes[source], image_hashes[output]
        if (source_hash != frozen_source["source_sha256"]
                or output_hash != review.get("output_sha256")
                or output_hash != row.get("output_sha256")):
            raise ValueError(f"Reviewed image checksum differs from frozen generator artifacts: {output}")
        key = (source_hash, output_hash, condition)
        if key in seen:
            raise ValueError(f"Duplicate reviewed source/output/condition: {output}")
        seen.add(key)
        joined.append({
            **review, "source_path": str(source), "output_path": str(output),
            "source_sha256": source_hash, "output_sha256": output_hash,
        })
    if not joined:
        raise ValueError("No reviews to calibrate")
    return joined, {
        "reviews_path": str(path), "reviews_sha256": file_sha256(path),
        "review_label_counts": dict(Counter(row["weather_present"] for row in joined)),
        "runs": [value[2] for _, value in sorted(runs.items())],
        "image_count": len(image_hashes),
    }


def _check_scores(row: dict) -> None:
    for field in ("weather_source_logit", "weather_generated_logit", "weather_shift"):
        if type(row.get(field)) not in (float, int) or not math.isfinite(row[field]):
            raise ValueError(f"Invalid/non-finite {field} in weather score cache")
    if not math.isclose(row["weather_shift"], row["weather_generated_logit"] - row["weather_source_logit"],
                        rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Cached weather_shift does not match its source/generated logits")


def join_scores(reviews: list[dict], *, legacy_probe: Path | None = None,
                scores_path: Path | None = None) -> tuple[list[dict], dict]:
    eligible = [row for row in reviews if row["weather_present"] != "uncertain"]
    definition = weather_definition()
    if legacy_probe is not None:
        scores = json.loads(legacy_probe.read_text(encoding="utf-8"))
        if not isinstance(scores, list) or len(scores) != len(eligible):
            raise ValueError("Legacy ordered probe count differs from non-uncertain reviews")
        if definition["weather_signal_model"] != "openai/clip-vit-base-patch32":
            raise ValueError("Legacy probe was computed with openai/clip-vit-base-patch32")
        provenance = "legacy_ordered_import_unverified_embedding_provenance"
        joined = []
        for review, score in zip(eligible, scores):
            if (score.get("cond") != review["condition"] or score.get("label") != review["weather_present"]
                    or score.get("passed") is not review["metric_passed"]):
                raise ValueError("Legacy ordered probe does not match review order/labels/metric verdicts")
            joined.append({**review, "weather_source_logit": score["src"],
                           "weather_generated_logit": score["gen"],
                           "weather_shift": score["gen"] - score["src"],
                           "weather_signal_model": definition["weather_signal_model"],
                           "weather_signal_definition": definition,
                           "weather_score_provenance": provenance})
        metadata = {
            "mode": provenance, "path": str(legacy_probe), "sha256": file_sha256(legacy_probe),
            "limitation": "Legacy scores have no image hashes/IDs. Ordered label/condition/verdict agreement cannot verify embedding provenance.",
        }
    elif scores_path is not None:
        cache = read_jsonl(scores_path)
        keyed = {}
        for row in cache:
            key = (row["source_sha256"], row["output_sha256"], row["condition"])
            if key in keyed:
                raise ValueError("Duplicate source/output/condition in hash-keyed weather score cache")
            cached_definition = row.get("weather_signal_definition")
            # Older historical caches included an exploratory IC-Light threshold
            # in the shared signal metadata. It did not affect their embeddings;
            # compare the unchanged model/text/scale definitions instead.
            canonical_definition = ({key: value for key, value in cached_definition.items()
                                     if key not in {"exploratory_min_weather_shift", "threshold_status"}}
                                    if isinstance(cached_definition, dict) else None)
            if canonical_definition != definition:
                raise ValueError("Weather score cache signal definition differs")
            if row.get("weather_score_provenance") not in {
                "recomputed_on_hash_verified_images", "legacy_ordered_import_unverified_embedding_provenance"
            }:
                raise ValueError("Weather score cache omits known embedding provenance")
            keyed[key] = row
        joined = []
        for review in eligible:
            key = (review["source_sha256"], review["output_sha256"], review["condition"])
            if key not in keyed:
                raise ValueError("Reviewed image is missing from hash-keyed weather score cache")
            score = keyed[key]
            joined.append({**review, **{field: score[field] for field in (
                "weather_source_logit", "weather_generated_logit", "weather_shift", "weather_signal_model",
                "weather_signal_definition", "weather_score_provenance",
            )}})
        metadata = {"mode": "hash_keyed_cache", "path": str(scores_path), "sha256": file_sha256(scores_path),
                    "embedding_provenance": sorted({row["weather_score_provenance"] for row in joined})}
    else:
        evaluator = WeatherSignalEvaluator()
        signal, joined = WeatherSignal(evaluator), []
        for review in eligible:
            with Image.open(review["source_path"]) as image:
                source = image.convert("RGB")
            with Image.open(review["output_path"]) as image:
                output = image.convert("RGB")
            measurements = signal.measure(source, output, review["condition"])
            joined.append({**review, **measurements, "weather_signal_definition": signal.definition,
                           "weather_score_provenance": "recomputed_on_hash_verified_images"})
        metadata = {"mode": "recomputed_on_hash_verified_images", "signal_definition": signal.definition}
    for row in joined:
        _check_scores(row)
    if not joined:
        raise ValueError("No non-uncertain reviews for weather calibration")
    return joined, metadata


def auc(labels: list[bool], scores: list[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def verdict_metrics(rows: list[dict], keep: list[bool], *, strict: bool = False) -> dict:
    positive = [row["weather_present"] == "yes" if strict else row["weather_present"] != "no" for row in rows]
    tp = sum(k and p for k, p in zip(keep, positive))
    fp = sum(k and not p for k, p in zip(keep, positive))
    fn = sum(not k and p for k, p in zip(keep, positive))
    tn = sum(not k and not p for k, p in zip(keep, positive))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {
        "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": ratio(tp, tp + fp), "recall": recall, "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "keep_rate": ratio(sum(keep), len(rows)),
        "keep_rate_by_label": {label: ratio(sum(k for k, r in zip(keep, rows) if r["weather_present"] == label),
                                                sum(r["weather_present"] == label for r in rows))
                               for label in ("yes", "weak", "no")},
        "metric_passed_count": sum(row["metric_passed"] for row in rows),
        "metric_passed_and_weather_kept_count": sum(row["metric_passed"] and k for row, k in zip(rows, keep)),
    }


def select_threshold(training: list[dict]) -> float | None:
    """Maximize yes|weak-vs-no balanced accuracy on training sources only.

    Ties prefer higher weather-present recall, then the lower threshold. No
    held-out scores or labels enter either the candidate grid or optimization.
    """
    if len({row["weather_present"] != "no" for row in training}) != 2:
        return None
    values = sorted({row["weather_shift"] for row in training})
    candidates = [math.nextafter(values[0], -math.inf), *values]
    ranked = []
    for threshold in candidates:
        metrics = verdict_metrics(training, [row["weather_shift"] > threshold for row in training])
        ranked.append((metrics["balanced_accuracy"], metrics["recall"], -threshold, threshold))
    return max(ranked)[-1]


def group_summary(rows: list[dict], thresholds: list[float]) -> dict:
    labels_present = [row["weather_present"] != "no" for row in rows]
    labels_yes = [row["weather_present"] == "yes" for row in rows]
    return {
        "n": len(rows), "source_count": len({row["source_sha256"] for row in rows}),
        "label_counts": dict(Counter(row["weather_present"] for row in rows)),
        "auc": {target: {field: auc(labels, [row[field] for row in rows])
                         for field in ("weather_shift", "weather_generated_logit")}
                for target, labels in (("yes_vs_weak_or_no", labels_yes), ("yes_or_weak_vs_no", labels_present))},
        "median_shift_by_label": {label: statistics.median(values) if values else None
                                  for label in ("yes", "weak", "no")
                                  for values in [[row["weather_shift"] for row in rows if row["weather_present"] == label]]},
        "thresholds": [{"threshold": threshold, "operator": ">", "status": "descriptive/exploratory",
                        "yes_or_weak_vs_no": verdict_metrics(rows, [row["weather_shift"] > threshold for row in rows]),
                        "yes_vs_weak_or_no": verdict_metrics(rows, [row["weather_shift"] > threshold for row in rows], strict=True)}
                       for threshold in thresholds],
    }


def summarize(rows: list[dict], thresholds: list[float]) -> dict:
    folds, predictions, test_rows = [], [], []
    sources = sorted({row["source_sha256"] for row in rows})
    for held_out in sources:
        train = [row for row in rows if row["source_sha256"] != held_out]
        test = [row for row in rows if row["source_sha256"] == held_out]
        threshold = select_threshold(train)
        keep = [row["weather_shift"] > threshold for row in test] if threshold is not None else None
        fold = {"held_out_source_sha256": held_out, "train_source_sha256": sorted({row["source_sha256"] for row in train}),
                "train_n": len(train), "test_n": len(test), "selected_threshold": threshold,
                "test_metrics": verdict_metrics(test, keep) if keep is not None else None}
        folds.append(fold)
        if keep is not None:
            predictions.extend(keep)
            test_rows.extend(test)
    per_condition = {}
    for condition in WEATHER_TEXT:
        group = [row for row in rows if row["condition"] == condition]
        if group:
            per_condition[condition] = group_summary(group, thresholds)
    return {
        "interpretation": "Exploratory agreement with existing single-assessor, non-blind, screen-level visual labels; no independent truth or VPR benefit claim.",
        "overall": group_summary(rows, thresholds), "per_condition": per_condition,
        "in_sample_selected_threshold": select_threshold(rows),
        "leave_one_source_out": {
            "selection": "training-sources-only maximum balanced accuracy for yes|weak versus no; strict shift > threshold; ties prefer recall then lower threshold",
            "folds": folds, "held_out_predictions_n": len(predictions),
            "aggregate_test_metrics": verdict_metrics(test_rows, predictions) if test_rows else None,
            "per_condition_test_metrics": {condition: verdict_metrics(
                [row for row in test_rows if row["condition"] == condition],
                [keep for row, keep in zip(test_rows, predictions) if row["condition"] == condition])
                for condition in per_condition},
            "limitation": "Only source-disjoint within the same small diagnosis, city, methods and assessor; not independent validation or cross-city generalization.",
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    cached = parser.add_mutually_exclusive_group()
    cached.add_argument("--legacy-probe", type=Path, help="Explicitly import Claude's legacy ordered JSON scores")
    cached.add_argument("--scores", type=Path, help="Reuse this script's hash-keyed joined_scores.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[3.0, 4.0, 5.0, EXPLORATORY_MIN_WEATHER_SHIFT, 7.0])
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) for value in args.thresholds):
        parser.error("Thresholds must be finite")
    for field in ("reviews", "legacy_probe", "scores", "out"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.expanduser().resolve())
    args.thresholds = sorted(set(args.thresholds))
    return args


def main(argv=None):
    args = parse_args(argv)
    use_adaptvpr()
    reviews, inputs = validate_reviews(args.reviews)
    rows, score_metadata = join_scores(reviews, legacy_probe=args.legacy_probe, scores_path=args.scores)
    config = {
        "schema_version": 1, "inputs": inputs, "scores": score_metadata,
        "signal_definition": weather_definition(), "thresholds": args.thresholds,
        "implementation_sha256": {str(path.relative_to(ADAPTVPR_ROOT)): file_sha256(path)
                                  for path in (Path(__file__), ADAPTVPR_ROOT / "experiments/qwen_curriculum/weather_signal.py",
                                               ADAPTVPR_ROOT / "verification/evaluator.py")},
    }
    config["fingerprint"] = fingerprint(config)
    config_path = args.out / "calibration_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Calibration configuration changed; use a new --out directory")
    write_json(config_path, config)
    write_jsonl(args.out / "joined_scores.jsonl", [{**row, "calibration_config_fingerprint": config["fingerprint"]} for row in rows])
    report = summarize(rows, args.thresholds)
    report.update({"config_fingerprint": config["fingerprint"], "input_provenance": inputs, "score_provenance": score_metadata,
                   "signal_definition": weather_definition(), "uncertain_labels_excluded": sum(r["weather_present"] == "uncertain" for r in reviews)})
    write_json(args.out / "summary.json", report)
    print(json.dumps({"out": str(args.out), "n": len(rows),
                      "leave_one_source_out": report["leave_one_source_out"]["aggregate_test_metrics"],
                      "score_provenance": score_metadata}, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
