#!/usr/bin/env python3
"""Versioned CPU-only recalibration/export from frozen v4.1 feature caches.

No DINO forwards, model downloads, generation or training. Existing v4.1 code,
artifacts and images remain unchanged. Every output must be a new location.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.qwen_curriculum import content_audit_v4 as legacy
from experiments.qwen_curriculum import content_audit_v42 as decision
from experiments.qwen_curriculum.audit_content_v4 import _new_dir, _new_file, _rows


def calibrate(args):
    rows = _rows(args.manifest)
    signals = {row["sample_id"]: row for row in _rows(args.signals)}
    artifact = decision.calibrate_controls_v42([(row, signals.get(row["sample_id"])) for row in rows],
                                               args.false_alarm_rate, min_pairs=args.min_pairs)
    artifact["input_manifest_sha256"] = legacy.sha256(args.manifest)
    artifact["input_signals_sha256"] = legacy.sha256(args.signals)
    artifact["artifact_sha256"] = legacy.digest({key: value for key, value in artifact.items()
                                               if key != "artifact_sha256"})
    _new_file(args.output, artifact)
    print(json.dumps({"version": decision.VERSION, "usable": artifact["usable"],
                      "fit": artifact["fit_summary"], "heldout": artifact["heldout_controls"],
                      "reject_test": artifact["heldout_reject_separation_test"],
                      "pass_test": artifact["heldout_pass_separation_test"]}, indent=2))
    if not artifact["usable"]:
        print("Heldout control validation failed; all downstream pairs abstain. Do not relax limits using Qwen/test outputs.")


def export(args):
    rows = _rows(args.manifest)
    signals = {row["sample_id"]: row for row in _rows(args.signals)} if args.signals else {}
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    out = _new_dir(args.output_dir)
    decisions, accepted, counts, groups = [], [], {}, {}
    for original in rows:
        row = dict(original)
        try:
            for field, sha_field in (("source_path", "source_sha256"), ("output_path", "output_sha256")):
                actual = legacy.sha256(row[field])
                if row.get(sha_field) and row[sha_field] != actual:
                    raise ValueError("Image hash mismatch: " + field)
                row[sha_field] = actual
            result = decision.audit_pair_v42(row, signals.get(row["sample_id"]), calibration)
        except (OSError, TypeError, ValueError, KeyError) as error:
            result = {"version": decision.VERSION, "decision": "unknown", "eligible_for_training": False,
                      "reasons": ["image_integrity_failure: " + str(error)]}
        record = {**row, "pre_content_eligible_for_training": original.get("eligible_for_training"),
                  "pre_content_passed": original.get("passed"),
                  "base_quality_passed": original.get("base_quality_passed", original.get("passed") is True and
                                            original.get("geometry_ok") is True and original.get("weather_ok") is True),
                  "passed": result["eligible_for_training"], "eligible_for_training": result["eligible_for_training"],
                  "content_audit": result, "content_status": result["decision"],
                  "input_row_sha256": legacy.digest(original)}
        if "content_audit" in original:
            record["previous_content_audit"] = original["content_audit"]
        if "result_sha256" in record:
            record["original_result_sha256"] = record.pop("result_sha256")
        record.pop("audit_record_sha256", None)
        record["audit_record_sha256"] = legacy.digest(record)
        decisions.append(record)
        if result["eligible_for_training"]:
            accepted.append(record)
        name = result["decision"]
        counts[name] = counts.get(name, 0) + 1
        key = f"{row.get('condition', 'unknown')}/{row.get('prompt_variant', 'unknown')}"
        group = groups.setdefault(key, {})
        group[name] = group.get(name, 0) + 1
    _new_file(out / "audited_manifest.jsonl", decisions, True)
    _new_file(out / "training_manifest.jsonl", accepted, True)
    _new_file(out / "report.json", {"version": decision.VERSION, "counts": counts, "groups": groups,
        "input_manifest": str(Path(args.manifest).resolve()), "input_manifest_sha256": legacy.sha256(args.manifest),
        "signals_sha256": legacy.sha256(args.signals) if args.signals else None,
        "calibration_sha256": legacy.sha256(args.calibration) if args.calibration else None,
        "training_manifest_sha256": legacy.sha256(out / "training_manifest.jsonl"),
        "interpretation": "Control-validated safety screen, not semantic correctness or real-night equivalence",
        "unknown_policy": "excluded", "quarantined_variants": ["shopfronts"]})
    print(f"Audit {decision.VERSION}: {counts}; eligible={len(accepted)}; {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    fit = sub.add_parser("calibrate")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--signals", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--false-alarm-rate", required=True, type=float)
    fit.add_argument("--min-pairs", type=int, default=20)
    fit.set_defaults(handler=calibrate)
    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--signals")
    audit.add_argument("--calibration")
    audit.add_argument("--output-dir", required=True)
    audit.set_defaults(handler=export)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
