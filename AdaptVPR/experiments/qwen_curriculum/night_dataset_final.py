"""Publish the latest selected night view for each place as an exact image pair."""
import csv
import fcntl
import itertools
import json
import os
import re
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from night_dataset import TARGET, WORKSPACE, atomic_json, exact_link, read_json, rows, sealed, sha256, status

OUTPUTS = WORKSPACE / "outputs"
FINAL = OUTPUTS / "night_dataset_final"
BASE = OUTPUTS / "night_dataset_release_v2/manifest.jsonl"
LATEST = OUTPUTS / "night_reviewed_regeneration_162_v1"
COLUMNS = ("sample_id", "city", "place_id", "style_id", "source_path", "source_link",
           "output_path", "raw_output_path", "generation_campaign")


def selected_rows():
    receipt = read_json(BASE.with_name(BASE.name + ".receipt.json"))
    if receipt["total"] != TARGET or sha256(BASE) != receipt["manifest_sha256"]:
        raise ValueError("Previous release manifest does not match its receipt")
    latest_receipt = read_json(LATEST / "complete.json")
    selected = list(rows(LATEST / "generated_results.jsonl"))
    latest = {item["review_sample_id"]: item["generated"] for item in selected}
    if len(latest) != len(selected) or len(latest) != latest_receipt["integrity_verified"] or len(latest) != latest_receipt["total"]:
        raise ValueError("Latest run and completion receipt disagree")
    for previous in rows(BASE):
        identity = previous["dataset_identity"]
        replacement = latest.pop(identity, None)
        if replacement is None:
            yield previous
            continue
        if (replacement["city"], replacement["place_id"]) != (previous["city"], previous["place_id"]):
            raise ValueError(f"Replacement place differs: {identity}")
        if replacement["review_sample_id"] != identity:
            raise ValueError(f"Replacement review identity differs: {identity}")
        yield {
            **replacement,
            "dataset_identity": identity,
            "dataset_stage": "reviewed_regeneration_162",
            "generation_campaign": LATEST.name,
            "replaces_release_sample_id": previous["sample_id"],
            "raw_quality_verdict": "not_audited",
            "quality_audit_status": "not_run_user_requested",
            "included_under_user_policy": True,
            "quality_gate_overridden_by_user": False,
            "prior_eligible_for_training": replacement["eligible_for_training"],
            "eligible_for_training": True,
            "eligibility_basis": previous["eligibility_basis"],
            "place_split_eligibility": previous["place_split_eligibility"],
            "dataset_policy_fingerprint": previous["dataset_policy_fingerprint"],
        }
    if latest:
        raise ValueError("Latest run lists places outside the existing cohort")


def link_pair(row, staging):
    city, place = row["city"], int(row["place_id"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", city) or place < 0:
        raise ValueError(f"Invalid place key: {city}/{place}")
    if row["condition"] != "night" or row["style_id"] == "mercury_vapor":
        raise ValueError(f"Unexpected condition or style: {row['sample_id']}")
    raw = Path(row["raw_output_path"])
    metadata = Path(row.get("metadata_path") or raw.with_suffix(".json"))
    source = Path(row["source_path"])
    if not source.name.startswith(f"{city}_{place:07d}_"):
        raise ValueError(f"Source filename does not match place: {source}")
    relative = Path("pairs") / city / str(place)
    paths = (
        (source, "source_link", "source.jpg", "source_sha256"),
        (Path(row["output_path"]), "output_path", "night.png", "output_sha256"),
        (raw, "raw_output_path", "night_full.png", "raw_output_sha256"),
        (metadata, "metadata_path", "generation.json", "metadata_sha256"),
    )
    record = dict(row)
    record["upstream_result_sha256"] = row["result_sha256"]
    record["release_input_output_path"] = row["output_path"]
    record["dataset_version"] = FINAL.name
    record.setdefault("generation_campaign", row["dataset_stage"])
    for old, field, name, digest in paths:
        target = staging / relative / name
        exact_link(old, target, symlink=field == "source_link")
        if sha256(target) != row[digest]:
            raise ValueError(f"Pair file hash mismatch: {old}")
        record[field] = str(FINAL / relative / name)
    return sealed(record)


def build():
    with (OUTPUTS / ".night_dataset_final.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if FINAL.exists():
            return status(FINAL)
        staging = FINAL.with_name("." + FINAL.name + ".building")
        staging.mkdir(exist_ok=True)
        manifest, tsv, index = (staging / name for name in ("manifest.jsonl", "pairs.tsv", "index.sqlite"))
        index.unlink(missing_ok=True)
        cities, styles, stages, verdicts = (Counter() for _ in range(4))
        identities = set()
        with sqlite3.connect(index) as db, manifest.open("w") as out, tsv.open("w", newline="") as table, ThreadPoolExecutor(max_workers=4) as pool:
            db.execute("CREATE TABLE pairs(sample_id TEXT PRIMARY KEY, city TEXT, place_id INTEGER, "
                       "style_id TEXT, source_path TEXT, source_link TEXT, output_path TEXT, raw_output_path TEXT, "
                       "generation_campaign TEXT, UNIQUE(city,place_id))")
            writer = csv.DictWriter(table, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            stream = selected_rows()
            while group := list(itertools.islice(stream, 128)):
                for row in pool.map(lambda item: link_pair(item, staging), group):
                    identity = row["dataset_identity"]
                    if identity in identities:
                        raise ValueError(f"Duplicate dataset identity: {identity}")
                    identities.add(identity)
                    out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    writer.writerow(row)
                    db.execute("INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?,?)", tuple(row[k] for k in COLUMNS))
                    cities[row["city"]] += 1
                    styles[row["style_id"]] += 1
                    stages[row["dataset_stage"]] += 1
                    verdicts[row["raw_quality_verdict"]] += 1
                if len(identities) % 4096 == 0:
                    print(f"{len(identities)}/{TARGET} pairs verified", flush=True)
            if len(identities) != TARGET or stages["reviewed_regeneration_162"] != 162:
                raise ValueError(f"Release coverage differs: {len(identities)} rows, {stages}")
        receipt = sealed({
            "version": FINAL.name, "count": len(identities), "expected": TARGET,
            "by_city": dict(cities), "by_style": dict(styles), "by_stage": dict(stages),
            "by_raw_quality_verdict": dict(verdicts), "gpu_used": False,
            "manifest_sha256": sha256(manifest), "pairs_tsv_sha256": sha256(tsv),
            "index_sha256": sha256(index), "base_manifest_sha256": sha256(BASE),
            "latest_results_sha256": sha256(LATEST / "generated_results.jsonl"),
            "source_and_generated_hashes_verified": True,
        })
        atomic_json(staging / "complete.json", receipt)
        os.replace(staging, FINAL)
        return status(FINAL)


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
