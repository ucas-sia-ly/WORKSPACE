"""Reuse verified, archived SALAD source reliability without loading a model.

Only the leave-one-out positive/reliability gate is reused.  No hardness
quantiles or nearest-neighbour labels choose places.  The archive binds scores
to source order and path/size/mtime inventories, not per-image content hashes;
all of those proofs must still match before current screened-image hashes can
be associated with the scores.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, source_screen_v3


DEFAULT_ARCHIVE = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/state/history/legacy.tar.gz"
ARRAY_KEYS = frozenset({"own_loo_positive_similarity", "hardest_negative_similarity", "hardness",
                        "centroid_positive_rank", "num_negative_places", "reliable"})


def _checked_config(value, label):
    if not isinstance(value, dict) or "fingerprint" not in value:
        raise ValueError(f"Missing archived configuration fingerprint: {label}")
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if common.fingerprint(body) != value["fingerprint"]:
        raise ValueError(f"Archived configuration checksum mismatch: {label}")
    return value


def _archive_entries(archive):
    entries = {}
    scope_pattern = re.compile(r"^(mining_4city|mining_23city/cities/[^/]+)/")
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            match = scope_pattern.match(member.name)
            if not match or not member.isfile():
                continue
            scope = match.group(1)
            relative = member.name[len(scope) + 1:]
            wanted = (relative == "mining_config.json"
                      or re.fullmatch(r"cache/[^/]+/(source_scores\.npz|scores_complete\.json)", relative))
            if not wanted:
                continue
            if member.size > 50 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large archived score/config entry: {member.name}")
            if member.name in entries:
                raise ValueError(f"Duplicate archived entry: {member.name}")
            entries[member.name] = stream.extractfile(member).read()
    return entries


def _rebuild_records(real_data, request):
    """Exact old mine_sources.load_training_sources identity/order, without models."""
    grouped = defaultdict(dict)
    csv_proofs = {}
    for city in request["cities"]:
        if Path(city).name != city:
            raise ValueError("Invalid archived city identity")
        metadata = real_data / "Dataframes" / f"{city}.csv"
        digest = common.file_sha256(metadata)
        if digest != request["metadata_sha256"].get(city):
            raise ValueError(f"Archived source CSV changed: {metadata}")
        csv_proofs[city] = digest
        with metadata.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["city_id"].strip() != city:
                    raise ValueError("Source city differs from archived city CSV")
                place_id = int(row["place_id"])
                filename = (f"{city}_{place_id:07d}_{int(row['year']):04d}_{int(row['month']):02d}_"
                            f"{int(row['northdeg']):03d}_{row['lat']}_{row['lon']}_{row['panoid']}.jpg")
                if Path(filename).name != filename:
                    raise ValueError("Invalid source filename")
                path = (real_data / "Images" / city / filename).resolve()
                if not path.is_relative_to(real_data / "Images"):
                    raise ValueError("Archived source inventory escaped the dataset")
                grouped[city, place_id][str(path)] = {
                    "source_path": str(path), "source_id": f"{city}/{filename}", "city": city,
                    "place_id": place_id, "latitude": float(row["lat"]), "longitude": float(row["lon"]),
                }
    digest = hashlib.sha256(b"[")
    inventory = hashlib.sha256()
    records, place_index = [], 0
    for _, views in sorted(grouped.items()):
        if len(views) < request["min_images_per_place"]:
            continue
        for _, view in sorted(views.items()):
            record = {**view, "place_index": place_index, "source_index": len(records)}
            if records:
                digest.update(b",")
            digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode())
            stat = Path(record["source_path"]).stat()
            inventory.update(json.dumps([record["source_path"], stat.st_size, stat.st_mtime_ns],
                                        separators=(",", ":")).encode() + b"\0")
            records.append(record)
        place_index += 1
    digest.update(b"]")
    if digest.hexdigest() != request["source_order_sha256"]:
        raise ValueError("Archived descriptor source-order checksum mismatch")
    expected = request["image_inventory"]
    if (expected.get("method") != "metadata_image_path_size_mtime" or expected.get("files") != len(records)
            or inventory.hexdigest() != expected.get("sha256")):
        raise ValueError("Archived source path/size/mtime inventory changed")
    return records, {"csv_sha256": csv_proofs, "source_order_sha256": digest.hexdigest(),
                     "image_inventory": expected, "places": place_index}


def verified_score_map(real_data, archive=DEFAULT_ARCHIVE):
    """Fail closed unless every archived scope has a current, complete mapping."""
    real_data, archive = Path(real_data).expanduser().resolve(), Path(archive).expanduser().resolve()
    entries = _archive_entries(archive)
    config_names = sorted(n for n in entries if n.endswith("/mining_config.json"))
    if not config_names:
        raise ValueError("No complete archived source-reliability configurations")
    mapping, places, proofs, cities = {}, defaultdict(list), [], set()
    for name in config_names:
        config = _checked_config(json.loads(entries[name]), name)
        request = _checked_config(config["descriptor_request"], name + ":descriptor_request")
        scoring = _checked_config(config["scoring"], name + ":scoring")
        if (request.get("real_data") != str(real_data) or request.get("min_images_per_place") != 4
                or scoring.get("descriptor_fingerprint") != request["fingerprint"]
                or scoring.get("min_positive_similarity") != 0.15):
            raise ValueError("Archived reliability scope does not match the expected source policy")
        if cities & set(request["cities"]):
            raise ValueError("Overlapping archived reliability city scopes")
        cities.update(request["cities"])
        scope = name.rsplit("/", 1)[0]
        score_name = f"{scope}/cache/{request['fingerprint'][:20]}/source_scores.npz"
        marker_name = score_name.rsplit("/", 1)[0] + "/scores_complete.json"
        if score_name not in entries or marker_name not in entries:
            raise ValueError("Missing archived score array or completion proof")
        raw = entries[score_name]
        marker = json.loads(entries[marker_name])
        score_sha = hashlib.sha256(raw).hexdigest()
        if marker.get("sha256") != score_sha or marker.get("config_fingerprint") != scoring["fingerprint"]:
            raise ValueError("Archived score array checksum/completion proof mismatch")
        records, proof = _rebuild_records(real_data, request)
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            if set(saved.files) != ARRAY_KEYS:
                raise ValueError("Unexpected archived reliability array fields")
            arrays = {k: saved[k] for k in ARRAY_KEYS}
        if any(a.shape != (len(records),) or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Invalid archived score shape or nonfinite measurements")
        if arrays["reliable"].dtype != np.dtype(bool):
            raise ValueError("Archived reliable flags are not boolean")
        for index, record in enumerate(records):
            key = record["source_path"]
            if key in mapping:
                raise ValueError("A source occurs in multiple archived score scopes")
            score = {"reliable": bool(arrays["reliable"][index]),
                     "own_loo_positive_similarity": float(arrays["own_loo_positive_similarity"][index]),
                     "num_negative_places": int(arrays["num_negative_places"][index])}
            if score["reliable"] and (score["own_loo_positive_similarity"] < 0.15 or score["num_negative_places"] <= 0):
                raise ValueError("Archived reliable flag contradicts its positive/negative measurements")
            mapping[key] = score
            places[record["city"], record["place_id"]].append(Path(key))
        proofs.append({**proof, "config_entry": name, "config_sha256": hashlib.sha256(entries[name]).hexdigest(),
                       "score_entry": score_name, "score_sha256": score_sha,
                       "completion_entry": marker_name,
                       "completion_sha256": hashlib.sha256(entries[marker_name]).hexdigest(),
                       "descriptor_fingerprint": request["fingerprint"], "scoring_fingerprint": scoring["fingerprint"],
                       "views": len(records), "reliable_views": int(arrays["reliable"].sum())})
    available = {p.stem for p in (real_data / "Dataframes").glob("*.csv")}
    if cities != available:
        raise ValueError("Archived reliability scopes do not cover exactly all current cities")
    report = {"archive": str(archive), "archive_sha256": common.file_sha256(archive),
              "verified_scopes": proofs, "cities": sorted(cities), "views": len(mapping),
              "places": len(places), "reliable_views": sum(s["reliable"] for s in mapping.values()),
              "unreliable_source_paths": sorted(path for path, score in mapping.items() if not score["reliable"]),
              "places_with_reliable_views": sum(any(mapping[str(p)]["reliable"] for p in paths)
                                                for paths in places.values()),
              "binding": "verified CSV + source order + path/size/mtime inventory; current selected bytes have SHA256",
              "selection": "own leave-one-out positive >= 0.15 and archived validity; no hardness quantile"}
    return mapping, dict(places), report


def refine_sources(sources, real_data, output_dir, *, seed=42):
    """Replace only unreliable selected views with a reliable view of the same place."""
    sources = list(sources)
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mapping, places, evidence = verified_score_map(real_data)
    config = {"schema_version": 1, "stage": "archived_salad_source_reliability_v1", "seed": seed,
              "unusable_place_policy": "record and skip only after all reliable distinct views fail CPU screening",
              "real_data": str(Path(real_data).expanduser().resolve()),
              "code_sha256": common.file_sha256(Path(__file__)),
              "cpu_screen_code_sha256": common.file_sha256(Path(source_screen_v3.__file__)),
              "input_sources_sha256": common.fingerprint(sources), "input_places": len(sources),
              "archive": evidence["archive"], "archive_sha256": evidence["archive_sha256"],
              "verified_scopes": evidence["verified_scopes"], "min_positive_similarity": 0.15}
    config["fingerprint"] = common.fingerprint(config)
    config_path = directory / "reliability_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Frozen reliability inputs, seed, implementation or archive changed")
    if not config_path.exists():
        common.write_json(config_path, config)
    source_keys, used_hashes = set(), set()
    for source in sources:
        key = source["city"], source["place_id"]
        if key in source_keys or source["source_sha256"] in used_hashes:
            raise ValueError("Input screened sources repeat a place or source contents")
        source_keys.add(key)
        used_hashes.add(source["source_sha256"])
        path = Path(source["source_path"])
        if (key not in places or path not in places[key]
                or common.file_sha256(path) != source["source_sha256"]):
            raise ValueError("Selected source identity or bytes differ from its verified archived mapping")
    refined, changes, skipped = [], [], []
    for source in sources:
        score = mapping[source["source_path"]]
        if score["reliable"]:
            refined.append({**source, "source_reliability": score})
            continue
        key = source["city"], source["place_id"]
        used_hashes.remove(source["source_sha256"])
        attempts, replacement = [], None
        for path in source_screen_v3.ordered_views(places[key], *key, seed):
            candidate_score = mapping[str(path)]
            if not candidate_score["reliable"]:
                attempts.append({"source_path": str(path), "reasons": ["archived_source_unreliable"], **candidate_score})
                continue
            verdict = source_screen_v3.screen_image(path)
            if verdict["passed"] and verdict["source_sha256"] in used_hashes:
                verdict = {**verdict, "passed": False, "reasons": ["source_content_already_selected"]}
            attempts.append(verdict)
            if verdict["passed"]:
                replacement = {**verdict, "city": key[0], "place_id": key[1], "source_reliability": candidate_score,
                               "replaced_source": {"source_path": source["source_path"],
                                                   "source_sha256": source["source_sha256"],
                                                   "source_reliability": score}}
                break
        if replacement is None:
            skipped.append({"city": key[0], "place_id": key[1],
                            "reason": "no_reliable_cpu_suitable_distinct_view", "attempts": attempts})
            continue
        used_hashes.add(replacement["source_sha256"])
        refined.append(replacement)
        changes.append({"city": key[0], "place_id": key[1], "old_source": source["source_path"],
                        "new_source": replacement["source_path"], "attempts": attempts})
    report = {**evidence, "schema_version": 1, "seed": seed, "config_fingerprint": config["fingerprint"],
              "code_sha256": config["code_sha256"], "cpu_screen_code_sha256": config["cpu_screen_code_sha256"],
              "input_sources_sha256": config["input_sources_sha256"], "input_places": len(sources),
              "output_places": len(refined), "replaced_views": len(changes), "changes": changes,
              "skipped_places": len(skipped), "skipped": skipped,
              "output_sources_fingerprint": common.fingerprint(refined)}
    common.write_jsonl(directory / "refined_sources.jsonl", refined)
    report["output_file_sha256"] = common.file_sha256(directory / "refined_sources.jsonl")
    common.write_json(directory / "reliability_report.json", report)
    return refined, report
