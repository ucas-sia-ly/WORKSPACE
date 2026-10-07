"""Generate K verified IC-Light candidates per released AdaptCities Global prompt.

Reuses AdaptVPR's frozen prompts, source resolver, pinned two-stage sampler,
rain denoise policy, Global negative prompt and DualTraitEvaluator thresholds.
Candidate seeds depend only on seed/sample/index, so released-generator arms
share the same pool. Resume validates configuration and candidate artifacts
before skipping work; interrupted final JSONL writes are safely recovered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

from common import ADAPTVPR_ROOT, candidate_seed, file_sha256, read_jsonl, use_adaptvpr, write_json, write_jsonl

use_adaptvpr()

CONFIG_VERSION = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="AdaptCities prompts JSONL")
    parser.add_argument("--image-root", type=Path, required=True, help="GSV-Cities Images/ directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cities", nargs="+", help="Restrict to these cities")
    parser.add_argument("--conditions", nargs="+", help="Restrict to these Global conditions")
    parser.add_argument("--offset", type=int, default=0, help="Start of this round's slice of the shuffled prompts")
    parser.add_argument("--num-sources", type=int, required=True, help="Prompts (source x condition) in this slice")
    parser.add_argument("--num-candidates", type=int, default=4, help="K candidates per prompt")
    parser.add_argument("--lora", type=Path, help="IC-Light LoRA; omit for the released generator")
    parser.add_argument("--seed", type=int, default=42, help="Prompt shuffle and candidate seed base")
    return parser.parse_args(argv)


def select_prompts(path: Path, cities, conditions, seed: int, offset: int, count: int) -> list[dict]:
    from generation.batch import safe_sample_id

    if offset < 0 or count <= 0:
        raise ValueError("--offset must be nonnegative and --num-sources must be positive")
    entries = [
        entry for entry in read_jsonl(path)
        if entry.get("route") == "global"
        and (not cities or entry.get("city") in cities)
        and (not conditions or entry.get("condition") in conditions)
    ]
    # Validate the whole filtered cohort before slicing: duplicate identities or
    # filename collisions must not slip into different rounds.
    seen = set()
    filenames = set()
    for entry in entries:
        sample_id = entry.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Global prompts require a nonempty string sample_id")
        filename = safe_sample_id(sample_id)
        if sample_id in seen or filename in filenames:
            raise ValueError(f"Duplicate sample_id or output filename collision: {sample_id!r}")
        if not isinstance(entry.get("prompt"), str) or not entry["prompt"].strip():
            raise ValueError(f"{sample_id}: missing Global prompt")
        if not isinstance(entry.get("condition"), str) or not entry["condition"].strip():
            raise ValueError(f"{sample_id}: missing Global condition")
        seen.add(sample_id)
        filenames.add(filename)
    # Filtering precedes one fixed shuffle; successive rounds take disjoint slices.
    random.Random(seed).shuffle(entries)
    if offset + count > len(entries):
        raise ValueError(f"Requested prompts [{offset}, {offset + count}) but only {len(entries)} match")
    return entries[offset:offset + count]


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> tuple[list[dict], bool]:
    """Recover only an interrupted last append, never malformed interior rows."""
    if not path.exists():
        return [], False
    lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    rewrite = bool(lines and not lines[-1].endswith(b"\n"))
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if number == len(lines) and not line.endswith(b"\n"):
                print(f"[resume] Recovering interrupted final row in {path}", flush=True)
                break
            raise ValueError(f"{path}:{number}: invalid candidate JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a candidate JSON object")
        rows.append(row)
    return rows, rewrite


def _request_config(args, entries, sources, adapter, negative_prompt, thresholds) -> dict:
    selected = [
        {"sample_id": entry["sample_id"], "source_id": entry["source_id"],
         "city": entry.get("city"), "condition": entry["condition"], "prompt": entry["prompt"],
         "source_path": str(sources[entry["sample_id"]]),
         "source_sha256": file_sha256(sources[entry["sample_id"]])}
        for entry in entries
    ]
    payload = {
        "schema_version": CONFIG_VERSION,
        "prompts": str(args.prompts), "prompts_sha256": file_sha256(args.prompts),
        "image_root": str(args.image_root), "output_dir": str(args.output_dir),
        "cities": sorted(set(args.cities)) if args.cities else None,
        "conditions": sorted(set(args.conditions)) if args.conditions else None,
        "offset": args.offset, "num_sources": args.num_sources,
        "num_candidates": args.num_candidates, "seed": args.seed,
        "lora": str(args.lora) if args.lora else None,
        "lora_sha256": file_sha256(args.lora) if args.lora else None,
        "generator": str(args.lora) if args.lora else "released_iclight",
        "sample_ids": [entry["sample_id"] for entry in entries], "selected_prompts": selected,
        "sampling": {
            "negative_prompt": negative_prompt,
            "highres_scale": adapter.DEFAULT_HIGHRES_SCALE,
            "default_highres_denoise": adapter.DEFAULT_HIGHRES_DENOISE,
            "rain_highres_denoise": adapter.RAIN_HIGHRES_DENOISE,
            "num_inference_steps": adapter.DEFAULT_INFERENCE_STEPS,
            "highres_steps": adapter.DEFAULT_HIGHRES_STEPS,
        },
        "verification": {
            "thresholds": thresholds,
            "matcher_name": os.getenv("ADAPTVPR_MATCHER_NAME", "superpoint-lightglue"),
            "clip_model_name": os.getenv("ADAPTVPR_CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
        },
        "model_paths": {
            key: str(Path(os.environ[key]).expanduser().resolve()) if os.getenv(key) else None
            for key in ("ICLIGHT_BASE_MODEL_PATH", "ICLIGHT_MODEL_PATH")
        },
        "implementation_sha256": {
            name: file_sha256(ADAPTVPR_ROOT / name)
            for name in (
                "adapters/iclight_sd15_fc.py", "generation/batch.py",
                "verification/evaluator.py", "prompts/rules.py",
                "experiments/vpr_guidance/common.py",
                "experiments/vpr_guidance/generate_candidates.py",
                "experiments/vpr_guidance/lora_utils.py",
            )
        },
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def _check_config(path: Path, expected: dict, has_candidates: bool) -> bool:
    """Return whether a matching legacy config needs migration."""
    if not path.is_file():
        if has_candidates:
            raise ValueError(f"Cannot resume candidates without their generation config: {path}")
        return False
    actual = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(actual, dict):
        raise ValueError(f"Invalid generation config: {path}")
    if "schema_version" in actual:
        saved = {key: value for key, value in actual.items() if key != "fingerprint"}
        if actual.get("fingerprint") != _fingerprint(saved):
            raise ValueError(f"Generation config fingerprint is invalid: {path}")
        if actual != expected:
            raise ValueError(f"Generation configuration changed; use a new --output-dir: {path.parent}")
        return False
    # Claude's original manifests predate fingerprints. Bind them to the current
    # request and verify every row below before recording their current provenance.
    for key in ("prompts", "image_root", "output_dir", "lora", "generator"):
        value = actual.get(key)
        if value is not None and value != "released_iclight":
            value = str(Path(value).expanduser().resolve())
        if value != expected[key]:
            raise ValueError(f"Legacy generation configuration differs in {key}; use a new --output-dir")
    for key in ("cities", "conditions"):
        value = sorted(set(actual[key])) if actual.get(key) else None
        if value != expected[key]:
            raise ValueError(f"Legacy generation configuration differs in {key}; use a new --output-dir")
    for key in ("offset", "num_sources", "num_candidates", "seed", "sample_ids"):
        if actual.get(key) != expected[key]:
            raise ValueError(f"Legacy generation configuration differs in {key}; use a new --output-dir")
    return True


def _validated_rows(rows: list[dict], config: dict, output_dir: Path, *, legacy: bool) -> list[dict]:
    from PIL import Image
    from generation.batch import safe_sample_id

    entries = {entry["sample_id"]: entry for entry in config["selected_prompts"]}
    seen = set()
    valid = []
    for row in rows:
        sample_id, index = row.get("sample_id"), row.get("candidate_index")
        if sample_id not in entries or type(index) is not int or not 0 <= index < config["num_candidates"]:
            raise ValueError("Candidate manifest contains an unexpected sample_id/candidate_index")
        key = (sample_id, index)
        if key in seen:
            raise ValueError(f"Duplicate candidate manifest row: {key}")
        seen.add(key)
        entry = entries[sample_id]
        denoise = config["sampling"]["rain_highres_denoise" if entry["condition"] == "rain" else "default_highres_denoise"]
        output_path = output_dir / "images" / f"{safe_sample_id(sample_id)}__k{index}.jpg"
        expected = {
            "sample_id": sample_id, "candidate_index": index,
            "seed": candidate_seed(config["seed"], sample_id, index),
            "route": "global", "city": entry["city"], "condition": entry["condition"],
            "prompt": entry["prompt"], "source_path": entry["source_path"],
            "output_path": str(output_path), "generator": config["generator"], "highres_denoise": denoise,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"{key}: candidate identity mismatch in {field}")
        if not legacy:
            if row.get("config_fingerprint") != config["fingerprint"] or row.get("source_sha256") != entry["source_sha256"]:
                raise ValueError(f"{key}: candidate provenance differs from generation config")
        for metric in ("s_geo", "s_div"):
            value = row.get(metric)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key}: invalid verifier metric {metric}")
        thresholds = config["verification"]["thresholds"]
        passed = row["s_geo"] >= thresholds["TAU_GEO"] and row["s_div"] >= thresholds["TAU_DIV"]
        if row.get("passed") is not passed or row.get("eligible_for_training") is not passed:
            raise ValueError(f"{key}: candidate verdict does not match Global thresholds")
        try:
            # Resolve before opening so an output symlink cannot stand in for an
            # unrelated artifact, even when the manifest spells the expected path.
            if output_path.resolve() != output_path or not output_path.is_file():
                raise OSError("missing candidate image or unexpected image symlink")
            with Image.open(output_path) as image:
                image.verify()
            image_hash = file_sha256(output_path)
            if not legacy and image_hash != row.get("output_sha256"):
                raise OSError("candidate image checksum differs")
        except (OSError, SyntaxError):
            print(f"[resume] Regenerating missing/corrupt image for {key}", flush=True)
            continue
        if legacy:
            row = {**row, "source_id": entry["source_id"], "source_sha256": entry["source_sha256"],
                   "output_sha256": image_hash, "config_fingerprint": config["fingerprint"],
                   "provenance": "legacy_identity_validated"}
        valid.append(row)
    return valid


def _generate_one(adapter, evaluator, source, source_path, entry, index, config, scratch, output_dir):
    from PIL import Image
    from generation.batch import safe_sample_id

    sample_id = entry["sample_id"]
    seed = candidate_seed(config["seed"], sample_id, index)
    denoise = config["sampling"]["rain_highres_denoise" if entry["condition"] == "rain" else "default_highres_denoise"]
    result = adapter.generate(adapter.GenerateRequest(
        image_path=str(source_path), prompt=entry["prompt"],
        negative_prompt=config["sampling"]["negative_prompt"], seed=seed, highres_denoise=denoise))
    result_path = Path(result["result_path"]).resolve()
    if not result_path.is_relative_to(scratch) or not result_path.is_file():
        raise ValueError(f"Adapter result is outside its scratch directory or missing: {result_path}")
    output_path = output_dir / "images" / f"{safe_sample_id(sample_id)}__k{index}.jpg"
    fd, name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with Image.open(result_path) as generated:
            generated.convert("RGB").save(temporary, format="JPEG", quality=95)
        # Verify exactly the JPEG that SALAD will consume, including compression.
        with Image.open(temporary) as generated:
            generated = generated.convert("RGB")
            verdict = evaluator.evaluate(source, generated, entry={"route": "global", "weather": entry["condition"]})
        for value in (verdict.s_geo, verdict.s_div):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{sample_id}: verifier returned an invalid score")
        thresholds = config["verification"]["thresholds"]
        passed = verdict.s_geo >= thresholds["TAU_GEO"] and verdict.s_div >= thresholds["TAU_DIV"]
        if bool(verdict.passed) != passed:
            raise ValueError(f"{sample_id}: verifier verdict does not match Global thresholds")
        image_hash = file_sha256(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
    selected = next(item for item in config["selected_prompts"] if item["sample_id"] == sample_id)
    return {
        "sample_id": sample_id, "candidate_index": index, "seed": seed,
        "city": entry.get("city"), "condition": entry["condition"], "route": "global",
        "prompt": entry["prompt"], "highres_denoise": denoise,
        "source_id": entry["source_id"], "source_path": str(source_path), "output_path": str(output_path),
        "source_sha256": selected["source_sha256"], "output_sha256": image_hash,
        "config_fingerprint": config["fingerprint"], "generator": config["generator"],
        "s_geo": verdict.s_geo, "s_div": verdict.s_div,
        "passed": bool(verdict.passed), "eligible_for_training": bool(verdict.passed),
    }


def main(argv=None):
    args = parse_args(argv)
    if args.num_candidates <= 0 or args.num_sources <= 0 or args.offset < 0:
        raise ValueError("--num-sources and --num-candidates must be positive; --offset must be nonnegative")
    for name in ("prompts", "image_root", "output_dir", "lora"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if not args.prompts.is_file() or not args.image_root.is_dir():
        raise ValueError("--prompts must be a file and --image-root must be an existing directory")
    if args.lora is not None and not args.lora.is_file():
        raise ValueError(f"--lora must be an existing checkpoint file: {args.lora}")

    from PIL import Image
    import adapters.iclight_sd15_fc as adapter
    from generation.batch import resolve_prompt_source
    from prompts.rules import global_negative_prompt
    from verification.evaluator import DualTraitEvaluator, ROUTE_THRESHOLDS

    entries = select_prompts(args.prompts, args.cities, args.conditions, args.seed, args.offset, args.num_sources)
    sources = {}
    for entry in entries:
        source_path = resolve_prompt_source(entry, args.image_root).resolve()
        if not source_path.is_relative_to(args.image_root):
            raise ValueError(f"{entry['sample_id']}: source image lies outside --image-root: {source_path}")
        with Image.open(source_path) as image:
            image.verify()
        sources[entry["sample_id"]] = source_path
    config = _request_config(args, entries, sources, adapter, global_negative_prompt(), ROUTE_THRESHOLDS["global"])
    output_dir = args.output_dir
    manifest_path, config_path = output_dir / "candidates.jsonl", output_dir / "generation_config.json"
    rows, repair_tail = _read_manifest(manifest_path)
    legacy = _check_config(config_path, config, bool(rows))
    validated = _validated_rows(rows, config, output_dir, legacy=legacy)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    if repair_tail or legacy or len(validated) != len(rows) or not manifest_path.exists():
        write_jsonl(manifest_path, validated)
    if legacy or not config_path.exists():
        write_json(config_path, config)
    done = {(row["sample_id"], row["candidate_index"]) for row in validated}
    complete_path = output_dir / "generation_complete.json"
    if len(done) != args.num_sources * args.num_candidates:
        complete_path.unlink(missing_ok=True)
        old_env = {key: os.environ.get(key) for key in ("ICLIGHT_OUTPUT_DIR", "ADAPTVPR_LORA_CHECKPOINT")}
        try:
            with tempfile.TemporaryDirectory(prefix=".adapter_tmp_", dir=output_dir) as scratch_name:
                scratch = Path(scratch_name).resolve()
                os.environ["ICLIGHT_OUTPUT_DIR"] = str(scratch)
                if args.lora:
                    os.environ["ADAPTVPR_LORA_CHECKPOINT"] = str(args.lora)
                else:
                    os.environ.pop("ADAPTVPR_LORA_CHECKPOINT", None)
                adapter.state.pipe_t2i, adapter.state.pipe_i2i, adapter.state.vae = adapter.load_pipeline()
                evaluator = DualTraitEvaluator()
                with manifest_path.open("a", encoding="utf-8") as manifest:
                    for position, entry in enumerate(entries, 1):
                        sample_id = entry["sample_id"]
                        pending = [index for index in range(args.num_candidates) if (sample_id, index) not in done]
                        if not pending:
                            continue
                        with Image.open(sources[sample_id]) as opened:
                            source = opened.convert("RGB")
                        for index in pending:
                            row = _generate_one(adapter, evaluator, source, sources[sample_id], entry,
                                                index, config, scratch, output_dir)
                            manifest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                            manifest.flush()
                            os.fsync(manifest.fileno())
                            validated.append(row)
                        print(f"[gen {position}/{len(entries)}] {sample_id} {entry['condition']}", flush=True)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    legacy_count = sum(row.get("provenance") == "legacy_identity_validated" for row in validated)
    complete = {
        "config_fingerprint": config["fingerprint"], "manifest_sha256": file_sha256(manifest_path),
        "num_sources": args.num_sources, "num_candidates": args.num_candidates,
        "candidate_count": len(validated), "passed_count": sum(row["passed"] for row in validated),
        "legacy_candidate_count": legacy_count,
    }
    if legacy_count:
        complete["legacy_provenance_note"] = (
            "Original source/model/code hashes were not recorded; existing verifier scores are retained."
        )
    write_json(complete_path, complete)
    print(f"[gen complete] {len(validated)} candidates, {sum(row['passed'] for row in validated)} verified", flush=True)


if __name__ == "__main__":
    main()
