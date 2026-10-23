"""CPU metadata preflight and explicit A/M/M_camera smoke command plans.

This helper does not load Torch models, scan the complete real pool, launch a
process or create outputs. Commands are published only after the supplied audit
export and image hashes validate. The training data check remains necessary:
we cannot infer full-pool eligibility from a generated-image manifest.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .camera_augmentation import CameraAugmentation


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _resolve(value, owner):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else owner.parent / path).resolve()


def build_training_preflight(*, manifest, exclude_places, real_root, checkpoint,
                             backbone_repo, profile, output_root, python=None,
                             audit_calibration=None, audit_report=None,
                             audit_checkpoint=None, batch_size=32,
                             synthetic_place_slots=8, seed=43, epochs=3,
                             learning_rate=3e-6, trainable_blocks=2,
                             smoke_batches=2, num_workers=2,
                             allowed_conditions=("night", "rainy_night")):
    """Return a JSON-ready report; invalid evidence returns blocked, never runs.

    ``audit_checkpoint`` defaults to the initial training checkpoint. Supply the
    separately frozen auditor checkpoint when that experiment deliberately uses
    different weights; its bytes must match the calibrated provider fingerprint.
    The real-only A reserves the same variant-bearing place stream as M. All
    methods have the same source preference, update/epoch limits, LR and seed.
    M_camera adds only its explicit profile.
    """
    errors, warnings = [], []
    result = dict(version=1, status="blocked", errors=errors, warnings=warnings,
                  commands=[], data_check_commands=[], checks={}, inputs={},
                  full_real_pool_eligibility="not_scanned_deferred_to_training_check_data",
                  gpu_status="not_checked_cpu_only_preflight", training_started=False,
                  statistical_scope="smoke_chain_only_not_a_frozen_formal_experiment")
    try:
        paths = {name: Path(value).expanduser().resolve() for name, value in dict(
            manifest=manifest, exclude_places=exclude_places, real_root=real_root,
            checkpoint=checkpoint, backbone_repo=backbone_repo,
            profile=profile, output_root=output_root).items()}
        paths["audit_report"] = Path(audit_report).resolve() if audit_report else paths["manifest"].parent / "report.json"
        paths["audit_calibration"] = (Path(audit_calibration).resolve() if audit_calibration
                                     else paths["manifest"].parent.parent / "calibration.json")
        paths["audit_checkpoint"] = Path(audit_checkpoint).resolve() if audit_checkpoint else paths["checkpoint"]
        result["inputs"] = {key: {"path": str(path)} for key, path in paths.items()}
        if type(batch_size) is not int or batch_size < 2:
            raise ValueError("batch_size must be an integer >=2")
        if type(synthetic_place_slots) is not int or not 0 < synthetic_place_slots < batch_size:
            raise ValueError("synthetic_place_slots must be an integer in [1,batch_size)")
        if (any(type(n) is not int or n < 1 for n in (epochs, smoke_batches)) or
                type(num_workers) is not int or num_workers < 0 or
                type(trainable_blocks) is not int or not 0 <= trainable_blocks <= 40 or
                type(seed) is not int or seed < 0):
            raise ValueError("Invalid epoch, smoke, worker, trainable-block or seed budget")
        if isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        known = {"night", "rainy_night", "rain", "snow", "fog", "sun", "overcast"}
        if (not isinstance(allowed_conditions, (tuple, list)) or not allowed_conditions or
                any(not isinstance(c, str) or c not in known for c in allowed_conditions)):
            raise ValueError("allowed_conditions must explicitly name supported weather conditions")
        for key in ("real_root", "backbone_repo"):
            if not paths[key].is_dir():
                raise ValueError(f"{key} directory is missing: {paths[key]}")
        for subdir in ("Images", "Dataframes"):
            if not (paths["real_root"] / subdir).is_dir():
                raise ValueError(f"Real root is missing {subdir}")
        hubconf = paths["backbone_repo"] / "hubconf.py"
        if not hubconf.is_file():
            raise ValueError("Local backbone repository has no hubconf.py")
        result["inputs"]["backbone_repo"]["hubconf_sha256"] = _sha256(hubconf)
        fingerprints = {}
        for key in ("manifest", "exclude_places", "checkpoint", "profile", "audit_report",
                    "audit_calibration", "audit_checkpoint"):
            path = paths[key]
            if not path.is_file():
                raise ValueError(f"{key} file is missing: {path}")
            fingerprints[key] = fingerprints.get(str(path)) or _sha256(path)
            fingerprints[str(path)] = fingerprints[key]
            result["inputs"][key]["sha256"] = fingerprints[key]
        calibration = json.loads(paths["audit_calibration"].read_text())
        definition = {k: v for k, v in calibration.items() if k != "artifact_sha256"}
        if calibration.get("artifact_sha256") != _digest(definition):
            raise ValueError("Audit calibration artifact seal is invalid")
        if calibration.get("usable") is not True:
            raise ValueError("Audit calibration is unusable; no images may be promoted to trainable")
        if calibration.get("version") == "content_safety_v4.2":
            pipeline = calibration.get("feature_pipeline")
            pipeline_sha256 = calibration.get("feature_pipeline_sha256")
            implementation_path = Path(__file__).resolve().parents[2] / \
                "AdaptVPR/experiments/qwen_curriculum/content_audit_v42.py"
            implementation_sha256 = _sha256(implementation_path)
            if calibration.get("decision_implementation_sha256") != implementation_sha256:
                raise ValueError("Audit decision implementation has changed since calibration")
            rules = calibration.get("rules")
            if (not isinstance(rules, dict) or rules.get("version") != calibration["version"]
                    or calibration.get("frozen_fit_rules_sha256") != _digest(rules)):
                raise ValueError("Audit frozen FIT decision rules seal is invalid")
            result["inputs"]["audit_decision_implementation"] = dict(
                path=str(implementation_path), sha256=implementation_sha256)
            result["checks"]["audit_decision_and_frozen_fit_rules"] = "verified"
        elif calibration.get("version") == "content_safety_v4.1":
            pipeline, pipeline_sha256 = calibration.get("pipeline"), calibration.get("pipeline_sha256")
        else:
            raise ValueError("Unsupported audit calibration schema version")
        if not isinstance(pipeline, dict) or pipeline_sha256 != _digest(pipeline):
            raise ValueError("Audit provider pipeline seal is invalid")
        provider = pipeline["provider"]
        if provider.get("checkpoint_sha256") != fingerprints["audit_checkpoint"]:
            raise ValueError("Audit provider checkpoint has changed since calibration")
        implementation = [hubconf] + sorted((paths["backbone_repo"] / "dinov2").rglob("*.py"))
        repository_code = {str(p.relative_to(paths["backbone_repo"])): _sha256(p)
                           for p in implementation if p.is_file()}
        if provider.get("repository_python_sha256") != _digest(repository_code):
            raise ValueError("Audit provider repository code has changed since calibration")
        result["checks"]["audit_calibration_and_provider"] = "verified"
        report = json.loads(paths["audit_report"].read_text())
        if report.get("training_manifest_sha256") != fingerprints["manifest"]:
            raise ValueError("Training manifest differs from its audit export receipt")
        if report.get("calibration_sha256") != fingerprints["audit_calibration"]:
            raise ValueError("Audit report refers to a different calibration artifact")
        excluded_raw = json.loads(paths["exclude_places"].read_text())
        if not isinstance(excluded_raw, list):
            raise ValueError("Excluded places must be an explicit list of [city,place_id]")
        excluded = set()
        for value in excluded_raw:
            if (not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], str)
                    or not value[0] or type(value[1]) is not int or value[1] < 0):
                raise ValueError("Malformed excluded place identity")
            excluded.add((value[0], value[1]))
        controls = calibration.get("control_provenance")
        if not isinstance(controls, list) or not controls:
            raise ValueError("Audit calibration requires explicit control place provenance")
        control_places = set()
        partitions = Counter()
        for control in controls:
            place_key = control.get("place_key") if isinstance(control, dict) else None
            if not isinstance(place_key, str) or "/" not in place_key:
                raise ValueError("Audit control provenance needs city/integer place_key identities")
            city, place = place_key.rsplit("/", 1)
            if not city or "/" in city or not place.isdecimal():
                raise ValueError("Audit control provenance needs city/integer place_key identities")
            identity = (city, int(place))
            if identity in excluded:
                raise ValueError(f"Audit control place overlaps the current calibration/validation exclusion: {place_key}")
            if calibration["version"] == "content_safety_v4.2":
                partition = control.get("partition")
                if partition not in ("fit", "heldout"):
                    raise ValueError("V4.2 control provenance requires its fit/heldout partition")
                partitions[partition] += 1
            control_places.add(identity)
        result["calibration_controls"] = dict(distinct_places=len(control_places),
            provenance_records=len(controls), partitions=dict(partitions))
        result["checks"]["calibration_control_place_exclusion"] = "verified"
        if str(calibration.get("holdout_limit", "")).startswith("reused"):
            warnings.append("Audit controls reuse the earlier v4.1 split; this is exploratory validation, not a fresh independent control holdout")
        warnings.append("A content-audit pass is a machine-control safety screen, not complete semantic validation or proof of real-night equivalence")
        rows = [json.loads(line) for line in paths["manifest"].read_text().splitlines() if line.strip()]
        result["counts"] = dict(rows=len(rows), excluded_places=len(excluded))
        if not rows:
            raise ValueError("Audited training manifest is empty; Unknown/rejected rows remain excluded")
        if report.get("counts", {}).get("pass") != len(rows):
            raise ValueError("Audit export pass count differs from the training manifest")
        ids, outputs, sources, places, conditions = set(), set(), set(), set(), Counter()
        image_hashes = {}
        for index, row in enumerate(rows, 1):
            where = f"Training manifest row {index}"
            if not isinstance(row, dict):
                raise ValueError(f"{where} must be an object")
            if row.get("audit_record_sha256") != _digest({k: v for k, v in row.items() if k != "audit_record_sha256"}):
                raise ValueError(f"{where} audit record seal is invalid")
            decision = row.get("content_audit", {})
            if (row.get("passed") is not True or row.get("eligible_for_training") is not True
                    or row.get("base_quality_passed") is not True or row.get("content_status") != "pass"
                    or decision.get("decision") != "pass" or decision.get("eligible_for_training") is not True
                    or decision.get("version") != calibration.get("version")
                    or decision.get("calibration_sha256") != calibration["artifact_sha256"]):
                raise ValueError(f"{where} lacks verified eligible content-audit pass evidence")
            if row.get("condition") not in allowed_conditions or row.get("prompt_variant") == "shopfronts":
                raise ValueError(f"{where} has an undeclared condition or quarantined variant")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
                raise ValueError(f"{where} sample_id is missing or duplicated")
            ids.add(sample_id)
            city, place = row.get("city"), row.get("place_id")
            if not isinstance(city, str) or not city or isinstance(place, bool):
                raise ValueError(f"{where} needs a stable city/place_id")
            if not (type(place) is int or isinstance(place, str) and place.isdecimal()):
                raise ValueError(f"{where} place_id must be an integer")
            try:
                place = int(place)
            except (ValueError, TypeError):
                raise ValueError(f"{where} place_id must be an integer") from None
            if place < 0 or (city, place) in excluded:
                raise ValueError(f"{where} leaks an excluded calibration/validation place")
            source, output = (_resolve(row[key], paths["manifest"]) for key in ("source_path", "output_path"))
            expected_city = paths["real_root"] / "Images" / city
            if source.parent != expected_city or not source.name.startswith(f"{city}_{place:07d}_"):
                raise ValueError(f"{where} exact source does not match its declared training place")
            if not (paths["real_root"] / "Dataframes" / f"{city}.csv").is_file():
                raise ValueError(f"{where} training city metadata is missing")
            if output == source or output in outputs or output.is_relative_to(paths["real_root"] / "Images"):
                raise ValueError(f"{where} generated image is duplicated or aliases the real pool")
            for path, key in ((source, "source_sha256"), (output, "output_sha256")):
                checksum = image_hashes.get(path) or _sha256(path)
                image_hashes[path] = checksum
                if row.get(key) != checksum:
                    raise ValueError(f"{where} {key} no longer matches the image bytes")
            sources.add(source); outputs.add(output); places.add((city, place))
            conditions[row["condition"]] += 1
        result["counts"].update(distinct_sources=len(sources), distinct_synthetic_places=len(places),
                                synthetic_images=len(outputs), by_condition=dict(conditions))
        minimum = synthetic_place_slots if synthetic_place_slots else batch_size
        if len(places) < minimum:
            raise ValueError(f"Need at least {minimum} distinct audited synthetic places for the declared sampler")
        transform = CameraAugmentation.from_json(paths["profile"])
        result["camera_profile"] = transform.provenance
        if not (set(conditions) & set(transform.profile["conditions"])):
            raise ValueError("Camera profile would never apply to the audited synthetic conditions")
        if transform.profile["status"] == "exploratory":
            warnings.append("Camera profile is exploratory; no real-night equivalence or calibrated camera pipeline is established")
        # One bounded preview verifies decoding, size/range and pure-input
        # behavior without loading a model or saving a generated image.
        from PIL import Image
        import numpy as np
        with Image.open(sorted(outputs)[0]) as opened:
            image = opened.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        original = image.tobytes()
        altered = transform.apply(image, seed=seed, condition=next(iter(conditions)))
        if image.tobytes() != original or altered.size != image.size or not np.isfinite(altered).all():
            raise ValueError("Camera transform changed the clean input or produced invalid output")
        result["checks"].update(audit_record_seals="verified", exact_source_and_image_hashes="verified",
                               heldout_place_exclusion="verified_for_all_audited_rows",
                               camera_profile_and_cpu_preview="verified")
        python = Path(python).expanduser().resolve() if python else Path("/home/admin123/miniconda3/envs/AdaptVPR/bin/python")
        if not python.is_file():
            raise ValueError("Python executable is missing")
        trainer = Path(__file__).resolve().parents[1] / "train_am.py"
        common = [str(python), "-B", str(trainer), "--real-data", str(paths["real_root"]),
            "--synthetic-manifest", str(paths["manifest"]), "--exclude-places", str(paths["exclude_places"]),
            "--init-checkpoint", str(paths["checkpoint"]), "--backbone-repo", str(paths["backbone_repo"]),
            "--real-sampling", "coverage_pk", "--batch-size", str(batch_size), "--images-per-place", "4",
            "--min-images-per-place", "4", "--synthetic-place-slots", str(synthetic_place_slots),
            "--prefer-synthetic-sources", "--learning-rate", str(learning_rate),
            "--num-trainable-blocks", str(trainable_blocks), "--epochs", str(epochs), "--seed", str(seed),
            "--precision", "bf16", "--device", "cuda", "--num-workers", str(num_workers),
            "--prefetch-factor", "1", "--image-size", "224", "224", "--weight-decay", "0",
            "--log-mined-pairs", "--profile-training", "--max-batches-per-epoch", str(smoke_batches),
            "--stop-after-epoch", "1"]
        for method in ("A", "M", "M_camera"):
            output_dir = paths["output_root"] / method
            if output_dir.exists():
                raise ValueError(f"Training output already exists; choose a fresh output root: {output_dir}")
            argv = common + ["--synthetic-mode", "mix" if method == "A" else "replace",
                             "--synthetic-fraction", "0" if method == "A" else "1",
                             "--output-dir", str(output_dir)]
            if method == "M_camera":
                argv += ["--synthetic-camera-profile", str(paths["profile"])]
            result["commands"].append(dict(method=method, argv=argv, output_dir=str(output_dir)))
            result["data_check_commands"].append(dict(method=method, argv=argv + ["--check-data"]))
        result["status"] = "ready_for_data_check"
        result["budget"] = dict(seed=seed, max_batches=smoke_batches, actual_epochs=1,
            scheduler_total_epochs=epochs, batch_places=batch_size, images_per_place=4,
            synthetic_place_slots=synthetic_place_slots, learning_rate=learning_rate,
            trainable_blocks=trainable_blocks, same_initial_checkpoint=True,
            nominal_replace_probability=1., observed_synthetic_fraction="must_measure_during_smoke")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(str(exc))
        result["commands"], result["data_check_commands"] = [], []
    return result
