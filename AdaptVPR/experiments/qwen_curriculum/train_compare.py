"""Freeze and run one matched REAL versus source-REPLACE fine-tune.

Preparation reads records and image bytes only. Training starts explicitly with
``run`` after Qwen generation is complete; SVOX evaluation is opt-in and uses
only the final fourth-epoch checkpoints, without checkpoint selection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.qwen_curriculum import common
else:
    from . import common

CITIES = ("Bangkok", "BuenosAires", "LosAngeles", "Medellin")
EPOCHS = 4


def _config(path):
    value = json.loads(path.read_text())
    expected = value.pop("fingerprint")
    if common.fingerprint(value) != expected:
        raise ValueError(f"Configuration checksum mismatch: {path}")
    return {**value, "fingerprint": expected}


def _seal(row, field):
    value = dict(row)
    expected = value.pop(field)
    if common.fingerprint(value) != expected:
        raise ValueError(f"Record checksum mismatch: {field}")


def _image(row, path_field, hash_field):
    path = Path(row[path_field])
    if not path.is_file() or common.file_sha256(path) != row[hash_field]:
        raise ValueError(f"Recorded image missing or changed: {path}")


def finalized_manifest(directory):
    """Validate a completed canonical generation, returning accepted rows.

    The training export must be exactly the accepted subset of sealed results,
    rather than a partially published or independently edited manifest.
    """
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("state") != "complete":
        raise ValueError("Qwen generation is not complete; finish generation before training preparation")
    execution = _config(directory / "execution_config.json")
    if execution.get("stage") != "generate":
        raise ValueError("Matched hard-source comparison requires a canonical generated run")
    plan = _config(directory / "plan_config.json")
    if execution.get("plan_fingerprint") != plan["fingerprint"]:
        raise ValueError("Generation execution and plan differ")
    if common.file_sha256(directory / "plan.jsonl") != plan["plan_sha256"]:
        raise ValueError("Plan manifest checksum mismatch")
    jobs = common.read_jsonl(directory / "plan.jsonl")
    planned = {row["sample_id"]: row for row in jobs}
    if len(planned) != len(jobs) or len(jobs) != plan["num_images"]:
        raise ValueError("Duplicate jobs or changed plan length")
    for job in jobs:
        _seal(job, "record_sha256")
        _image(job, "source_path", "source_sha256")
    results = common.read_jsonl(directory / "results.jsonl")
    if (summary.get("planned") != len(jobs) or summary.get("completed") != len(jobs)
            or len(results) != len(jobs) or {row["sample_id"] for row in results} != set(planned)):
        raise ValueError("Completed summary does not cover every planned source exactly once")
    seen = set()
    for row in results:
        _seal(row, "result_sha256")
        if row["sample_id"] in seen:
            raise ValueError("Duplicate result")
        seen.add(row["sample_id"])
        job = planned[row["sample_id"]]
        if (row.get("execution_fingerprint") != execution["fingerprint"]
                or any(row.get(key) != value for key, value in job.items())):
            raise ValueError("Saved result differs from its frozen execution or source job")
        for path_field, hash_field in (("output_path", "output_sha256"),
                                      ("raw_output_path", "raw_output_sha256")):
            if row.get(path_field):
                _image(row, path_field, hash_field)
    attempts = common.read_jsonl(directory / "attempts.jsonl")
    attempt_ids = set()
    for number, row in enumerate(attempts, 1):
        _seal(row, "attempt_sha256")
        job = planned.get(row.get("sample_id"))
        if (job is None or row["sample_id"] in attempt_ids or row.get("call_number") != number
                or row.get("execution_fingerprint") != execution["fingerprint"]
                or row.get("record_sha256") != job["record_sha256"]):
            raise ValueError("Invalid durable generation call ledger")
        attempt_ids.add(row["sample_id"])
    if (attempt_ids != set(planned) or len(attempts) > execution["max_calls"]
            or summary.get("http_calls_reserved") != len(attempts)):
        raise ValueError("Completed generation has an inconsistent call ledger")
    accepted = [row for row in results if row.get("passed") is True and row.get("eligible_for_training") is True]
    if not accepted or len(accepted) != summary.get("accepted"):
        raise ValueError("No accepted images or accepted count changed")
    if common.read_jsonl(directory / "training_manifest.jsonl") != accepted:
        raise ValueError("Training manifest differs from accepted sealed results")
    for row in accepted:
        if (not row.get("output_path") or not row.get("raw_output_path")
                or row.get("geometry_ok") is not True or row.get("weather_ok") is not True
                or ("plausible" in row and row["plausible"] is not True)):
            raise ValueError("Accepted row is missing quality flags or image provenance")
    return accepted


def training_command(args, arm, snapshot):
    command = [sys.executable, str(common.SALAD_ROOT / "train_salad.py"),
               "--real-data", str(args.real_data), "--output-dir", str(args.output_dir / arm),
               "--init-checkpoint", str(args.checkpoint), "--cities", *CITIES,
               "--epochs", str(EPOCHS), "--batch-size", str(args.train_batch_size),
               "--images-per-place", "4", "--min-images-per-place", "4",
               "--num-trainable-blocks", "0", "--image-size", "224", "224", "--no-augment",
               "--learning-rate", "1e-6", "--weight-decay", "0", "--device", args.device,
               "--precision", "32", "--num-workers", "0", "--seed", str(args.seed), "--save-every", "4",
               "--backbone-repo", str(args.backbone_repo)]
    if arm == "replace":
        command.extend(["--synthetic-manifest", str(snapshot), "--synthetic-mode", "replace",
                        "--synthetic-fraction", "0.5"])
    return command


def wait_for_generation(args):
    """Poll only JSON and systemd state; never import a model or open images."""
    while True:
        path = args.run_dir / "summary.json"
        summary = json.loads(path.read_text()) if path.exists() else {}
        state = subprocess.run(["systemctl", "--user", "show", args.generation_service,
                                "--property=ActiveState", "--value"], check=True,
                               capture_output=True, text=True).stdout.strip()
        if summary.get("state") == "complete" and state not in {"active", "activating", "deactivating"}:
            return
        if state not in {"active", "activating", "deactivating"}:
            raise RuntimeError(f"Generation worker {args.generation_service} is {state or 'missing'}; "
                               "summary is not complete, so training will not start")
        if str(summary.get("state", "")).startswith("stopped_") or summary.get("state") == "call_budget_exhausted":
            raise RuntimeError(f"Generation stopped with state={summary['state']}; training will not start")
        print(f"Waiting for generation: {summary.get('completed', 0)}/{summary.get('planned', '?')}; "
              f"worker={state}", flush=True)
        time.sleep(60)


def stop_qwen_service():
    """Stop only the known workspace-owned Qwen systemd unit."""
    unit = "adaptvpr-lightx2v.service"
    execution = subprocess.run(["systemctl", "--user", "show", unit, "--property=ExecStart", "--value"],
                               check=True, capture_output=True, text=True).stdout
    adapter = str(common.ADAPTVPR_ROOT / "adapters/lightx2v_qwen_image_edit.py")
    if adapter not in execution:
        raise RuntimeError("Refusing to stop Qwen unit: ExecStart does not name this workspace's Qwen adapter")
    subprocess.run(["systemctl", "--user", "stop", unit], check=True)


def curate_manifest(accepted, exclusions_path):
    if exclusions_path is None:
        return accepted, {"path": None, "sha256": None, "excluded_images": 0, "rows": []}
    rows = common.read_jsonl(exclusions_path)
    accepted_ids = {row["sample_id"]: row for row in accepted}
    excluded = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in accepted_ids:
            raise ValueError("Review exclusion names an unknown or machine-rejected sample_id")
        if sample_id in excluded:
            raise ValueError("Duplicate review exclusion sample_id")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("Review exclusion requires a nonempty human-review reason")
        if any(row.get(key) != accepted_ids[sample_id][key] for key in ("source_sha256", "output_sha256")):
            raise ValueError("Review exclusion image/source hash differs from accepted sealed bytes")
        excluded.add(sample_id)
    curated = [row for row in accepted if row["sample_id"] not in excluded]
    if not curated:
        raise ValueError("Human review excluded every machine-accepted image")
    return curated, {"path": str(exclusions_path), "sha256": common.file_sha256(exclusions_path),
                     "excluded_images": len(excluded), "rows": rows,
                     "provenance": "explicit hash-bound human visual review; machine records/images remain intact"}


def validate_training_pool(args, accepted):
    common.use_salad()
    from workflow.training_data import MixedGSVCitiesDataset
    options = {"real_data": args.real_data, "cities": list(CITIES), "images_per_place": 4,
               "min_images_per_place": 4, "augment": False}
    real = MixedGSVCitiesDataset(**options)
    with tempfile.TemporaryDirectory(prefix="qwen-training-data-check-") as temporary:
        manifest = Path(temporary) / "accepted.jsonl"
        common.write_jsonl(manifest, accepted)
        replaced = MixedGSVCitiesDataset(**options, synthetic_manifest=manifest,
                                        synthetic_mode="replace", synthetic_fraction=0.5)
    replaced.summary["synthetic_manifest"] = str(args.output_dir / "accepted_manifest.jsonl")
    keys = lambda dataset: [(place.city, place.place_id, place.real_paths) for place in dataset.places]
    if len(real) < 2 or keys(real) != keys(replaced):
        raise ValueError("REAL and REPLACE do not share the same eligible real training pool")
    eligible = {(place.city, place.place_id) for place in replaced.places}
    for row in accepted:
        key = replaced.source_index.get(Path(row["source_path"]).resolve())
        if key != (row["city"], row["place_id"]) or key not in eligible:
            raise ValueError("Accepted source label is not an eligible exact GSV metadata view")
    if replaced.summary["num_synthetic_images"] != len(accepted):
        raise ValueError("Accepted training variants were duplicated or dropped by the source loader")
    return {"real": real.summary, "replace": replaced.summary}


def native_svox_manifest(directory):
    expected = {"gallery": 17166, "queries": 14278, "queries_night": 823}
    result = {}
    for name, count in expected.items():
        folder = directory / "images" / "test" / name
        paths = sorted(path for path in folder.iterdir() if path.is_file()
                       and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(paths) != count:
            raise ValueError(f"Native SVOX test/{name} expects {count} images, found {len(paths)}")
        result[name] = {"count": count, "ordered_filenames_sha256": common.fingerprint([path.name for path in paths])}
    return result


@contextmanager
def _lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".comparison.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another comparison process owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare(args):
    machine_accepted = finalized_manifest(args.run_dir)
    accepted, review = curate_manifest(machine_accepted, args.review_exclusions)
    if any(row["city"] not in CITIES for row in accepted):
        raise ValueError("Accepted source cities differ from the fixed four-city training pool")
    for path in (args.checkpoint, args.backbone_repo / "hubconf.py"):
        if not path.is_file():
            raise ValueError(f"Required training input missing: {path}")
    metadata = {city: common.file_sha256(args.real_data / "Dataframes" / f"{city}.csv") for city in CITIES}
    svox = native_svox_manifest(args.dataset_root) if args.evaluate_svox else None
    summaries = validate_training_pool(args, accepted)
    with _lock(args.output_dir):
        snapshot = args.output_dir / "accepted_manifest.jsonl"
        if snapshot.exists() and common.read_jsonl(snapshot) != accepted:
            raise ValueError("Frozen accepted training snapshot changed; choose a new output directory")
        source_files = [Path(__file__), Path(common.__file__), common.SALAD_ROOT / "train_salad.py",
                        *sorted((common.SALAD_ROOT / "workflow").glob("*.py"))]
        definition = {
            "schema_version": 1, "generation_run": str(args.run_dir),
            "generation_inputs_sha256": {name: common.file_sha256(args.run_dir / name) for name in
                ("summary.json", "plan_config.json", "plan.jsonl", "execution_config.json", "results.jsonl",
                 "attempts.jsonl", "training_manifest.jsonl")},
            "accepted_manifest": str(snapshot), "accepted_rows_sha256": common.fingerprint(accepted),
            "accepted_manifest_sha256": hashlib.sha256("".join(json.dumps(row, ensure_ascii=False,
                allow_nan=False) + "\n" for row in accepted).encode()).hexdigest(),
            "accepted_images": len(accepted), "real_data": str(args.real_data), "cities": list(CITIES),
            "machine_accepted_images": len(machine_accepted), "review_exclusions": review,
            "real_metadata_sha256": metadata, "init_checkpoint": str(args.checkpoint),
            "dataset_summaries": summaries,
            "init_checkpoint_sha256": common.file_sha256(args.checkpoint), "backbone_repo": str(args.backbone_repo),
            "backbone_implementation_sha256": {str(path.relative_to(args.backbone_repo)): common.file_sha256(path)
                                                for path in sorted(args.backbone_repo.rglob("*.py"))},
            "protocol": {"epochs": EPOCHS, "learning_rate": 1e-6, "weight_decay": 0,
                         "trainable_backbone_blocks": 0, "image_size": [224, 224], "augment": False,
                         "images_per_place": 4, "min_images_per_place": 4, "places_per_batch": args.train_batch_size,
                         "num_workers": 0, "precision": "32", "seed": args.seed,
                         "real_pool": "all eligible places in the same four cities, including places without synthetic images",
                         "replace_probability": 0.5, "minimum_real_views_per_place": 1,
                         "synthetic_mapping": "exact original source_path; no duplicated source slots",
                         "checkpoint_selection": "final epoch 4 only; no test-set hyperparameter or checkpoint selection"},
            "device": args.device, "cpu_threads": args.cpu_threads,
            "evaluation": {"enabled": args.evaluate_svox, "dataset_root": str(args.dataset_root),
                           "split": "test", "positive_radius_meters": 25,
                           "query_subdirs": {"SVOX": ["queries"], "SVOX_night": ["queries_night"]},
                           "batch_size": 32, "num_workers": 0, "image_size": [224, 224], "dtype": "float32"},
            "svox_native_test_manifest": svox,
            "implementation_sha256": {str(path.relative_to(common.WORKSPACE_ROOT)): common.file_sha256(path)
                                      for path in source_files},
            "training_commands": {arm: training_command(args, arm, snapshot) for arm in ("real", "replace")},
        }
        config = {**definition, "fingerprint": common.fingerprint(definition)}
        path = args.output_dir / "comparison_config.json"
        if path.exists():
            if _config(path) != config:
                raise ValueError("Immutable comparison settings changed; choose a new output directory")
        elif any((args.output_dir / arm).exists() for arm in ("real", "replace")):
            raise ValueError("Training outputs exist without frozen comparison settings")
        else:
            common.write_jsonl(snapshot, accepted)
            common.write_json(path, config)
        if not snapshot.exists():
            common.write_jsonl(snapshot, accepted)
        if common.file_sha256(snapshot) != config["accepted_manifest_sha256"]:
            raise ValueError("Frozen training snapshot byte checksum differs")
        return config


def _subprocess(command, log_path, threads):
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    print(shlex.join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f"Command failed; see {log_path}")


def _check_final(args, arm, config):
    # This path runs only for explicit training, never for prepare/check-data.
    common.use_salad()
    from workflow.model import read_checkpoint
    checkpoint = read_checkpoint(args.output_dir / arm / "checkpoint.pt")
    training = checkpoint["training_config"]
    snapshot = args.output_dir / "accepted_manifest.jsonl"
    expected = {"real_data": str(args.real_data), "cities": sorted(CITIES), "synthetic_places_only": False,
                "synthetic_manifest": str(snapshot) if arm == "replace" else None,
                "synthetic_mode": "replace" if arm == "replace" else "mix", "synthetic_fraction": 0.5,
                "init_checkpoint": str(args.checkpoint), "init_checkpoint_sha256": config["init_checkpoint_sha256"],
                "learning_rate": 1e-6, "weight_decay": 0.0, "augment": False, "images_per_place": 4,
                "min_images_per_place": 4, "batch_size": args.train_batch_size, "num_workers": 0,
                "seed": args.seed, "max_batches_per_epoch": 0}
    if (any(training.get(key) != value for key, value in expected.items())
            or checkpoint.get("total_epochs") != EPOCHS or checkpoint.get("precision") != "32"
            or checkpoint["model_config"]["backbone_config"]["num_trainable_blocks"] != 0
            or checkpoint["model_config"]["image_size"] != [224, 224]):
        raise ValueError(f"Saved {arm} checkpoint differs from fixed comparison protocol")
    return checkpoint


def run(args, config):
    with _lock(args.output_dir):
        report = args.output_dir / "comparison.json"
        if report.exists():
            saved_report = json.loads(report.read_text())
            _seal(saved_report, "report_sha256")
            if saved_report.get("config_fingerprint") != config["fingerprint"]:
                raise ValueError("Comparison report belongs to different settings")
            if saved_report.get("state") not in {"complete", "training_complete_evaluation_disabled"}:
                raise ValueError("Unexpected saved comparison completion state")
            for arm in ("real", "replace"):
                if saved_report["checkpoints"][arm]["path"] != str(args.output_dir / arm / "checkpoint.pt"):
                    raise ValueError("Comparison report checkpoint path differs from its training arm")
                _image(saved_report["checkpoints"][arm], "path", "sha256")
            print(f"Comparison already complete: {report}", flush=True)
            return
        if args.stop_qwen_service:
            stop_qwen_service()
        for arm, original in config["training_commands"].items():
            command = list(original)
            checkpoint = args.output_dir / arm / "checkpoint.pt"
            if checkpoint.exists():
                saved = _check_final(args, arm, config)
                if saved["epoch"] == EPOCHS:
                    print(f"{arm}: final checkpoint already complete", flush=True)
                    continue
                if not 0 < saved["epoch"] < EPOCHS:
                    raise ValueError(f"Invalid saved epoch for {arm}")
                index = command.index("--init-checkpoint")
                command[index:index + 2] = ["--resume", str(checkpoint)]
            _subprocess(command, args.output_dir / f"{arm}_training.log", args.cpu_threads)
            if _check_final(args, arm, config)["epoch"] != EPOCHS:
                raise ValueError(f"{arm} did not complete the fixed fourth epoch")
        # Revalidate every frozen input after the two sequential training arms.
        prepare_after, review_after = curate_manifest(finalized_manifest(args.run_dir), args.review_exclusions)
        if common.fingerprint(prepare_after) != config["accepted_rows_sha256"]:
            raise ValueError("Accepted source/image manifest changed during training")
        if review_after != config["review_exclusions"]:
            raise ValueError("Frozen human review exclusions changed during training")
        for city, digest in config["real_metadata_sha256"].items():
            if common.file_sha256(args.real_data / "Dataframes" / f"{city}.csv") != digest:
                raise ValueError("Real training metadata changed during comparison")
        if common.file_sha256(args.checkpoint) != config["init_checkpoint_sha256"]:
            raise ValueError("Initial SALAD checkpoint changed during comparison")
        if args.evaluate_svox:
            evaluate(args, config)
        else:
            write_report(args.output_dir / "comparison.json", {
                "config_fingerprint": config["fingerprint"], "state": "training_complete_evaluation_disabled",
                "checkpoints": {arm: {"path": str(args.output_dir / arm / "checkpoint.pt"),
                    "sha256": common.file_sha256(args.output_dir / arm / "checkpoint.pt")} for arm in ("real", "replace")}})


def evaluate(args, config):
    if native_svox_manifest(args.dataset_root) != config["svox_native_test_manifest"]:
        raise ValueError("Native SVOX evaluation file order changed after preparation")
    common.use_salad()
    import torch
    from workflow.comparison import evaluate_svox_pair
    torch.set_num_threads(args.cpu_threads)
    pairs = {}
    for arm in ("real", "replace"):
        pairs[arm] = evaluate_svox_pair(args.output_dir / arm / "checkpoint.pt", args.dataset_root,
                                       args.output_dir / "evaluation" / arm, args.backbone_repo,
                                       device=args.device, batch_size=32, num_workers=0)
    datasets = {}
    for name, real in pairs["real"].items():
        replaced = pairs["replace"][name]
        if any(real[key] != replaced[key] for key in ("protocol", "num_queries", "num_references")):
            raise ValueError("REAL and REPLACE SVOX protocols differ")
        datasets[name] = {"protocol": real["protocol"], "num_queries": real["num_queries"],
                          "num_references": real["num_references"], "real": real["recall"],
                          "replace": replaced["recall"], "replace_minus_real_percentage_points": {
                              key: 100 * (replaced["recall"][key] - value) for key, value in real["recall"].items()}}
    write_report(args.output_dir / "comparison.json", {"config_fingerprint": config["fingerprint"],
        "state": "complete", "checkpoint_selection": config["protocol"]["checkpoint_selection"],
        "checkpoints": {arm: {"path": str(args.output_dir / arm / "checkpoint.pt"),
            "sha256": common.file_sha256(args.output_dir / arm / "checkpoint.pt")} for arm in ("real", "replace")},
        "datasets": datasets})


def write_report(path, value):
    common.write_json(path, {**value, "report_sha256": common.fingerprint(value)})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "check-data", "run"))
    parser.add_argument("--generation-run-dir", "--run-dir", dest="run_dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=common.SALAD_ROOT / "checkpoint/dino_salad.ckpt")
    parser.add_argument("--real-data", type=Path, default=common.WORKSPACE_ROOT / "dataset/gsv-cities")
    parser.add_argument("--backbone-repo", type=Path,
                        default=Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--dataset-root", type=Path, default=common.WORKSPACE_ROOT / "dataset/svox")
    parser.add_argument("--evaluate-svox", action="store_true",
                        help="Evaluate native day/night only after both fixed fourth-epoch checkpoints finish")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Places per batch; four views per place")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-exclusions", type=Path,
                        help="JSONL of sample_id/source_sha256/output_sha256/reason; exclude reviewed failures from frozen snapshot")
    parser.add_argument("--wait-for-generation", action="store_true",
                        help="Poll every 60 seconds before input validation; fail if the worker stops before completion")
    parser.add_argument("--generation-service", default="qwen-curriculum-1000.service")
    parser.add_argument("--stop-qwen-service", action="store_true",
                        help="After completed-generation/data validation, stop only workspace-owned adaptvpr-lightx2v.service")
    args = parser.parse_args(argv)
    for name in ("run_dir", "output_dir", "checkpoint", "real_data", "backbone_repo", "dataset_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.review_exclusions is not None:
        args.review_exclusions = args.review_exclusions.expanduser().resolve()
    if args.train_batch_size < 2 or args.cpu_threads < 1:
        parser.error("Training batch must contain at least two places; CPU threads must be positive")
    if (args.wait_for_generation or args.stop_qwen_service) and args.command != "run":
        parser.error("Generation waiting/service stopping applies only to the explicit run command")
    if not args.generation_service.endswith(".service") or "/" in args.generation_service:
        parser.error("Generation service must be a systemd service name")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.wait_for_generation:
        wait_for_generation(args)
    config = prepare(args)
    if args.command == "prepare":
        print(json.dumps({"fingerprint": config["fingerprint"], "accepted_images": config["accepted_images"],
                          "training_commands": {arm: shlex.join(command) for arm, command in
                                                config["training_commands"].items()}}, indent=2))
    elif args.command == "check-data":
        for arm, command in config["training_commands"].items():
            _subprocess([*command, "--check-data"], args.output_dir / f"{arm}_data_check.log", args.cpu_threads)
    else:
        run(args, config)


if __name__ == "__main__":
    main()
