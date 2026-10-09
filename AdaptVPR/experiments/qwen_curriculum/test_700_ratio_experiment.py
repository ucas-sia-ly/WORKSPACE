"""Four fixed-pool SALAD runs: generated 8:1/4:1 and their paired all-real controls.

Freeze 700 successful edits, including automatic quality rejects as requested.
Each edit replaces its exact source once per epoch. Every source slot and batch
order is identical within a pair; ratios describe exact exposure, not Bernoulli
probabilities. No source/generated duplicate is added to the positive group.
VPR training starts fresh: local pretrained DINOv2, random SALAD aggregator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import unittest
import tempfile
import time
from unittest.mock import patch

from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, train_compare

CITIES = train_compare.CITIES
DOMAINS = {"day": "queries", "night": "queries_night", "rain": "queries_rain",
           "snow": "queries_snow", "sun": "queries_sun", "overcast": "queries_overcast"}
ARMS = {"generated_8to1": (8, True), "true_8to1": (8, False),
        "generated_4to1": (4, True), "true_4to1": (4, False)}
INIT_POLICY = "random_aggregator_pretrained_backbone"
BACKBONE = "dinov2_vitb14"


def svox_inventory(root):
    if (root / "evaluation_manifest.json").exists():
        raise ValueError("Native SVOX evaluation must not have a manifest override")
    counts = {"gallery": 17166, "queries": 14278, "queries_night": 823,
              "queries_rain": 937, "queries_snow": 870, "queries_sun": 854, "queries_overcast": 872}
    inventory = {}
    for folder, count in counts.items():
        paths = sorted(p for p in (root / "images/test" / folder).iterdir()
                       if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"})
        if len(paths) != count:
            raise ValueError(f"Native SVOX {folder}: expected {count}, found {len(paths)}")
        inventory[folder] = {"count": len(paths), "ordered_filename_size_mtime_sha256": common.fingerprint(
            [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths])}
    return inventory


def seal(value):
    return {**value, "fingerprint": common.fingerprint(value)}


def config_file(path):
    return train_compare._config(path)


def load_successful(args):
    execution = config_file(args.generation_run_dir / "execution_config.json")
    plan = config_file(args.generation_run_dir / "plan_config.json")
    if execution["plan_fingerprint"] != plan["fingerprint"]:
        raise ValueError("Generation and plan fingerprints differ")
    if common.file_sha256(args.generation_run_dir / "plan.jsonl") != plan["plan_sha256"]:
        raise ValueError("Generation plan changed")
    jobs = {row["sample_id"]: row for row in common.read_jsonl(args.generation_run_dir / "plan.jsonl")}
    rows = [row for row in common.read_jsonl(args.generation_run_dir / "results.jsonl")
            if row.get("output_path") and row.get("status") in {"passed", "rejected"}][:args.num_images]
    if len(rows) != args.num_images:
        raise ValueError(f"Need {args.num_images} successfully generated images; found {len(rows)}")
    sources, outputs, ids = set(), set(), set()
    for row in rows:
        train_compare._seal(row, "result_sha256")
        job = jobs[row["sample_id"]]
        train_compare._seal(job, "record_sha256")
        if row["execution_fingerprint"] != execution["fingerprint"] or any(row.get(k) != v for k, v in job.items()):
            raise ValueError("Generated row differs from its sealed source job")
        for path, digest in (("source_path", "source_sha256"), ("output_path", "output_sha256"),
                             ("raw_output_path", "raw_output_sha256")):
            train_compare._image(row, path, digest)
        for seen, field in ((sources, "source_path"), (outputs, "output_path"), (ids, "sample_id")):
            if row[field] in seen:
                raise ValueError(f"Duplicate generation {field}")
            seen.add(row[field])
    return rows, execution["fingerprint"], plan["fingerprint"]


def build_groups(places, generated, count, seed):
    """Nested, distinct-source pools of N*5 and N*9 views, in four-view bags."""
    if count % 4:
        raise ValueError("Generated count must be divisible by four for exact four-view exposure")
    rng = random.Random(seed)
    selected = defaultdict(list)
    for row in generated:
        selected[(row["city"], row["place_id"])].append(row["source_path"])
    available = {key: sorted(str(p) for p in paths) for key, paths in places.items()}
    groups = []
    for key in sorted(available):
        rng.shuffle(available[key])
    for key in sorted(selected):
        required = selected[key]
        if key not in available or not set(required) <= set(available[key]):
            raise ValueError("Generated source is not an eligible metadata view of its claimed place")
        if len(required) >= 4:
            raise ValueError("At least one real context view is required in each source group")
        remaining = [p for p in available[key] if p not in required]
        if len(remaining) < 4 - len(required):
            raise ValueError("Generated place lacks four distinct source views")
        slots = required + remaining[:4 - len(required)]
        available[key] = remaining[4 - len(required):]
        groups.append({"city": key[0], "place_id": key[1], "sources": slots})
    target4, target8 = count * 5 // 4, count * 9 // 4
    if len(groups) > target4:
        raise ValueError("Not enough real slots for the required same-place positive context")
    # Balanced city round-robin fill; all sources are consumed at most once.
    by_city = defaultdict(list)
    for key, paths in available.items():
        for start in range(0, len(paths) - 3, 4):
            by_city[key[0]].append({"city": key[0], "place_id": key[1], "sources": paths[start:start + 4]})
    for city in sorted(by_city):
        rng.shuffle(by_city[city])
    offset = 0
    while len(groups) < target8:
        added = False
        for city in sorted(by_city):
            if offset < len(by_city[city]) and len(groups) < target8:
                groups.append(by_city[city][offset])
                added = True
        if not added:
            raise ValueError("Insufficient distinct four-view source groups for the requested exact ratios")
        offset += 1
    return {4: groups[:target4], 8: groups}


def request_definition(args):
    return {"generation_run_dir": str(args.generation_run_dir), "num_images": args.num_images,
            "real_data": str(args.real_data), "backbone_weights": str(args.backbone_weights),
            "initialization": INIT_POLICY, "backbone": BACKBONE,
            "backbone_repo": str(args.backbone_repo), "dataset_root": str(args.dataset_root),
            "epochs": args.epochs, "learning_rate": args.learning_rate,
            "trainable_blocks": args.trainable_blocks, "batch_size": args.batch_size,
            "seed": args.seed, "device": args.device,
            "reliability": reliability_definition(args),
            "baseline_dir": str(args.baseline_dir) if getattr(args, "baseline_dir", None) else None}


def reliability_definition(args):
    if not getattr(args, "reliability_ot", False):
        return {"enabled": False}
    return {"enabled": True, "lambda": args.reliability_lambda,
            "hidden_dim": args.reliability_hidden_dim,
            "head_learning_rate": args.reliability_head_lr or args.learning_rate,
            "loss_weight": args.reliability_loss_weight,
            "real_prior_weight": args.reliability_real_prior_weight,
            "coverage_weight": args.reliability_coverage_weight,
            "coverage_floor": args.reliability_coverage_floor}


def validate(config, directory):
    for name, digest in config["implementation_sha256"].items():
        if common.file_sha256(Path(name)) != digest:
            raise ValueError(f"Experiment implementation changed: {name}")
    if common.file_sha256(Path(config["request"]["backbone_weights"])) != config["backbone_weights_sha256"]:
        raise ValueError("Pretrained DINOv2 weights changed")
    for name, digest in config["files_sha256"].items():
        if common.file_sha256(directory / name) != digest:
            raise ValueError(f"Frozen experiment input changed: {name}")
    for name, digest in config.get("baseline", {}).get("snapshot_files_sha256", {}).items():
        if common.file_sha256(directory / "baseline" / name) != digest:
            raise ValueError(f"Historical baseline snapshot changed: {name}")
    for row in common.read_jsonl(directory / "image_inventory.jsonl"):
        if common.file_sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"Frozen image bytes changed: {row['path']}")
    for city, digest in config["metadata_sha256"].items():
        if common.file_sha256(Path(config["request"]["real_data"]) / "Dataframes" / f"{city}.csv") != digest:
            raise ValueError("Real metadata changed")
    if svox_inventory(Path(config["request"]["dataset_root"])) != config["svox_inventory"]:
        raise ValueError("Native SVOX input inventory changed")


def prepare(args):
    path = args.output_dir / "experiment_config.json"
    if path.exists():
        config = config_file(path)
        if config["request"] != request_definition(args):
            raise ValueError("Frozen experiment options changed; choose another output directory")
        validate(config, args.output_dir)
        return config
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = None
    if getattr(args, "reliability_ot", False):
        from experiments.qwen_curriculum.reliability_comparison import freeze_baseline
        baseline = freeze_baseline(args.baseline_dir, args.output_dir, args)
        old = config_file(args.output_dir / "baseline/experiment_config.json")
        for name in old["files_sha256"]:
            destination = args.output_dir / name
            if destination.exists() and common.file_sha256(destination) != old["files_sha256"][name]:
                raise ValueError("Partial snapshot differs from historical fixed image pool")
            shutil.copyfile(args.baseline_dir / name, destination)
        rows = common.read_jsonl(args.output_dir / "generated_700.jsonl")
        execution, plan = old["generation_execution_fingerprint"], old["generation_plan_fingerprint"]
    else:
        rows, execution, plan = load_successful(args)
    common.use_salad()
    from workflow.training_data import MixedGSVCitiesDataset
    dataset = MixedGSVCitiesDataset(args.real_data, cities=list(CITIES), augment=False)
    places = {(p.city, p.place_id): p.real_paths for p in dataset.places}
    groups = ({ratio: common.read_jsonl(args.output_dir / f"groups_{ratio}to1.jsonl") for ratio in (4, 8)}
              if baseline else build_groups(places, rows, args.num_images, args.seed))
    generated = {row["source_path"]: row for row in rows}
    labels = {key: index for index, key in enumerate(sorted(places))}
    for ratio, bags in groups.items():
        for index, bag in enumerate(bags):
            key = (bag["city"], bag["place_id"])
            if baseline:
                if bag["group_id"] != index or bag["label"] != labels[key] or not set(bag["sources"]) <= set(map(str, places[key])):
                    raise ValueError("Frozen baseline source slots differ from current CSV metadata")
            else:
                bag.update(group_id=index, label=labels[key])
        if not baseline:
            common.write_jsonl(args.output_dir / f"groups_{ratio}to1.jsonl", bags)
    if not baseline:
        common.write_jsonl(args.output_dir / "generated_700.jsonl", rows)
    source_paths = {p for bag in groups[8] for p in bag["sources"]}
    inventory = [{"path": p, "sha256": common.file_sha256(Path(p)), "kind": "source"} for p in sorted(source_paths)]
    inventory += [{"path": row["output_path"], "sha256": row["output_sha256"], "kind": "generated"} for row in rows]
    if not baseline:
        common.write_jsonl(args.output_dir / "image_inventory.jsonl", inventory)
    # Validate every selected source label independently of filenames or retrieval.
    for row in rows:
        if dataset.source_index[Path(row["source_path"])] != (row["city"], row["place_id"]):
            raise ValueError("Generated source label differs from exact CSV metadata")
    for ratio, bags in groups.items():
        paths = [p for bag in bags for p in bag["sources"]]
        if len(paths) != len(set(paths)) or len(paths) != args.num_images * (ratio + 1):
            raise ValueError("Pool is not an exact distinct-source ratio")
        if sum(p in generated for p in paths) != args.num_images:
            raise ValueError("Generated source count changed between ratios")
        if len(bags) % args.batch_size == 1:
            raise ValueError("Batch size would drop one group; choose a batch size with remainder zero or >=2")
    code = [Path(__file__).resolve(), Path(common.__file__), Path(train_compare.__file__),
            common.SALAD_ROOT / "train_salad.py", *sorted((common.SALAD_ROOT / "workflow").glob("*.py"))]
    code += sorted((common.SALAD_ROOT / "models").rglob("*.py"))
    code += sorted(args.backbone_repo.rglob("*.py"))
    if baseline:
        from experiments.qwen_curriculum import reliability_comparison
        code.append(Path(reliability_comparison.__file__))
    names = ["groups_4to1.jsonl", "groups_8to1.jsonl", "generated_700.jsonl", "image_inventory.jsonl"]
    definition = {"schema_version": 3, "request": request_definition(args),
                  "reliability": reliability_definition(args),
                  "generation_execution_fingerprint": execution, "generation_plan_fingerprint": plan,
                  "selection": "first N completed successful outputs in execution order, including automatic quality rejects",
                  "quality_counts": dict(Counter(row["status"] for row in rows)),
                  "generated_conditions": dict(Counter(row["condition"] for row in rows)),
                  "arms": {arm: {"ratio": ratio, "replace": replace,
                       "source_slots": args.num_images * (ratio + 1),
                       "generated_per_epoch": args.num_images if replace else 0,
                       "true_per_epoch": args.num_images * (ratio if replace else ratio + 1),
                       "schedule": f"groups_{ratio}to1.jsonl"} for arm, (ratio, replace) in ARMS.items()},
                  "backbone_weights_sha256": common.file_sha256(args.backbone_weights),
                  "metadata_sha256": {city: common.file_sha256(args.real_data / "Dataframes" / f"{city}.csv") for city in CITIES},
                  "implementation_sha256": {str(p.resolve()): common.file_sha256(p) for p in code},
                  "files_sha256": {name: common.file_sha256(args.output_dir / name) for name in names},
                  "training": {"views_per_group": 4, "workers": 0, "precision": "32", "augment": False,
                               "sampling": "one pass over fixed source groups per epoch; identical DataLoader seed within each pair"},
                  "svox_domains": DOMAINS, "evaluation_protocol": "full native test; 25m UTM positives; final checkpoint only"}
    definition["svox_inventory"] = svox_inventory(args.dataset_root)
    if baseline:
        definition["baseline"] = baseline
        if definition["files_sha256"] != old["files_sha256"] or definition["svox_inventory"] != old["svox_inventory"]:
            raise ValueError("New module experiment changed the historical image pool or SVOX protocol")
    config = seal(definition)
    common.write_json(path, config)
    print(json.dumps({"arms": config["arms"], "quality_counts": config["quality_counts"]}, indent=2), flush=True)
    return config


class FixedPairedDataset:
    """Fixed source slots; only the mapped image path differs in a generated arm."""
    def __init__(self, directory, config, arm):
        spec = config["arms"][arm]
        self.groups = common.read_jsonl(directory / spec["schedule"])
        self.generated = {row["source_path"]: row for row in common.read_jsonl(directory / "generated_700.jsonl")}
        self.replace = spec["replace"]
        self.reliability_pairs = config.get("reliability", {}).get("enabled", False)
        self.summary = {"fixed_experiment": "700_source_paired_ratios", "arm": arm,
                        "source_identity_sha256": common.fingerprint(self.groups),
                        "num_groups": len(self.groups), "num_source_images": spec["source_slots"],
                        "real_exposure_per_epoch": spec["true_per_epoch"],
                        "synthetic_exposure_per_epoch": spec["generated_per_epoch"],
                        "schedule_sha256": config["files_sha256"][spec["schedule"]],
                        "image_inventory_sha256": config["files_sha256"]["image_inventory.jsonl"]}
        if self.reliability_pairs:
            self.summary.update(reliability_pairs=True, reliability_pair_source="exact_fixed_source_slot",
                                paired_synthetic_exposure_per_epoch=spec["generated_per_epoch"],
                                companions_enter_metric_loss=False)

    def __len__(self):
        return len(self.groups)

    def selected(self, index):
        return [(self.generated[p]["output_path"] if self.replace and p in self.generated else p,
                 bool(self.replace and p in self.generated)) for p in self.groups[index]["sources"]]

    def __getitem__(self, index):
        import numpy as np
        import torch
        tensors, flags, companions = [], [], []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        def read_tensor(path):
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                                    dtype=np.float32).copy() / 255.0
            return (torch.from_numpy(pixels).permute(2, 0, 1) - mean) / std
        for source, (path, flag) in zip(self.groups[index]["sources"], self.selected(index)):
            tensor = read_tensor(path)
            tensors.append(tensor)
            flags.append(flag)
            if self.reliability_pairs:
                companions.append(read_tensor(source) if flag else tensor)
        result = (torch.stack(tensors), torch.full((4,), self.groups[index]["label"], dtype=torch.long),
                  torch.tensor(flags, dtype=torch.bool))
        if self.reliability_pairs:
            return (*result, torch.stack(companions), torch.tensor(flags, dtype=torch.bool))
        return result


def train_arm(args, config):
    common.use_salad()
    import torch
    import train_salad
    import workflow.training_data as data
    import workflow.model as model_api
    torch.set_num_threads(4)
    arm = args.arm
    spec = config["arms"][arm]
    directory = args.output_dir / arm
    checkpoint = directory / "checkpoint.pt"
    if checkpoint.exists():
        saved = verify_checkpoint(args, config, arm)
        reconcile_training_log(args, config, arm, saved)
        if saved["dataset_summary"] != FixedPairedDataset(args.output_dir, config, arm).summary:
            raise ValueError("Checkpoint belongs to another fixed source schedule")
        if saved["epoch"] == args.epochs:
            check_exposure(args, config, arm)
            return
    argv = ["--real-data", str(args.real_data), "--output-dir", str(directory),
            "--cities", *CITIES, "--synthetic-manifest", str(args.output_dir / spec["schedule"]),
            "--synthetic-mode", "replace", "--synthetic-fraction", str(1 / (spec["ratio"] + 1) if spec["replace"] else 0),
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--images-per-place", "4",
            "--num-workers", "0", "--num-trainable-blocks", str(args.trainable_blocks),
            "--learning-rate", str(args.learning_rate), "--weight-decay", "0", "--no-augment",
            "--precision", "32", "--device", args.device, "--seed", str(args.seed),
            "--save-every", str(args.epochs), "--backbone-repo", str(args.backbone_repo),
            "--backbone", BACKBONE, "--init-policy", INIT_POLICY]
    argv += (["--resume", str(checkpoint)] if checkpoint.exists()
             else ["--backbone-weights", str(args.backbone_weights)])
    reliability = config.get("reliability", {"enabled": False})
    if reliability["enabled"]:
        argv += reliability_arguments(reliability)
    # Scoped injection leaves SALAD source files unchanged. Its normal optimizer,
    # loss, checkpoint/resume and DataLoader are used with the fixed dataset.
    original = data.MixedGSVCitiesDataset
    original_model = model_api.SALADModel
    def dataset_factory(**kwargs):
        if kwargs["images_per_place"] != 4 or kwargs["augment"]:
            raise ValueError("Fixed paired training requires four views and no augmentation")
        if kwargs.get("reliability_pairs", False) != reliability["enabled"]:
            raise ValueError("Reliability training must request exact-source companions")
        return FixedPairedDataset(args.output_dir, config, arm)
    data.MixedGSVCitiesDataset = dataset_factory
    def model_factory(*model_args, **kwargs):
        model = original_model(*model_args, **kwargs)
        if not checkpoint.exists():
            if not kwargs["pretrained_backbone"] or kwargs["backbone_weights"] != args.backbone_weights:
                raise ValueError("Fresh VPR training must use only the specified DINOv2 backbone weights")
            standard_state = {name: tensor for name, tensor in model.aggregator.state_dict().items()
                              if not name.startswith("reliability_")}
            standard_sha = tensor_state_sha256(standard_state)
            backbone_sha = tensor_state_sha256(model.backbone.state_dict())
            if "baseline" in config:
                old = config_file(args.output_dir / "baseline" / f"{arm}_initialization.json")
                if (standard_sha != old["initial_aggregator_state_sha256"]
                        or backbone_sha != old["initial_backbone_state_sha256"]):
                    raise ValueError("New module run changed original SALAD/backbone initialization")
            common.write_json(directory / "initialization.json", seal({
                "experiment_fingerprint": config["fingerprint"], "init_policy": INIT_POLICY,
                "backbone_weights_sha256": config["backbone_weights_sha256"],
                "initial_backbone_state_sha256": backbone_sha,
                "standard_aggregator_state_sha256": standard_sha,
                "initial_aggregator_state_sha256": tensor_state_sha256(model.aggregator.state_dict())}))
        return model
    model_api.SALADModel = model_factory
    try:
        train_salad.main(argv)
    finally:
        data.MixedGSVCitiesDataset = original
        model_api.SALADModel = original_model
    check_exposure(args, config, arm)
    verify_checkpoint(args, config, arm)


def reliability_arguments(reliability):
    names = {"lambda": "lambda", "hidden_dim": "hidden-dim", "head_learning_rate": "head-lr",
             "loss_weight": "loss-weight", "real_prior_weight": "real-prior-weight",
             "coverage_weight": "coverage-weight", "coverage_floor": "coverage-floor"}
    return ["--reliability-ot", *[part for key, flag in names.items()
                                   for part in (f"--reliability-{flag}", str(reliability[key]))]]


def tensor_state_sha256(state):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_initialization(args, config, arm):
    row = config_file(args.output_dir / arm / "initialization.json")
    if (row["experiment_fingerprint"] != config["fingerprint"] or row["init_policy"] != INIT_POLICY
            or row["backbone_weights_sha256"] != config["backbone_weights_sha256"]):
        raise ValueError("Initialization provenance differs from this fresh VPR experiment")
    return row


def check_shared_initialization(args, config):
    rows = [verify_initialization(args, config, arm) for arm in ARMS]
    for key in ("initial_backbone_state_sha256", "initial_aggregator_state_sha256", "standard_aggregator_state_sha256"):
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"Four arms did not start with identical weights: {key}")
    return rows[0]


def verify_checkpoint(args, config, arm):
    common.use_salad()
    from workflow.model import read_checkpoint
    saved = read_checkpoint(args.output_dir / arm / "checkpoint.pt")
    training = saved["training_config"]
    expected = {"seed": args.seed, "learning_rate": args.learning_rate, "num_workers": 0,
                "batch_size": args.batch_size, "images_per_place": 4, "augment": False,
                "max_batches_per_epoch": 0, "init_checkpoint": None, "init_checkpoint_sha256": None}
    if (saved["total_epochs"] != args.epochs or saved["precision"] != "32"
            or any(training.get(key) != value for key, value in expected.items())
            or saved["model_config"]["backbone_config"]["num_trainable_blocks"] != args.trainable_blocks
            or saved["model_config"]["backbone_arch"] != BACKBONE
            or saved["dataset_summary"] != FixedPairedDataset(args.output_dir, config, arm).summary):
        raise ValueError("Saved checkpoint differs from this arm's fixed training protocol")
    reliability = config.get("reliability", {"enabled": False})
    agg = saved["model_config"]["agg_config"]
    if agg.get("reliability_ot", False) != reliability["enabled"]:
        raise ValueError("Checkpoint did not train the requested reliability module")
    if reliability["enabled"]:
        from workflow.model import checkpoint_state_and_config
        checkpoint_state_and_config(saved)
        expected_reliability = {
            "loss_weight": reliability["loss_weight"], "real_prior_weight": reliability["real_prior_weight"],
            "coverage_weight": reliability["coverage_weight"], "coverage_floor": reliability["coverage_floor"],
            "head_learning_rate": reliability["head_learning_rate"],
            "teacher": "detached_local_self_similarity_reciprocal_structure_v1"}
        if (agg.get("reliability_lambda") != reliability["lambda"]
                or agg.get("reliability_hidden_dim") != reliability["hidden_dim"]
                or training.get("reliability") != expected_reliability):
            raise ValueError("Checkpoint module settings differ from the fixed experiment")
    verify_initialization(args, config, arm)
    return saved


def reconcile_training_log(args, config, arm, saved):
    """Repair only the checkpoint-before-log crash window from trusted metrics."""
    path = args.output_dir / arm / "training_log.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    logs = []
    truncated = False
    for index, line in enumerate(lines):
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("Training log has an interior malformed record")
            truncated = True
    epoch = saved["epoch"]
    epochs = [row["epoch"] for row in logs]
    if not truncated and epochs == list(range(1, epoch + 1)):
        if logs[-1] != saved["metrics"]:
            raise ValueError("Latest training log differs from checkpoint metrics")
        return
    metrics, spec = saved["metrics"], config["arms"][arm]
    if (epochs != list(range(1, epoch)) or metrics["epoch"] != epoch
            or (metrics["real_exposure"], metrics["synthetic_exposure"]) !=
               (spec["true_per_epoch"], spec["generated_per_epoch"])
            or config.get("reliability", {}).get("enabled", False) and
               metrics.get("reliability", {}).get("paired_synthetic_exposure") != spec["generated_per_epoch"]):
        raise ValueError("Training log gap cannot be recovered from the checkpoint's final epoch")
    common.write_jsonl(path, [*logs, metrics])
    print(f"[{arm}] Recovered epoch {epoch} log from its saved checkpoint metrics", flush=True)


def check_exposure(args, config, arm):
    spec = config["arms"][arm]
    logs = common.read_jsonl(args.output_dir / arm / "training_log.jsonl")
    if [row["epoch"] for row in logs] != list(range(1, args.epochs + 1)):
        raise ValueError("Training has not completed all configured epochs exactly once")
    for row in logs:
        if (row["real_exposure"], row["synthetic_exposure"]) != (spec["true_per_epoch"], spec["generated_per_epoch"]):
            raise ValueError("Observed training exposure does not match the exact requested ratio")
        if config.get("reliability", {}).get("enabled", False):
            if row.get("reliability", {}).get("paired_synthetic_exposure") != spec["generated_per_epoch"]:
                raise ValueError("Reliability teacher did not see every generated view's exact source")


def evaluate_arm(args, config, arm):
    common.use_salad()
    import torch
    from workflow.model import load_checkpoint_model
    from workflow.evaluation import load_evaluation_set, extract_descriptors, exact_retrieval, build_results
    torch.set_num_threads(4)
    checkpoint = args.output_dir / arm / "checkpoint.pt"
    output = args.output_dir / arm / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    digest = common.file_sha256(checkpoint)
    datasets = {domain: load_evaluation_set("SVOX", args.dataset_root, split="test", positive_radius=25,
                                           query_subdirs=[folder]) for domain, folder in DOMAINS.items()}
    expected_gallery = [(image.id, image.path) for image in datasets["day"].references]
    for domain, dataset in datasets.items():
        if dataset.protocol != {"ground_truth": "utm_radius", "positive_radius_meters": 25,
                                "split": "test", "query_subdirs": [DOMAINS[domain]]}:
            raise ValueError("Requires native SVOX test metadata, not an evaluation manifest override")
        if [(image.id, image.path) for image in dataset.references] != expected_gallery:
            raise ValueError("Weather domains do not share the same ordered gallery")
    results = {}
    for domain in DOMAINS:
        path = output / f"SVOX_{domain}.json"
        if path.exists():
            row = config_file(path)
            if row["checkpoint_sha256"] != digest or row["experiment_fingerprint"] != config["fingerprint"]:
                raise ValueError("Cached SVOX results belong to different weights or experiment")
            results[domain] = row
    if len(results) < len(DOMAINS):
        model = load_checkpoint_model(checkpoint, device=args.device, backbone_repo=args.backbone_repo)
        if model.aggregator.reliability_ot != config.get("reliability", {}).get("enabled", False):
            raise ValueError("SVOX inference loaded a different model variant")
        gallery = extract_descriptors(model, datasets["day"].references, (224, 224), args.device, 32, 0)
        for domain, dataset in datasets.items():
            if domain in results:
                continue
            queries = extract_descriptors(model, dataset.queries, (224, 224), args.device, 32, 0)
            retrieval = exact_retrieval(gallery, queries, dataset.positives, device=args.device)
            row = seal({**build_results(dataset, retrieval, save_hard_cases=True),
                        "checkpoint_sha256": digest, "experiment_fingerprint": config["fingerprint"],
                        "image_size": [224, 224], "query_folder": DOMAINS[domain]})
            common.write_json(output / f"SVOX_{domain}.json", row)
            results[domain] = row
            print(f"[{arm}/{domain}] {row['recall']}", flush=True)
            del queries, retrieval
        del model, gallery
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def run_all(args, config):
    with train_compare._lock(args.output_dir):
        report = args.output_dir / "comparison.json"
        if report.exists():
            saved = config_file(report)
            if saved["experiment_fingerprint"] != config["fingerprint"] or saved["state"] != "complete":
                raise ValueError("Saved report belongs to another experiment")
            for arm, digest in saved["checkpoint_sha256"].items():
                if common.file_sha256(args.output_dir / arm / "checkpoint.pt") != digest:
                    raise ValueError("Reported final checkpoint changed")
            publish_module_comparison(config, saved, args.output_dir)
            print(f"Already complete: {report}", flush=True)
            return
        with generation_guard(args, config):
            if args.stop_qwen_service:
                train_compare.stop_qwen_service()
            execute_experiment(args, config)


@contextmanager
def generation_guard(args, config):
    """Use the generator's own flock; this also detects foreground workers."""
    with (args.generation_run_dir / ".run.lock").open("a") as handle:
        locked = False
        try:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except BlockingIOError:
                    pass
                path = args.generation_run_dir / "summary.json"
                summary = json.loads(path.read_text()) if path.exists() else {}
                if locked:
                    if args.wait_for_generation and summary.get("state") != "complete":
                        raise RuntimeError("Generation worker exited before completing; resume generation first or omit --wait-for-generation")
                    break
                if not args.wait_for_generation:
                    raise RuntimeError("Qwen generation is active (including foreground workers); use --wait-for-generation")
                common.write_json(args.output_dir / "progress.json", {
                    "stage": "waiting_for_generation", "experiment_fingerprint": config["fingerprint"],
                    "frozen_generated_images": args.num_images, "planned_training_runs": len(ARMS)})
                print(f"Waiting for generation: {summary.get('completed', 0)}/{summary.get('planned', '?')}", flush=True)
                time.sleep(30)
            yield
        finally:
            if locked:
                fcntl.flock(handle, fcntl.LOCK_UN)


def publish_module_comparison(config, report, directory):
    if "baseline" not in config:
        return
    from experiments.qwen_curriculum.reliability_comparison import build_module_comparison
    comparison = build_module_comparison(config, report, directory)
    common.write_json(directory / "module_comparison.json", comparison)
    lines = ["# 新 reliability_ot 模块与原版 SALAD 比较", "",
             "R@1 为百分比；差值为百分点。R@5/10 的完整数据见 module_comparison.json。", ""]
    for ratio in ("8", "4"):
        lines += [f"## {ratio}:1", "",
                  "| SVOX | 原版全真 | 新版全真 | 原版生成 | 新版生成 | 全真变化 | 生成变化 | 生成配对收益变化 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for domain, row in comparison["comparisons"][ratio].items():
            real, generated = row["arms"]["true"], row["arms"]["generated"]
            benefit = row["paired_benefit_percentage_points"]["benefit_change"]["R@1"]
            lines.append(f"| {domain} | {100 * real['old']['R@1']:.2f} | {100 * real['new']['R@1']:.2f} "
                         f"| {100 * generated['old']['R@1']:.2f} | {100 * generated['new']['R@1']:.2f} "
                         f"| {real['delta_percentage_points']['R@1']:+.2f} "
                         f"| {generated['delta_percentage_points']['R@1']:+.2f} | {benefit:+.2f} |")
        lines.append("")
    common._atomic_write(directory / "module_comparison.md", ["\n".join(lines)])


def execute_experiment(args, config):
    # Fixed configuration was prepared while generation was still running.
    validate(config, args.output_dir)
    common_args = ["--generation-run-dir", str(args.generation_run_dir), "--output-dir", str(args.output_dir),
                   "--num-images", str(args.num_images), "--real-data", str(args.real_data),
                   "--backbone-weights", str(args.backbone_weights), "--backbone-repo", str(args.backbone_repo),
                   "--dataset-root", str(args.dataset_root), "--epochs", str(args.epochs),
                   "--learning-rate", str(args.learning_rate), "--trainable-blocks", str(args.trainable_blocks),
                   "--batch-size", str(args.batch_size), "--seed", str(args.seed), "--device", args.device]
    if config.get("reliability", {}).get("enabled", False):
        common_args += ["--baseline-dir", str(args.baseline_dir), *reliability_arguments(config["reliability"])]
    for arm in ARMS:
        common.write_json(args.output_dir / "progress.json", {"stage": "training", "arm": arm})
        train_compare._subprocess([sys.executable, str(Path(__file__).resolve()), "train-arm", *common_args,
                                   "--arm", arm], args.output_dir / f"{arm}.log", 4)
        check_exposure(args, config, arm)
    initialization = check_shared_initialization(args, config)
    results = {}
    for arm in ARMS:
        common.write_json(args.output_dir / "progress.json", {"stage": "evaluation", "arm": arm})
        results[arm] = evaluate_arm(args, config, arm)
    comparisons = {}
    for ratio in (8, 4):
        generated, real = results[f"generated_{ratio}to1"], results[f"true_{ratio}to1"]
        comparisons[str(ratio)] = {domain: {"true": real[domain]["recall"],
            "generated": generated[domain]["recall"], "delta_percentage_points": {
                metric: 100 * (generated[domain]["recall"][metric] - value)
                for metric, value in real[domain]["recall"].items()}} for domain in DOMAINS}
    validate(config, args.output_dir)
    report = seal({"experiment_fingerprint": config["fingerprint"],
                      "state": "complete", "comparisons": comparisons,
                      "shared_initialization": initialization,
                      "reliability": config.get("reliability", {"enabled": False}),
                      "checkpoint_sha256": {arm: common.file_sha256(args.output_dir / arm / "checkpoint.pt") for arm in ARMS}})
    if config.get("reliability", {}).get("enabled", False):
        report.pop("fingerprint")
        report["final_reliability_diagnostics"] = {arm: common.read_jsonl(
            args.output_dir / arm / "training_log.jsonl")[-1]["reliability"] for arm in ARMS}
        report = seal(report)
    common.write_json(args.output_dir / "comparison.json", report)
    publish_module_comparison(config, report, args.output_dir)
    common.write_json(args.output_dir / "progress.json", {"stage": "complete"})

class FixedRatioTests(unittest.TestCase):
    def fixture(self):
        places = {("City", i): [f"source_{i}_{j}" for j in range(8)] for i in range(12)}
        generated = [{"city": "City", "place_id": i, "source_path": f"source_{i}_0"} for i in range(4)]
        return places, generated

    def test_exact_ratios_and_nested_unique_source_pools(self):
        places, generated = self.fixture()
        groups = build_groups(places, generated, 4, 42)
        paths = {ratio: [p for bag in bags for p in bag["sources"]] for ratio, bags in groups.items()}
        targets = {row["source_path"] for row in generated}
        self.assertEqual(groups[4], groups[8][:5])
        for ratio in (4, 8):
            self.assertEqual(len(paths[ratio]), (ratio + 1) * 4)
            self.assertEqual(len(set(paths[ratio])), len(paths[ratio]))
            self.assertEqual(len(set(paths[ratio]) & targets), 4)
            self.assertTrue(all(len(set(bag["sources"])) == 4 for bag in groups[ratio]))

    def test_reproducibility_and_insufficient_context(self):
        places, generated = self.fixture()
        self.assertEqual(build_groups(places, generated, 4, 42), build_groups(places, generated, 4, 42))
        self.assertNotEqual(build_groups(places, generated, 4, 42), build_groups(places, generated, 4, 43))
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            build_groups({key: value for key, value in list(places.items())[:4]}, generated, 4, 42)
        with self.assertRaisesRegex(ValueError, "eligible metadata"):
            build_groups(places, [{**generated[0], "source_path": "wrong_source"}, *generated[1:]], 4, 42)

    def test_paired_sampler_only_changes_generated_slots(self):
        places, generated = self.fixture()
        groups = build_groups(places, generated, 4, 42)
        for ratio in (4, 8):
            real, edited = FixedPairedDataset.__new__(FixedPairedDataset), FixedPairedDataset.__new__(FixedPairedDataset)
            for dataset, replace in ((real, False), (edited, True)):
                dataset.groups, dataset.replace = groups[ratio], replace
                dataset.generated = {row["source_path"]: {"output_path": "generated_" + row["source_path"]} for row in generated}
            changes = total = 0
            for i in range(len(real)):
                for (source, false), (output, flag) in zip(real.selected(i), edited.selected(i)):
                    self.assertFalse(false)
                    self.assertEqual(source != output, flag)
                    self.assertEqual(output, "generated_" + source if flag else source)
                    changes += flag
                    total += 1
            self.assertEqual((changes, total - changes), (4, ratio * 4))

    def test_real_trainer_observes_exact_ratios_with_tiny_cpu_model(self):
        # Exercise the actual SALAD optimizer/DataLoader/checkpoint path without
        # loading DINOv2. This catches failed dataset injection and dropped bags.
        common.use_salad()
        import torch
        import train_salad
        torch.set_num_threads(2)
        class TinyModel(torch.nn.Module):
            def __init__(self, *_args, **kwargs):
                super().__init__()
                self.backbone = torch.nn.Linear(3, 3)
                self.aggregator = torch.nn.Linear(3, 8)
                if kwargs["pretrained_backbone"]:
                    self.backbone.load_state_dict(torch.load(kwargs["backbone_weights"], weights_only=True))
            def forward(self, images):
                return torch.nn.functional.normalize(self.aggregator(self.backbone(images.mean(dim=(-1, -2)))), dim=-1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, real = root / "images", root / "real"
            images.mkdir()
            (real / "Dataframes").mkdir(parents=True)
            for city in CITIES:
                (real / "Dataframes" / f"{city}.csv").write_text("test metadata\n")
            places = {}
            for place in range(12):
                paths = []
                for view in range(8):
                    path = images / f"source_{place}_{view}.png"
                    Image.new("RGB", (8, 8), (place * 17, view * 20, 100)).save(path)
                    paths.append(str(path))
                places[(CITIES[0], place)] = paths
            rows = []
            for place in range(4):
                output = images / f"generated_{place}.png"
                Image.new("RGB", (8, 8), (200, place * 40, 20)).save(output)
                rows.append({"city": CITIES[0], "place_id": place, "source_path": places[(CITIES[0], place)][0],
                             "output_path": str(output)})
            groups = build_groups(places, rows, 4, 42)
            for ratio, bags in groups.items():
                for bag in bags:
                    bag["label"] = bag["place_id"]
                common.write_jsonl(root / f"groups_{ratio}to1.jsonl", bags)
            common.write_jsonl(root / "generated_700.jsonl", rows)
            backbone_weights = root / "dinov2_fixture.pt"
            torch.save(torch.nn.Linear(3, 3).state_dict(), backbone_weights)
            args = argparse.Namespace(output_dir=root, real_data=real, backbone_weights=backbone_weights,
                        backbone_repo=root, epochs=1, batch_size=3, trainable_blocks=0, learning_rate=1e-6,
                        seed=42, device="cpu", arm=None)
            config = {"backbone_weights_sha256": common.file_sha256(backbone_weights), "arms": {
                arm: {"ratio": ratio, "replace": replace, "source_slots": 4 * (ratio + 1),
                      "true_per_epoch": 4 * (ratio if replace else ratio + 1),
                      "generated_per_epoch": 4 if replace else 0, "schedule": f"groups_{ratio}to1.jsonl"}
                      for arm, (ratio, replace) in ARMS.items()}, "files_sha256": {
                          f"groups_{ratio}to1.jsonl": common.file_sha256(root / f"groups_{ratio}to1.jsonl") for ratio in (4, 8)}}
            config["files_sha256"]["image_inventory.jsonl"] = "fixture inventory"
            config = seal(config)
            with patch("workflow.model.SALADModel", TinyModel):
                for arm in ARMS:
                    args.arm = arm
                    train_arm(args, config)
                    check_exposure(args, config, arm)
                    self.assertEqual(verify_checkpoint(args, config, arm)["epoch"], 1)
                    # A completed arm is skipped without initializing another model.
                    train_arm(args, config)
            shared = check_shared_initialization(args, config)
            self.assertEqual(shared["initial_backbone_state_sha256"],
                             tensor_state_sha256(torch.load(backbone_weights, weights_only=True)))
            saved_path = root / "generated_8to1/checkpoint.pt"
            saved = torch.load(saved_path, weights_only=True)
            saved["training_config"]["init_checkpoint"] = "unexpected_pretrained_vpr.ckpt"
            torch.save(saved, saved_path)
            with self.assertRaisesRegex(ValueError, "fixed training protocol"):
                verify_checkpoint(args, config, "generated_8to1")

    def test_initialization_digest_handles_scalar_parameters(self):
        import torch
        self.assertEqual(tensor_state_sha256({"scalar": torch.tensor(1.)}),
                         tensor_state_sha256({"scalar": torch.tensor(1.)}))
        self.assertNotEqual(tensor_state_sha256({"scalar": torch.tensor(1.)}),
                            tensor_state_sha256({"scalar": torch.tensor(2.)}))

    def test_foreground_generation_lock_blocks_training_and_incomplete_wait(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common.write_json(root / "summary.json", {"state": "running", "completed": 1, "planned": 2})
            args = argparse.Namespace(generation_run_dir=root, output_dir=root, wait_for_generation=False,
                                      num_images=700)
            with (root / ".run.lock").open("a") as worker:
                fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "including foreground workers"):
                    with generation_guard(args, {"fingerprint": "fixture"}):
                        self.fail("Active foreground generator must block training")
                fcntl.flock(worker, fcntl.LOCK_UN)
            args.wait_for_generation = True
            with self.assertRaisesRegex(RuntimeError, "exited before completing"):
                with generation_guard(args, {"fingerprint": "fixture"}):
                    self.fail("An interrupted generation cannot satisfy an explicit completion wait")

    def test_waiting_for_foreground_generation_then_reserving_gpu_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common.write_json(root / "summary.json", {"state": "running", "completed": 1, "planned": 2})
            args = argparse.Namespace(generation_run_dir=root, output_dir=root, wait_for_generation=True,
                                      num_images=700)
            with (root / ".run.lock").open("a") as worker:
                fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
                def finish(_seconds):
                    common.write_json(root / "summary.json", {"state": "complete", "completed": 2, "planned": 2})
                    fcntl.flock(worker, fcntl.LOCK_UN)
                with patch.object(time, "sleep", side_effect=finish) as sleep:
                    with generation_guard(args, {"fingerprint": "fixture"}):
                        sleep.assert_called_once()
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_checkpoint_log_gap_is_repaired_but_other_gaps_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(output_dir=root)
            spec = {"true_per_epoch": 8, "generated_per_epoch": 1}
            config = {"arms": {"arm": spec}}
            one = {"epoch": 1, "real_exposure": 8, "synthetic_exposure": 1}
            two = {"epoch": 2, "real_exposure": 8, "synthetic_exposure": 1}
            saved = {"epoch": 2, "metrics": two}
            path = root / "arm/training_log.jsonl"
            common.write_jsonl(path, [one])
            reconcile_training_log(args, config, "arm", saved)
            self.assertEqual(common.read_jsonl(path), [one, two])
            path.write_text(json.dumps(one) + '\n{"epoch": 2')
            reconcile_training_log(args, config, "arm", saved)
            self.assertEqual(common.read_jsonl(path), [one, two])
            common.write_jsonl(path, [])
            with self.assertRaisesRegex(ValueError, "cannot be recovered"):
                reconcile_training_log(args, config, "arm", saved)
            common.write_jsonl(path, [one, one])
            with self.assertRaisesRegex(ValueError, "cannot be recovered"):
                reconcile_training_log(args, config, "arm", saved)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("command", choices=["prepare", "run", "train-arm", "self-test"])
    p.add_argument("--generation-run-dir", type=Path, default=common.WORKSPACE_ROOT / "outputs/qwen_curriculum/generation_1000")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--baseline-dir", type=Path, help="Completed original-SALAD fixed experiment; required for module comparison")
    p.add_argument("--real-data", type=Path, default=common.WORKSPACE_ROOT / "dataset/gsv-cities")
    p.add_argument("--backbone-weights", type=Path, default=Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
                   help="DINOv2-only weights; SALAD is initialized randomly, without a VPR checkpoint")
    p.add_argument("--backbone-repo", type=Path, default=Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main")
    p.add_argument("--dataset-root", type=Path, default=common.WORKSPACE_ROOT / "dataset/svox")
    p.add_argument("--num-images", type=int, default=700)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=6e-5)
    p.add_argument("--trainable-blocks", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reliability-ot", action="store_true", help="Enable the new module in every arm and compare with historical SALAD")
    p.add_argument("--reliability-lambda", type=float, default=2.0)
    p.add_argument("--reliability-hidden-dim", type=int, default=64)
    p.add_argument("--reliability-head-lr", type=float)
    p.add_argument("--reliability-loss-weight", type=float, default=0.1)
    p.add_argument("--reliability-real-prior-weight", type=float, default=0.01)
    p.add_argument("--reliability-coverage-weight", type=float, default=0.1)
    p.add_argument("--reliability-coverage-floor", type=float, default=0.5)
    p.add_argument("--arm", choices=list(ARMS))
    p.add_argument("--wait-for-generation", action="store_true")
    p.add_argument("--stop-qwen-service", action="store_true")
    args = p.parse_args(argv)
    if args.reliability_ot:
        args.baseline_dir = args.baseline_dir or common.WORKSPACE_ROOT / "outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1"
    elif args.baseline_dir:
        p.error("--baseline-dir requires --reliability-ot")
    args.output_dir = args.output_dir or common.WORKSPACE_ROOT / "outputs/qwen_curriculum" / (
        "ratio_700_reliability_ot_8to1_4to1" if args.reliability_ot else "ratio_700_vpr_scratch_8to1_4to1")
    if args.baseline_dir:
        args.baseline_dir = args.baseline_dir.expanduser().resolve()
    for name in ("generation_run_dir", "output_dir", "real_data", "backbone_weights", "backbone_repo", "dataset_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if (args.num_images <= 0 or args.num_images % 4 or args.epochs <= 0 or args.batch_size < 2
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 <= args.trainable_blocks <= 12):
        p.error("Positive counts/finite lr required; generated count divisible by 4, batch >=2, trainable blocks in [0,12]")
    if args.command == "train-arm" and not args.arm:
        p.error("train-arm requires --arm")
    if (args.reliability_hidden_dim < 1 or not math.isfinite(args.reliability_lambda) or args.reliability_lambda < 0
            or any(not math.isfinite(v) or v < 0 for v in (args.reliability_loss_weight,
                args.reliability_real_prior_weight, args.reliability_coverage_weight))
            or not math.isfinite(args.reliability_coverage_floor) or not 0 <= args.reliability_coverage_floor <= 1
            or args.reliability_head_lr is not None and (not math.isfinite(args.reliability_head_lr) or args.reliability_head_lr <= 0)):
        p.error("Invalid reliability module settings")
    if args.output_dir == args.baseline_dir:
        p.error("New module output must differ from the historical baseline directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "self-test":
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FixedRatioTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    config = prepare(args)
    if args.command == "run":
        run_all(args, config)
    elif args.command == "train-arm":
        with train_compare._lock(args.output_dir / args.arm):
            train_arm(args, config)


if __name__ == "__main__":
    main()
