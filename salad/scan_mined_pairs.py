#!/usr/bin/env python3
"""Forward-only, fixed-input MS saturation scan; never creates an optimizer.

P=8/16/32 share descriptor prefixes from one P32 forward. Empty mining masks
certify zero MS gradients; gradients themselves are not calculated. No recall
evaluation, training, checkpoint writes, or teacher feature caching occurs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from run_ccr_feasibility import digest, run_child, write_json

ROOT = Path(__file__).resolve().parent


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class FixedScanDataset:
    """Freeze Python-RNG augmentation per place/epoch, independently of workers."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key
        place = self.dataset.places[index]
        seed = int.from_bytes(hashlib.sha256(
            f"mining_scan_v1/{self.seed}/{epoch}/{place.city}/{place.place_id}".encode()).digest()[:8], "big")
        previous = random.getstate()
        random.seed(seed)
        try:
            images, labels, synthetic = self.dataset[key]
        finally:
            random.setstate(previous)
        selected = self.dataset.sample_paths(index, epoch=epoch)
        return images, labels, synthetic, {"city": place.city, "place_id": place.place_id,
            "epoch": epoch, "paths": [str(path) for path, _ in selected]}


def make_schedule(num_places, batches, seed):
    """Unique places within each P32 batch; reshuffle if a pool is exhausted."""
    if num_places < 32 or batches < 1:
        raise ValueError("Scan needs at least 32 places and one superbatch")
    rng, schedule, epoch = random.Random(seed), [], 0
    while len(schedule) < batches:
        order = list(range(num_places))
        rng.shuffle(order)
        for start in range(0, num_places - 31, 32):
            schedule.append([(epoch, index) for index in order[start:start + 32]])
            if len(schedule) == batches:
                break
        epoch += 1
    return schedule


def summarize(rows):
    import statistics
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in ("checkpoint", "cohort", "composition", "mode", "P"))
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in groups.items():
        count = len(group)
        row = dict(zip(("checkpoint", "cohort", "composition", "mode", "P"), key))
        row.update(batches=count, empty_mining_batches=sum(r["empty_mining_mask"] for r in group),
                   zero_ms_gradient_certified_fraction=sum(r["empty_mining_mask"] for r in group) / count,
                   numerically_zero_loss_with_nonempty_mask=sum(r["numerically_zero_loss_with_nonempty_mask"] for r in group),
                   anchors=sum(r["batch_size"] for r in group),
                   active_anchors=sum(sum(s["active_anchors"] for s in r["source_anchors"].values()) for r in group),
                   common_8_active_anchors=sum(r["common_8_place_anchors"]["active_anchors"] for r in group),
                   common_8_anchors=32 * count, mean_loss=statistics.mean(r["loss"] for r in group),
                   mean_synthetic_fraction=statistics.mean(r["source_anchors"]["synthetic"]["anchors"] / r["batch_size"] for r in group))
        row["active_anchor_ratio"] = row["active_anchors"] / row["anchors"]
        row["common_8_active_anchor_ratio"] = row["common_8_active_anchors"] / row["common_8_anchors"]
        for source in ("real_real", "real_synthetic", "synthetic_real", "synthetic_synthetic"):
            for kind in ("positive", "negative", "mined_positive", "mined_negative"):
                row[source + "_" + kind] = sum(r["source_pairs"][source][kind] for r in group)
        for source in ("real", "synthetic"):
            row[source + "_anchors"] = sum(r["source_anchors"][source]["anchors"] for r in group)
            row[source + "_active_anchors"] = sum(r["source_anchors"][source]["active_anchors"] for r in group)
        result.append(row)
    return result


def save_results(out, rows, protocol):
    summary = summarize(rows)
    write_json(out / "summary.json", summary)
    write_json(out / "protocol.json", protocol)
    with (out / "mined_pair_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else [])
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Forward-only MS saturation scan", "",
        "No optimizer, backward pass, or parameter updates. Empty mining masks certify zero MS gradients at the scanned descriptors; a numerically zero loss alone is not used as a certificate.", "",
        "Nested P values reuse the same P32-forward descriptors. Common-8 statistics hold anchors fixed while increasing candidate negatives. These are descriptive sampled-batch statistics, not a whole-training-set gradient measurement.", "",
        "| Checkpoint | Cohort | Composition | Mode | P | Batches | Empty masks | Active anchors | Common-8 active | Mean MS |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['checkpoint']} | {row['cohort']} | {row['composition']} | {row['mode']} | {row['P']} | {row['batches']} | {row['empty_mining_batches']}/{row['batches']} | {row['active_anchor_ratio']:.3%} | {row['common_8_active_anchor_ratio']:.3%} | {row['mean_loss']:.6g} |")
    lines.extend(["", "Source-pair counts in the CSV are directed (anchor → reference); real→synthetic and synthetic→real are distinct. `batch_manifest.jsonl` records exact paths and normalized tensor SHA256 for identical input verification across checkpoints.", "",
        "Train BF16 combines training-mode Dropout and mixed precision. Eval FP32 differs in both; use the Stage 1 local diagnostic to isolate Dropout from precision. Dropout is sampled once per superbatch/mode/checkpoint with the same seed. No confidence interval or significance claim is inferred from nested, correlated P comparisons."])
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run_worker(args, out):
    import torch
    from torch.utils.data import DataLoader, default_collate
    from workflow.mining_diagnostics import source_aware_mined_pair_statistics, common_anchor_statistics
    from workflow.model import SALADModel, read_checkpoint, checkpoint_state_and_config
    from workflow.resource_limits import configure_cuda_budget
    from workflow.training_data import MixedGSVCitiesDataset
    started = time.perf_counter()
    plan_path = args.experiment_dir / "plan.json"
    plan_hash = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    c = plan["config"]
    manifest_hash = digest(Path(plan["train_manifest"]))
    device = torch.device(args.device)
    if device.type == "cpu":
        import os
        os.environ["XFORMERS_DISABLED"] = "1"
    if "train_bf16" in args.modes and device.type != "cuda":
        raise ValueError("train_bf16 scan requires CUDA; use --modes eval_fp32 on CPU")
    torch.set_num_threads(args.torch_threads)
    budget = configure_cuda_budget(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial = Path(c["checkpoint"])
    checkpoint_paths = {name: initial if name == "initial" else
        args.experiment_dir / "runs" / name / str(args.training_seed) / "checkpoint.pt"
        for name in args.checkpoints}
    hashes = {name: digest(path) for name, path in checkpoint_paths.items()}
    full = MixedGSVCitiesDataset(Path(c["real_data"]), Path(plan["train_manifest"]), cities=c["cities"],
        images_per_place=4, min_images_per_place=4, synthetic_fraction=0., image_size=(224, 224),
        augment=not args.no_augmentation, real_sampling="coverage_pk", seed=args.seed)
    excluded = {tuple(key) for key in plan.get("heldout_places", [])}
    excluded.update((r["city"], r["place_id"]) for r in plan.get("calibration_pairs", []))
    if plan.get('training_exclude_places'):
        excluded.update(tuple(row) for row in json.loads(Path(plan['training_exclude_places']).read_text()))
    full.places = [p for p in full.places if (p.city, p.place_id) not in excluded]
    panels = {}
    for cohort in args.cohorts:
        dataset = copy.copy(full)
        if cohort == "training_subset":
            dataset.places = [p for p in full.places if p.synthetic_paths]
        panels[cohort] = dataset
    schedules = {cohort: make_schedule(len(dataset), args.batches, args.seed)
                 for cohort, dataset in panels.items()}
    rows, input_hashes, states, all_grad_absent = [], {}, {}, True
    manifest_path = out / "batch_manifest.jsonl"
    with manifest_path.open("w") as manifest:
        for checkpoint_name, path in checkpoint_paths.items():
            state, config = checkpoint_state_and_config(read_checkpoint(path))
            # Match the original fine-tuning mode, including initial weights.
            config["backbone_config"]["num_trainable_blocks"] = c["num_trainable_blocks"]
            model = SALADModel(config, pretrained_backbone=False, backbone_repo=Path(c["backbone_repo"]))
            model.load_state_dict(state, strict=True)
            del state
            model.to(device)
            model.requires_grad_(False)
            before = state_digest(model)
            versions = {name: parameter._version for name, parameter in model.named_parameters()}
            for cohort, real_dataset in panels.items():
                for composition in args.compositions:
                    dataset = copy.copy(real_dataset)
                    dataset.synthetic_fraction = c["mix_fraction"] if composition == "mix" else 0.
                    loader_kwargs = dict(batch_sampler=schedules[cohort], num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))
                    if args.num_workers:
                        loader_kwargs["prefetch_factor"] = 1
                    loader = DataLoader(FixedScanDataset(dataset, args.seed), **loader_kwargs)
                    for batch_index, (images, labels, synthetic, metadata) in enumerate(loader):
                        images = images.flatten(0, 1)
                        labels, synthetic = labels.flatten(), synthetic.flatten()
                        if labels.unique().numel() != 32 or len(labels) != 128:
                            raise RuntimeError("A scan superbatch must contain P32/K4")
                        identity = f"{cohort}/{composition}/{batch_index}"
                        fingerprint = tensor_digest(images)
                        if identity in input_hashes and input_hashes[identity] != fingerprint:
                            raise RuntimeError("Fixed input tensor changed across checkpoints")
                        if identity not in input_hashes:
                            input_hashes[identity] = fingerprint
                            selected = [dict(city=metadata["city"][i], place_id=int(metadata["place_id"][i]),
                                epoch=int(metadata["epoch"][i]), paths=[column[i] for column in metadata["paths"]])
                                for i in range(32)]
                            manifest.write(json.dumps(dict(id=identity, tensor_sha256=fingerprint,
                                synthetic=synthetic.tolist(), places=selected)) + "\n")
                            manifest.flush()
                        for mode in args.modes:
                            model.train(mode == "train_bf16")
                            rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
                            with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
                                torch.manual_seed(args.seed + batch_index)
                                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=mode == "train_bf16"):
                                    descriptors = torch.cat([model(chunk.to(device, non_blocking=True)).float().cpu()
                                        for chunk in images.split(args.forward_batch_size)])
                            for p in (8, 16, 32):
                                stats = source_aware_mined_pair_statistics(descriptors[:p * 4], labels[:p * 4], synthetic[:p * 4])
                                stats["common_8_place_anchors"] = common_anchor_statistics(stats, 32)
                                if not args.save_per_anchor:
                                    stats.pop("per_anchor")
                                rows.append(dict(checkpoint=checkpoint_name, cohort=cohort, composition=composition,
                                                 mode=mode, P=p, batch=batch_index, **stats))
                        if (batch_index + 1) % 10 == 0 or batch_index + 1 == args.batches:
                            print(f"{checkpoint_name}/{cohort}/{composition}: {batch_index + 1}/{args.batches} fixed superbatches", flush=True)
                    del loader
            after = state_digest(model)
            unchanged = before == after and all(p._version == versions[n] for n, p in model.named_parameters())
            gradients_absent = all(p.grad is None for p in model.parameters())
            if not unchanged or not gradients_absent:
                raise RuntimeError("No-update invariant failed")
            states[checkpoint_name] = dict(before=before, after=after, parameter_versions_unchanged=unchanged,
                                           parameter_gradients_absent=gradients_absent)
            all_grad_absent &= gradients_absent
            with (out / (checkpoint_name + "_batch_statistics.jsonl")).open("w") as handle:
                for row in rows:
                    if row["checkpoint"] == checkpoint_name:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if any(digest(path) != hashes[name] for name, path in checkpoint_paths.items()):
        raise RuntimeError("Checkpoint file changed during the scan")
    if digest(plan_path) != plan_hash or digest(Path(plan["train_manifest"])) != manifest_hash:
        raise RuntimeError("Experiment plan or training manifest changed during the scan")
    protocol = dict(optimizer_created=False, optimizer_steps=0, backward_passes=0,
        parameter_state_unchanged=True, parameter_gradients_absent=all_grad_absent,
        checkpoint_paths={name: str(path) for name, path in checkpoint_paths.items()},
        checkpoint_sha256=hashes, parameter_state_sha256=states, plan_sha256=plan_hash,
        train_manifest_sha256=manifest_hash,
        scanner_sha256=digest(Path(__file__)), helper_sha256=digest(ROOT / "workflow/mining_diagnostics.py"),
        fixed_batch_manifest_sha256=digest(manifest_path), seed=args.seed,
        augmentation="fixed_once_per_place_epoch_flip_color" if not args.no_augmentation else "resize_only",
        batches_per_panel=args.batches, places={key: len(value) for key, value in panels.items()},
        excluded_places=len(excluded), cohorts=args.cohorts, compositions=args.compositions, modes=args.modes,
        nested_P=[8, 16, 32], K=4, common_anchor_count=32, forward_batch_size=args.forward_batch_size,
        dropout_sampling="single_P32_forward_prefixes" if args.forward_batch_size >= 128 else "microbatch_P32_forward_prefixes",
        zero_gradient_certificate="empty exact mined masks only; no measured gradients",
        cuda_budget=budget, elapsed_seconds=time.perf_counter() - started,
        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.)
    save_results(out, rows, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "outputs/ccr_accepted_13710_svox_categories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", choices=["initial", "A", "M", "B0", "B4", "B5"], default=["initial"])
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3043)
    parser.add_argument("--cohorts", nargs="+", choices=["training_subset", "full_real"], default=["training_subset"])
    parser.add_argument("--compositions", nargs="+", choices=["real", "mix"], default=["real"])
    parser.add_argument("--modes", nargs="+", choices=["train_bf16", "eval_fp32"], default=["train_bf16"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--forward-batch-size", type=int, default=128)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--save-per-anchor", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("batches", "torch_threads", "forward_batch_size"):
        if getattr(args, name) < 1:
            parser.error(name + " must be positive")
    if args.num_workers < 0 or args.forward_batch_size > 128:
        parser.error("num_workers must be nonnegative and forward_batch_size <= 128")
    for name in ("checkpoints", "cohorts", "compositions", "modes"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(name + " must be unique")
    args.experiment_dir = args.experiment_dir.resolve()
    out = args.output_dir.resolve()
    if args.worker:
        run_worker(args, out)
        return
    plan = json.loads((args.experiment_dir / "plan.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    run_child([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
              out / "scan.log", args.torch_threads, limits=plan["config"].get("resource_limits"))
    print(f"No-update scan report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
