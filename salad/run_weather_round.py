#!/usr/bin/env python3
"""Stage 5: A / A+retention / M / B0 under RandAugment, scored on natural weather.

Every method starts from the same initial checkpoint and uses one place stream:
``--synthetic-place-slots`` reserves places with accepted variants in each batch
and ``--prefer-synthetic-sources`` puts their exact sources into the K views, for
real-only controls too. Methods differ only in what happens to those sources:

  A            real only
  A_retention  real only + frozen-initial descriptor retention
  M            sources replaced by their accepted variants inside the MS batch
  B0           sources kept; variants added as InfoNCE queries (per-anchor scaled)

Real-only methods run in ``mix`` mode with fraction 0 like B0, so A and B0 see
byte-identical main batches; M differs from A only where a source is replaced.

Every epoch is scored on ordinary SVOX val and on the SVOX-train weather manifest
(``build_weather_validation.py``); externals run on the final epoch only. SVOX
test is never read. Default phase only freezes the plan; training needs --execute.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import shlex
import statistics
import sys

import run_ccr_feasibility as ccr
from run_qwen_finetune import TRAINING_FIELDS
from workflow.am_statistics import holm, paired, query_ranks

ROOT = ccr.ROOT
METHOD_KEYS = {"synthetic_mode", "synthetic_fraction", "synthetic_aux", "synthetic_global_weight",
               "synthetic_exposure_prob", "real_global_distill_weight"}
REQUIRED = {"output_dir", "real_data", "checkpoint", "backbone_repo", "exclude_places", "synthetic_manifests",
            "svox_root", "weather_validation", "external_validation", "training", "methods", "reference_method",
            "seeds", "device", "num_workers", "prefetch_factor", "torch_threads", "eval_batch_size",
            "eval_query_chunk_size", "eval_reference_chunk_size", "eval_reference_cache_mib", "resource_limits"}
TRAINING_KEYS = {"batch_size", "images_per_place", "learning_rate", "num_trainable_blocks", "epochs",
                 "precision", "synthetic_place_slots", "prefer_synthetic_sources", "retention_batch_size"}


def load_config(path):
    c = json.loads(path.read_text())
    if set(c) != REQUIRED:
        raise ValueError(f"Config keys missing={REQUIRED - set(c)}, unknown={set(c) - REQUIRED}")
    if set(c["training"]) != TRAINING_KEYS:
        raise ValueError("Unexpected training keys")
    resolve = lambda value: str((path.parent / value).resolve())
    for key in ("output_dir", "real_data", "checkpoint", "exclude_places", "svox_root", "weather_validation"):
        c[key] = resolve(c[key])
    c["synthetic_manifests"] = [resolve(v) for v in c["synthetic_manifests"]]
    c["external_validation"] = [resolve(v) for v in c["external_validation"]]
    if c["reference_method"] not in c["methods"] or not c["seeds"]:
        raise ValueError("Reference method must be declared and seeds nonempty")
    for name, options in c["methods"].items():
        if set(options) - METHOD_KEYS:
            raise ValueError(f"Unknown options for method {name}: {set(options) - METHOD_KEYS}")
        if options.get("synthetic_aux") and options.get("synthetic_mode", "mix") != "mix":
            raise ValueError("B0-style auxiliary methods use an all-real main batch (mix mode, fraction 0)")
    return c


def snapshot(c, out):
    """Freeze accepted rows of every manifest; image bytes are verified once."""
    inputs = out / "inputs"
    target = inputs / "accepted_manifest.jsonl"
    if target.exists():
        return json.loads((inputs / "snapshot.json").read_text())
    rows, seen, counts = [], set(), Counter()
    for manifest in c["synthetic_manifests"]:
        for line in Path(manifest).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            counts["rows"] += 1
            if row.get("passed") is not True or row.get("eligible_for_training") is not True:
                counts["not_eligible"] += 1
                continue
            if row["output_path"] in seen:
                counts["duplicate"] += 1
                continue
            for key in ("source", "output"):
                path = Path(row[key + "_path"])
                if not path.is_file() or (row.get(key + "_sha256") and ccr.digest(path) != row[key + "_sha256"]):
                    raise ValueError(f"Missing or changed image: {path}")
            seen.add(row["output_path"])
            rows.append({k: row[k] for k in TRAINING_FIELDS if k in row})
    inputs.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    summary = dict(manifests={m: ccr.digest(m) for m in c["synthetic_manifests"]}, counts=dict(counts),
                   accepted=len(rows), conditions=dict(Counter(r["condition"] for r in rows)),
                   places=len({(r["city"], r["place_id"]) for r in rows}), sha256=ccr.digest(target))
    ccr.write_json(inputs / "snapshot.json", summary)
    return summary


def evaluations(c):
    result = [dict(name="svox_val", dataset="SVOX", root=c["svox_root"], split="val", role="validation",
                   cadence="every", kind="real", query_subdirs=["queries"]),
              dict(name="svox_train_weather", dataset="manifest", root=c["svox_root"], split="train",
                   role="validation", cadence="every", kind="natural_weather", manifest=c["weather_validation"])]
    for directory in c["external_validation"]:
        protocol = json.loads((Path(directory) / "protocol.json").read_text())
        result.append(dict(name=Path(directory).name, dataset="manifest", root=protocol["root"], split="test",
                           role="external_exploratory", cadence="final", kind="external",
                           manifest=str(Path(directory) / "manifest.json")))
    return result


def source_paths():
    paths = [Path(__file__), ROOT / "train_am.py", ROOT / "train_salad.py", ROOT / "evaluate_salad.py",
             Path(ccr.__file__)]
    return paths + sorted((ROOT / "workflow").glob("*.py")) + sorted((ROOT / "models/aggregators").glob("*.py"))


def prepare(c):
    out = Path(c["output_dir"])
    snap = snapshot(c, out)
    inputs = [c["checkpoint"], c["exclude_places"], c["weather_validation"],
              *[str(Path(d) / "manifest.json") for d in c["external_validation"]]]
    plan = dict(version=1, config=c, snapshot=snap, evaluations=evaluations(c),
                input_hashes={p: ccr.digest(p) for p in inputs},
                source_hashes={str(p): ccr.digest(p) for p in source_paths()},
                protocol=dict(
                    stream="All methods share the stratified place stream and preferred exact sources",
                    selection="No automatic winner; final-epoch paired tests are the declared comparison",
                    statistics="Per-seed exact McNemar on R@1, Holm over seeds x contrasts x groups; seeds never pooled",
                    test="SVOX test is not read by this runner"))
    ccr.immutable(out / "plan.json", ccr.encoded(plan))
    return plan


def validate(plan):
    for group in ("input_hashes", "source_hashes"):
        for name, expected in plan[group].items():
            if ccr.digest(name) != expected:
                raise ValueError(f"Frozen {group} changed: {name}; use a new output_dir")
    if snapshot(plan["config"], Path(plan["config"]["output_dir"])) != plan["snapshot"]:
        raise ValueError("Synthetic snapshot changed")


def command(plan, method, seed, directory, stop_epoch, resume):
    c, t = plan["config"], plan["config"]["training"]
    options = {"synthetic_mode": "mix", "synthetic_fraction": 0.0, **c["methods"][method]}
    k = str(t["images_per_place"])
    cmd = [sys.executable, str(ROOT / "train_am.py"), "--real-data", c["real_data"], "--output-dir", str(directory),
           "--synthetic-manifest", str(Path(c["output_dir"]) / "inputs/accepted_manifest.jsonl"),
           "--exclude-places", c["exclude_places"], "--real-sampling", "coverage_pk",
           "--images-per-place", k, "--min-images-per-place", k, "--batch-size", str(t["batch_size"]),
           "--learning-rate", str(t["learning_rate"]), "--num-trainable-blocks", str(t["num_trainable_blocks"]),
           "--weight-decay", "0", "--miner-margin", "0.1", "--seed", str(seed), "--epochs", str(t["epochs"]),
           "--stop-after-epoch", str(stop_epoch), "--save-every", "1", "--num-workers", str(c["num_workers"]),
           "--prefetch-factor", str(c["prefetch_factor"]), "--image-size", "224", "224",
           "--precision", t["precision"], "--device", c["device"], "--backbone-repo", c["backbone_repo"],
           "--log-mined-pairs", "--profile-training", "--synthetic-place-slots", str(t["synthetic_place_slots"]),
           "--synthetic-mode", options["synthetic_mode"], "--synthetic-fraction", str(options["synthetic_fraction"])]
    if t["prefer_synthetic_sources"]:
        cmd.append("--prefer-synthetic-sources")
    if options.get("synthetic_aux"):
        cmd += ["--synthetic-aux", "--isolate-auxiliary-rng",
                "--synthetic-global-weight", str(options.get("synthetic_global_weight", 1.0)),
                "--synthetic-exposure-prob", str(options.get("synthetic_exposure_prob", 1.0))]
    if options.get("real_global_distill_weight"):
        cmd += ["--teacher-checkpoint", c["checkpoint"],
                "--real-global-distill-weight", str(options["real_global_distill_weight"]),
                "--real-global-distill-batch-size", str(t["retention_batch_size"])]
    cmd += ["--resume", str(directory / "checkpoint.pt")] if resume else ["--init-checkpoint", c["checkpoint"]]
    return cmd


def check_state(saved, plan, method, seed):
    c, t, config = plan["config"], plan["config"]["training"], saved["training_config"]
    options = {"synthetic_mode": "mix", "synthetic_fraction": 0.0, **c["methods"][method]}
    expected = dict(seed=seed, batch_size=t["batch_size"], learning_rate=t["learning_rate"],
                    synthetic_place_slots=t["synthetic_place_slots"],
                    prefer_synthetic_sources=t["prefer_synthetic_sources"],
                    synthetic_mode=options["synthetic_mode"],
                    synthetic_fraction=0.0 if options.get("synthetic_aux") else options["synthetic_fraction"],
                    init_checkpoint_sha256=plan["input_hashes"][c["checkpoint"]])
    wrong = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if bool(config.get("synthetic_aux")) != bool(options.get("synthetic_aux")):
        wrong["synthetic_aux"] = True
    if config.get("real_descriptor_retention", {}).get("weight", 0) != options.get("real_global_distill_weight", 0):
        wrong["retention"] = True
    if wrong:
        raise ValueError(f"Checkpoint violates {method}/seed {seed}: {wrong}")


def run_one(plan, method, seed):
    from workflow.model import read_checkpoint
    c = plan["config"]
    out = Path(c["output_dir"])
    directory = out / "runs" / str(seed) / method
    directory.mkdir(parents=True, exist_ok=True)
    ccr.immutable(directory / "contract.json", ccr.encoded(dict(method=method, seed=seed,
                  plan_sha256=ccr.digest(out / "plan.json"))))
    ccr.evaluate_checkpoint(plan, Path(c["checkpoint"]), out / "initial")
    epochs = c["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        checkpoint = directory / f"checkpoint_epoch_{epoch:03d}.pt"
        if not checkpoint.exists():
            resume = (directory / "checkpoint.pt").exists()
            if resume:
                saved = read_checkpoint(directory / "checkpoint.pt")
                check_state(saved, plan, method, seed)
                if saved["epoch"] != epoch - 1:
                    raise ValueError("Latest checkpoint is not the previous epoch; inspect manually")
                del saved
            ccr.run_child(command(plan, method, seed, directory, epoch, resume), directory / "train.log",
                          c["torch_threads"], limits=c["resource_limits"])
        saved = read_checkpoint(checkpoint)
        check_state(saved, plan, method, seed)
        del saved
        ccr.evaluate_checkpoint(plan, checkpoint, directory / "validation" / str(epoch), final=epoch == epochs)


def group_of(evaluation, query_id):
    if evaluation == "svox_train_weather":
        return query_id.split("/")[-2].removeprefix("queries_")
    return query_id.split("/")[0] if evaluation == "tokyo247_reduced" else "all"


def split_groups(name, result):
    ranks = query_ranks(result)
    groups = {}
    for query, rank in ranks.items():
        groups.setdefault(group_of(name, query), {})[query] = rank
    return {g: dict(query_results=[dict(query_id=q, rank=r) for q, r in sorted(rows.items())],
                    recall={f"R@{k}": sum(r is not None and r <= k for r in rows.values())
                            / max(1, sum(r is not None for r in rows.values())) for k in (1, 5, 10)})
            for g, rows in groups.items()}


def load_result(directory, evaluation, checkpoint_sha):
    path = directory / (evaluation["name"] + ".json")
    receipt = directory / (evaluation["name"] + ".receipt.json")
    if not path.exists() or not receipt.exists():
        return None
    saved = json.loads(receipt.read_text())
    if (saved["checkpoint_sha256"] != checkpoint_sha or saved["protocol"] != evaluation
            or saved["result_sha256"] != ccr.digest(path)):
        raise ValueError(f"Stale evaluation: {path}")
    return json.loads(path.read_text())


def summarize(plan):
    c = plan["config"]
    out = Path(c["output_dir"])
    epochs, reference = c["training"]["epochs"], c["reference_method"]
    results, rows, audits = {}, [], []
    targets = [("initial", None, 0, out / "initial", c["checkpoint"])]
    for seed in c["seeds"]:
        for method in c["methods"]:
            for epoch in range(1, epochs + 1):
                checkpoint = out / "runs" / str(seed) / method / f"checkpoint_epoch_{epoch:03d}.pt"
                if checkpoint.exists():
                    targets.append((method, seed, epoch, checkpoint.parent / "validation" / str(epoch), checkpoint))
    for method, seed, epoch, directory, checkpoint in targets:
        checksum = ccr.digest(checkpoint)
        for evaluation in plan["evaluations"]:
            result = load_result(directory, evaluation, checksum)
            if result is None:
                continue
            for group, part in split_groups(evaluation["name"], result).items():
                results[method, seed, epoch, evaluation["name"], group] = part
                rows.append(dict(method=method, seed=seed, epoch=epoch, evaluation=evaluation["name"],
                                 group=group, queries=len(part["query_results"]), recall=part["recall"]))
        if method != "initial":
            log = {json.loads(line)["epoch"]: json.loads(line)
                   for line in (checkpoint.parent / "training_log.jsonl").read_text().splitlines() if line.strip()}
            metrics = log[epoch]
            audits.append(dict(method=method, seed=seed, epoch=epoch, loss=metrics["loss"],
                               synthetic_fraction_observed=metrics["synthetic_fraction_observed"],
                               synthetic_weather=metrics["sampling"].get("synthetic_weather_exposures"),
                               active_anchor_fraction=metrics.get("mining", {}).get("active_anchor_fraction"),
                               clip_batch_fraction=metrics.get("mining", {}).get("clip_batch_fraction"),
                               synthetic_aux=metrics.get("synthetic_aux"),
                               retention=metrics.get("real_descriptor_retention")))
    groups = sorted({(r["evaluation"], r["group"]) for r in rows if r["method"] == "initial"})
    comparisons = []
    for seed in c["seeds"]:
        for method in c["methods"]:
            for baseline in ("initial", reference):
                if baseline == method:
                    continue
                for evaluation, group in groups:
                    left = results.get((baseline, None if baseline == "initial" else seed,
                                        0 if baseline == "initial" else epochs, evaluation, group))
                    right = results.get((method, seed, epochs, evaluation, group))
                    if left is None or right is None:
                        continue
                    for k in (1, 5, 10):
                        comparisons.append(dict(seed=seed, method=method, reference=baseline, evaluation=evaluation,
                                                group=group, k=k, **paired(left, right, k=k)))
    primary = [r for r in comparisons if r["k"] == 1]
    family = len(c["seeds"]) * (2 * len(c["methods"]) - 1) * len(groups)
    for row, p in zip(primary, holm([r["p_exact"] for r in primary], max(family, len(primary)))):
        row["p_holm"] = p
    distributions = []
    for method in c["methods"]:
        for baseline in ("initial", reference):
            for evaluation, group in groups:
                values = [r["delta_pp"] for r in primary if (r["method"], r["reference"], r["evaluation"],
                          r["group"]) == (method, baseline, evaluation, group)]
                if values and baseline != method:
                    distributions.append(dict(method=method, reference=baseline, evaluation=evaluation, group=group,
                                              seeds=len(values), mean_pp=statistics.mean(values),
                                              std_pp=statistics.stdev(values) if len(values) > 1 else None,
                                              positive_seeds=sum(v > 0 for v in values)))
    report = dict(status="complete" if all((m, s, epochs) in {(r["method"], r["seed"], r["epoch"]) for r in rows}
                                            for m in c["methods"] for s in c["seeds"]) else "partial",
                  rows=rows, training=audits, paired_statistics=comparisons, holm_family_size=family,
                  seed_distributions=distributions, snapshot=plan["snapshot"])
    ccr.write_json(out / "report.json", report)
    lines = ["# Stage 5 weather round", "", f"Status: {report['status']}. SVOX test is not read.",
             f"Synthetic snapshot: {plan['snapshot']['accepted']} images, {plan['snapshot']['places']} places, "
             f"{plan['snapshot']['conditions']}", "",
             "## Final-epoch R@1 (%)", "", "| Method | Seed | " + " | ".join(f"{e}/{g}" for e, g in groups) + " |",
             "|---|---|" + "---:|" * len(groups)]
    for method, seed in [("initial", None)] + [(m, s) for s in c["seeds"] for m in c["methods"]]:
        epoch = 0 if method == "initial" else epochs
        cells = [results.get((method, seed, epoch, e, g)) for e, g in groups]
        if any(cells):
            lines.append(f"| {method} | {seed if seed is not None else '—'} | " + " | ".join(
                f"{100 * x['recall']['R@1']:.2f}" if x else "—" for x in cells) + " |")
    lines += ["", "## Mean R@1 change over seeds (pp)", "", "| Method | vs | " + " | ".join(f"{e}/{g}" for e, g in groups) + " |",
              "|---|---|" + "---:|" * len(groups)]
    index = {(d["method"], d["reference"], d["evaluation"], d["group"]): d for d in distributions}
    for method in c["methods"]:
        for baseline in ("initial", reference):
            if baseline == method:
                continue
            cells = [index.get((method, baseline, e, g)) for e, g in groups]
            if any(cells):
                lines.append(f"| {method} | {baseline} | " + " | ".join(
                    f"{d['mean_pp']:+.2f} ({d['positive_seeds']}/{d['seeds']})" if d else "—" for d in cells) + " |")
    lines += ["", "## Training exposure (final epoch)", "", "| Method | Seed | Synthetic share | Active anchors | Clipped batches | Weather |",
              "|---|---|---:|---:|---:|---|"]
    for a in audits:
        if a["epoch"] == epochs:
            weather = a["synthetic_weather"] or (a["synthetic_aux"] or {}).get("condition_exposure")
            lines.append(f"| {a['method']} | {a['seed']} | {100 * a['synthetic_fraction_observed']:.2f}% | "
                         f"{100 * (a['active_anchor_fraction'] or 0):.2f}% | {100 * (a['clip_batch_fraction'] or 0):.2f}% | {weather} |")
    lines += ["", "(positive seeds / seeds) follow each mean. Per-seed exact McNemar with Holm-adjusted p values are in report.json.",
              "B0's synthetic share is 0 in the MS batch; its auxiliary query counts are in report.json.",
              "Externals are reduced or probe galleries: exploratory, never official benchmark numbers."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/weather_round.json")
    parser.add_argument("--phase", choices=("prepare", "run", "report"), default="prepare")
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    c = load_config(args.config.resolve())
    out = Path(c["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "plan.json").exists():
            plan = json.loads((out / "plan.json").read_text())
            if plan["config"] != c:
                raise ValueError("Frozen config changed; use a new output_dir")
            if args.phase != "report":
                validate(plan)
        else:
            plan = prepare(c)
        if args.phase == "run":
            methods = [args.method] if args.method else list(c["methods"])
            seeds = [args.seed] if args.seed is not None else c["seeds"]
            if set(methods) - set(c["methods"]) or set(seeds) - set(c["seeds"]):
                raise ValueError("Method/seed not declared in the config")
            for seed in seeds:
                for method in methods:
                    directory = out / "runs" / str(seed) / method
                    if args.execute:
                        run_one(plan, method, seed)
                        summarize(plan)
                    else:
                        print(shlex.join(command(plan, method, seed, directory, 1, False)))
        report = summarize(plan)
        print(f"Status: {report['status']}; report={out / 'report.md'}")
        if not args.execute:
            print("No GPU or training launched.")


if __name__ == "__main__":
    main()
