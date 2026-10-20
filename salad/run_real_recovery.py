#!/usr/bin/env python3
"""Real-only forgetting controls. Default: freeze a plan, never launch GPU."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import shlex
import statistics
import sys

import run_am_multiseed as am
from workflow.am_statistics import paired, holm

ccr = am.ccr
CASES = {
    "weak": ("weak", False),
    "randaugment": ("randaugment", False),
    "weak_retention": ("weak", True),
    "randaugment_retention": ("randaugment", True),
}
CONTRASTS = (("randaugment", "weak"), ("weak_retention", "weak"),
             ("randaugment_retention", "randaugment"),
             ("randaugment_retention", "weak_retention"))


def load_config(path):
    c = json.loads(path.read_text())
    if set(c) != {"stage3_dir", "output_dir", "window_steps", "budgets", "retention_weight",
                  "retention_batch_size", "num_workers"}:
        raise ValueError("Unexpected recovery config keys")
    for key in ("stage3_dir", "output_dir"):
        c[key] = str((path.parent / c[key]).resolve())
    for key in ("window_steps", "retention_batch_size", "num_workers"):
        if type(c[key]) is not int or c[key] < (0 if key == "num_workers" else 1):
            raise ValueError("Invalid " + key)
    if (not c["budgets"] or c["budgets"] != sorted(set(c["budgets"]))
            or any(type(n) is not int or n < 1 or n % c["window_steps"] for n in c["budgets"])):
        raise ValueError("Budgets must be increasing multiples of window_steps")
    if not math.isfinite(c["retention_weight"]) or c["retention_weight"] <= 0:
        raise ValueError("Retention weight is a positive declared hypothesis")
    return c


def prepare(c):
    parent_dir = Path(c["stage3_dir"])
    out = Path(c["output_dir"])
    if out == parent_dir or parent_dir in out.parents:
        raise ValueError("Use an independent recovery output directory")
    parent = json.loads((parent_dir / "plan.json").read_text())
    am.validate_plan(parent, parent_dir)
    formal = json.loads((parent_dir / "formal_protocol.json").read_text())
    if formal["plan_sha256"] != ccr.digest(parent_dir / "plan.json"):
        raise ValueError("Stage 3 protocol does not match its data plan")
    if parent["config"]["precision"] not in ("bf16", "32"):
        raise ValueError("Recovery requires BF16/FP32")
    # Reuse only validated ordinary validation, never historical test.
    if any(e["split"] != "val" or e.get("role") != "validation"
           or e.get("query_subdirs") != ["queries"] or e["dataset"] != "SVOX"
           for e in parent["evaluations"]):
        raise ValueError("Expected Stage 3 ordinary SVOX validation only")
    sources = [Path(__file__).resolve(), ccr.ROOT / "train_real_recovery.py"]
    plan = dict(
        config=c, parent=parent, formal=formal,
        parent_plan_sha256=ccr.digest(parent_dir / "plan.json"),
        parent_formal_sha256=ccr.digest(parent_dir / "formal_protocol.json"),
        source_hashes={str(p): ccr.digest(p) for p in sources},
        cases=CASES, missing_independent_natural_weather_validation=True,
        protocol=dict(
            objective="Separate augmentation, update budget and real descriptor retention",
            budgets="Predeclared checkpoints on one continuous fixed-horizon trajectory",
            scheduler="Cosine over windows, common horizon=max(budgets); not restarted at 500",
            sampling="Full eligible real pool, continuous coverage order across checkpoint windows",
            initialization="Every case/seed starts independently from frozen initial checkpoint",
            randomization="Keyed augmentation; teacher initialization restores Torch RNG",
            selection="No automatic winner, best-checkpoint selection or weather-success claim",
            validation="Ordinary SVOX val only. Synthetic proxies cannot certify natural-weather retention",
            statistics="Per-seed paired queries, Holm across planned contrasts/budgets/seeds; no pooled queries",
        ),
    )
    am.immutable(out / "plan.json", ccr.encoded(plan))
    am.immutable(out / "config.json", ccr.encoded(c))
    return plan


def validate(plan):
    c = plan["config"]
    root = Path(c["stage3_dir"])
    if (ccr.digest(root / "plan.json") != plan["parent_plan_sha256"]
            or ccr.digest(root / "formal_protocol.json") != plan["parent_formal_sha256"]):
        raise ValueError("Inherited protocol changed")
    am.validate_plan(plan["parent"], root)
    for path, sha in plan["source_hashes"].items():
        if ccr.digest(path) != sha:
            raise ValueError("Frozen recovery source changed; use a new output directory")


def parent_runtime(plan):
    parent = plan["parent"]
    return {**parent, "config": {**parent["config"], "num_workers": plan["config"]["num_workers"]}}


def command(plan, case, seed, directory, until, resume=False):
    c = plan["config"]
    horizon = max(c["budgets"])
    parent = parent_runtime(plan)
    parent = {**parent, "config": {**parent["config"], "epochs": horizon // c["window_steps"]}}
    cmd = am.baseline.training_command(parent, plan["formal"]["spec"], seed, directory,
                                       until // c["window_steps"], resume)
    cmd[1] = str(ccr.ROOT / "train_real_recovery.py")
    # Preserve numbered endpoints, with cheap overwrite-only intermediate saves.
    cmd[cmd.index("--save-every") + 1] = str(math.gcd(*c["budgets"]) // c["window_steps"])
    policy, retention = CASES[case]
    cmd += ["--augmentation-policy", policy, "--window-steps", str(c["window_steps"]),
            "--max-steps", str(horizon)]
    if retention:
        cmd += ["--teacher-checkpoint", parent["config"]["checkpoint"],
                "--real-global-distill-weight", str(c["retention_weight"]),
                "--real-global-distill-batch-size", str(c["retention_batch_size"])]
    return cmd


def check_checkpoint(saved, plan, case, seed):
    c, t, spec = plan["config"], saved["training_config"], plan["formal"]["spec"]
    pc = plan["parent"]["config"]
    policy, retention = CASES[case]
    if (saved["global_step"] != saved["epoch"] * c["window_steps"]
            or saved["total_epochs"] != max(c["budgets"]) // c["window_steps"]
            or saved["global_step"] > max(c["budgets"])
            or t["seed"] != seed or t["recovery"]["augmentation"] != policy
            or t["recovery"]["num_workers"] != c["num_workers"]
            or t["max_steps"] != max(c["budgets"])
            or t["batch_size"] != spec["batch_size"]
            or t["learning_rate"] != spec["learning_rate"]
            or t["model_config"]["backbone_config"]["num_trainable_blocks"] != spec["num_trainable_blocks"]
            or t["init_checkpoint_sha256"] != plan["parent"]["input_hashes"][pc["checkpoint"]]
            or t.get("real_descriptor_retention", {}).get("weight", 0) != (c["retention_weight"] if retention else 0)
            or t["synthetic_manifest"] or t["synthetic_fraction"]):
        raise ValueError("Recovery checkpoint violates case/seed/budget/provenance")
    if saved["scheduler_state_dict"]["last_epoch"] != saved["epoch"]:
        raise ValueError("Scheduler and checkpoint window disagree")


def checked_result(path, evaluation, checkpoint_sha):
    from workflow.am_statistics import query_ranks
    result = json.loads(path.read_text())
    receipt = json.loads(path.with_suffix(".receipt.json").read_text())
    if (receipt["result_sha256"] != ccr.digest(path)
            or receipt["checkpoint_sha256"] != checkpoint_sha
            or receipt["protocol"] != evaluation):
        raise ValueError("Stale or incompatible validation audit/receipt")
    query_ranks(result)
    return result


def execute(plan, case, seed, until):
    from workflow.model import read_checkpoint
    c, pc = plan["config"], plan["parent"]["config"]
    directory = Path(c["output_dir"]) / "runs" / str(seed) / case
    am.immutable(directory / "contract.json", ccr.encoded(dict(
        case=case, seed=seed, plan_sha256=ccr.digest(Path(c["output_dir"]) / "plan.json"))))
    path = directory / "checkpoint.pt"
    done = 0
    if path.exists():
        saved = read_checkpoint(path)
        check_checkpoint(saved, plan, case, seed)
        done = saved["global_step"]
        del saved
    if done < until:
        ccr.run_child(command(plan, case, seed, directory, until, path.exists()),
                      directory / "train.log", pc["torch_threads"], limits=pc["resource_limits"])
    saved = read_checkpoint(path)
    check_checkpoint(saved, plan, case, seed)
    if saved["global_step"] < until:
        raise RuntimeError("Requested budget did not complete")
    del saved
    ep = am.ep_plan(parent_runtime(plan))
    ccr.evaluate_checkpoint(ep, Path(pc["checkpoint"]), Path(c["output_dir"]) / "initial")
    for budget in c["budgets"]:
        if budget > until:
            continue
        checkpoint = directory / f"checkpoint_epoch_{budget // c['window_steps']:03d}.pt"
        saved = read_checkpoint(checkpoint)
        check_checkpoint(saved, plan, case, seed)
        if saved["global_step"] != budget:
            raise ValueError("Wrong numbered checkpoint for budget")
        payload = dict(case=case, seed=seed, optimizer_updates=budget,
                       checkpoint_sha256=ccr.digest(checkpoint),
                       latest_metrics=saved["metrics"],
                       scheduler=saved["scheduler_state_dict"],
                       training_config=saved["training_config"],
                       window_count=saved["epoch"],
                       windows_are_not_full_epochs=True)
        windows = {}
        for line in (directory / "training_log.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["global_step"] <= budget:
                windows[row["epoch"]] = row
        windows[saved["epoch"]] = saved["metrics"]
        payload["training_windows"] = [windows[k] for k in sorted(windows)]
        payload["complete_window_log"] = sorted(windows) == list(range(1, saved["epoch"] + 1))
        del saved
        target = directory / "validation" / str(budget)
        am.immutable(target / "audit.json", ccr.encoded(payload))
        ccr.evaluate_checkpoint(ep, checkpoint, target)


def summarize(plan):
    out = Path(plan["config"]["output_dir"])
    c = plan["config"]
    seeds = [plan["parent"]["config"]["screen_seed"], *plan["formal"]["seeds"]]
    runs, comparisons, lookup = [], [], {}
    evaluation = plan["parent"]["evaluations"][0]
    initial_path = out / "initial/svox_val.json"
    initial = checked_result(initial_path, evaluation,
        plan["parent"]["input_hashes"][plan["parent"]["config"]["checkpoint"]]) if initial_path.exists() else None
    for seed in seeds:
        for case in CASES:
            for budget in c["budgets"]:
                directory = out / "runs" / str(seed) / case / "validation" / str(budget)
                if not (directory / "svox_val.json").exists() or not (directory / "audit.json").exists():
                    continue
                audit = json.loads((directory / "audit.json").read_text())
                result = checked_result(directory / "svox_val.json", evaluation, audit["checkpoint_sha256"])
                lookup[seed, case, budget] = result
                runs.append(dict(seed=seed, case=case, budget=budget, recall=result["recall"], audit=audit))
    for seed in seeds:
        for budget in c["budgets"]:
            for method, reference in CONTRASTS:
                if (seed, method, budget) in lookup and (seed, reference, budget) in lookup:
                    for k in (1, 5, 10):
                        comparisons.append(dict(seed=seed, budget=budget, method=method, reference=reference, k=k,
                            **paired(lookup[seed, reference, budget], lookup[seed, method, budget], k=k)))
    for (seed, case, budget), result in lookup.items():
        if initial is not None:
            for k in (1, 5, 10):
                comparisons.append(dict(seed=seed, budget=budget, method=case, reference="initial", k=k,
                                        **paired(initial, result, k=k)))
    family_size = len(seeds) * len(c["budgets"]) * (len(CONTRASTS) + len(CASES)) * 3
    for row, corrected in zip(comparisons, holm([r["p_exact"] for r in comparisons], family_size)):
        row["p_holm"] = corrected
    distributions = []
    for budget in c["budgets"]:
        for method, reference in (*CONTRASTS, *((case, "initial") for case in CASES)):
            rows = [r for r in comparisons if r["budget"] == budget and r["method"] == method
                    and r["reference"] == reference and r["k"] == 1 and r["seed"] in plan["formal"]["seeds"]]
            values = [r["delta_pp"] for r in rows]
            if values:
                distributions.append(dict(budget=budget, method=method, reference=reference, seeds=[r["seed"] for r in rows],
                    delta_pp=values, mean_pp=statistics.mean(values),
                    std_pp=statistics.stdev(values) if len(values)>1 else None,
                    complete=len(values)==len(plan["formal"]["seeds"])))
    report = dict(status="pilot_results" if runs else "prepared_not_trained",
                  weather_retention_status="unverified_missing_independent_natural_weather_validation",
                  runs=runs, paired_statistics=comparisons, planned_holm_family_size=family_size,
                  seed_distributions=distributions, pooled_seed_query_test=False, automatic_selection=False)
    ccr.write_json(out / "report.json", report)
    lines = ["# Real-only recovery controls", "", "Natural-weather retention: UNVERIFIED.",
             "Ordinary validation and synthetic proxies cannot certify it. No test is read.",
             "Checkpoint windows are not full data epochs. All budgets share the same cosine horizon.", "",
             "| Seed | Case | Updates | Val R1 | MS / total loss | Active anchors |",
             "|---|---|---:|---:|---:|---:|"]
    for r in runs:
        metrics = r["audit"]["latest_metrics"]
        lines.append(f"| {r['seed']} | {r['case']} | {r['budget']} | {100*r['recall']['R@1']:.4f}% | {metrics['loss']:.5f} | {100*metrics['mining']['active_anchor_fraction']:.2f}% |")
    lines += ["", "No best checkpoint or winner is selected. Each seed is paired separately.",
              "See report.json for retention loss, cosine, clipping, coverage, GPU peaks and planned Holm tests.",
              "A non-regressing ordinary validation score is not proof of preserved weather robustness."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ccr.ROOT / "configs/real_recovery.json")
    parser.add_argument("--phase", choices=("prepare", "run", "report"), default="prepare")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--until", type=int, default=500)
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
                raise ValueError("Frozen config changed")
            if args.phase != "report":
                validate(plan)
        else:
            plan = prepare(c)
        if args.phase == "run":
            if not args.case or args.until not in c["budgets"]:
                raise ValueError("Choose an explicit case and a predeclared budget")
            if args.seed not in [plan["parent"]["config"]["screen_seed"], *plan["formal"]["seeds"]]:
                raise ValueError("Seed is not in the inherited protocol")
            directory = out / "runs" / str(args.seed) / args.case
            if args.execute:
                execute(plan, args.case, args.seed, args.until)
            else:
                print(shlex.join(command(plan, args.case, args.seed, directory, args.until,
                                         (directory / "checkpoint.pt").exists())))
        report = summarize(plan)
        print(f"Status: {report['status']}; report={out / 'report.md'}")
        if not args.execute:
            print("No GPU or training launched.")


if __name__ == "__main__":
    main()
