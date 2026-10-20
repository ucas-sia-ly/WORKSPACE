#!/usr/bin/env python3
"""Post-hoc, read-only SVOX category audit of completed recovery checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import run_real_baseline as baseline
import run_am_multiseed as am
from workflow.am_statistics import paired, holm, query_ranks

ccr = baseline.ccr
DOMAINS = ("default", "night", "rain", "snow", "sun", "overcast")
CASES = ("weak", "randaugment", "weak_retention", "randaugment_retention")


def verified(path, checksum, protocol):
    receipt_path = path.with_suffix(".receipt.json")
    if not path.exists() or not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("checkpoint_sha256") != checksum or receipt.get("protocol") != protocol
            or receipt.get("result_sha256") != ccr.digest(path)):
        raise ValueError("Stale SVOX result or changed evaluation protocol: " + str(path))
    result = json.loads(path.read_text())
    query_ranks(result)
    return result


def setup(root):
    plan = json.loads((root / "plan.json").read_text())
    parent = plan["parent"]
    initial = Path(parent["config"]["checkpoint"])
    if ccr.digest(initial) != parent["input_hashes"][str(initial)]:
        raise ValueError("Initial checkpoint changed")
    # Follow the existing historical SVOX test protocol exactly.
    ep = baseline.evaluation_plan(parent, test=True)
    runtime = dict(ep["config"], num_workers=0, eval_batch_size=32,
                   eval_reference_cache_mib=512, prefetch_factor=1)
    return plan, dict(config=runtime, evaluations=ep["evaluations"])


def completed(root, case, seed, budget):
    directory = root / "runs" / str(seed) / case / "validation" / str(budget)
    audit_path = directory / "audit.json"
    checkpoint = root / "runs" / str(seed) / case / f"checkpoint_epoch_{budget//100:03d}.pt"
    if not audit_path.exists() or not checkpoint.exists():
        raise ValueError("Completed, audited training checkpoint required: " + str(directory))
    audit = json.loads(audit_path.read_text())
    if (audit.get("optimizer_updates") != budget or audit.get("case") != case
            or audit.get("seed") != seed or audit.get("checkpoint_sha256") != ccr.digest(checkpoint)):
        raise ValueError("Training audit/checkpoint mismatch")
    return checkpoint


def evaluate(plan, evaluation, root, cases, seed, budget):
    destination = root / "posthoc_svox_test" / str(seed)
    for case in cases:
        checkpoint = completed(root, case, seed, budget)
        ccr.evaluate_checkpoint(evaluation, checkpoint, destination / f"{case}_{budget}")


def summarize(plan, evaluation, root, seed, budget):
    initial_checksum = plan["parent"]["input_hashes"][plan["parent"]["config"]["checkpoint"]]
    source = Path(plan["config"]["stage3_dir"]).parent / "real_baseline_v1" / "test" / "initial"
    # Historical initial files must match the same checkpoint and exact query protocol.
    reference = {}
    for protocol in evaluation["evaluations"]:
        name = protocol["name"]
        reference[name] = verified(source / (name + ".json"), initial_checksum, protocol)
    if any(v is None for v in reference.values()):
        raise ValueError("Historical initial test results are missing; run the same grouped protocol for initial")

    out = root / "posthoc_svox_test" / str(seed)
    rows, comparisons = [], []
    results = {}
    for case in CASES:
        checkpoint = root / "runs" / str(seed) / case / f"checkpoint_epoch_{budget//100:03d}.pt"
        if not checkpoint.exists():
            continue
        sha = ccr.digest(checkpoint)
        for protocol in evaluation["evaluations"]:
            name = protocol["name"]
            path = out / f"{case}_{budget}" / (name + ".json")
            current = verified(path, sha, protocol)
            if current is None:
                continue
            results[case,name] = current
            rows.append(dict(case=case, domain=name, budget=budget, checkpoint_sha256=sha,
                             queries=current["num_evaluated_queries"], recall=current["recall"]))
            for k in (1,5,10):
                comparisons.append(dict(case=case, reference="initial", domain=name, k=k,
                                        **paired(reference[name],current,k=k)))
    for name in ("svox_" + d for d in DOMAINS):
        for trial, baseline_case in (("randaugment","weak"),
                                     ("weak_retention","weak"),
                                     ("randaugment_retention","randaugment"),
                                     ("randaugment_retention","weak_retention")):
            if (trial,name) in results and (baseline_case,name) in results:
                for k in (1,5,10):
                    comparisons.append(dict(case=trial,reference=baseline_case,domain=name,k=k,
                        **paired(results[baseline_case,name],results[trial,name],k=k)))
    # Full planned family including unavailable results: incomplete runs cannot gain significance.
    family = len(DOMAINS) * (len(CASES)+4) * 3
    for row,p in zip(comparisons,holm([r["p_exact"] for r in comparisons],family)):
        row["p_holm"] = p
    report = dict(status="complete" if len(rows)==len(DOMAINS)*len(CASES) else "partial",
                  use="post-hoc descriptive SVOX test; forbidden for tuning/checkpoint selection",
                  seed=seed,budget=budget,initial_source=str(source),initial_sha256=initial_checksum,
                  planned_holm_family_size=family,rows=rows,paired_statistics=comparisons)
    ccr.write_json(out / f"report_{budget}.json",report)
    lines = ["# SVOX test by category: recovery controls", "",
             f"Status: {report['status']}; seed={seed}; optimizer updates={budget}",
             "Historical test: descriptive only. Do not choose a training config from this table.",
             "", "| Category | Queries | Initial | Weak | RandAugment | Weak + retention | RandAugment + retention |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    lookup = {(r["domain"],r["case"]):r for r in rows}
    for domain in DOMAINS:
        name = "svox_" + domain
        base = reference[name]
        fields = []
        for case in CASES:
            row = lookup.get((name,case))
            fields.append("—" if row is None else f"{100*row['recall']['R@1']:.3f}%")
        lines.append(f"| {domain} | {base['num_evaluated_queries']} | {100*base['recall']['R@1']:.3f}% | " + " | ".join(fields) + " |")
    lines += ["", "Paired improved/regressed query counts, exact McNemar and Holm values are in report JSON.",
              "These are six real SVOX test categories; the training report's ordinary val is separate.",
              "No seed pooling, winner selection, or test-driven training change is performed."]
    (out / f"report_{budget}.md").write_text("\n".join(lines)+"\n")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ccr.ROOT.parent / "outputs/qwen_curriculum/state/finetuning/real_recovery_v1")
    parser.add_argument("--case",choices=CASES,action="append")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--budget",type=int,default=500)
    parser.add_argument("--execute",action="store_true",help="Run GPU evaluation; default reads available results only")
    args=parser.parse_args(argv)
    root=args.root.resolve();plan,evaluation=setup(root)
    if args.budget not in plan["config"]["budgets"]:
        raise ValueError("Budget was not predeclared")
    if args.seed not in [plan["parent"]["config"]["screen_seed"],*plan["formal"]["seeds"]]:
        raise ValueError("Seed is outside frozen protocol")
    cases=args.case or list(CASES)
    for case in cases:
        completed(root,case,args.seed,args.budget)
    if args.execute:
        try:
            evaluate(plan,evaluation,root,cases,args.seed,args.budget)
        finally:
            summarize(plan,evaluation,root,args.seed,args.budget)
    else:
        report=summarize(plan,evaluation,root,args.seed,args.budget)
        print("Available cases:", len({r['case'] for r in report['rows']}), "of",len(CASES),"; no GPU evaluation launched")


if __name__=="__main__": main()
