#!/usr/bin/env python3
"""Summarize completed zero-update CCR diagnostics using only the standard library.

Reads the five checkpoint JSON files and protocol.json produced by
diagnose_ccr_stage1.py. This command never imports torch, loads checkpoints,
instantiates models, or runs a GPU workload.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


CHECKPOINTS = ("initial", "A", "B0", "B4", "B5")
LOSSES = ("real", "syn", "feature", "score", "ot")
ORIGINAL_VARIANTS = {"initial": "real_only", "A": "real_only", "B0": "global_010",
                     "B4": "feature_ot", "B5": "feature_ot"}
WEIGHTS = {"real": 1.0, "syn": 0.1, "feature": 0.1, "score": 0.0, "ot": 0.1}


def distribution(values):
    """Linear quantiles; undefined values are counted explicitly, never as zero."""
    values = list(values)
    finite = sorted(float(value) for value in values if value is not None)
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("Nonfinite diagnostic input; refusing a misleading summary")
    result = {"count": len(finite), "undefined_count": len(values) - len(finite),
              "mean": None, "std": None, "min": None, "p05": None, "median": None,
              "p95": None, "max": None}
    if finite:
        def quantile(q):
            index = q * (len(finite) - 1)
            lower = int(index)
            return finite[lower] + (finite[math.ceil(index)] - finite[lower]) * (index - lower)
        result.update(mean=statistics.fmean(finite), std=statistics.pstdev(finite),
                      min=finite[0], p05=quantile(.05), median=quantile(.5),
                      p95=quantile(.95), max=finite[-1])
    return result


def angle_summary(records):
    records = list(records)
    values = [record.get("value") for record in records]
    valid = [value for value in values if value is not None]
    return {"cosine": distribution(values),
            "angle_degrees": distribution(record.get("angle_degrees") for record in records),
            "defined_count": len(valid), "undefined_count": len(values) - len(valid),
            "negative_cosine_count": sum(value < 0 for value in valid),
            "negative_cosine_fraction_defined": sum(value < 0 for value in valid) / len(valid) if valid else None}


def ratio_summary(numerators, denominators):
    pairs = list(zip(numerators, denominators))
    return {"finite_positive_denominator": distribution(n / d if d > 0 else None for n, d in pairs),
            "both_zero_count": sum(n == 0 and d == 0 for n, d in pairs),
            "nonzero_numerator_zero_real_count": sum(n > 0 and d == 0 for n, d in pairs),
            "zero_real_count": sum(d == 0 for _, d in pairs),
            "samples": len(pairs)}


def gradient_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = "|".join(str(row[name]) for name in ("panel", "precision", "local_mode", "confidence_mode"))
        grouped[key].append(row)
    output = {}
    for key, selected in grouped.items():
        result = {"rows": len(selected), "synthetic_queries": distribution(r["synthetic_queries"] for r in selected),
                  "zero_synthetic_query_rows": sum(r["synthetic_queries"] == 0 for r in selected),
                  "losses": {name: distribution(r["losses"][name] for r in selected) for name in LOSSES},
                  "groups": {}, "variants": {}}
        for group in selected[0]["groups"]:
            groups = [row["groups"][group] for row in selected]
            real = [g["losses"]["real"]["raw_norm"] for g in groups]
            result["groups"][group] = {
                "parameter_elements": groups[0]["parameter_elements"],
                "real_zero_norm_rows": sum(value == 0 for value in real),
                "losses": {name: {
                    "raw_norm": distribution(g["losses"][name]["raw_norm"] for g in groups),
                    "weighted_norm": distribution(g["losses"][name]["weighted_norm"] for g in groups),
                    "gradient_parameter_tensors": distribution(g["losses"][name]["gradient_parameter_tensors"] for g in groups),
                    "weighted_to_real_ratio": ratio_summary(
                        (g["losses"][name]["weighted_norm"] for g in groups), real),
                } for name in LOSSES},
                "angles": {pair: angle_summary(g["cosines"][pair] for g in groups)
                           for pair in groups[0]["cosines"]},
            }
        for variant in selected[0]["combined_variants"]:
            values = [row["combined_variants"][variant] for row in selected]
            result["variants"][variant] = {
                "norm": distribution(value["norm"] for value in values),
                "would_clip_count": sum(value["would_clip"] for value in values),
                "would_clip_fraction": sum(value["would_clip"] for value in values) / len(values),
                "clip_scale": distribution(value["clip_scale"] for value in values),
                "vs_real": angle_summary({"value": value["vs_real_cosine"],
                                           "angle_degrees": value["vs_real_angle_degrees"]} for value in values),
            }
        stat_names = sorted({name for row in selected for name in row["target_statistics"]})
        result["target_statistics"] = {name: distribution(row["target_statistics"].get(name) for row in selected)
                                       for name in stat_names}
        output[key] = result
    return output


def anchor_statistics(anchors):
    anchors = list(anchors)
    totals = {name: sum(row[name] for row in anchors)
              for name in ("positive", "negative", "mined_positive", "mined_negative")}
    return {"anchors": len(anchors), "directed_pair_totals": totals,
            "active_anchor_count": sum(row["ms_active"] for row in anchors),
            "active_anchor_fraction": sum(row["ms_active"] for row in anchors) / len(anchors) if anchors else None,
            "both_sides_active_anchor_fraction": sum(row["both_sides_active"] for row in anchors) / len(anchors) if anchors else None,
            "pooled_positive_retention": totals["mined_positive"] / totals["positive"] if totals["positive"] else None,
            "pooled_negative_retention": totals["mined_negative"] / totals["negative"] if totals["negative"] else None,
            "mined_positive_per_anchor": distribution(row["mined_positive"] for row in anchors),
            "mined_negative_per_anchor": distribution(row["mined_negative"] for row in anchors),
            "hardest_negative_similarity": distribution(row["hardest_negative_similarity"] for row in anchors),
            "hardest_positive_similarity": distribution(row["hardest_positive_similarity"] for row in anchors)}


def mining_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[f'{row["panel"]}|{row["mode"]}|P{row["P"]}'].append(row)
    scalars = ("loss", "active_anchor_ratio", "both_sides_active_anchor_ratio", "positive_retention_ratio",
               "negative_retention_ratio", "mined_positive_pairs", "mined_negative_pairs")
    return {key: {"batches": len(selected),
                  "batch_distributions": {name: distribution(row[name] for row in selected) for name in scalars},
                  "all_anchors": anchor_statistics(anchor for row in selected for anchor in row["per_anchor"]),
                  "common_first_8_place_anchors": anchor_statistics(
                      anchor for row in selected for anchor in row["per_anchor"][:32])}
            for key, selected in grouped.items()}


def local_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = "|".join(str(row[name]) for name in ("view", "target", "correspondence", "precision", "student_mode"))
        grouped[key].append(row)
    output = {}
    for key, selected in grouped.items():
        result = {"rows": len(selected), "unique_pairs": len({row["pair"] for row in selected}), "losses": {}, "support": {}}
        for loss in ("feature", "score", "ot", "feature_cosine"):
            mass = ("feature" if loss == "feature_cosine" else loss) + "_mass"
            result["losses"][loss] = distribution(row[loss] if row[mass] > 0 else None for row in selected)
        for name in ("positive_fraction", "feature_support_fraction", "score_support_fraction", "ot_support_fraction",
                     "low_teacher_keep_fraction", "confidence_mean", "teacher_keep_mean", "teacher_keep_min"):
            result["support"][name] = distribution(row.get(name) for row in selected)
        output[key] = result
    return output


def correspondence_summary(rows):
    result = {}
    for weather in ("all", "night", "rain", "snow", "fog"):
        selected = rows if weather == "all" else [row for row in rows if row["condition"].lower() == weather]
        patches = sum(row["patches"] for row in selected)
        result[weather] = {
            "pairs": len(selected), "patches": patches,
            "positive_a_patches": sum(row["positive_a_patches"] for row in selected),
            "positive_a_fraction": sum(row["positive_a_patches"] for row in selected) / patches if patches else None,
            "geometry_pairs_with_points": sum(row["geometry_valid_points"] > 0 for row in selected),
            "geometry_status_counts": {status: sum(row["geometry_status"] == status for row in selected)
                                       for status in sorted({row["geometry_status"] for row in selected})},
            "distributions": {name: distribution(row[name] for row in selected)
                              for name in ("positive_a_fraction", "confidence_mass", "unknown_fraction",
                                           "negative_fraction", "excluded_b_patches", "geometry_valid_points")},
        }
    return result


def load_completed(directory):
    required = [directory / f"{name}.json" for name in CHECKPOINTS] + [directory / "protocol.json"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError("诊断尚未完成，缺少：" + ", ".join(missing))
    payloads = {name: json.loads((directory / f"{name}.json").read_text()) for name in CHECKPOINTS}
    protocol = json.loads((directory / "protocol.json").read_text())
    if (protocol.get("optimizer_created") is not False or protocol.get("optimizer_steps") != 0
            or protocol.get("parameter_state_unchanged") is not True
            or protocol.get("checkpoint_sha256_before") != protocol.get("checkpoint_sha256_after")):
        raise ValueError("零更新或 checkpoint 未修改检查未通过")
    for name, payload in payloads.items():
        if (payload.get("state_sha256_before") != payload.get("state_sha256_after")
                or payload.get("parameter_versions_unchanged") is not True):
            raise ValueError(f"{name} 的内存参数未修改检查未通过")
        if not payload.get("gradients") or not payload.get("mining") or not payload.get("local", {}).get("rows"):
            raise ValueError(f"{name} 诊断数据为空")
    return payloads, protocol


def summarize(directory):
    payloads, protocol = load_completed(directory)
    checkpoints = {}
    for name, payload in payloads.items():
        gradients = gradient_summary(payload["gradients"])
        branch_checks = {}
        for group, loss in (("content", "feature"), ("score", "feature"), ("token", "feature"),
                            ("content", "score"), ("content", "ot")):
            values = [row["groups"][group]["losses"][loss]["raw_norm"] for row in payload["gradients"]
                      if group in row["groups"]]
            branch_checks[f"{loss}_on_{group}"] = {"maximum_raw_norm": max(values) if values else None,
                                                     "nonzero_rows": sum(value > 0 for value in values),
                                                     "rows": len(values)}
        checkpoints[name] = {
            "original_variant": ORIGINAL_VARIANTS[name],
            "gradients": gradients, "mining": mining_summary(payload["mining"]),
            "local": local_summary(payload["local"]["rows"]),
            "correspondence": correspondence_summary(payload["local"]["correspondence"]),
            "branch_gradient_checks": branch_checks, "dropout_inventory": payload["dropout_inventory"],
            "state_unchanged": True,
        }
    files = [directory / f"{name}.json" for name in CHECKPOINTS] + [directory / "protocol.json"]
    return {"version": 1, "protocol": protocol,
            "inputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            "gradient_reference_weights": WEIGHTS, "checkpoints": checkpoints,
            "limitations": [
                "固定少量 batch 的描述性统计，不是全训练分布或统计显著性检验。",
             "角度仅在双方梯度范数均大于零时定义；未定义值不记作 0 度或余弦 0。",
                "加权辅助/真实梯度比只在真实范数大于零时计算；真实为零的样本单独计数。",
                "梯度范数和夹角反映当前固定参数位置，不等于 AdamW 历史更新归因。",
                "Train BF16 mining 以 microbatch16 提取 P32 描述子后嵌套切片；P8 梯度前向整批32图，Dropout realization 不同，计数与零梯度不必逐一对应。",
                "裁剪率是固定样本在 max_norm=1 下的假想触发率，未实际裁剪或更新。",
                "training_subset 是原生成覆盖地点子集；full_real 是全真实池的独立固定样本。",
                "受控面板强制精确源图进入 K 且触发概率为 1，不能用来估计自然曝光。",
                "同图 own-eval 对照隔离随机性；固定初始教师的同图对照另包含权重漂移。",
                "多次 Dropout 重复共享图像，不作为独立地点样本，不进行 p 值检验。",
                "未计算或扣除所谓 Loss 下限；没有训练、检索评估或 checkpoint 写入。",
            ]}


def number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}g}"


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def quantiles(d):
    return f'{number(d["median"])} [{number(d["p05"])}, {number(d["p95"])}]'


def markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows)]


def make_report(summary):
    checkpoints = summary["checkpoints"]
    protocol = summary["protocol"]
    lines = ["# CCR Stage 1 零更新诊断：详细分析", "",
             "诊断读取初始与 A/B0/B4/B5 checkpoint，在固定输入上执行前向和梯度计算。"
             "未创建 optimizer，更新次数为 0；所有 checkpoint 文件 SHA256 与模型内存状态前后均一致。", "",
             f'固定样本：原训练地点子集 {protocol.get("natural_batches")} 个 P32 super-batch；'
             f'全真实池 {protocol.get("full_real_batches")} 个；受控配对 {protocol.get("controlled_batches")} 个 P8 batch。'
             f'局部对照 {protocol.get("dropout_pairs")} 对图像，每对 train 模式重复 {protocol.get("dropout_repeats")} 次。',
             "P8/16/32 使用同一 super-batch 的嵌套前缀；相同 checkpoint 的同图 own-eval 目标隔离随机性和精度。"
             "真实—生成目标使用固定初始教师和原有校准。输入变换以 protocol 的记录为准，所有 checkpoint 复用同一组已固定张量。", "",
             f'输入变换记录：`{protocol.get("input_transform")}`。', "",
             "以下表格给出中位数 [P05, P95]，标注“均值”的列除外。— 表示未定义或没有有效监督。"
             "完整范数、余弦、角度、比值和支持率分布保存在 summary.json，原始样本在各 checkpoint JSON。", "",
             "## 1. 主任务困难样本：全部 anchor", "",
             "mined pair 按有向 (anchor, reference) 计数。保留率的分母为全部可用正/负 pair。"
             "有效 anchor 指至少一侧存在 mined pair；同时保留了双侧均有效比例。", "",
             "Train BF16 mining 使用 microbatch16 提取 P32 描述子，再按 P8/16/32 嵌套切片。"
             "P8 梯度诊断则整批前向 32 张图，两者的 Dropout realization 不同；"
             "所以 mining 的有效 pair 计数与梯度面板的零梯度次数不必逐 batch 一一对应。", ""]
    initial = checkpoints["initial"]
    natural = next((value for key, value in initial["gradients"].items()
                    if key.startswith("training_subset|bf16|train|")), None)
    if natural:
        all_group = natural["groups"]["all"]
        self_control = initial["local"].get("source_self|student_eval_fp32|identity|32|train")
        precision_control = initial["local"].get("source_self|student_eval_fp32|identity|bf16|eval")
        support = initial["correspondence"]["all"]
        original = natural["variants"]["feature_ot"]
        observations = ["## 本次固定样本最直接的观察", "",
                        f'- 初始模型自然 P8 梯度样本 {natural["rows"]} 个，其中 '
                        f'{all_group["real_zero_norm_rows"]} 个真实 MS 梯度严格为 0。'
                        f'真实/全局方向只有 {all_group["angles"]["real|syn"]["defined_count"]} 个样本可定义；'
                        "不能用很少的有效角度宣告长期梯度冲突。",
                        '- 初始模型自然 P8 的原始梯度范数均值：'
                        + "，".join(f'{loss}={number(all_group["losses"][loss]["raw_norm"]["mean"])}'
                                    for loss in ("real", "syn", "feature", "ot"))
                        + "。权重见后表；这是当前固定参数点上的证据，不是 AdamW 累积位移的分解。",
                        f'- 初始模型假想 Feature+OT 组合在自然 P8 面板的裁剪触发为 '
                        f'{original["would_clip_count"]}/{natural["rows"]}，'
                        "该计数不能替代历史训练的实际裁剪率。"]
        if self_control and precision_control:
            observations += ['- 初始同图 FP32 train/eval 的 Loss 均值：'
                             + "，".join(f'{loss}={number(self_control["losses"][loss]["mean"])}'
                                         for loss in ("feature", "score", "ot"))
                             + '；同图 BF16 eval/FP32 eval 为 '
                             + "，".join(f'{loss}={number(precision_control["losses"][loss]["mean"])}'
                                         for loss in ("feature", "score", "ot"))
                             + "。Feature 与 Score/OT 应分开解释，后两项的随机扰动和精度偏差也不能相互替代。"]
        observations += [f'- 局部对照 {support["pairs"]} 对图像，可信 A 支持 '
                         f'{support["positive_a_patches"]}/{support["patches"]} Patch（{percent(support["positive_a_fraction"])}），'
                         f'具有几何点证据的图对为 {support["geometry_pairs_with_points"]}。'
                         "本次支持稀疏性可被直接观察；尚未测得检索收益。", ""]
        full = initial["mining"].get("full_real|train_bf16|P32")
        subset = initial["mining"].get("training_subset|train_bf16|P32")
        if full and subset:
            observations.insert(-1, '- 初始 Train BF16 的 P32 有效 anchor：原地点子集 '
                                f'{percent(subset["all_anchors"]["active_anchor_fraction"])}，'
                                f'全真实池固定样本 {percent(full["all_anchors"]["active_anchor_fraction"])}。'
                                "因此，扩大真实地点池本身不保证随机 batch 自动获得更多难样本；"
                                "两面板样本组成不同，也不能据此判定哪一个总体更容易。")
        position = lines.index("## 1. 主任务困难样本：全部 anchor")
        lines[position:position] = observations
    for mode, title in (("eval_fp32", "Eval FP32"), ("train_bf16", "Train BF16")):
        lines += [f"### {title}", ""]
        rows = []
        for name, data in checkpoints.items():
            for panel in ("training_subset", "full_real"):
                for p in (8, 16, 32):
                    value = data["mining"].get(f"{panel}|{mode}|P{p}")
                    if value:
                        anchor = value["all_anchors"]
                        totals = anchor["directed_pair_totals"]
                        rows.append((name, panel, p, value["batches"], percent(anchor["active_anchor_fraction"]),
                                     percent(anchor["pooled_positive_retention"]), percent(anchor["pooled_negative_retention"]),
                                     f'{totals["mined_positive"]}/{totals["mined_negative"]}',
                                     number(value["batch_distributions"]["loss"]["mean"])))
        lines += markdown_table(("模型", "面板", "P", "batch", "有效 anchor", "正 pair 保留", "负 pair 保留", "mined 正/负", "MS 均值"), rows) + [""]
    lines += ["### 相同前 8 地点 anchor 的候选池变化", "",
              "此处 anchor 恒为同一批 32 张真实图，仅增加其他地点 reference。可用于排除 P 改变时 anchor 组成变化。", ""]
    rows = []
    for name, data in checkpoints.items():
        for panel in ("training_subset", "full_real"):
            for mode in ("eval_fp32", "train_bf16"):
                values = [data["mining"].get(f"{panel}|{mode}|P{p}") for p in (8, 16, 32)]
                if all(values):
                    anchor = [v["common_first_8_place_anchors"] for v in values]
                    rows.append((name, panel, mode, *(percent(v["active_anchor_fraction"]) for v in anchor),
                                 *(number(v["mined_negative_per_anchor"]["mean"]) for v in anchor)))
    lines += markdown_table(("模型", "面板", "模式", "P8 有效", "P16 有效", "P32 有效", "P8 负/anchor", "P16 负/anchor", "P32 负/anchor"), rows) + [""]

    lines += ["## 2. 梯度规模与原配置的假想裁剪", "",
              "所有模型统一只对最后一个 Backbone block、norm 和 SALAD 参数求梯度。"
              "梯度使用 autograd.grad，模型 .grad 保持空；没有实际裁剪。"
              "表中辅助加权采用 global=0.1、Feature=0.1、Score=0、OT=0.1。"
              "Score 原始梯度只作诊断，不属于原 B4/B5 训练目标。", ""]
    rows = []
    for name, data in checkpoints.items():
        for key, entry in data["gradients"].items():
            panel, precision, mode, confidence = key.split("|")
            if precision != "bf16" or mode != "train":
                continue
            group = entry["groups"]["all"]
            variant = data["original_variant"]
            combined = entry["variants"][variant]
            rows.append((name, panel, entry["rows"], group["real_zero_norm_rows"],
                         *(quantiles(group["losses"][loss]["weighted_norm"]) for loss in ("real", "syn", "feature", "ot")),
                         variant, f'{combined["would_clip_count"]}/{entry["rows"]}', quantiles(combined["norm"])))
    lines += markdown_table(("模型", "面板", "batch", "真实零梯度", "真实 ∥g∥", "0.1 全局 ∥g∥", "0.1 Feature ∥g∥", "0.1 OT ∥g∥", "原目标", "会裁剪", "总 ∥g∥"), rows) + ["",
              "initial 的 real_only 仅为假设继续做真实训练的参考；未声称它在本次诊断前经历了该微调。"
              "其余模型也同时测量了各独立 Loss，不意味着 A 实际训练过辅助项。", "",
              "### 辅助 / 真实梯度比值及方向（自然面板，Train BF16）", "",
              "真实梯度为 0 时，比值不写成无穷大，也不纳入有限比值的分位数；余弦和角度同样记为未定义。"
              "负余弦比例仅在双方非零的样本中计算。", ""]
    rows = []
    for name, data in checkpoints.items():
        for key, entry in data["gradients"].items():
            panel, precision, mode, _ = key.split("|")
            if panel != "training_subset" or precision != "bf16" or mode != "train":
                continue
            group = entry["groups"]["all"]
            for loss in ("syn", "feature", "ot"):
                ratio = group["losses"][loss]["weighted_to_real_ratio"]
                angles = group["angles"][f"real|{loss}"]
                rows.append((name, loss, quantiles(ratio["finite_positive_denominator"]),
                             ratio["finite_positive_denominator"]["count"], ratio["zero_real_count"],
                             ratio["nonzero_numerator_zero_real_count"], quantiles(angles["cosine"]),
                             quantiles(angles["angle_degrees"]),
                             f'{angles["defined_count"]}/{entry["rows"]}', percent(angles["negative_cosine_fraction_defined"])))
    lines += markdown_table(("模型", "辅助项", "加权/真实比", "比值有效 n", "真实零 n", "辅助非零且真实零 n", "余弦", "角度°", "方向有效", "负余弦"), rows) + ["",
              "### SALAD 与 Backbone 分支（自然面板，Train BF16）", ""]
    rows = []
    for name, data in checkpoints.items():
        for key, entry in data["gradients"].items():
            if not key.startswith("training_subset|bf16|train|"):
                continue
            for group_name in ("backbone", "score", "content", "token", "dustbin"):
                group = entry["groups"].get(group_name)
                if group:
                    angle = group["angles"]["real|syn"]
                    rows.append((name, group_name, *(quantiles(group["losses"][loss]["raw_norm"]) for loss in LOSSES),
                                 quantiles(angle["cosine"]), f'{angle["defined_count"]}/{entry["rows"]}'))
    lines += markdown_table(("模型", "分支", "真实原始范数", "全局原始范数", "Feature 原始范数", "Score 原始范数", "OT 原始范数", "真实/全局余弦", "有效方向"), rows) + ["",
              "Feature→aggregator、Score/OT→content 的直接梯度边界检查：", ""]
    rows = [(name, check, number(value["maximum_raw_norm"]), f'{value["nonzero_rows"]}/{value["rows"]}')
            for name, data in checkpoints.items() for check, value in data["branch_gradient_checks"].items()]
    lines += markdown_table(("模型", "边界", "最大原始范数", "非零样本"), rows) + ["",
              "全局辅助的梯度可进入 Backbone、score、content、token；detach 只切断经真实 reference 的直接梯度。"
              "固定 OT 边际下，标量 Dustbin score 接近一个 gauge 方向，其很小梯度不能解释为没有 Patch 被分配至 Dustbin。", "",
              "### 权重候选的假想裁剪（同一已测梯度，不更新权重）", ""]
    rows = []
    for name, data in checkpoints.items():
        for key, entry in data["gradients"].items():
            if not key.startswith("training_subset|bf16|train|"):
                continue
            for variant, value in entry["variants"].items():
                rows.append((name, variant, quantiles(value["norm"]),
                             f'{value["would_clip_count"]}/{entry["rows"]}', quantiles(value["vs_real"]["cosine"]),
                             quantiles(value["vs_real"]["angle_degrees"])))
    lines += markdown_table(("模型", "目标组合", "总梯度范数", "会裁剪", "总梯度/真实余弦", "角度°"), rows) + [""]

    lines += ["## 3. 同图随机性与精度对照", "",
              "source_self：每个 checkpoint 自己的源图 FP32 eval 输出作为目标。"
              "FP32 eval/eval 应近零；FP32 train/eval 反映模型随机模式差异；BF16 还含精度差异。"
              "Feature 为 Backbone 局部特征，不能把 SALAD score/content 的 Dropout 归因到 Feature。", ""]
    rows = []
    for name, data in checkpoints.items():
        for precision in ("32", "bf16"):
            for mode in ("eval", "train"):
                value = data["local"].get(f"source_self|student_eval_fp32|identity|{precision}|{mode}")
                if value:
                    rows.append((name, precision, mode, f'{value["unique_pairs"]}/{value["rows"]}',
                                 *(quantiles(value["losses"][loss]) for loss in ("feature", "score", "ot"))))
    lines += markdown_table(("模型", "精度", "学生模式", "图对/重复行", "Feature", "Score KL", "Conditional OT KL"), rows) + ["",
              "## 4. 固定教师漂移、跨域局部一致性与支持区域", "",
              "source_copy 使用固定初始教师的真实源图目标，因此训练后模型即使 eval 也可能非零。"
              "real_generated 使用生成图学生输出和真实图教师目标；same_coordinate 仅是未经验证的参照，"
              "verified_uniform 与 verified 才使用原 A 类可信对应支持。没有有效支持的 Loss 零值不纳入 Loss 分布，支持率仍完整记录。", ""]
    rows = []
    for name, data in checkpoints.items():
        for view, correspondence in (("source_copy", "same_coordinate"), ("real_generated", "same_coordinate"),
                                      ("real_generated", "verified_uniform"), ("real_generated", "verified")):
            for precision, mode in (("32", "eval"), ("bf16", "train")):
                value = data["local"].get(f"{view}|teacher_source|{correspondence}|{precision}|{mode}")
                if value:
                    rows.append((name, view, correspondence, f"{precision}/{mode}",
                                 *(quantiles(value["losses"][loss]) for loss in ("feature", "score", "ot", "feature_cosine")),
                                 percent(value["support"]["feature_support_fraction"]["mean"])))
    lines += markdown_table(("模型", "输入", "对应", "精度/模式", "Feature", "Score KL", "OT KL", "Feature 余弦", "支持率均值"), rows) + [""]
    rows = []
    # Teacher/correspondence inputs are fixed; retain every model in summary.json,
    # show the initial copy once and verify equality before claiming identical support.
    correspondence = [data["correspondence"] for data in checkpoints.values()]
    same = all(value == correspondence[0] for value in correspondence[1:])
    lines += [f'各 checkpoint 的教师可信区域统计一致：{"是" if same else "否，应检查输入/校准差异"}。', ""]
    for weather, value in checkpoints["initial"]["correspondence"].items():
        rows.append((weather, value["pairs"], value["patches"], value["positive_a_patches"],
                     percent(value["positive_a_fraction"]), percent(value["distributions"]["unknown_fraction"]["mean"]),
                     percent(value["distributions"]["negative_fraction"]["mean"]), value["geometry_pairs_with_points"]))
    lines += markdown_table(("天气", "图对", "Patch", "Positive A", "Positive 支持率", "Unknown 均值", "Negative 均值", "有几何点图对"), rows) + ["",
              "## 5. 结果的解释边界与后续选择", "",
              "- 如果 P8 主任务梯度经常严格为零，而辅助梯度非零，说明当前 batch 的数据梯度可完全来自辅助项；"
              "AdamW 的参数更新仍可能受历史动量和 weight decay 影响。"
              "这比比较 Loss 数值更直接，但不能外推为整轮所有更新都如此。",
              "- 如果 P32 对共同 anchor 明显增加有效 mined pair，支持优先扩大地点负样本池；"
              "全真实池与原易地点子集是不同样本，二者差异也受 batch 组成影响。",
              "- 如果同图 FP32 train/eval 的 Score/OT 已较大，局部监督前向关闭相应 Dropout 值得单独比较。"
              "不要从 Loss 中减去常数基线：减常数不改变梯度，截断则会引入新门控。",
              "- 辅助权重 0.01/0.03 的表格只是在同一参数点上的梯度重组，不是这些权重的训练结果。"
              "角度冲突、范数及裁剪应结合判断，不按单个阈值自动宣告某路线失败。",
              "- q 的有效支持率与 Loss 必须一起看；低支持率或无几何证据时，不能据 Loss 下降推断困难结构蒸馏有效。",
              "- 本诊断没有新增 Recall 结果，不对检索收益、自然天气泛化或多种子显著性作结论。", "",
              "## 6. 复现与资源", "",
              f'固定输入 seed：{protocol.get("batch_seed")}；统一可训练 Backbone blocks：{protocol.get("trainable_backbone_blocks")}。',
              f'诊断 worker 耗时：{number(protocol.get("seconds"))} 秒；'
              f'GPU allocated 峰值：{number(protocol.get("gpu_peak_allocated_bytes", 0) / 1024**3)} GiB；'
              f'reserved 峰值：{number(protocol.get("gpu_peak_reserved_bytes", 0) / 1024**3)} GiB。',
              "CPU 汇总入口：`python salad/summarize_ccr_diagnostics.py --input-dir <诊断目录>`。"
              "该入口要求五个 checkpoint JSON 和 protocol.json 完整，不会启动 GPU 或重跑诊断。", ""]
    lines += ["限制："] + [f"- {item}" for item in summary["limitations"]] + [""]
    return "\n".join(lines)


def reference_comparison(current, reference, directory):
    """Compact descriptive comparison of two completed fixed-input protocols."""
    for key in ("batch_seed", "K", "checkpoint_sha256_before"):
        if current["protocol"].get(key) != reference["protocol"].get(key):
            raise ValueError(f"参考目录的 {key} 不一致，不能当作同 checkpoint 对照")
    result = {"directory": str(directory), "input_transform": reference["protocol"].get("input_transform"),
              "rows": []}
    for name, data in current["checkpoints"].items():
        other = reference["checkpoints"][name]
        for key, gradient in data["gradients"].items():
            if not key.startswith("training_subset|bf16|train|") or key not in other["gradients"]:
                continue
            base = other["gradients"][key]
            group, base_group = gradient["groups"]["all"], base["groups"]["all"]
            result["rows"].append({
                "checkpoint": name, "condition": key, "current_rows": gradient["rows"], "reference_rows": base["rows"],
                "current_zero_real": group["real_zero_norm_rows"], "reference_zero_real": base_group["real_zero_norm_rows"],
                "current_raw_norm_means": {loss: group["losses"][loss]["raw_norm"]["mean"] for loss in LOSSES},
                "reference_raw_norm_means": {loss: base_group["losses"][loss]["raw_norm"]["mean"] for loss in LOSSES},
                "current_feature_ot_clip": gradient["variants"]["feature_ot"]["would_clip_count"],
                "reference_feature_ot_clip": base["variants"]["feature_ot"]["would_clip_count"],
            })
    return result


def reference_report(comparison):
    lines = ["", "## 独立参考面板对照", "",
             f'参考目录：`{comparison["directory"]}`；参考输入变换：`{comparison["input_transform"]}`。',
             "下表当前/参考是两套固定输入面板的描述性对照，不作显著性检验；"
             "增强面板的梯度与未增强面板分开，避免把固定 Resize 诊断当作原增强训练的完整替代。", ""]
    rows = []
    for row in comparison["rows"]:
        rows.append((row["checkpoint"],
                     f'{row["current_zero_real"]}/{row["current_rows"]}',
                     f'{row["reference_zero_real"]}/{row["reference_rows"]}',
                     *(f'{number(row["current_raw_norm_means"][loss])}/{number(row["reference_raw_norm_means"][loss])}'
                       for loss in ("real", "syn", "feature", "ot")),
                     f'{row["current_feature_ot_clip"]}/{row["current_rows"]}',
                     f'{row["reference_feature_ot_clip"]}/{row["reference_rows"]}'))
    lines += markdown_table(("模型", "当前真实零", "参考真实零", "真实均值 当前/参考", "全局均值 当前/参考",
                             "Feature 均值 当前/参考", "OT 均值 当前/参考", "当前 Feature+OT 会裁剪", "参考会裁剪"), rows) + [""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path,
                        help="Optional completed baseline directory for a separate fixed-input comparison")
    args = parser.parse_args(argv)
    directory = args.input_dir.expanduser().resolve()
    try:
        result = summarize(directory)
        if args.reference_dir:
            reference = args.reference_dir.expanduser().resolve()
            result["reference_comparison"] = reference_comparison(result, summarize(reference), reference)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    # Both products are derived files in the owned diagnostic directory.
    (directory / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    report = make_report(result)
    if "reference_comparison" in result:
        report += reference_report(result["reference_comparison"])
    (directory / "detailed_report.md").write_text(report)
    print(directory / "detailed_report.md")


if __name__ == "__main__":
    main()
