"""Read-only controls for local distillation, precision and model randomness.

Teacher targets and correspondence support are computed once in FP32 eval mode.
An own-checkpoint FP32 eval target separates stochastic/precision discrepancy
from drift relative to the initial teacher. No loss baseline is subtracted.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math

import torch

from .pair_correspondence import CorrespondenceCalibration, build_pair_correspondence
from .synthetic_losses import local_distillation_losses, local_target_weights


_KEYS = ("backbone_local_features", "score_logits", "log_assignment")


@contextmanager
def _preserved_state(*models):
    """Preserve mixed module modes and buffers, including possible BN statistics."""
    modes = [(module, module.training) for model in models for module in model.modules()]
    buffers = [(value, value.detach().clone()) for model in models for value in model.buffers()]
    try:
        yield
    finally:
        for value, saved in buffers:
            value.copy_(saved)
        for module, training in modes:
            module.training = training


def _aux(model, inputs, device, precision="32"):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        _, aux = model(inputs.to(device=device, dtype=torch.float32), return_aux=True)
    result = {key: aux[key].detach().float() for key in _KEYS}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise RuntimeError("Nonfinite local diagnostic model output")
    return result


def _coordinate_index(source, generated):
    """Same relative full-frame coordinate, also for unequal feature-grid sizes."""
    sh, sw = source["backbone_local_features"].shape[-2:]
    gh, gw = generated["backbone_local_features"].shape[-2:]
    y = ((torch.arange(gh, device=source["score_logits"].device) + .5) * sh / gh).long()
    x = ((torch.arange(gw, device=y.device) + .5) * sw / gw).long()
    return (y[:, None] * sw + x[None, :])[None]


def _geometry_for(pair, geometry):
    if "geometry" in pair:
        return pair["geometry"]
    if geometry is None:
        return None
    if "status" in geometry:
        return geometry
    images = pair["source_image"], pair["generated_image"]
    key = (hashlib.sha256(images[0].tobytes()).hexdigest(),
           hashlib.sha256(images[1].tobytes()).hexdigest(), images[0].size, images[1].size)
    return geometry.get(key, geometry.get(str(pair.get("generated_path", ""))))


def _distribution(values):
    if not values:
        return dict(n=0, mean=None, std=None, min=None, p05=None, median=None, p95=None, max=None)
    x = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(x).all():
        raise RuntimeError("Nonfinite local diagnostic statistic")
    return dict(n=len(values), mean=float(x.mean()), std=float(x.std(unbiased=False)),
                min=float(x.min()), p05=float(x.quantile(.05)), median=float(x.median()),
                p95=float(x.quantile(.95)), max=float(x.max()))


def summarize_local_rows(rows):
    """Across pair/repeat distributions; zero-support losses remain identified."""
    groups = {}
    for row in rows:
        key = "|".join(str(row[k]) for k in ("view", "target", "correspondence", "precision", "student_mode"))
        groups.setdefault(key, []).append(row)
    result = {}
    for key, values in groups.items():
        # Exclude unsupported loss zeros from the loss distribution. Coverage
        # distributions still include all pairs, so abstention remains visible.
        metrics = {}
        for name in ("feature", "score", "ot", "feature_cosine"):
            mass = ("feature" if name == "feature_cosine" else name) + "_mass"
            metrics[name] = _distribution([r[name] for r in values if r[mass] > 0])
        for name in ("positive_fraction", "feature_support_fraction", "score_support_fraction",
                     "ot_support_fraction", "low_teacher_keep_fraction"):
            metrics[name] = _distribution([r[name] for r in values])
        result[key] = dict(rows=len(values), distributions=metrics)
    return result


def _loss_row(student, target, index, q, metadata, min_teacher_keep):
    losses = local_distillation_losses(student, target, index, q, min_teacher_keep=min_teacher_keep)
    weights = local_target_weights(target, index, q, min_teacher_keep=min_teacher_keep)
    result = {key: float(value) for key, value in losses.items()}
    result.update(metadata)
    result.update(patches=q.numel(), positive_patches=int((q > 0).sum()),
                  positive_fraction=float((q > 0).float().mean()),
                  confidence_mass=float(q.sum()), confidence_mean=float(q.mean()))
    for key, value in weights.items():
        result[key + "_support_patches"] = int((value > 0).sum())
        result[key + "_support_fraction"] = float((value > 0).float().mean())
    result["low_teacher_keep_fraction"] = result["low_teacher_keep_patches"] / max(1, result["score_support_patches"])
    route = target["log_assignment"].flatten(2)
    gathered = route.gather(2, index.flatten(1).clamp_min(0)[:, None].expand(-1, route.shape[1], -1))
    keep = (torch.logsumexp(gathered[:, :-1], dim=1) - torch.logsumexp(gathered, dim=1)).exp()
    supported = keep[q.flatten(1) > 0]
    result["teacher_keep_mean"] = float(supported.mean()) if supported.numel() else None
    result["teacher_keep_min"] = float(supported.min()) if supported.numel() else None
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise RuntimeError("Nonfinite local diagnostic loss/statistic")
    return result


@torch.no_grad()
def local_consistency_probe(student, teacher, pairs, calibration, *, device, repeats=4, seed=31,
                            geometry=None, min_teacher_keep=None, precisions=("32", "bf16")):
    """Probe one checkpoint without changing parameters, buffers, grads or RNG.

    Each pair supplies CPU [1,3,H,W] source_tensor/generated_tensor, PIL RGB
    source_image/generated_image, optional paths, quality and masks. ``geometry``
    accepts the existing RGB-hash/size cache mapping or per-generated-path data;
    pair['geometry'] overrides it. Calibration may be an object, dictionary or
    None (verified support is then empty). All verified rows use Stage 2C's A
    support; both uniform and confidence-weighted reductions are reported.

    Rows whose target is ``student_eval_fp32`` measure same-checkpoint
    stochastic/precision discrepancies. Rows targeted at ``teacher_source``
    additionally contain checkpoint drift and/or real/generated differences.
    Train-mode values are repeated with deterministic, isolated random seeds.
    BF16 controls use autocast; all losses are evaluated in FP32.
    """
    device = torch.device(device)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if "32" not in precisions or any(value not in ("32", "bf16") for value in precisions):
        raise ValueError("precisions must include 32 and contain only 32/bf16")
    if isinstance(calibration, dict):
        calibration = CorrespondenceCalibration(**calibration)
    if device.type == "cuda" and "bf16" in precisions and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 diagnostic requested on an unsupported CUDA device")
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    rows, correspondence_rows = [], []
    with _preserved_state(student, teacher), torch.random.fork_rng(devices=cuda_devices):
        teacher.eval()
        for pair_number, pair in enumerate(pairs):
            for key in ("source_tensor", "generated_tensor"):
                value = pair[key]
                if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
                    raise ValueError("Diagnostic pair tensors must be [1,3,H,W]")
            source_target = _aux(teacher, pair["source_tensor"], device)
            generated_target = _aux(teacher, pair["generated_tensor"], device)
            correspondence = build_pair_correspondence(pair["source_image"], pair["generated_image"],
                source_target, generated_target, calibration=calibration, geometry=_geometry_for(pair, geometry),
                quality=pair.get("quality"), **pair.get("masks", {}))
            diagnostic = correspondence["match_diagnostics"]
            q_verified = (correspondence["positive_conf"] * diagnostic["positive_a"])[None].to(device)
            verified_index = correspondence["match_index"][None].to(device)
            correspondence_rows.append(dict(pair=pair_number, condition=pair.get("condition", "unknown"),
                source_path=str(pair.get("source_path", "")), generated_path=str(pair.get("generated_path", "")),
                patches=q_verified.numel(), positive_a_patches=int((q_verified > 0).sum()),
                positive_a_fraction=float((q_verified > 0).float().mean()),
                confidence_mass=float(q_verified.sum()), excluded_b_patches=diagnostic["positive_b_count"],
                unknown_fraction=float(((q_verified[0] == 0) & (correspondence["negative_conf"] == 0)).float().mean()),
                stage2b_unknown_fraction=float(correspondence["unknown_mask"].float().mean()),
                negative_fraction=float((correspondence["negative_conf"] > 0).float().mean()),
                source_grid_hw=list(diagnostic["source_grid_hw"]), generated_grid_hw=list(diagnostic["generated_grid_hw"]),
                geometry_status=diagnostic["geometry_status"], geometry_valid_points=diagnostic["geometry_valid_points"]))
            student.eval()
            own_source = _aux(student, pair["source_tensor"], device)
            own_generated = _aux(student, pair["generated_tensor"], device)
            source_index = _coordinate_index(source_target, own_source)
            generated_index = _coordinate_index(source_target, own_generated)
            source_q, generated_q = torch.ones_like(source_index, dtype=torch.float32), torch.ones_like(generated_index, dtype=torch.float32)
            own_source_index = _coordinate_index(own_source, own_source)
            own_generated_index = _coordinate_index(own_generated, own_generated)
            base = dict(pair=pair_number, condition=pair.get("condition", "unknown"),
                source_path=str(pair.get("source_path", "")), generated_path=str(pair.get("generated_path", "")))
            for precision in precisions:
                for training in (False, True):
                    for repeat in range(repeats if training else 1):
                        # Reuse the same seed across precision controls. The
                        # modes consume random numbers only inside this fork.
                        sample_seed = seed + pair_number * 100_003 + repeat * 1_009
                        # torch.manual_seed would also change RNG state on
                        # unrelated CUDA devices outside this fork's scope.
                        torch.random.default_generator.manual_seed(sample_seed)
                        if cuda_devices:
                            torch.cuda.default_generators[cuda_devices[0]].manual_seed(sample_seed)
                        student.train(training)
                        source = _aux(student, pair["source_tensor"], device, precision)
                        generated = _aux(student, pair["generated_tensor"], device, precision)
                        meta = dict(base, precision=precision, student_mode="train" if training else "eval", repeat=repeat,
                                    seed=sample_seed)
                        comparisons = [
                            (source, source_target, source_index, source_q, "source_copy", "teacher_source", "same_coordinate"),
                            (source, own_source, own_source_index, source_q, "source_self", "student_eval_fp32", "identity"),
                            (generated, own_generated, own_generated_index, generated_q, "generated_self", "student_eval_fp32", "identity"),
                            (generated, source_target, generated_index, generated_q, "real_generated", "teacher_source", "same_coordinate"),
                            (generated, source_target, verified_index, (q_verified > 0).float(), "real_generated", "teacher_source", "verified_uniform"),
                            (generated, source_target, verified_index, q_verified, "real_generated", "teacher_source", "verified"),
                        ]
                        for output, target, index, q, view, target_name, policy in comparisons:
                            rows.append(_loss_row(output, target, index, q,
                                dict(meta, view=view, target=target_name, correspondence=policy), min_teacher_keep))
                        del source, generated
            del source_target, generated_target, own_source, own_generated, correspondence
    return dict(rows=rows, summaries=summarize_local_rows(rows), correspondence=correspondence_rows,
                protocol=dict(version=1, repeats=repeats, precisions=list(precisions), teacher_mode="eval_fp32",
                    correspondence="fixed_teacher_fp32_stage2c_A_only", rng="fork_rng_seeded_per_pair_repeat",
                    feature="backbone_local_features", losses="unweighted_FP32_no_baseline_subtraction",
                    min_teacher_keep=(torch.finfo(torch.float32).eps if min_teacher_keep is None else min_teacher_keep),
                    buffers_restored=True, optimizer_steps=0))
