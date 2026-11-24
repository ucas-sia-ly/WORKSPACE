"""CCR-VPR synthetic global queries and independent local distillation losses."""

import math

import torch
import torch.nn.functional as F


def asymmetric_multi_positive_infonce(queries, query_labels, references, reference_labels,
                                     temperature=0.1, pair_valid=None, denominator=None):
    """-log(sum positive exp / sum all real exp) with cosine logits, averaged.

    The default averages over valid queries. Pass ``denominator`` (the main
    batch's image count) to sum valid queries and divide by it instead, so each
    synthetic query weighs like one real MS anchor regardless of how few
    queries the batch has. All same-place real images are positives, including
    the exact source. Invalid/unmatched queries are excluded. Empty batches return
    a differentiable zero. Detaching references isolates this loss's DIRECT
    reference gradient; the synthetic branch still updates shared student weights.
    """
    if not math.isfinite(temperature) or temperature < torch.finfo(torch.float32).tiny:
        raise ValueError("temperature must be finite and positive in FP32")
    if queries.ndim != 2 or references.ndim != 2 or queries.shape[1] != references.shape[1]:
        raise ValueError("queries and references must have matching descriptor dimensions")
    if query_labels.shape != (len(queries),) or reference_labels.shape != (len(references),):
        raise ValueError("one label is required per descriptor")
    if pair_valid is not None and (pair_valid.shape != (len(queries),) or pair_valid.dtype != torch.bool):
        raise ValueError("pair_valid must be a boolean mask over synthetic queries")
    # Disabling autocast is essential: .float() alone would not prevent AMP
    # from casting matmul back to half precision.
    with torch.autocast(device_type=queries.device.type, enabled=False):
        q, r = queries.float(), references.detach().float()
        positives = query_labels[:, None] == reference_labels[None, :]
        valid = positives.any(dim=1)
        if pair_valid is not None:
            valid = valid & pair_valid
        q, positives = q[valid], positives[valid]
        if not len(q):
            return q.sum() * 0.0
        if not torch.isfinite(q).all() or not torch.isfinite(r).all():
            raise RuntimeError("Nonfinite descriptors in synthetic global loss")
        logits = (F.normalize(q, dim=1) @ F.normalize(r, dim=1).T) / temperature
        # Centering also reduces cancellation for low temperatures.
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        positive_logits = logits.masked_fill(~positives, -torch.inf)
        losses = torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
        return losses.mean() if denominator is None else losses.sum() / denominator


def _tokens(value, name):
    if value.ndim not in (3, 4) or any(n < 1 for n in value.shape[1:]):
        raise ValueError(f"{name} must have shape [B,C,N] or [B,C,H,W]")
    value = value.flatten(2).float()
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Nonfinite {name} in local distillation")
    return value


def _target(teacher, match_index, positive_conf, student_tokens):
    """Gather only in the source grid; unknown indices cannot supervise token 0."""
    teacher = _tokens(teacher.detach(), "teacher target")
    if match_index.ndim not in (2, 3) or match_index.dtype != torch.long:
        raise ValueError("match_index must be int64 [B,N] or [B,H,W]")
    if positive_conf.shape != match_index.shape:
        raise ValueError("positive_conf and match_index must have the same shape")
    if teacher.device != match_index.device or teacher.device != positive_conf.device:
        raise ValueError("Teacher targets, indices and confidence must share a device")
    index = match_index.detach().flatten(1)
    q = positive_conf.detach().flatten(1).float()
    if index.shape != (teacher.shape[0], student_tokens):
        raise ValueError("Correspondence grid must match student tokens and teacher batch")
    if not torch.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("positive_conf must be finite in [0,1]")
    valid = (index >= 0) & (index < teacher.shape[-1])
    if ((q > 0) & ~valid).any():
        raise ValueError("Positive confidence requires an in-range source index")
    gathered = teacher.gather(2, index.clamp(0, teacher.shape[-1] - 1)[:, None].expand(-1, teacher.shape[1], -1))
    return gathered, q * valid


def _inputs(student, teacher, index, confidence):
    student = _tokens(student, "student output")
    target, q = _target(teacher, index, confidence, student.shape[-1])
    if target.shape != student.shape or target.device != student.device:
        raise ValueError("Student/teacher channels, batch and devices must match")
    return student, target, q


def _weighted(value, q, eps):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Distillation eps must be finite and positive")
    result = (value * q).sum() / (q.sum() + eps)
    if not torch.isfinite(result):
        raise RuntimeError("Nonfinite local distillation loss")
    return result


def _feature_patch_values(student, teacher, index, confidence):
    student, target, q = _inputs(student, teacher, index, confidence)
    # An exactly empty teacher feature cannot define a cosine target.
    q = q * (target.norm(dim=1) > torch.finfo(torch.float32).tiny)
    cosine = (F.normalize(student, dim=1, eps=1e-6)
              * F.normalize(target, dim=1, eps=1e-6)).sum(1).clamp(-1, 1)
    return 1 - cosine, cosine, q


def _kl_patch_values(student, teacher, index, confidence, conditional, min_teacher_keep):
    student, target, q = _inputs(student, teacher, index, confidence)
    low_keep = torch.zeros_like(q, dtype=torch.bool)
    if conditional:
        if student.shape[1] < 2:
            raise ValueError("Conditional OT needs valid clusters plus one dustbin")
        floor = torch.finfo(torch.float32).eps if min_teacher_keep is None else min_teacher_keep
        if not math.isfinite(floor) or not 0 <= floor < 1:
            raise ValueError("min_teacher_keep must be finite in [0,1)")
        # Compute mass in log space; never exp the full assignment then log it.
        log_keep = torch.logsumexp(target[:, :-1], dim=1) - torch.logsumexp(target, dim=1)
        cutoff = math.log(max(floor, torch.finfo(torch.float32).tiny))
        low_keep = log_keep <= cutoff
        q = q * ~low_keep
        student, target = student[:, :-1], target[:, :-1]
    log_student = F.log_softmax(student, dim=1)
    log_teacher = F.log_softmax(target, dim=1)
    kl = F.kl_div(log_student, log_teacher, reduction="none", log_target=True).sum(1).clamp_min(0)
    if not torch.isfinite(kl).all():
        raise RuntimeError("Nonfinite FP32 local KL")
    return kl, q, low_keep


def feature_distillation_loss(student_features, teacher_features, match_index, positive_conf, eps=1e-8):
    """Weighted 1-cosine of backbone patch features, teacher and q detached."""
    with torch.autocast(device_type=student_features.device.type, enabled=False):
        values, _, q = _feature_patch_values(student_features, teacher_features, match_index, positive_conf)
        return _weighted(values, q, eps)


def score_distillation_loss(student_logits, teacher_logits, match_index, positive_conf, eps=1e-8):
    """KL(teacher || student) over pre-Sinkhorn clusters, independently usable."""
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        values, q, _ = _kl_patch_values(student_logits, teacher_logits, match_index, positive_conf, False, None)
        return _weighted(values, q, eps)


def conditional_ot_distillation_loss(student_log_assignment, teacher_log_assignment, match_index,
                                    positive_conf, eps=1e-8, min_teacher_keep=None):
    """KL over valid clusters conditioned on non-dustbin assignment.

    Only the TEACHER's near-zero valid-cluster mass masks routing supervision;
    a student's dustbin collapse cannot hide its error by switching q off.
    The default floor is FP32 epsilon, a numerical guard, not a semantic cutoff.
    """
    with torch.autocast(device_type=student_log_assignment.device.type, enabled=False):
        values, q, _ = _kl_patch_values(student_log_assignment, teacher_log_assignment, match_index,
                                       positive_conf, True, min_teacher_keep)
        return _weighted(values, q, eps)


@torch.no_grad()
def local_target_weights(teacher_aux, match_index, positive_conf, min_teacher_keep=None):
    """Weights known before any student forward, for exact microbatch reduction."""
    n = match_index.flatten(1).shape[1]
    f, q = _target(teacher_aux["backbone_local_features"], match_index, positive_conf, n)
    route, _ = _target(teacher_aux["log_assignment"], match_index, positive_conf, n)
    floor = torch.finfo(torch.float32).eps if min_teacher_keep is None else min_teacher_keep
    if not math.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("min_teacher_keep must be finite in [0,1)")
    log_keep = torch.logsumexp(route[:, :-1], 1) - torch.logsumexp(route, 1)
    return {"feature": q * (f.norm(dim=1) > torch.finfo(torch.float32).tiny), "score": q,
            "ot": q * (log_keep > math.log(max(floor, torch.finfo(torch.float32).tiny)))}


def local_distillation_losses(student_aux, teacher_aux, match_index, positive_conf, *, eps=1e-8,
                              min_teacher_keep=None):
    """Three independent scalar losses and detached diagnostics; no weight mixing."""
    with torch.autocast(device_type=student_aux["backbone_local_features"].device.type, enabled=False):
        feature, cosine, fq = _feature_patch_values(student_aux["backbone_local_features"],
            teacher_aux["backbone_local_features"], match_index, positive_conf)
        score, sq, _ = _kl_patch_values(student_aux["score_logits"], teacher_aux["score_logits"],
                                       match_index, positive_conf, False, None)
        route, oq, low_keep = _kl_patch_values(student_aux["log_assignment"], teacher_aux["log_assignment"],
                                              match_index, positive_conf, True, min_teacher_keep)
        return {"feature": _weighted(feature, fq, eps), "score": _weighted(score, sq, eps),
                "ot": _weighted(route, oq, eps), "feature_cosine": _weighted(cosine, fq, eps).detach(),
                "feature_mass": fq.sum().detach(), "score_mass": sq.sum().detach(), "ot_mass": oq.sum().detach(),
                "low_teacher_keep_patches": (low_keep & (sq > 0)).sum().detach()}
