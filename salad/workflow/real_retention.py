"""Optional real-domain descriptor retention against an initial frozen model.

This is a restorative constraint after the student moves. At identical weights,
identical inputs and deterministic FP32 eval forwards, both its loss and gradient
are zero; it does not create a useful task gradient on an already matched pair.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

import torch
import torch.nn.functional as F


def _descriptor_distances(student, teacher):
    if (student.ndim != 2 or teacher.shape != student.shape
            or student.shape[1] < 1 or teacher.device != student.device):
        raise ValueError("Retention descriptors must have matching [B,D] shapes and devices")
    if not student.is_floating_point() or not teacher.is_floating_point():
        raise ValueError("Retention descriptors must be floating point")
    with torch.autocast(device_type=student.device.type, enabled=False):
        student, teacher = student.float(), teacher.detach().float()
        if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
            raise RuntimeError("Nonfinite descriptor in real retention loss")
        teacher_scale = teacher.abs().amax(dim=1, keepdim=True)
        valid = teacher_scale[:, 0] > 0
        # A zero target cannot specify a cosine direction. Its row supplies no
        # constraint. Real SALAD descriptors are normally nonzero unit vectors.
        # Scale before the norm to avoid overflow even for finite extreme
        # BF16/FP32 inputs. Scaling cancels out of the cosine direction.
        tiny = torch.finfo(torch.float32).tiny
        student_scale = student.abs().amax(dim=1, keepdim=True)
        s = F.normalize(student / student_scale.clamp_min(tiny), dim=1, eps=1e-12)
        t = F.normalize(teacher / teacher_scale.clamp_min(tiny), dim=1, eps=1e-12)
        # Equivalent to 1-cosine for nonzero normalized descriptors. The squared
        # form has exactly zero value AND derivative when student equals target,
        # avoiding self-cosine roundoff as a fictitious preservation gradient.
        values = 0.5 * (s - t).square().sum(dim=1) * valid
        return values, (s * t).sum(dim=1), valid


def descriptor_retention_loss(student, teacher):
    """FP32 mean cosine distance with detached teacher targets.

    Empty batches and zero teacher rows contribute differentiable zero. Zero
    rows remain in the batch denominator so microbatch sums have the exact same
    reduction as a single full-batch call. Nonfinite inputs are rejected.
    """
    values, _, _ = _descriptor_distances(student, teacher)
    loss = values.sum() / max(len(values), 1)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite real retention loss")
    return loss


@contextmanager
def local_eval_with_grad(model):
    """Temporarily use eval behavior while retaining student autograd.

    Exact nested train/eval flags, buffers, and Torch CPU/CUDA RNG states are
    restored even on failure. This prevents an optional auxiliary forward from
    advancing the real training branch's Dropout RNG or BatchNorm statistics.
    Parameters are not copied and no ``no_grad`` wraps the yielded forward.
    """
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    devices = sorted({tensor.device.index for tensor in list(model.parameters()) + list(model.buffers())
                      if tensor.device.type == "cuda"})
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            yield
        finally:
            with torch.no_grad():
                for buffer, saved in buffers:
                    if not torch.equal(buffer, saved):
                        buffer.copy_(saved)
            # Calling .train() recursively would overwrite deliberately mixed
            # modes, e.g. a frozen backbone and an independently trained head.
            for module, was_training in modes:
                module.training = was_training


def backward_real_retention(model, teacher, images, scaler, weight, chunk_size=8):
    """Accumulate a bounded FP32 retention backward; never step an optimizer.

    ``images`` are the same normalized real main-batch inputs for both models.
    The teacher must implement ``teacher(images, return_aux=False)``. Each chunk
    is reduced against the entire real batch, including a short final chunk.
    The caller performs the eventual single unscale/clip/optimizer step shared
    by main and auxiliary losses. An OOM is deliberately not retried after
    backward because some chunks may already have accumulated gradients.
    """
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("real retention weight must be finite and nonnegative")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("real retention chunk_size must be a positive integer")
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError("Real retention inputs must be floating RGB [B,3,H,W]")
    metrics = dict(loss=0., weighted_total=0., cosine=0., images=0, valid_targets=0,
                   invalid_targets=0, chunks=0, seconds=0.)
    if weight == 0 or len(images) == 0:
        return metrics
    started = time.perf_counter()
    cosine_sum = 0.
    with local_eval_with_grad(model), torch.autocast(device_type=images.device.type, enabled=False):
        for start in range(0, len(images), chunk_size):
            inputs = images[start:start + chunk_size].float()
            with torch.no_grad():
                target = teacher(inputs, return_aux=False)
            student = model(inputs)
            values, cosine, valid = _descriptor_distances(student, target)
            if not valid.all():
                raise RuntimeError("Zero teacher descriptor cannot provide a real retention target")
            loss = values.sum() / len(images)
            weighted = weight * loss
            if not torch.isfinite(weighted):
                raise RuntimeError("Nonfinite weighted real retention loss")
            scaler.scale(weighted).backward()
            metrics["loss"] += float(loss.detach())
            metrics["weighted_total"] += float(weighted.detach())
            cosine_sum += float(cosine[valid].sum().detach())
            metrics["valid_targets"] += int(valid.sum())
            metrics["images"] += len(inputs)
            metrics["chunks"] += 1
            del target, student, values, cosine, valid, loss, weighted
    metrics["cosine"] = cosine_sum / max(metrics["valid_targets"], 1)
    metrics["seconds"] = time.perf_counter() - started
    return metrics
