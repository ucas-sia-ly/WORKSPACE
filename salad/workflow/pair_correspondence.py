"""Detached, conservative same-view patch evidence; no training loss.

Indices map generated patches to flattened source patches. Images/grids cover
the whole frame (resize only, no undisclosed crop). A is calibrated reciprocal
teacher/relative-structure evidence; B is an independently supported geometric
candidate when the teacher rule fails. Neither is semantic landmark truth.
Only explicit destruction/occlusion masks produce negatives. Missing matches,
low cosine, OT keep, image edges and whole-image quality never create negatives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from .reliability import _shift


@dataclass(frozen=True)
class CorrespondenceCalibration:
    """Empirical null cutoffs, fitted on separate, mismatched-place pairs.

    tail_probability is an explicit exploratory operating point, NOT a proven
    false-match rate. There is deliberately no default cosine/margin cutoff.
    Recalibrate after changing teacher, preprocessing, grid or search radius.
    """
    cosine_min: float
    margin_min: float
    structure_max: float
    tail_probability: float
    samples: int
    search_radius: int
    provenance: dict[str, Any]

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.cosine_min, self.margin_min,
                                               self.structure_max, self.tail_probability))
                or not -1 <= self.cosine_min <= 1 or not 0 <= self.margin_min <= 2
                or not 0 <= self.structure_max <= 2 or not 0 < self.tail_probability < 1
                or type(self.samples) is not int or self.samples < 2
                or type(self.search_radius) is not int or self.search_radius < 1):
            raise ValueError("Invalid correspondence calibration")

    def to_dict(self):
        return asdict(self)


def _feature(aux):
    x = aux["backbone_local_features"]
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError("One finite teacher feature grid [C,H,W] is required")
    return x.detach().float()


def _centers(shape, device):
    h, w = shape
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return torch.stack(((x.flatten() + .5) / w, (y.flatten() + .5) / h), -1)


def _signature(feature):
    # Reuse reliability.py's spatial primitive, but not its uncalibrated .08/.025
    # thresholds. Invalid neighbors are explicitly excluded, including borders.
    centered = feature - feature.mean((-2, -1), keepdim=True)
    valid = centered.norm(dim=0, keepdim=True) > torch.finfo(torch.float32).eps
    normalized = F.normalize(centered, dim=0)
    values, masks = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                values.append((normalized * _shift(normalized, dy, dx)).sum(0))
                masks.append((valid & _shift(valid, dy, dx))[0])
    return torch.stack(values).flatten(1).T, torch.stack(masks).flatten(1).T


def _direction(query, reference, radius):
    qh, qw = query.shape[-2:]
    rh, rw = reference.shape[-2:]
    qc, rc = _centers((qh, qw), query.device), _centers((rh, rw), query.device)
    # Compare against the nearest same-relative-coordinate source patch.
    anchor = (qc * qc.new_tensor([rw, rh])).floor().long()
    ridx = torch.arange(rh * rw, device=query.device)
    delta = torch.stack((ridx % rw, ridx // rw), -1)[None] - anchor[:, None]
    allowed = delta.abs().amax(-1) <= radius
    similarity = F.normalize(query.flatten(1), dim=0).T @ F.normalize(reference.flatten(1), dim=0)
    # Prefer the same coordinate only on exact ties; do not hide ambiguity by
    # adding a positional bonus to the cosine or the best/second-best margin.
    order = (qc[:, None] - rc[None]).square().sum(-1).argsort(dim=1, stable=True)
    scores = similarity.masked_fill(~allowed, -torch.inf).gather(1, order)
    top = scores.argsort(dim=1, descending=True, stable=True)[:, :min(2, rh * rw)]
    index = order.gather(1, top[:, :1]).squeeze(1)
    best = similarity.gather(1, index[:, None]).squeeze(1).clamp(-1, 1)
    second = scores.gather(1, top[:, -1:]).squeeze(1)
    margin = torch.where(allowed.sum(1) >= 2, best - second, torch.zeros_like(best)).clamp(0, 2)
    qsig, qvalid = _signature(query)
    rsig, rvalid = _signature(reference)
    shared = qvalid & rvalid[index]
    structure = (((qsig - rsig[index]).square() * shared).sum(-1)
                 / shared.sum(-1).clamp_min(1)).sqrt()
    structure_valid = shared.any(-1)
    displacement = delta.gather(1, index[:, None, None].expand(-1, 1, 2)).squeeze(1).abs().amax(-1)
    return dict(index=index, cosine=best, margin=margin, structure=structure,
                structure_valid=structure_valid, displacement=displacement,
                candidate_count=allowed.sum(1))


@torch.no_grad()
def correspondence_measurements(source_aux, generated_aux, search_radius=1):
    """Unthresholded two-way evidence, usable for calibration and diagnostics."""
    if type(search_radius) is not int or search_radius < 1:
        raise ValueError("search_radius must be an integer >= 1")
    source, generated = _feature(source_aux), _feature(generated_aux)
    if source.device != generated.device or source.shape[0] != generated.shape[0]:
        raise ValueError("Teacher feature channels and devices must agree")
    with torch.autocast(device_type=source.device.type, enabled=False):
        forward = _direction(generated, source, search_radius)
        reverse = _direction(source, generated, search_radius)
        idx = forward["index"]
        forward["reciprocal"] = reverse["index"][idx] == torch.arange(idx.numel(), device=idx.device)
        forward["reverse_margin"] = reverse["margin"][idx]
        forward["margin"] = torch.minimum(forward["margin"], forward["reverse_margin"])
        forward["reverse_displacement"] = reverse["displacement"][idx]
        return forward


def fit_calibration(measurements, *, tail_probability, search_radius, provenance,
                    positive_controls=None):
    """Fit disjoint nulls and optional known-copy controls, never weather labels.

    Copy controls calibrate the numerical/texture ambiguity floor. Null margin
    tails alone can exceed the margins of perfectly aligned smooth patches.
    Both choices and their observed values are recorded, not silently tuned.
    """
    if not 0 < tail_probability < 1:
        raise ValueError("tail_probability must be in (0,1)")
    if not measurements:
        raise ValueError("Calibration requires mismatched-pair measurements")
    data = {key: torch.cat([m[key][m["structure_valid"]].detach().float().cpu()
                           for m in measurements]) for key in ("cosine", "margin", "structure")}
    n = len(data["cosine"])
    if n < math.ceil(1 / tail_probability):
        raise ValueError("Insufficient calibration samples for requested tail probability")
    if any(not torch.isfinite(x).all() for x in data.values()):
        raise ValueError("Calibration measurements must be finite")
    null_margin = float(torch.quantile(data["margin"], 1 - tail_probability, interpolation="higher"))
    margin = null_margin
    provenance = dict(provenance, null_margin_quantile=null_margin, margin_rule="null_upper_tail")
    if positive_controls:
        margins = torch.cat([m["margin"][m["reciprocal"] & m["structure_valid"]].detach().float().cpu()
                             for m in positive_controls])
        if margins.numel() < math.ceil(1 / tail_probability) or not torch.isfinite(margins).all():
            raise ValueError("Insufficient finite copy-control samples")
        margin = float(torch.quantile(margins, tail_probability, interpolation="lower"))
        provenance.update(margin_rule="known_copy_lower_tail", copy_control_samples=len(margins))
    return CorrespondenceCalibration(
        float(torch.quantile(data["cosine"], 1 - tail_probability, interpolation="higher")),
        margin,
        float(torch.quantile(data["structure"], tail_probability, interpolation="lower")),
        tail_probability, n, search_radius, provenance)


def points_to_patch(points, canvas_hw, grid_hw):
    """Map matcher-space XY pixel centers through full-frame resize to a grid.

    canvas_hw MUST be the actual matcher tensor size, not the original image
    size or a guessed 512 square. Coordinates outside the canvas are invalid,
    not clamped into a boundary patch. No crop/padding transform is inferred.
    """
    points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2)
    if any(type(v) is not int or v < 1 for v in (*canvas_hw, *grid_hw)):
        raise ValueError("Canvas/grid sizes must be positive integers")
    h, w = canvas_hw
    gh, gw = grid_hw
    valid = torch.isfinite(points).all(1) & (points >= 0).all(1)
    valid &= (points[:, 0] < w) & (points[:, 1] < h)
    # Coordinates refer to pixel centers indexed from zero, hence +0.5.
    xy = ((torch.nan_to_num(points) + .5) * points.new_tensor([gw / w, gh / h])).floor().long()
    valid &= (xy[:, 0] < gw) & (xy[:, 1] < gh)
    index = xy[:, 1] * gw + xy[:, 0]
    return index.masked_fill(~valid, -1), valid


def _geometry(evidence, source_shape, generated_shape, radius, device):
    ns, ng = math.prod(source_shape), math.prod(generated_shape)
    empty = dict(index=torch.full((ng,), -1, device=device, dtype=torch.long),
                 confidence=torch.zeros(ng, device=device), votes=torch.zeros(ng, device=device),
                 valid_points=0, status="absent")
    if evidence is None:
        return empty
    if evidence.get("status") != "ok":
        return {**empty, "status": evidence.get("status", "unverified")}
    if evidence.get("coordinate_convention") != "xy_pixel_centers_full_frame_resize":
        raise ValueError("Geometry requires explicit full-frame pixel coordinate convention")
    p0 = torch.as_tensor(evidence["inlier_kpts0"], dtype=torch.float32).reshape(-1, 2)
    p1 = torch.as_tensor(evidence["inlier_kpts1"], dtype=torch.float32).reshape(-1, 2)
    if p0.shape != p1.shape:
        raise ValueError("Paired inlier arrays must have identical lengths")
    # Duplicate correspondences cannot manufacture corroboration.
    pairs = torch.unique(torch.cat([p0, p1], 1), dim=0)
    p0, p1 = pairs[:, :2], pairs[:, 2:]
    homography = torch.as_tensor(evidence.get("H", []), dtype=torch.float32)
    reprojection_limit = evidence.get("ransac_reprojection_threshold", 0)
    if (homography.shape != (3, 3) or not torch.isfinite(homography).all()
            or torch.linalg.matrix_rank(homography) < 3
            or not math.isfinite(reprojection_limit) or reprojection_limit <= 0):
        return {**empty, "status": "invalid_homography_or_ransac_metadata"}
    projected = torch.cat([p0, torch.ones(len(p0), 1)], 1) @ homography.T
    denominator = projected[:, 2:]
    h_valid = denominator[:, 0].abs() > torch.finfo(torch.float32).eps
    reprojection_error = (projected[:, :2] / denominator - p1).norm(dim=1)
    i0, valid0 = points_to_patch(p0, tuple(evidence["source_canvas_hw"]), source_shape)
    i1, valid1 = points_to_patch(p1, tuple(evidence["generated_canvas_hw"]), generated_shape)
    valid = valid0 & valid1 & h_valid & (reprojection_error <= reprojection_limit)
    counts = torch.zeros(ng, ns, dtype=torch.float32)
    counts.index_put_((i1[valid], i0[valid]), torch.ones(int(valid.sum())), accumulate=True)
    best_votes, index = counts.max(1)
    reverse = counts.argmax(0)
    own = torch.arange(ng)
    # Strict two-way majority and >=2 distinct local observations. This is
    # corroboration, not a semantic score derived from keypoint density.
    total_g, total_s = counts.sum(1), counts.sum(0)[index]
    mutual = ((reverse[index] == own) & (best_votes >= 2)
              & (2 * best_votes > total_g) & (2 * best_votes > total_s))
    gc, sc = _centers(generated_shape, counts.device), _centers(source_shape, counts.device)
    source_scale = counts.new_tensor([source_shape[1], source_shape[0]])
    generated_scale = counts.new_tensor([generated_shape[1], generated_shape[0]])
    disp_s = ((sc[index] * source_scale).floor() - (gc * source_scale).floor()).abs()
    disp_g = ((sc[index] * generated_scale).floor() - (gc * generated_scale).floor()).abs()
    mutual &= (disp_s.amax(1) <= radius) & (disp_g.amax(1) <= radius)
    confidence = torch.minimum(best_votes / total_g.clamp_min(1), best_votes / total_s.clamp_min(1))
    return dict(index=index.masked_fill(~mutual, -1).to(device),
                confidence=(confidence * mutual).to(device), votes=best_votes.to(device),
                valid_points=int(valid.sum()), status="ok")


def _mask(mask, image_hw, grid_hw, device):
    if mask is None:
        return torch.zeros(math.prod(grid_hw), device=device)
    mask = torch.as_tensor(mask, device=device).detach().float()
    if (tuple(mask.shape) != tuple(image_hw) or not torch.isfinite(mask).all()
            or (mask < 0).any() or (mask > 1).any()):
        raise ValueError("Masks must be [image H,image W] finite values in [0,1]")
    # Exact full-frame area overlap, including non-divisible native sizes.
    # Adaptive pooling would double-count pixels crossing uneven bin edges.
    def weights(pixels, patches):
        edges = torch.arange(patches + 1, device=device, dtype=torch.float32) * (pixels / patches)
        cells = torch.arange(pixels, device=device, dtype=torch.float32)
        overlap = (torch.minimum(edges[1:, None], cells[None] + 1)
                   - torch.maximum(edges[:-1, None], cells[None])).clamp_min(0)
        return overlap / (pixels / patches)
    with torch.autocast(device_type=device.type, enabled=False):
        return (weights(image_hw[0], grid_hw[0]) @ mask
                @ weights(image_hw[1], grid_hw[1]).T).flatten().clamp(0, 1)


def _teacher_diagnostics(aux, shape):
    result = {}
    for key in ("score_logits", "log_assignment", "assignment_keep"):
        value = aux.get(key)
        if value is None:
            continue
        value = value.detach().float()
        if not torch.isfinite(value).all() or value.numel() % math.prod(shape):
            raise ValueError(f"Invalid teacher {key}")
        if value.shape[-1] != math.prod(shape):
            raise ValueError(f"Teacher {key} does not match patch grid")
        value = value.reshape(-1, math.prod(shape))
        result[key + "_mean"] = float(value.mean())
        if key == "log_assignment":
            probability = value.exp()
            if not torch.isfinite(probability).all():
                raise ValueError("Teacher log_assignment overflows probability space")
            result["ot_patch_mass_max_error"] = float((probability.sum(0) - 1).abs().max())
            result["teacher_keep"] = probability[:-1].sum(0)
    return result


def _image_aliases(image, grid_hw, radius, device, patch_stride):
    """Veto exactly repeated local RGB patches (including feature position bias).

    This is an ambiguity veto, never positive/negative structure evidence.
    Exact equality needs no tuned similarity threshold. Near repetitions must
    still be handled by feature margins/geometry and inspected in diagnostics.
    """
    import numpy as np
    from PIL import Image

    h, w = grid_hw
    resized = image.resize((w * patch_stride, h * patch_stride), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.asarray(resized).copy())
    patches = pixels.reshape(h, patch_stride, w, patch_stride, 3).permute(1, 3, 4, 0, 2).reshape(-1, h, w)
    alias = torch.zeros(h, w, dtype=torch.bool)
    valid = torch.ones(1, h, w, dtype=torch.bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy or dx:
                alias |= (patches == _shift(patches, dy, dx)).all(0) & _shift(valid, dy, dx)[0]
    return alias.flatten().to(device)


@torch.no_grad()
def build_pair_correspondence(source_image, generated_image, source_aux, generated_aux, *,
                              calibration=None, search_radius=1, alignment_tolerance=1,
                              geometry=None, quality=None, edit_mask=None,
                              occlusion_mask=None, destruction_mask=None, patch_stride=14):
    """Return generated-grid [H,W] indices, confidences, unknown and diagnostics.

    Images are PIL RGB images. Masks are in the original generated-image canvas;
    edit_mask means edited/uncertain (veto only), whereas occlusion/destruction
    must be explicit observed evidence. Partial masked patches receive their
    masked area fraction as negative weight; all overlap vetoes positives.
    Without calibration, teacher-only positives are disabled. B candidates can
    still be proposed by independent inlier evidence. No default pseudo labels.
    """
    if type(alignment_tolerance) is not int or not 0 <= alignment_tolerance <= search_radius:
        raise ValueError("alignment_tolerance must be in [0, search_radius]")
    if type(patch_stride) is not int or patch_stride < 1:
        raise ValueError("patch_stride must be a positive integer")
    source, generated = _feature(source_aux), _feature(generated_aux)
    sh, gh = tuple(source.shape[-2:]), tuple(generated.shape[-2:])
    if getattr(source_image, "mode", None) != "RGB" or getattr(generated_image, "mode", None) != "RGB":
        raise ValueError("Source and generated images must be PIL RGB full frames")
    m = correspondence_measurements(source_aux, generated_aux, search_radius)
    if calibration is not None and calibration.search_radius != search_radius:
        raise ValueError("Calibration search radius does not match")
    geo = _geometry(geometry, sh, gh, alignment_tolerance, generated.device)
    near = (m["displacement"] <= alignment_tolerance) & (m["reverse_displacement"] <= alignment_tolerance)
    eps = torch.finfo(torch.float32).eps
    teacher_ok = torch.zeros_like(near)
    teacher_conf = torch.zeros_like(m["cosine"])
    geometric_a = torch.zeros_like(near)
    has_geo = geo["index"] >= 0
    geometric_cosine = (F.normalize(generated.flatten(1), dim=0)
                        * F.normalize(source.flatten(1), dim=0)[:, geo["index"].clamp_min(0)]).sum(0).clamp(-1, 1)
    low_teacher = has_geo.clone()
    if calibration is not None:
        c = calibration
        teacher_ok = (m["reciprocal"] & near & m["structure_valid"]
                      & (m["cosine"] > c.cosine_min) & (m["margin"] > max(eps, c.margin_min))
                      & (m["structure"] <= c.structure_max))
        # Continuous evidence strengths, NOT calibrated posterior probabilities.
        # Normalize distinctiveness against the same source's self-match margin.
        # A copied, locally unique patch has strength near one; a tied wall has
        # zero margin and cannot pass the ambiguity gate.
        with torch.autocast(device_type=source.device.type, enabled=False):
            self_margin = _direction(source, source, search_radius)["margin"][m["index"]]
        contrast = ((m["margin"] - c.margin_min) / (self_margin - c.margin_min).clamp_min(eps)).clamp(0, 1)
        appearance = ((m["cosine"] - c.cosine_min) / max(eps, 1 - c.cosine_min)).clamp(0, 1)
        teacher_conf = contrast * appearance
        geometric_a = (has_geo & (geo["index"] == m["index"]) & m["reciprocal"] & near
                       & (m["margin"] > max(eps, c.margin_min)) & (m["cosine"] > c.cosine_min))
        low_teacher = geometric_cosine <= c.cosine_min
    conflict = has_geo & (geo["index"] != m["index"]) & teacher_ok
    a = (teacher_ok | geometric_a) & ~conflict
    # B is intentionally marked as a candidate, not an error/certain positive.
    b = has_geo & low_teacher & ~teacher_ok & ~a
    index = torch.where(b, geo["index"], m["index"])
    positive = torch.where(a, teacher_conf, torch.where(b, geo["confidence"], 0.))
    if quality is not None and not isinstance(quality, dict):
        raise ValueError("Manifest quality must be a dictionary")
    quality = quality or {}
    blocked = any(key in quality and quality[key] is not True
                  for key in ("geometry_ok", "passed", "eligible_for_training"))
    if blocked:
        a.zero_(); b.zero_(); positive.zero_()
    image_hw = (generated_image.height, generated_image.width)
    edited = _mask(edit_mask, image_hw, gh, generated.device)
    occluded = _mask(occlusion_mask, image_hw, gh, generated.device)
    destroyed = _mask(destruction_mask, image_hw, gh, generated.device)
    negative = torch.maximum(occluded, destroyed)
    if occlusion_mask is not None and destruction_mask is not None:
        negative = _mask(torch.maximum(torch.as_tensor(occlusion_mask, device=generated.device),
                                        torch.as_tensor(destruction_mask, device=generated.device)),
                         image_hw, gh, generated.device)
    source_alias = _image_aliases(source_image, sh, search_radius, source.device, patch_stride)
    generated_alias = _image_aliases(generated_image, gh, search_radius, generated.device, patch_stride)
    alias = generated_alias | source_alias[index.clamp_min(0)]
    veto = (edited > 0) | (negative > 0) | alias
    positive[veto] = 0
    a &= ~veto; b &= ~veto
    unknown = (positive == 0) & (negative == 0)
    # Unknown/negative patches have no supervision correspondence. Candidate
    # indices remain inspectable in diagnostics, never passed as valid labels.
    index = index.masked_fill(positive == 0, -1)
    diagnostics = {**m, "positive_a": a.reshape(gh), "positive_b_candidate": b.reshape(gh),
                   "geometry_conflict": conflict.reshape(gh), "geometry_index": geo["index"].reshape(gh),
                   "geometry_votes": geo["votes"].reshape(gh), "geometry_status": geo["status"],
                   "geometry_cosine": geometric_cosine.reshape(gh),
                   "appearance_alias": alias.reshape(gh),
                   "geometry_valid_points": geo["valid_points"], "quality_blocked": blocked,
                   "quality": quality, "calibration": calibration.to_dict() if calibration else None,
                   "source_grid_hw": sh, "generated_grid_hw": gh,
                   "source_image_hw": (source_image.height, source_image.width), "generated_image_hw": image_hw,
                   "source_teacher": _teacher_diagnostics(source_aux, sh),
                   "generated_teacher": _teacher_diagnostics(generated_aux, gh),
                   "positive_a_count": int(a.sum()), "positive_b_count": int(b.sum()),
                   "negative_count": int((negative > 0).sum()), "unknown_count": int(unknown.sum()),
                   "confidence_semantics": "evidence strength; B is a hard-positive candidate, not semantic truth"}
    return dict(match_index=index.reshape(gh), positive_conf=positive.reshape(gh),
                negative_conf=negative.reshape(gh), unknown_mask=unknown.reshape(gh),
                match_diagnostics=diagnostics)
