"""Conservative, calibrated content safety checks; never a hallucination oracle.

Low DINO cosine, new OCR text, and changing edges are *signals*, not labels.
Only a joint outlier beyond a control-calibrated safety envelope is excluded.
An absent/invalid detector or missing required evidence yields UNKNOWN, which
is excluded from training. Existing image and quality ledgers remain untouched.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

VERSION = "content_safety_v4.1"
DENIED_DATASETS = {"svox", "robotcar", "robotcar-seasons", "tokyo", "tokyo247",
                   "tokyo24/7", "sf-xl", "sfxl", "nordland", "aachen"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def require_training_calibration(row):
    """Defense in depth; dataset/split declarations are mandatory, never inferred."""
    if row.get("split") != "calibration" or row.get("origin_split") != "train":
        raise ValueError("Content calibration requires split=calibration, origin_split=train")
    dataset = str(row.get("dataset", "")).strip().lower()
    if not dataset or dataset in DENIED_DATASETS:
        raise ValueError("Known evaluation corpora cannot calibrate this detector")
    for field in ("source_path", "output_path"):
        parts = {part.lower() for part in Path(str(row.get(field, ""))).parts}
        if parts & (DENIED_DATASETS | {"test", "val", "validation", "evaluation"}):
            raise ValueError("Evaluation/test paths cannot calibrate content decisions")
    if row.get("control") not in {"benign", "controlled_negative"}:
        raise ValueError("Calibration labels must come from programmatic controls")
    if not row.get("control_recipe") or not row.get("place_key"):
        raise ValueError("Control recipe and training place provenance required")


def pipeline_definition(provider, side, radius):
    if side <= 0 or radius < 0 or int(radius) != radius:
        raise ValueError("Positive input side and non-negative integer radius required")
    return {"version": VERSION, "provider": provider, "side": int(side),
            "search_radius": int(radius), "mapping": "inverse_full_patch_footprint",
            "structure": "aligned_local_standardized_gray_cosine_v1",
            "ocr": "unmatched_boxes_diagnostic_only_v1"}


class FrozenDINOProvider:
    """Explicit local model loading only; no torch hub network or downloads.

    Native DINO state dictionaries and full SALAD states containing the
    backbone.model.* prefix are supported. Returned maps have shape H,W,D.
    """
    def __init__(self, repository, checkpoint, architecture="dinov2_vitb14", device="cpu"):
        import torch
        repository, checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
        if not repository.is_dir() or not checkpoint.is_file():
            raise ValueError("A local DINO repository and existing weights are required")
        self.torch, self.device = torch, device
        self.model = torch.hub.load(str(repository), architecture, source="local", pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state.get("model", state))
        prefixed = {key[len("backbone.model."):]: value for key, value in state.items()
                    if key.startswith("backbone.model.")}
        self.model.load_state_dict(prefixed or state, strict=True)
        self.model.eval().requires_grad_(False).to(device)
        implementation = [repository / "hubconf.py"] + sorted((repository / "dinov2").rglob("*.py"))
        code = {str(path.relative_to(repository)): sha256(path) for path in implementation if path.is_file()}
        if not code:
            raise ValueError("Cannot fingerprint the local DINO implementation")
        self.definition = {"architecture": architecture, "checkpoint_sha256": sha256(checkpoint),
                           "normalization": "ImageNet", "repository": str(repository),
                           "repository_python_sha256": digest(code), "torch_version": str(torch.__version__)}

    def __call__(self, image, side):
        torch = self.torch
        patch = self.model.patch_size
        patch = patch[0] if isinstance(patch, (tuple, list)) else patch
        if side % patch:
            raise ValueError("DINO input side must be divisible by patch size")
        array = np.array(image.convert("RGB").resize((side, side), Image.Resampling.BICUBIC), copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        self.model.eval()
        with torch.no_grad():
            tokens = self.model.forward_features(((tensor - mean) / std)[None].to(self.device))
            tokens = tokens["x_norm_patchtokens"].float().cpu().numpy()[0]
        return tokens.reshape(side // patch, side // patch, -1)


def _project(points, matrix):
    values = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1) @ matrix.T
    denominator = values[..., 2:]
    return np.divide(values[..., :2], denominator, out=np.full_like(values[..., :2], np.nan),
                     where=np.abs(denominator) > 1e-12)


def validate_homography(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Finite non-singular 3x3 source-to-generated homography required")
    matrix = matrix / np.linalg.norm(matrix)
    corners = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]])
    values = (corners @ matrix.T)[:, 2]
    if not (np.all(values > 1e-12) or np.all(values < -1e-12)):
        raise ValueError("Homography horizon crosses the normalized frame")
    return matrix


def normalized_homography(row):
    """Historical quality geometry resizes BOTH images to a square matcher side."""
    side = row.get("matcher_image_size")
    if not isinstance(side, (int, float)) or side <= 0:
        raise ValueError("Matcher coordinate size is required; original dimensions cannot substitute")
    matrix = np.asarray(row.get("estimated_H"), dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Cached source-to-generated homography missing")
    to_normal = np.diag([1. / side, 1. / side, 1.])
    return validate_homography(to_normal @ matrix @ np.linalg.inv(to_normal))


def patch_distances(source, generated, matrix, radius=1):
    """Local DINO search, excluding patches whose warped footprint leaves source.

    A source feature may be selected by multiple generated patches: this is a
    safety signal, deliberately not ground-truth correspondence supervision.
    """
    source, generated = np.asarray(source, np.float32), np.asarray(generated, np.float32)
    if source.ndim != 3 or generated.ndim != 3 or source.shape[2] != generated.shape[2]:
        raise ValueError("Feature maps must have shape H,W,D and matching feature dimensions")
    if radius < 0 or int(radius) != radius:
        raise ValueError("Search radius must be a nonnegative integer")
    matrix = validate_homography(matrix)
    if not np.isfinite(source).all() or not np.isfinite(generated).all():
        raise ValueError("Non-finite DINO features")
    sh, sw, _ = source.shape
    gh, gw, _ = generated.shape
    yy, xx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    centers = np.stack(((xx + .5) / gw, (yy + .5) / gh), -1)
    mapped = _project(centers, np.linalg.inv(matrix))
    valid = np.ones((gh, gw), dtype=bool)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        corner = _project(np.stack(((xx + dx) / gw, (yy + dy) / gh), -1), np.linalg.inv(matrix))
        valid &= np.isfinite(corner).all(-1) & (corner >= -1e-9).all(-1) & (corner <= 1 + 1e-9).all(-1)
    norms0, norms1 = np.linalg.norm(source, axis=-1), np.linalg.norm(generated, axis=-1)
    norm0 = source / np.maximum(norms0[..., None], 1e-12)
    norm1 = generated / np.maximum(norms1[..., None], 1e-12)
    ix = np.floor(np.nan_to_num(mapped[..., 0], nan=-1) * sw).astype(int)
    iy = np.floor(np.nan_to_num(mapped[..., 1], nan=-1) * sh).astype(int)
    best, found = np.full((gh, gw), -1., np.float32), np.zeros((gh, gw), bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cx, cy = ix + dx, iy + dy
            in_bounds = (cx >= 0) & (cx < sw) & (cy >= 0) & (cy < sh)
            cx, cy = np.clip(cx, 0, sw - 1), np.clip(cy, 0, sh - 1)
            good = in_bounds & (norms0[cy, cx] > 1e-12) & (norms1 > 1e-12)
            score = (norm0[cy, cx] * norm1).sum(-1)
            best = np.where(good, np.maximum(best, score), best)
            found |= good
    valid &= found
    return np.clip(1 - best, 0, 2), valid


def structural_distances(source, generated, matrix, grid_shape, side):
    """Aligned local contrast shape; brightness/contrast invariance, no semantics.

    Uniform/fully clipped patches are unknown, never structural destruction.
    Night relighting can change this signal; it cannot independently reject.
    """
    matrix = validate_homography(matrix)
    scale = np.diag([side, side, 1.])
    inverse = np.linalg.inv(scale @ matrix @ np.linalg.inv(scale))
    inverse /= inverse[2, 2]
    src = source.convert("L").resize((side, side), Image.Resampling.BICUBIC)
    gen = generated.convert("L").resize((side, side), Image.Resampling.BICUBIC)
    warped = src.transform((side, side), Image.Transform.PERSPECTIVE, inverse.ravel()[:8],
                           resample=Image.Resampling.BILINEAR)
    # Mild fixed blur removes subpixel interpolation noise, not a pass/reject threshold.
    aa = np.asarray(warped.filter(ImageFilter.GaussianBlur(.5)), dtype=np.float32)
    bb = np.asarray(gen.filter(ImageFilter.GaussianBlur(.5)), dtype=np.float32)
    gh, gw = grid_shape
    distances, valid = np.zeros((gh, gw), np.float32), np.zeros((gh, gw), bool)
    for y in range(gh):
        for x in range(gw):
            box = (slice(y * side // gh, (y + 1) * side // gh),
                   slice(x * side // gw, (x + 1) * side // gw))
            a, b = aa[box].ravel(), bb[box].ravel()
            a, b = a - a.mean(), b - b.mean()
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na > 1e-6 and nb > 1e-6:
                distances[y, x] = np.clip(1 - (a @ b) / (na * nb), 0, 2)
                valid[y, x] = True
    return distances, valid


def ocr_diagnostics(cache, matrix):
    """OCR boxes use normalized xyxy coordinates in each ORIGINAL image.

    Newly detected text may have existed but become visible after relighting.
    Text differences and unmatched boxes therefore never constitute rejection.
    """
    if cache is None:
        return {"available": False, "newly_detected": None, "rejection_evidence": False}
    matrix = validate_homography(matrix)
    projected = []
    for entry in cache.get("source", []):
        x0, y0, x1, y1 = _ocr_box(entry)
        corners = _project(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]), matrix)
        projected.append([*corners.min(0), *corners.max(0)])
    unmatched = 0
    for entry in cache.get("generated", []):
        x0, y0, x1, y1 = _ocr_box(entry)
        # Any geometric overlap is descriptive support, not OCR text correctness.
        overlaps = [max(0, min(x1, b[2]) - max(x0, b[0])) *
                    max(0, min(y1, b[3]) - max(y0, b[1])) > 0 for b in projected]
        unmatched += not any(overlaps)
    return {"available": True, "source_count": len(projected),
            "generated_count": len(cache.get("generated", [])), "newly_detected": int(unmatched),
            "rejection_evidence": False, "interpretation": "newly_visible_is_not_newly_created"}


def _ocr_box(entry):
    box = np.asarray(entry["box_xyxy_normalized"], dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or (box < 0).any() or (box > 1).any():
        raise ValueError("OCR boxes require finite normalized xyxy coordinates")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("OCR box must have positive area")
    return box


def pair_signals(row, provider, side=448, radius=1, ocr_cache=None):
    source, generated = Path(row["source_path"]), Path(row["output_path"])
    hashes = {"source_sha256": sha256(source), "output_sha256": sha256(generated)}
    for key, value in hashes.items():
        if row.get(key) and row[key] != value:
            raise ValueError(f"Image changed relative to generation ledger: {key}")
    if ocr_cache is not None and any(ocr_cache.get(key) != value for key, value in hashes.items()):
        raise ValueError("OCR evidence is not bound to these exact image bytes")
    matrix = normalized_homography(row)
    with Image.open(source) as a, Image.open(generated) as b:
        f0, f1 = provider(a, side), provider(b, side)
        distance, valid = patch_distances(f0, f1, matrix, radius)
        structure, structure_valid = structural_distances(a, b, matrix, distance.shape, side)
    pipeline = pipeline_definition(provider.definition, side, radius)
    return {"sample_id": row.get("sample_id"), **hashes, "pipeline": pipeline,
            "pipeline_sha256": digest(pipeline), "grid_shape": list(distance.shape),
            "dino_distance": distance.tolist(), "structure_distance": structure.tolist(),
            "valid": valid.tolist(), "structure_valid": (valid & structure_valid).tolist(),
            "valid_fraction": float(valid.mean()), "ocr": ocr_diagnostics(ocr_cache, matrix),
            "geometry_sha256": digest({"estimated_H": row.get("estimated_H"),
                                        "matcher_image_size": row.get("matcher_image_size")})}


def _arrays(signals):
    distance = np.asarray(signals["dino_distance"], np.float32)
    structure = np.asarray(signals["structure_distance"], np.float32)
    valid, structure_valid = np.asarray(signals["valid"], bool), np.asarray(signals["structure_valid"], bool)
    if (distance.ndim != 2 or not distance.size or
            any(a.shape != distance.shape for a in (structure, valid, structure_valid)) or
            not np.isfinite(distance).all() or not np.isfinite(structure).all()):
        raise ValueError("Invalid/nonfinite patch signal arrays")
    if ((distance < 0) | (distance > 2)).any() or ((structure < 0) | (structure > 2)).any():
        raise ValueError("Cosine distance outside mathematical range")
    return distance, structure, valid, structure_valid & valid


def largest_component(mask):
    mask = np.asarray(mask, bool)
    pending = set(zip(*np.nonzero(mask)))
    best = 0
    while pending:
        stack = [pending.pop()]
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
        best = max(best, count)
    return best


def summarize(signals, thresholds):
    distance, structure, valid, structural_valid = _arrays(signals)
    size = int(valid.sum())
    dino_mask = valid & (distance > thresholds["dino_patch"])
    joint = dino_mask & structural_valid & (structure > thresholds["structure_patch"])
    return {"valid_fraction": float(valid.mean()), "valid_patches": size,
            "structure_fraction": float(structural_valid.mean()),
            "dino_mean": float(distance[valid].mean()) if size else None,
            "dino_anomaly_fraction": float(dino_mask.sum() / max(size, 1)),
            "joint_component_fraction": float(largest_component(joint) / max(size, 1)),
            "ocr": signals.get("ocr", {"available": False})}


def calibrate_controls(rows_and_signals, false_alarm_rate, *, min_pairs=20):
    """Control-driven envelope, not semantic truth; report held-out sensitivity.

    Place hashing splits controls into fit and held-out sets. Confidence or
    coverage is not silently inferred from a high match score. The operating
    false alarm level must be explicitly chosen by the caller.
    """
    if not 0 < false_alarm_rate < 1 or min_pairs < 2:
        raise ValueError("Explicit false_alarm_rate in (0,1), min_pairs>=2 required")
    rows_and_signals = list(rows_and_signals)
    pipeline_ids = set()
    grouped = {"fit": [], "heldout": []}
    controls_by_place = {}
    for row, signals in rows_and_signals:
        require_training_calibration(row)
        _arrays(signals)
        for key in ("source_sha256", "output_sha256"):
            if not row.get(key) or signals.get(key) != row[key]:
                raise ValueError("Calibration signals must match exact control image hashes")
        if signals.get("pipeline_sha256") != digest(signals.get("pipeline")):
            raise ValueError("Calibration pipeline fingerprint invalid")
        pipeline_ids.add(signals["pipeline_sha256"])
        place = controls_by_place.setdefault(row["place_key"], {})
        if row["control"] in place:
            raise ValueError("Exactly one benign and one controlled_negative control per training place required")
        place[row["control"]] = (row, signals)
        partition = "heldout" if int(digest(row["place_key"])[:8], 16) % 4 == 0 else "fit"
        grouped[partition].append((row, signals))
    for place in controls_by_place.values():
        if set(place) != {"benign", "controlled_negative"}:
            raise ValueError("Every training place requires a complete benign/controlled_negative pair")
        benign_row, _ = place["benign"]
        negative_row, _ = place["controlled_negative"]
        if any(benign_row.get(key) != negative_row.get(key)
               for key in ("source_path", "source_sha256", "estimated_H", "matcher_image_size")):
            raise ValueError("Paired controls must share exact source and geometric transform")
        if any(negative_row["control_recipe"].get(key) != value
               for key, value in benign_row["control_recipe"].items()):
            raise ValueError("Paired controls must share their benign transform recipe")
    if len(pipeline_ids) != 1:
        raise ValueError("Exactly one feature provider/preprocessing pipeline required")
    for partition, pairs in grouped.items():
        for label in ("benign", "controlled_negative"):
            if sum(row["control"] == label for row, _ in pairs) < min_pairs:
                raise ValueError(f"Need at least {min_pairs} {partition} {label} controls from training places")
    benign = [signals for row, signals in grouped["fit"] if row["control"] == "benign"]
    patch_values, structure_values = [], []
    for signals in benign:
        d, s, v, sv = _arrays(signals)
        patch_values.extend(d[v].tolist())
        structure_values.extend(s[sv].tolist())
    if not patch_values or not structure_values:
        raise ValueError("Benign controls lack valid DINO/structure patches")
    # The component envelope catches spatially coherent joint outliers, not one
    # marginal low-cosine patch; all numeric limits come from benign controls.
    thresholds = {"dino_patch": float(np.quantile(patch_values, 1 - false_alarm_rate)),
                  "structure_patch": float(np.quantile(structure_values, 1 - false_alarm_rate))}
    summary = [summarize(signals, thresholds) for signals in benign]
    if any(s["valid_fraction"] <= 0 or s["structure_fraction"] <= 0 for s in summary):
        raise ValueError("Every benign fit control requires valid DINO and structural overlap")
    thresholds.update({"joint_component": max(s["joint_component_fraction"] for s in summary),
                       "dino_anomaly": max(s["dino_anomaly_fraction"] for s in summary),
                       "min_valid_fraction": min(s["valid_fraction"] for s in summary),
                       "min_structure_fraction": min(s["structure_fraction"] for s in summary)})
    artifact = {"version": VERSION, "pipeline": benign[0]["pipeline"],
                "pipeline_sha256": next(iter(pipeline_ids)), "thresholds": thresholds,
                "false_alarm_rate_requested": false_alarm_rate,
                "scope": "programmatic_training_controls_only_not_hallucination_truth",
                "control_provenance": [{"sample_id": row.get("sample_id"),
                   "place_key": row["place_key"], "control": row["control"],
                   "source_sha256": row["source_sha256"], "output_sha256": row["output_sha256"],
                   "recipe_sha256": digest(row["control_recipe"])} for row, _ in rows_and_signals]}
    metrics = {}
    for label in ("benign", "controlled_negative"):
        decisions = [_decide(summarize(sig, thresholds), thresholds)
                     for row, sig in grouped["heldout"] if row["control"] == label]
        metrics[label] = {"n": len(decisions), **{name: decisions.count(name) for name in ("pass", "reject", "unknown")}}
    artifact["heldout_controls"] = metrics
    b, n = metrics["benign"], metrics["controlled_negative"]
    # Each insertion control shares a place/source/benign transform with its
    # benign control. Use exact *paired* discordances, not independent rates.
    discordance = {"negative_only_reject": 0, "benign_only_reject": 0,
                   "both_reject": 0, "neither_reject": 0}
    for key, place in controls_by_place.items():
        if int(digest(key)[:8], 16) % 4 != 0:
            continue
        br = _decide(summarize(place["benign"][1], thresholds), thresholds) == "reject"
        nr = _decide(summarize(place["controlled_negative"][1], thresholds), thresholds) == "reject"
        outcome = ("both_reject" if br and nr else "benign_only_reject" if br else
                   "negative_only_reject" if nr else "neither_reject")
        discordance[outcome] += 1
    successes = discordance["negative_only_reject"]
    discordant_n = successes + discordance["benign_only_reject"]
    p_separation = (sum(math.comb(discordant_n, k) for k in range(successes, discordant_n + 1)) /
                    2 ** discordant_n) if discordant_n else 1.
    artifact["heldout_separation_test"] = {"name": "exact_paired_discordance_one_sided",
                                           **discordance, "p": p_separation}
    artifact["usable"] = bool(p_separation <= false_alarm_rate and
                               b["reject"] / b["n"] <= false_alarm_rate and n["reject"] > 0)
    artifact["certification"] = "experimental_control_envelope_no_semantic_guarantee"
    artifact["artifact_sha256"] = digest(artifact)
    return artifact


def _decide(summary, thresholds):
    if (not summary["valid_patches"] or
            summary["valid_fraction"] < thresholds["min_valid_fraction"] or
            summary["structure_fraction"] < thresholds["min_structure_fraction"]):
        return "unknown"
    component_outlier = summary["joint_component_fraction"] > thresholds["joint_component"]
    dino_outlier = summary["dino_anomaly_fraction"] > thresholds["dino_anomaly"]
    if component_outlier and dino_outlier:
        return "reject"
    if component_outlier or dino_outlier:
        return "unknown"
    return "pass"


def audit_pair(row, signals, calibration, *, quarantine_variants=("shopfronts",)):
    result = {"version": VERSION, "decision": "unknown", "eligible_for_training": False,
              "reasons": [], "signals_are_semantic_truth": False}
    if row.get("prompt_variant") in quarantine_variants:
        result.update(decision="reject", reasons=["prompt_variant_quarantined_by_policy"])
        return result
    base_passed = row.get("base_quality_passed", row.get("geometry_ok") is True and
                          row.get("weather_ok") is True and row.get("passed") is True)
    if base_passed is not True:
        result["reasons"] = ["base_geometry_weather_not_verified"]
        return result
    if signals is None or calibration is None:
        result["reasons"] = ["required_signals_or_calibration_missing"]
        return result
    try:
        definition = {key: value for key, value in calibration.items() if key != "artifact_sha256"}
        if (calibration.get("version") != VERSION or
                calibration.get("artifact_sha256") != digest(definition)):
            raise ValueError("Calibration artifact fingerprint invalid")
        if calibration.get("usable") is not True:
            raise ValueError("Control detector did not separate held-out content insertions from relighting")
        if (signals.get("pipeline_sha256") != calibration["pipeline_sha256"] or
                signals.get("pipeline_sha256") != digest(signals["pipeline"])):
            raise ValueError("Provider/preprocessing differs from calibration")
        for key in ("source_sha256", "output_sha256"):
            if not row.get(key) or row[key] != signals.get(key):
                raise ValueError("Signals do not match exact image bytes")
        geometry_digest = digest({"estimated_H": row.get("estimated_H"),
                                  "matcher_image_size": row.get("matcher_image_size")})
        if signals.get("geometry_sha256") != geometry_digest:
            raise ValueError("Signals do not match the current homography")
        summary = summarize(signals, calibration["thresholds"])
        decision = _decide(summary, calibration["thresholds"])
        result.update(decision=decision, eligible_for_training=decision == "pass", diagnostics=summary,
                      calibration_sha256=calibration["artifact_sha256"])
        result["reasons"] = [{"pass": "inside_control_calibrated_safety_envelope",
                              "reject": "joint_content_safety_outlier_not_hallucination_proof",
                              "unknown": "ambiguous_or_insufficient_valid_overlap"}[decision]]
    except (ValueError, TypeError, KeyError, IndexError) as error:
        result["reasons"] = ["invalid_required_evidence: " + str(error)]
    return result
