"""Online fixed-resize teacher pairing and bounded student microbatches.

No feature cache or matcher model is owned here. Stage 3 optionally supervises
the existing reliability head, with independent per-step controlled occlusions.
The Stage 1 synthetic query augmentation remains intact; this is a separate
local view of the same exact source/generated files selected by that sampler.
"""
from __future__ import annotations

import hashlib
from contextlib import nullcontext
import json
from pathlib import Path
import time

from PIL import Image
import torch

from .pair_correspondence import CorrespondenceCalibration, build_pair_correspondence
from .reliability_cache import file_sha256, preprocess_teacher_image
from .synthetic_losses import local_distillation_losses, local_target_weights
from .training_data import _resolve_manifest_file


class LocalDistillationSupervisor:
    def __init__(self, teacher, dataset, args):
        self.teacher = teacher
        if teacher.provenance["model_config"]["agg_config"].get("reliability_ot", False):
            raise ValueError("Stage 2C requires a teacher without a Reliability Head")
        self.size = tuple(args.image_size)
        self.mode = args.pair_confidence_mode
        self.student_eval = getattr(args,'local_distill_eval',False)
        self.min_keep = args.ot_min_teacher_keep
        self.chunk_size = args.teacher_batch_size
        self.reliability_weight = args.reliability_generated_weight
        self.occlusion_weight = args.reliability_generated_occlusion_weight
        self.seed = args.seed
        self.device = next(teacher.parameters()).device
        path = args.pair_calibration.expanduser().resolve()
        calibration_bytes = path.read_bytes()
        payload = json.loads(calibration_bytes)
        if isinstance(payload, list):
            if not 0 <= args.pair_calibration_index < len(payload):
                raise ValueError("pair calibration index is out of range")
            payload = payload[args.pair_calibration_index]
        elif args.pair_calibration_index != 0:
            raise ValueError("A single calibration object only supports index 0")
        self.calibration = CorrespondenceCalibration(**payload)
        recorded = self.calibration.provenance.get("teacher", {})
        if (recorded.get("checkpoint_sha256") != teacher.provenance["checkpoint_sha256"]
                or recorded.get("image_preprocessing") != teacher.provenance["image_preprocessing"]):
            raise ValueError("Pair calibration teacher/preprocessing does not match this frozen teacher")
        if self.calibration.search_radius != 1:
            raise ValueError("Stage 2C fixed-resize baseline requires radius-1 calibration")
        self.manifest = Path(dataset.synthetic_manifest) if hasattr(dataset, "synthetic_manifest") else args.synthetic_manifest
        expected = {str(output): str(source) for place in dataset.places
                    for output, source in place.source_by_synthetic_path.items()}
        self.rows = {}
        if self.manifest is not None:
            self.manifest = self.manifest.resolve()
            for line_number, line in enumerate(self.manifest.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("passed") is not True or row.get("eligible_for_training") is not True:
                    continue
                raw = Path(row["output_path"]).expanduser()
                if raw.is_absolute() and str(raw.resolve()) not in expected:
                    continue
                output = _resolve_manifest_file(row["output_path"], manifest=self.manifest,
                    line_number=line_number, field_name="output_path", real_data=args.real_data)
                if str(output) not in expected:
                    continue
                source = _resolve_manifest_file(row["source_path"], manifest=self.manifest,
                    line_number=line_number, field_name="source_path", real_data=args.real_data)
                if str(source) != expected[str(output)]:
                    raise ValueError("Local supervision must use the exact manifest source")
                self.rows[str(output)] = {**row, "manifest_line": line_number}
        self.real_data = args.real_data
        self.geometry = {}
        geometry_digest = None
        if args.pair_geometry_cache is not None:
            cache = args.pair_geometry_cache.expanduser().resolve()
            if not cache.is_dir():
                raise ValueError("pair geometry cache directory does not exist")
            digest = hashlib.sha256()
            for item in sorted(cache.glob("*.json")):
                contents = item.read_bytes()
                digest.update(item.name.encode() + b"\0" + contents)
                saved = json.loads(contents)
                identity = saved["identity"]
                key = self._geometry_key(identity)
                if key in self.geometry and self.geometry[key] != saved["evidence"]:
                    raise ValueError("Ambiguous geometry cache entries for the same image pair")
                self.geometry[key] = saved["evidence"]
            geometry_digest = digest.hexdigest()
        self.provenance = dict(version=1, confidence_mode=self.mode, positive_policy="stage2b_A_only",
            feature="backbone_local_features", correspondence="stage2b_radius1_tolerance1",
            pair_transform="full_frame_fixed_resize_rgb_bilinear_imagenet_v1",
            student_local_microbatch=1, reduction="sum_weighted_patches_over_entire_aux_batch",
            calibration_path=str(path), calibration_sha256=hashlib.sha256(calibration_bytes).hexdigest(),
            calibration_index=args.pair_calibration_index,
            geometry_cache=str(args.pair_geometry_cache.resolve()) if args.pair_geometry_cache else None,
            geometry_cache_sha256=geometry_digest, missing_geometry="teacher_structure_A_only",
            min_teacher_keep=self.min_keep,
            feature_weight=args.feature_distill_weight, score_weight=args.score_distill_weight,
            ot_weight=args.ot_distill_weight)
        if self.student_eval: self.provenance['student_local_mode']='eval_with_grad'

    @staticmethod
    def _geometry_key(identity):
        return (identity["source_rgb_sha256"], identity["generated_rgb_sha256"],
                tuple(identity["source_size"]), tuple(identity["generated_size"]))

    def _masks(self, row, generated):
        import numpy as np
        result = {}
        for name in ("edit_mask", "occlusion_mask", "destruction_mask"):
            if not row.get(name + "_path"):
                continue
            path = _resolve_manifest_file(row[name + "_path"], manifest=self.manifest,
                line_number=row["manifest_line"], field_name=name + "_path", real_data=self.real_data)
            if row.get(name + "_sha256") and file_sha256(path) != row[name + "_sha256"]:
                raise ValueError("Local distillation mask content changed")
            with Image.open(path) as image:
                if image.size != generated.size:
                    raise ValueError("Local masks must use generated native image coordinates")
                result[name] = torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.)
        return result

    @torch.no_grad()
    def prepare(self, auxiliary_batch):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prepared = []
        totals = dict(feature=0., score=0., ot=0.)
        if self.reliability_weight:
            totals.update(positive=0., negative=0.)
        stats = dict(pairs=0, patches=0, positive_a_patches=0, excluded_b_patches=0,
                     geometry_cache_hits=0, feature_patches=0, score_patches=0, ot_patches=0,
                     low_teacher_keep_patches=0)
        for i, (source_path, generated_path) in enumerate(zip(auxiliary_batch["source_paths"], auxiliary_batch["synthetic_paths"])):
            if not bool(auxiliary_batch["pair_valid"][i]):
                continue
            row = self.rows.get(str(Path(generated_path).resolve()))
            if row is None:
                raise ValueError("Auxiliary image missing its exact local manifest record")
            recorded_source = _resolve_manifest_file(row["source_path"], manifest=self.manifest,
                line_number=row["manifest_line"], field_name="source_path", real_data=self.real_data)
            if recorded_source != Path(source_path).resolve():
                raise ValueError("Auxiliary local source does not match manifest")
            for path, key in ((source_path, "source_sha256"), (generated_path, "output_sha256")):
                if row.get(key) and file_sha256(Path(path)) != row[key]:
                    raise ValueError("Local distillation image content differs from manifest")
            with Image.open(source_path) as handle:
                source = handle.convert("RGB")
            with Image.open(generated_path) as handle:
                generated = handle.convert("RGB")
            identity = dict(source_rgb_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                generated_rgb_sha256=hashlib.sha256(generated.tobytes()).hexdigest(),
                source_size=source.size, generated_size=generated.size)
            geometry = self.geometry.get(self._geometry_key(identity))
            stats["geometry_cache_hits"] += int(geometry is not None)
            inputs = torch.stack([preprocess_teacher_image(im, self.size) for im in (source, generated)])
            targets = []
            for _, descriptors, aux in self.teacher.iter_targets(inputs.to(self.device), self.chunk_size):
                for j in range(len(descriptors)):
                    targets.append({k: v[j:j+1].detach().cpu() for k, v in aux.items()})
                del descriptors, aux
            source_aux, generated_aux = targets
            correspondence = build_pair_correspondence(source, generated, source_aux, generated_aux,
                calibration=self.calibration, geometry=geometry, quality=row, **self._masks(row, generated))
            d = correspondence["match_diagnostics"]
            # B was only a candidate in Stage 2B, never silently promote it to
            # verified supervision. Both modes use exactly the same A support.
            confidence = correspondence["positive_conf"] * d["positive_a"]
            positive = confidence[None, None]
            negative = correspondence["negative_conf"][None, None]
            if self.reliability_weight:
                totals["positive"] += float(positive.sum())
                totals["negative"] += float(negative.sum())
            if self.mode == "uniform":
                confidence = (confidence > 0).float()
            index = correspondence["match_index"].unsqueeze(0)
            confidence = confidence.unsqueeze(0)
            weights = local_target_weights(source_aux, index, confidence, self.min_keep)
            for name, value in weights.items():
                totals[name] += float(value.sum())
                stats[name + "_patches"] += int((value > 0).sum())
            stats["low_teacher_keep_patches"] += int(((weights["score"] > 0) & (weights["ot"] == 0)).sum())
            stats["positive_a_patches"] += int((confidence > 0).sum())
            stats["excluded_b_patches"] += d["positive_b_count"]
            stats["patches"] += confidence.numel()
            stats["pairs"] += 1
            prepared.append(dict(image=inputs[1:2], teacher={k: source_aux[k] for k in
                ("backbone_local_features", "score_logits", "log_assignment")}, index=index, q=confidence,
                masses={k: float(v.sum()) for k, v in weights.items()}, positive=positive, negative=negative))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        stats["pair_prepare_seconds"] = time.perf_counter() - started
        return prepared, totals, stats

    def backward(self, model, prepared, totals, scaler, precision, weights, *, global_step=0):
        metrics = dict(feature=0., score=0., ot=0., feature_cosine=0., weighted_total=0.)
        eps = 1e-8
        for pair_number, pair in enumerate(prepared):
            if (not any(weights[k] > 0 and pair["masses"][k] > 0 for k in weights)
                    and not self.reliability_weight and not self.occlusion_weight):
                continue
            teacher = {k: v.to(self.device) for k, v in pair["teacher"].items()}
            index, q = pair["index"].to(self.device), pair["q"].to(self.device)
            dtype = torch.float16 if precision == "16" else torch.bfloat16
            if self.student_eval:
                from .real_retention import local_eval_with_grad
            context=local_eval_with_grad(model) if self.student_eval else nullcontext()
            with context:
                with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=precision != "32"):
                    descriptor, student = model(pair["image"].to(self.device), return_aux=True)
            losses = local_distillation_losses(student, teacher, index, q, eps=eps, min_teacher_keep=self.min_keep)
            # This rescales each microbatch mean into the exact combined-batch
            # weighted mean, including the final epsilon only once.
            pieces = {k: losses[k] * (losses[k + "_mass"] + eps) / (totals[k] + eps) for k in weights}
            total = sum(weights[k] * pieces[k] for k in weights)
            if self.reliability_weight or self.occlusion_weight:
                from .reliability import tristate_reliability_loss, reliability_region_statistics
                positive, negative = pair["positive"].to(self.device), pair["negative"].to(self.device)
                diagnostics = reliability_region_statistics(student["reliability_logits"], positive, negative,
                                                            student["assignment_dustbin"])
                for key, value in diagnostics.items():
                    metrics["generated_region_" + key] = metrics.get("generated_region_" + key, 0.) + value
                if self.reliability_weight:
                    reliability = tristate_reliability_loss(student["reliability_logits"], positive, negative)
                    classes = max(1, int(totals["positive"] > 0) + int(totals["negative"] > 0))
                    rel_loss = sum(reliability[name + "_numerator"] / (totals[name] + eps)
                                   for name in ("positive", "negative")) / classes
                    total = total + self.reliability_weight * rel_loss
                    metrics["generated_reliability_loss"] = metrics.get("generated_reliability_loss", 0.) + float(rel_loss.detach())
            if not torch.isfinite(total):
                raise RuntimeError("Nonfinite weighted local distillation loss")
            scaler.scale(total).backward()
            for k, value in pieces.items():
                metrics[k] += float(value.detach())
            metrics["feature_cosine"] += float(losses["feature_cosine"] * (losses["feature_mass"] + eps)
                                               / (totals["feature"] + eps))
            metrics["weighted_total"] += float(total.detach())
            del descriptor, student, losses, pieces, total, teacher, index, q
            if self.occlusion_weight:
                from .reliability_corruption import make_generated_occlusions, corrupted_patch_loss
                generator = torch.Generator(device=self.device).manual_seed(
                    (self.seed + 7_000_019 + global_step * 1_000_003 + pair_number) % (2 ** 63))
                altered, mask = make_generated_occlusions(pair["image"].to(self.device), generator=generator)
                training = model.training
                try:
                    model.eval()
                    with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=precision != "32"):
                        _, occluded = model(altered, return_aux=True)
                    # No outside pseudo-targets: unverified generated patches
                    # remain unknown, including on the controlled copy.
                    corruption = corrupted_patch_loss(occluded["reliability_logits"], mask)
                    occlusion_loss = corruption["corruption_loss"] / max(1, len(prepared))
                    weighted = self.occlusion_weight * occlusion_loss
                    if not torch.isfinite(weighted):
                        raise RuntimeError("Nonfinite controlled generated occlusion loss")
                    scaler.scale(weighted).backward()
                    metrics["generated_occlusion_loss"] = metrics.get("generated_occlusion_loss", 0.) + float(occlusion_loss.detach())
                    metrics["weighted_total"] += float(weighted.detach())
                    # Only already verified clean positives outside the inserted mask
                    # remain diagnostic positives; the rest are still unknown.
                    stats = reliability_region_statistics(occluded["reliability_logits"],
                        positive * ~mask, mask.float(), occluded["assignment_dustbin"])
                    for key, value in stats.items():
                        metrics["occluded_region_" + key] = metrics.get("occluded_region_" + key, 0.) + value
                finally:
                    model.train(training)
                del occluded, corruption, weighted, occlusion_loss, altered, mask
        return metrics
