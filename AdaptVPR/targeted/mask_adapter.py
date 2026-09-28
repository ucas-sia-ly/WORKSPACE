"""Deterministic compact-mask search at original resolution; Pillow + stdlib only."""

from __future__ import annotations

from array import array
import heapq
import math

from PIL import Image


def _binary_image(mask):
    if not isinstance(mask, Image.Image):
        # Pillow accepts the array interface without importing numpy here.
        try:
            mask = Image.fromarray(mask)
        except (TypeError, AttributeError) as exc:
            raise ValueError("vulnerability_mask must be a binary PIL image or bool/uint8 2D array") from exc
    if mask.mode not in ("1", "L"):
        raise ValueError("vulnerability_mask must be single-channel binary (1 or L)")
    image = mask.convert("L")
    values = {i for i, count in enumerate(image.histogram()) if count}
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError("vulnerability_mask must contain {0,1} or {0,255}; no thresholding")
    return image.point([0] + [1] * 255)


def _geometry(data, width, height):
    area = sx = sy = squared = 0
    for i, value in enumerate(data):
        if value:
            y, x = divmod(i, width)
            area += 1
            sx += x
            sy += y
            squared += x * x + y * y
    if not area:
        return 0, None, 0.0
    cx, cy = sx / area, sy / area
    # Include each unit pixel's continuous second moment (1/6 per pixel).
    moment = squared - (sx * sx + sy * sy) / area + area / 6
    compactness = min(1.0, area * area / (2 * math.pi * moment))
    return area, [cx + .5, cy + .5], compactness


def _component_centers(data, width, height):
    """8-connected component centroids, in deterministic row-major order."""
    visited = bytearray(len(data))
    centers = []
    for start, value in enumerate(data):
        if not value or visited[start]:
            continue
        visited[start] = 1
        stack = [start]
        n = sx = sy = 0
        while stack:
            index = stack.pop()
            y, x = divmod(index, width)
            n += 1
            sx += x
            sy += y
            for yy in range(max(0, y - 1), min(height, y + 2)):
                base = yy * width
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = base + xx
                    if data[neighbor] and not visited[neighbor]:
                        visited[neighbor] = 1
                        stack.append(neighbor)
        centers.append((n, sx / n + .5, sy / n + .5))
    return centers


def _row_prefix(data, width, height):
    prefixes = []
    for y in range(height):
        row = array("I", [0])
        total = 0
        for value in data[y * width:(y + 1) * width]:
            total += value
            row.append(total)
        prefixes.append(row)
    return prefixes


def _ellipse(width, height):
    """Rasterize a filled ellipse using pixel centers, no resizing or bbox fill."""
    spans = []
    image = Image.new("L", (width, height), 0)
    for y in range(height):
        normalized_y = (y + .5 - height / 2) / (height / 2)
        half = width / 2 * math.sqrt(max(0.0, 1 - normalized_y * normalized_y))
        left = max(0, math.ceil(width / 2 - half - .5))
        right = min(width, math.floor(width / 2 + half - .5) + 1)
        if left < right:
            spans.append((y, left, right))
            image.paste(1, (left, y, right, y + 1))
    area, centroid, compactness = _geometry(image.tobytes(), width, height)
    return spans, area, centroid, compactness


def adapt_generation_mask(
    vulnerability_mask,
    target_ratio=0.06,
    min_overlap=0.70,
    *,
    image_height=None,
    image_width=None,
    area_tolerance=0.05,
    aspect_ratios=(1.0, 0.75, 4 / 3, 0.5, 2.0),
    search_stride=None,
    refine_candidates=8,
    min_compactness_gain=0.0,
):
    """Return ``(generation_mask, diagnostics)``; failure returns ``(None, d)``.

    Output is a NEW PIL L image with exact values {0,1}. Ratio is relative to
    full image area, constrained to [0.04,0.08]. ``area_tolerance`` is relative
    to the requested area (default +/-5%), further clipped to the 4%-8% band.

    Search filled ellipses, using exact pixel overlap as local density, then
    proximity to the vulnerability centroid as tie-breaker. Coarse positions
    include image borders and major component centroids; best positions are
    refined at single-pixel precision. Failure is explicit, not a claim that
    all conceivable shapes/centers are infeasible. No random state is used.

    Compactness is A^2/(2*pi*J), with continuous pixel polar moment J about the
    centroid: translation/scale normalized, a disk is optimal (1). Require no
    compactness regression (or a configured positive gain); report both values.
    Existing nearly optimal masks may therefore match rather than strictly gain.
    """
    for name, value in (("target_ratio", target_ratio), ("min_overlap", min_overlap),
                        ("area_tolerance", area_tolerance), ("min_compactness_gain", min_compactness_gain)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not .04 <= target_ratio <= .08:
        raise ValueError("target_ratio must be between 0.04 and 0.08")
    if not 0 < min_overlap <= 1 or not 0 <= area_tolerance <= .25 or not 0 <= min_compactness_gain <= 1:
        raise ValueError("invalid overlap/tolerance/compactness constraint")
    if type(refine_candidates) is not int or refine_candidates < 1:
        raise ValueError("refine_candidates must be positive")
    if not aspect_ratios or any(isinstance(r, bool) or not isinstance(r, (int, float)) or
                               not math.isfinite(r) or not .5 <= r <= 2 for r in aspect_ratios):
        raise ValueError("aspect_ratios must be nonempty and within [0.5,2]")
    image = _binary_image(vulnerability_mask)
    width, height = image.size
    if (image_height is None) != (image_width is None):
        raise ValueError("Specify both image_height and image_width")
    if image_height is not None:
        if type(image_height) is not int or type(image_width) is not int or (image_width, image_height) != image.size:
            raise ValueError("image H/W must match vulnerability mask; implicit resize is forbidden")
    if not width or not height:
        raise ValueError("image dimensions must be positive")
    if search_stride is None:
        search_stride = max(1, min(width, height) // 32)
    if type(search_stride) is not int or search_stride < 1:
        raise ValueError("search_stride must be positive")
    data = image.tobytes()
    vuln_area, centroid, compact_before = _geometry(data, width, height)
    components = _component_centers(data, width, height)
    total_pixels = width * height
    target_area = target_ratio * total_pixels
    minimum = max(1, math.ceil(max(.04 * total_pixels, target_area * (1 - area_tolerance))))
    maximum = math.floor(min(.08 * total_pixels, target_area * (1 + area_tolerance)))
    diagnostics = dict(
        schema_version=1, status="failure", failure_reason=None,
        vulnerability_area=vuln_area, generation_area=0, overlap_pixels=0, overlap_ratio=0.0,
        centroid_vulnerability=centroid, centroid_generation=None, centroid_distance_normalized=None,
        num_components_before=len(components), num_components_after=0, connectivity=8,
        vulnerability_area_ratio=vuln_area / total_pixels, generation_area_ratio=0.0,
        target_ratio=target_ratio, target_area=target_area, area_tolerance=area_tolerance,
        allowed_generation_area=[minimum, maximum], min_overlap=min_overlap,
        compactness_before=compact_before, compactness_after=None,
        compactness_metric="A^2/(2*pi*polar_second_moment); unit-pixel intrinsic moment included",
        min_compactness_gain=min_compactness_gain, image_height=height, image_width=width,
        centroid_coordinate_convention="[x,y], pixel centers at (column+0.5,row+0.5)",
        method="filled_ellipse_density_search_v1", aspect_ratios=list(aspect_ratios),
        search_stride=search_stride, refine_candidates=refine_candidates, candidates_evaluated=0,
        morphological_smoothing="none: analytic filled ellipse boundary; no dilation or resizing",
    )

    def failure(reason):
        diagnostics["failure_reason"] = reason
        return None, diagnostics

    if not vuln_area:
        return failure("empty_vulnerability_mask")
    if maximum < minimum:
        return failure("no_integer_area_in_requested_tolerance")
    if vuln_area < math.ceil(min_overlap * minimum):
        return failure("insufficient_vulnerability_area_for_required_overlap")
    prefixes = _row_prefix(data, width, height)
    templates = []
    for ratio in sorted(set(aspect_ratios)):
        ideal_width = math.sqrt(4 * target_area * ratio / math.pi)
        ideal_height = math.sqrt(4 * target_area / (ratio * math.pi))
        choices = []
        for w in range(max(1, round(ideal_width) - 2), min(width, round(ideal_width) + 2) + 1):
            for h in range(max(1, round(ideal_height) - 2), min(height, round(ideal_height) + 2) + 1):
                if not .5 <= w / h <= 2:
                    continue
                spans, area, center, compactness = _ellipse(w, h)
                if (minimum <= area <= maximum and
                        compactness + 1e-12 >= compact_before + min_compactness_gain):
                    choices.append((abs(area - target_area), abs(w / h - ratio), w, h, spans, area, center, compactness))
        if choices:
            templates.append(min(choices, key=lambda t: t[:4]))
    if not templates:
        return failure("no_template_meets_area_size_and_compactness_constraints")
    best = None
    best_any_overlap = 0.0
    for _, _, tw, th, spans, area, local_center, compactness in templates:
        xmax, ymax = width - tw, height - th
        xs = sorted(set(range(0, xmax + 1, search_stride)) | {xmax})
        ys = sorted(set(range(0, ymax + 1, search_stride)) | {ymax})
        seen = set()
        ranked = []

        def evaluate(x, y):
            nonlocal best, best_any_overlap
            if (x, y) in seen:
                return
            seen.add((x, y))
            overlap = sum(prefixes[y + dy][x + right] - prefixes[y + dy][x + left] for dy, left, right in spans)
            overlap_ratio = overlap / area
            gx, gy = x + local_center[0], y + local_center[1]
            distance2 = (gx - centroid[0])**2 + (gy - centroid[1])**2
            rank = (-overlap_ratio, distance2, abs(area - target_area), -compactness, y, x)
            ranked.append((rank, x, y))
            best_any_overlap = max(best_any_overlap, overlap_ratio)
            if overlap_ratio >= min_overlap and (best is None or rank < best[0]):
                best = (rank, x, y, spans, area, overlap, gx, gy, compactness, tw, th)

        for y in ys:
            for x in xs:
                evaluate(x, y)
        centers = [(vuln_area, *centroid)] + sorted(components, reverse=True)[:32]
        for _, cx, cy in centers:
            evaluate(max(0, min(xmax, round(cx - local_center[0]))),
                     max(0, min(ymax, round(cy - local_center[1]))))
        # Fixed refinement neighborhood; ranking is deterministic, no RNG state.
        for _, cx, cy in heapq.nsmallest(refine_candidates, ranked):
            for y in range(max(0, cy - search_stride), min(ymax, cy + search_stride) + 1):
                for x in range(max(0, cx - search_stride), min(xmax, cx + search_stride) + 1):
                    evaluate(x, y)
        diagnostics["candidates_evaluated"] += len(seen)
    diagnostics["best_candidate_overlap_ratio"] = best_any_overlap
    if best is None:
        return failure("no_compact_candidate_meets_overlap_in_configured_search")
    _, x, y, spans, area, overlap, gx, gy, compactness, tw, th = best
    generation = Image.new("L", image.size, 0)
    for dy, left, right in spans:
        generation.paste(1, (x + left, y + dy, x + right, y + dy + 1))
    after = len(_component_centers(generation.tobytes(), width, height))
    if after != 1:
        return failure("internal_candidate_not_connected")
    diagnostics.update(
        status="success", failure_reason=None, generation_area=area,
        generation_area_ratio=area / total_pixels, overlap_pixels=overlap,
        overlap_ratio=overlap / area, centroid_generation=[gx, gy],
        centroid_distance_normalized=math.hypot(gx - centroid[0], gy - centroid[1]) / math.hypot(width, height),
        num_components_after=after, compactness_after=compactness,
        compactness_gain=compactness - compact_before,
        ellipse_bounds_xywh=[x, y, tw, th],
        target_area_relative_error=abs(area - target_area) / target_area,
    )
    return generation, diagnostics
