"""Versioned, editable family footprint priors; never supplied by a VLM."""

import hashlib
import json
import math
from pathlib import Path

FAMILY_SHAPES = {
    "parked_vehicle": "horizontal_compact", "construction_barrier": "horizontal_compact",
    "traffic_cones": "cone_triangle", "temporary_sign": "vertical_rectangle",
    "vegetation": "compact_irregular_blob", "scaffolding": "facade_rectangle",
    "construction_tarp": "facade_rectangle",
}
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/targeted_family_constraints.json"


def finite(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid {name}: require a finite number in [{low},{high}]")


def validate_constraints(config):
    if config["schema_version"] != 1 or set(config["families"]) != set(FAMILY_SHAPES):
        raise ValueError("Require version 1 and the complete seven-family taxonomy")
    if config["weight_map"] not in ("raw_attention_map", "attention_map", "intervention_map", "fused_map"):
        raise ValueError("Unknown explicitly configured token weight map")
    if config["weighted_coverage_scope"] != "full_map":
        raise ValueError("Thresholded weighted coverage uses full-map mass; ROI-conditioned coverage is diagnostic only")
    finite(config["tau_target_precision"], "tau_target_precision", 0, 1)
    if config["weighted_coverage_thresholds"] != [.3, .5, .7]:
        raise ValueError("Dev must report all three weighted coverage thresholds [0.3,0.5,0.7]")
    search = config["search"]
    if not search["center_offsets"] or len(set(search["center_offsets"])) != len(search["center_offsets"]):
        raise ValueError("Require unique deterministic center offsets")
    for value in search["center_offsets"]:
        finite(value, "center offset", -1, 1)
    if type(search["roi_anchors"]) is not int or not 0 <= search["roi_anchors"] <= 32:
        raise ValueError("roi_anchors must be an integer in [0,32]")
    finite(search["anchor_separation_tokens"], "anchor separation", 0, 16)
    for family, spec in config["families"].items():
        if spec["shape"] != FAMILY_SHAPES[family]:
            raise ValueError(f"{family}: incompatible footprint shape")
        low, high = spec["area_range"]
        finite(low, "minimum area fraction", .001, .3)
        finite(high, "maximum area fraction", low, .3)
        if type(spec["area_samples"]) is not int or not 2 <= spec["area_samples"] <= 10:
            raise ValueError("area_samples must be an integer in [2,10]")
        if not spec["aspect_ratios"] or not spec["angle_offsets_degrees"]:
            raise ValueError("Require explicit aspect ratios and angles")
        for ratio in spec["aspect_ratios"]:
            finite(ratio, "aspect ratio (width/height)", .2, 6)
            if spec["shape"] == "horizontal_compact" and ratio <= 1:
                raise ValueError("Vehicle/barrier footprint must be horizontal")
            if spec["shape"] in ("vertical_rectangle", "cone_triangle") and ratio >= 1:
                raise ValueError("Sign/cone footprint must be vertical")
        for angle in spec["angle_offsets_degrees"]:
            finite(angle, "angle offset", -15, 15)
    render = config["render"]
    finite(render["radius_fraction_of_short_side"], "render radius fraction", 0, .05)
    finite(render["max_added_area_ratio"], "render added area / core area", 0, 1)
    finite(render["max_image_area_fraction"], "render total area fraction cap", .001, 1)
    if type(render["max_radius_pixels"]) is not int or not 0 <= render["max_radius_pixels"] <= 64:
        raise ValueError("max_radius_pixels must be an integer in [0,64]")
    return config


def load_constraints(path=DEFAULT_CONFIG):
    return validate_constraints(json.loads(Path(path).read_text(encoding="utf-8")))


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
