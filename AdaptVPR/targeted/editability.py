"""Strict VLM response contract and conservative spatial editability policy."""

from dataclasses import asdict, dataclass, replace
import json
import math

from .prompt_family import FAMILIES, OBJECT_NAMES

REGION_TYPES = (
    "road", "sidewalk", "parking_area", "grass", "vegetation", "building_front",
    "wall_or_fence", "sky", "unknown",
)
SUPPORT_SURFACES = (
    "paved_ground", "soil_ground", "building_base", "facade_attachment", "none", "unknown",
)
DECISION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["editable", "region_type", "support_surface", "occluder_family", "object_name", "confidence", "reason"],
    "properties": {
        "editable": {"type": "boolean"},
        "region_type": {"type": "string", "enum": list(REGION_TYPES)},
        "support_surface": {"type": "string", "enum": list(SUPPORT_SURFACES)},
        "occluder_family": {"type": "string", "enum": list(FAMILIES)},
        "object_name": {"type": "string", "enum": list(OBJECT_NAMES.values())},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
}
ALLOWED_REGIONS = {
    "parked_vehicle": {"road", "parking_area"},
    "construction_barrier": {"road", "sidewalk"},
    "traffic_cones": {"road", "sidewalk"},
    "temporary_sign": {"road", "sidewalk", "parking_area", "grass", "wall_or_fence"},
    "vegetation": {"grass", "vegetation"},
    "scaffolding": {"building_front"},
    "construction_tarp": {"building_front"},
    "none": set(),
}
ALLOWED_SUPPORTS = {
    "parked_vehicle": {"paved_ground"},
    "construction_barrier": {"paved_ground"},
    "traffic_cones": {"paved_ground"},
    "temporary_sign": {"paved_ground", "soil_ground", "facade_attachment"},
    "vegetation": {"soil_ground"},
    "scaffolding": {"building_base"},
    "construction_tarp": {"building_base", "facade_attachment"},
    "none": set(),
}
REGION_SUPPORTS = {
    "road": {"paved_ground"}, "sidewalk": {"paved_ground"},
    "parking_area": {"paved_ground"}, "grass": {"soil_ground"},
    "vegetation": {"soil_ground"},
    "building_front": {"building_base", "facade_attachment"},
    "wall_or_fence": {"facade_attachment"}, "sky": set(), "unknown": set(),
}


class DecisionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EditabilityDecision:
    editable: bool
    region_type: str
    support_surface: str
    occluder_family: str
    object_name: str
    confidence: float
    reason: str

    def to_dict(self):
        return asdict(self)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DecisionValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_decision(raw: str) -> EditabilityDecision:
    """Reject prose, code fences, extra fields, coercions, duplicates and non-finite values."""
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object)
    except (TypeError, ValueError) as exc:
        raise DecisionValidationError(f"Expected one strict JSON object: {exc}") from exc
    if type(data) is not dict:
        raise DecisionValidationError("Decision must be one JSON object")
    required = set(DECISION_SCHEMA["required"])
    if set(data) != required:
        raise DecisionValidationError(f"Exactly seven fields required; missing={sorted(required - set(data))}, extra={sorted(set(data) - required)}; no coordinates or prompts")
    if type(data["editable"]) is not bool:
        raise DecisionValidationError("editable must be a JSON boolean")
    for key, choices in (("region_type", REGION_TYPES), ("support_surface", SUPPORT_SURFACES),
                         ("occluder_family", FAMILIES), ("object_name", tuple(OBJECT_NAMES.values()))):
        if type(data[key]) is not str or data[key] not in choices:
            raise DecisionValidationError(f"Invalid {key}")
    confidence = data["confidence"]
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise DecisionValidationError("confidence must be a finite number in [0,1]")
    if type(data["reason"]) is not str or not 1 <= len(data["reason"].strip()) <= 1200:
        raise DecisionValidationError("reason must be nonempty and at most 1200 characters")
    if data["object_name"] != OBJECT_NAMES[data["occluder_family"]]:
        raise DecisionValidationError("object_name must equal the canonical family object name")
    if data["editable"] != (data["occluder_family"] != "none"):
        raise DecisionValidationError("Rejected decisions must use family/object none; editable decisions require an object")
    return EditabilityDecision(**data)


def apply_hard_rules(decision: EditabilityDecision, min_confidence: float = .70):
    """Rules may only reject; never choose a different object, region, or mask."""
    if type(min_confidence) not in (float, int) or not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0,1]")
    # Revalidate even programmatically constructed dataclasses.
    parse_decision(json.dumps(decision.to_dict(), allow_nan=False))
    reasons = []
    if not decision.editable:
        reasons.append("vlm_rejected")
    if decision.region_type == "sky":
        reasons.append("sky_not_editable")
    if decision.confidence < min_confidence:
        reasons.append("low_confidence")
    if decision.region_type == "unknown":
        reasons.append("unknown_region_not_supported")
    if decision.support_surface in ("none", "unknown"):
        reasons.append("no_visible_support")
    if decision.editable:
        if decision.region_type not in ALLOWED_REGIONS[decision.occluder_family]:
            reasons.append("family_region_incompatible")
        if (decision.support_surface not in ALLOWED_SUPPORTS[decision.occluder_family]
                or decision.support_surface not in REGION_SUPPORTS[decision.region_type]):
            reasons.append("support_incompatible")
    if reasons:
        return replace(decision, editable=False, occluder_family="none", object_name="none"), reasons
    return decision, []
