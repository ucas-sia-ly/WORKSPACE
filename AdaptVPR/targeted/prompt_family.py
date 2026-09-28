"""Versioned, deterministic prompts. No VLM text is interpolated into prompts."""

FAMILIES = (
    "parked_vehicle", "construction_barrier", "traffic_cones", "temporary_sign",
    "vegetation", "scaffolding", "construction_tarp", "none",
)
OBJECT_NAMES = dict(zip(FAMILIES, (
    "parked car", "construction barrier", "traffic cones", "temporary sign",
    "shrub", "scaffolding", "construction tarp", "none",
)))
REALISM_CONSTRAINTS = (
    "The occluder must be physically grounded or visibly secured to its support, "
    "with correct scale and correct perspective. Match lighting and match shadows "
    "to the source photograph. Preserve scene geometry. Do not modify outside the mask. "
    "Fit the complete visible occluder and its necessary contact shadow inside the fixed mask; "
    "do not move, expand, or reinterpret the edit region. Photorealistic local occlusion."
)
COMMON_NEGATIVE = (
    "floating object, incorrect perspective, oversized object, distorted geometry, "
    "warped building, unrealistic shadow, duplicate object, cartoon, painting, "
    "object intersecting walls, deformed vehicle, text artifacts"
)
TEMPLATES = {
    "parked_vehicle": "Place one ordinary parked car on the visible ground within the masked region.",
    "construction_barrier": "Place one temporary construction barrier resting on the visible ground within the masked region.",
    "traffic_cones": "Place a small group of traffic cones resting on the visible ground within the masked region.",
    "temporary_sign": "Place one plain temporary sign with a blank face, visibly supported, within the masked region.",
    "vegetation": "Add one compact natural shrub rooted in the visible soil or vegetation within the masked region.",
    "scaffolding": "Add a small section of realistic scaffolding at the building front, supported at its visible base, within the masked region.",
    "construction_tarp": "Add a construction tarp visibly secured to the building front within the masked region.",
    "none": "",
}
FAMILY_NEGATIVE = {
    "parked_vehicle": "moving vehicle, impossible wheels",
    "construction_barrier": "unsupported barrier, road reconstruction",
    "traffic_cones": "giant cone, excessive cones",
    "temporary_sign": "legible lettering, logos, unsupported sign",
    "vegetation": "floating roots, artificial foliage",
    "scaffolding": "unsupported scaffolding, impossible structural joints",
    "construction_tarp": "unattached tarp, floating fabric",
    "none": "unrequested edit",
}


def prompt_for_family(family: str) -> dict:
    if family not in FAMILIES:
        raise ValueError(f"Unknown occluder family: {family!r}")
    return {
        "template_id": f"targeted_occlusion_v1/{family}",
        "prompt": f"{TEMPLATES[family]} {REALISM_CONSTRAINTS}" if family != "none" else "",
        "negative_prompt": f"{COMMON_NEGATIVE}, {FAMILY_NEGATIVE[family]}",
    }
