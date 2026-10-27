"""Opt-in, full-scene night relighting for same-view VPR editing.

V4 and historical generations remain frozen. These instructions are a candidate
policy, not evidence that an edit is faithful or photographically realistic.
Night appearance and content preservation require separate automated checks.
"""
from __future__ import annotations

import copy
import hashlib

from experiments.qwen_curriculum import prompt_bank as legacy
from experiments.qwen_curriculum import prompt_bank_v4


POLICY_VERSION = "qwen_weather_bank_v5.0"

HEADER = (
    "Relight this exact source photograph, preserving its viewpoint and physical scene. "
    "Keep buildings, windows, signs, lettering, road markings, vegetation and every object in the same "
    "positions and shapes. Do not add, remove or rewrite structures, shops, text, lamps, vehicles or people. "
    "Only light fixtures, windows and vehicle lights already present in the source may emit light."
)

NIGHT_RELIGHTING = (
    "Make a photograph taken after full nightfall, including when little or no sky is visible. "
    "Relight walls, facades, roads, pavements and vegetation throughout the image. Remove daytime "
    "illumination, direct sunlight, sunlit highlights and solar cast shadows; retain physical edges and "
    "surface markings. A dark sky alone is insufficient. Keep any visible sky dark, not dusk or blue hour. "
    "Light falls locally from existing sources; unlit areas receive only weak night ambient light. "
    "If no light source is present, use weak diffuse night ambient light without adding a source or "
    "preserving daylight exposure. Reduced detail visibility in unlit areas is allowed; do not remove objects. "
    "Dry road and dry pavement, with no wet gloss, puddles or rain reflections."
)

FOOTER = (
    "Use restrained, natural colours and a plausible night exposure. "
    "No global colour wash, decorative lighting or uniformly bright facades."
)

# Other weather domains retain the v4 instructions, including their own header
# and footer via prompt(). This revision changes strict night only.
VARIANTS = copy.deepcopy(prompt_bank_v4.VARIANTS)
VARIANTS["night"] = {
    "led_lamps": (
        "Existing light fixtures may cast localized neutral or cool-white light on nearby surfaces. "
        "Surfaces outside their reach remain dim; do not spread the lamp illumination across the scene."
    ),
    "sodium_lamps": (
        "Existing outdoor lamps may cast localized warm amber light. "
        "Keep surfaces beyond those light pools subdued and neutral, without an orange tint over the whole image."
    ),
    "sparse_lamps": (
        "Use sparse, low illumination from existing fixtures or windows, with dark intervals between lit areas. "
        "Where no source is present, let dim ambient light reveal only limited surface detail."
    ),
    "existing_headlights": (
        "Only vehicles already present may have their existing headlights or tail lights lit. "
        "Their spill stays near the corresponding vehicle. If no vehicle is present, use the existing "
        "scene's lighting or weak night ambient light; do not add traffic or beams."
    ),
}

# V4's four non-blue-hour variants had equal weights. Keep that allocation,
# renormalized after removing blue hour, rather than infer untested new weights.
# No variant relies on adding a light source: all use the no-source fallback.
VARIANT_WEIGHTS = copy.deepcopy(prompt_bank_v4.VARIANT_WEIGHTS)
VARIANT_WEIGHTS["night"] = {
    "led_lamps": 1,
    "sodium_lamps": 1,
    "sparse_lamps": 1,
    "existing_headlights": 1,
}
DOMAIN_WEIGHTS = dict(prompt_bank_v4.DOMAIN_WEIGHTS)
CLIP_TEXT = dict(prompt_bank_v4.CLIP_TEXT)
CLIP_NEUTRAL = dict(prompt_bank_v4.CLIP_NEUTRAL)
LOW_LIGHT = prompt_bank_v4.LOW_LIGHT


def prompt(domain: str, variant: str) -> str:
    """Keep hard content/lighting constraints in the positive prompt itself.

    Some deployed editing configurations ignore a negative-prompt channel. The
    policy therefore does not rely on one for any essential constraint.
    """
    variant_text = VARIANTS[domain][variant]
    if domain == "night":
        return f"{HEADER}\n\n{NIGHT_RELIGHTING}\n\n{variant_text}\n\n{FOOTER}"
    return prompt_bank_v4.prompt(domain, variant)


def choose_variant(domain: str, key: str) -> str:
    """Deterministic weighted draw without touching caller RNG state."""
    weights = VARIANT_WEIGHTS[domain]
    digest = hashlib.sha256(f"{POLICY_VERSION}/{domain}/{key}".encode()).digest()
    point = int.from_bytes(digest, "big") * sum(weights.values()) // (1 << 256)
    for name, weight in sorted(weights.items()):
        if point < weight:
            return name
        point -= weight
    raise AssertionError("Invalid variant weights")


def matches_record(row: dict) -> bool:
    domain, variant = row.get("condition"), row.get("prompt_variant")
    return (row.get("prompt_policy") == POLICY_VERSION and domain in VARIANTS
            and variant in VARIANTS[domain] and row.get("prompt") == prompt(domain, variant))


def definition() -> dict:
    return copy.deepcopy({
        "policy_version": POLICY_VERSION,
        "header": HEADER,
        "night_relighting": NIGHT_RELIGHTING,
        "footer": FOOTER,
        "variants": VARIANTS,
        "variant_weights": VARIANT_WEIGHTS,
        "variant_sampling": "sha256 stateless weighted draw; equal non-blue-hour v4 weights renormalized",
        "domain_weights": DOMAIN_WEIGHTS,
        "clip_text": CLIP_TEXT,
        "clip_neutral": CLIP_NEUTRAL,
        "low_light_geometry": sorted(LOW_LIGHT),
        "scope": "strict night revised; non-night prompts inherited verbatim from v4",
        "inherited_non_night_policy": prompt_bank_v4.definition(),
        "content_constraints": {
            "new_objects": False,
            "removed_objects": False,
            "new_light_sources": False,
            "rewritten_text": False,
            "night_road": "dry",
            "no_visible_source": "weak night ambient light; reduced visibility allowed; no invented lights",
            "low_sky": "relight walls, road, pavement and vegetation regardless of visible sky",
            "blue_hour": "excluded from strict night",
            "sensor_defects": "separate calibrated train-time augmentation",
            "negative_prompt_required": False,
        },
        "legacy_reuse": "no implicit v3/v4 reuse or relabelling",
    })
