"""Opt-in content-conservative prompts; the frozen v3.1 bank is unchanged.

These are proposed editing instructions, not a guarantee of faithful content or
camera realism. Sensor artefacts belong in a separately calibrated augmentation
policy. Existing v3 images must be re-audited separately, never relabelled v4.
"""
from __future__ import annotations

import copy
import hashlib

from experiments.qwen_curriculum import prompt_bank as legacy

POLICY_VERSION = "qwen_weather_bank_v4.0"

HEADER = (
    "Edit the weather and illumination of this exact source photograph. Preserve the camera viewpoint, "
    "buildings, facades, windows, road layout, lane markings, vegetation, poles and every existing object. "
    "Do not add or remove shops, windows, signs, text, neon, lamps, vehicles, people or structures. "
    "Preserve the shapes and lettering of existing signs; do not invent or rewrite text. "
    "Only lamps, windows, signs and vehicles already visible in the source may emit light. "
    "If a requested light source is absent, do not create one; use subtle ambient illumination instead."
)
FOOTER = (
    "Photorealistic street-level photograph with muted, natural colours. Preserve the location's identity. "
    "No new objects, no removed structures, no warping, no decorative lighting."
)

VARIANTS = copy.deepcopy(legacy.VARIANTS)
VARIANTS["night"] = {
    "sodium_lamps": (
        "Late night with a dark sky. If street lamps are already present, give only their nearby surfaces "
        "localized warm sodium illumination. Keep neutral colours outside those pools of light; no global "
        "orange tint. Keep facade edges and road layout readable. Dry road and dry pavement: no puddles, "
        "wet gloss or rain reflections."
    ),
    "led_lamps": (
        "Late night with a dark sky and restrained cool-white illumination from existing light fixtures. "
        "Keep neutral, muted colours and readable building contours. Dry road and dry pavement: no "
        "puddles, wet gloss or rain reflections."
    ),
    "sparse_lamps": (
        "Night with sparse illumination from only the fixtures and windows already visible in the source. "
        "Use low ambient light between lit areas; keep facade boundaries and road layout faintly readable. "
        "Dry road and dry pavement: no puddles, wet gloss or rain reflections."
    ),
    "existing_headlights": (
        "Night. Only if vehicles are already visible, their existing headlights or tail lights may be lit, "
        "with localized spill on nearby surfaces. Preserve each vehicle's exact position and shape. If no "
        "vehicle is present, retain ambient night illumination without adding traffic. Dry road and dry "
        "pavement: no puddles, wet gloss or rain reflections."
    ),
    "blue_hour": (
        "Blue hour shortly after sunset with a deep blue sky and low ambient light. Only existing lamps "
        "and windows may be lit. Keep facade colours natural and contours readable. Dry road and dry "
        "pavement: no puddles, wet gloss or rain reflections."
    ),
}
# Wet appearance is explicit only in wet-weather domains. No prompt asks the
# editor to synthesise grain, motion blur, compression or sensor defects.
VARIANTS["rainy_night"] = {
    "wet_reflections": (
        "Rainy night with a dark sky and wet asphalt. Existing lights may reflect on the wet road; "
        "reflections must originate only from light sources already present. Preserve every object and "
        "use restrained colours, visible road contours and building boundaries."
    ),
    "lens_drops": (
        "Rainy night with wet road and a few translucent lens droplets. Existing lights may reflect "
        "and bloom locally; do not invent lights or vehicles. Keep most place-defining structure visible."
    ),
}
VARIANTS["rain"]["downpour_spray"] = (
    "Downpour with rain streaks, dim grey diffuse light, wet road and water collecting along existing "
    "gutters. Keep buildings and road boundaries recognizable. Do not add traffic or objects."
)
VARIANTS["sun"]["harsh_midday"] = (
    "Harsh midday sun with bright surfaces and hard shadows cast by the existing trees, balconies "
    "and vehicles. Preserve natural colours and all source objects."
)

# Integer weights make the proposed 10% blue-hour allocation explicit and
# reproducible. This is a starting design, not a test-set-calibrated optimum.
VARIANT_WEIGHTS = {
    domain: {name: 1 for name in variants} for domain, variants in VARIANTS.items()
}
VARIANT_WEIGHTS["night"] = {
    "sodium_lamps": 225, "led_lamps": 225, "sparse_lamps": 225,
    "existing_headlights": 225, "blue_hour": 100,
}
DOMAIN_WEIGHTS = dict(legacy.DOMAIN_WEIGHTS)
CLIP_TEXT = dict(legacy.CLIP_TEXT)
CLIP_NEUTRAL = dict(legacy.CLIP_NEUTRAL)
LOW_LIGHT = legacy.LOW_LIGHT


def prompt(domain: str, variant: str) -> str:
    return f"{HEADER}\n\n{VARIANTS[domain][variant]}\n\n{FOOTER}"


def choose_variant(domain: str, key: str) -> str:
    """Stateless weighted draw; unrelated jobs cannot change this job's variant."""
    weights = VARIANT_WEIGHTS[domain]
    digest = hashlib.sha256(f"{POLICY_VERSION}/{domain}/{key}".encode()).digest()
    # Avoid modulo bias without requiring an RNG or mutable queue.
    total = sum(weights.values())
    point = int.from_bytes(digest, "big") * total // (1 << 256)
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
        "policy_version": POLICY_VERSION, "header": HEADER, "footer": FOOTER,
        "variants": VARIANTS, "variant_weights": VARIANT_WEIGHTS,
        "variant_sampling": "sha256 stateless weighted draw; night blue_hour weight=100/1000",
        "domain_weights": DOMAIN_WEIGHTS, "clip_text": CLIP_TEXT,
        "clip_neutral": CLIP_NEUTRAL, "low_light_geometry": sorted(LOW_LIGHT),
        "content_constraints": {"new_shopfronts": False, "new_light_sources": False,
            "night_road": "dry", "sensor_defects": "separate calibrated train-time augmentation"},
        "legacy_reuse": "no implicit v3 reuse or relabelling",
    })
