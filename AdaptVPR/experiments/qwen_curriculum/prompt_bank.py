"""Weather prompt bank v3: several appearance regimes per domain.

Why v3. The v2 plan used one conservative template per domain ("dim street
lighting", "very light rain streaks"). Accepted images moved SALAD descriptors
about half as far as real SVOX weather for night and ~40% for rain (see
salad/probe_synthetic_difficulty.py), and every night image looked like the same
moonlit GSV scene. v3 therefore varies the *lighting regime and sensor response*
rather than simply darkening: light sources, glare, wet reflections, hard sun
shadows and high-ISO grain are what change appearance in real weather while the
facades that identify a place stay visible. Darkness without light sources only
removes place evidence, so no variant asks for an unlit scene.

Each planned job stores ``prompt_variant``; the difficulty probe groups by it, so
variants can be calibrated (kept, strengthened or dropped) from a small pilot.
Domain weights favour what the targets lack most and what the pilot showed to
move descriptors: night and wet night (SVOX night, Tokyo247 N, SF-XL night), then
rain and snow. Fog is kept small: no target benchmark has fog queries.
"""
from __future__ import annotations

import hashlib

POLICY_VERSION = "qwen_weather_bank_v3.1"

HEADER = ("Preserve the exact scene geometry, camera viewpoint, buildings, facades, windows, signs, "
          "road layout, lane markings, poles and parked vehicles. Change only lighting, weather and "
          "camera response.")
FOOTER = "Photorealistic street-level photograph. No new buildings, no removed structures, no warping."

VARIANTS = {
    "night": {
        "sodium_lamps": "Late night. Orange sodium-vapour street lamps cast warm pools of light on the road and "
                        "the lower facades; the space between lamps falls into deep shadow. Black sky. Bright lamp "
                        "heads with glare halos. Visible high-ISO grain in dark areas.",
        "led_lamps": "Late night under cold white LED street lights. Bluish-white light with hard falloff, very dark "
                     "sky, overexposed lamp heads with lens flare, strong contrast between lit and unlit surfaces, "
                     "sensor noise in the shadows.",
        "shopfronts": "Night in an urban street. Lit shop windows, illuminated signs and lit apartment windows are "
                      "the main light sources and reflect on the pavement; upper facades and sky are dark. "
                      "Slight bloom around bright signs.",
        "sparse_lamps": "Dark night with sparse lighting: only a few street lamps and some lit windows, large dark "
                        "areas, but facade edges, rooflines and the road remain faintly readable. Strong grain, "
                        "slightly soft focus like a long exposure.",
        "headlights": "Night with traffic. Car headlights and red tail lights with strong glare and starbursts, light "
                      "spill on the road surface and nearby walls, dark surroundings, slight motion blur on lights.",
        "blue_hour": "Blue hour shortly after sunset: deep blue sky, street lamps and windows already switched on, "
                     "low ambient light, cool shadows and warm artificial light on the facades.",
    },
    "rainy_night": {
        "wet_reflections": "Rainy night. Soaked black asphalt mirrors street lamps, signs and headlights as long "
                           "streaky reflections; puddles; rain streaks visible in the lamp light; glare halos.",
        "lens_drops": "Rainy night seen through a lens with scattered raindrops: blurred bright blobs around lights, "
                      "wet reflective road, dark sky, some areas slightly smeared by water.",
    },
    "rain": {
        "heavy_overcast": "Heavy rain on a dark overcast day. Dark grey sky, soaked darker asphalt with puddles and "
                          "mirror-like reflections, visible rain streaks, wet glossy facades and cars, reduced "
                          "contrast and haze in the distance, no shadows.",
        "lens_drops": "Rainy day photographed through a lens with water droplets: scattered droplets and slightly "
                      "smeared patches over parts of the image, grey diffuse light, wet reflective road.",
        "downpour_spray": "Downpour: dense rain streaks, splashing on the road, water spray from cars, flooded "
                          "gutters, dim grey light, low saturation.",
    },
    "sun": {
        "low_sun_glare": "Low sun near the horizon in front of the camera: strong glare and lens flare, overexposed "
                         "sky, long hard shadows across the road, warm high-contrast light.",
        "harsh_midday": "Harsh midday sun: very bright surfaces with partly blown-out highlights, deep black hard "
                        "shadows under trees, balconies and cars, saturated colours.",
        "backlit": "Backlit scene against a bright hazy sky: facades and trees in shadow and partly silhouetted, "
                   "veiling glare, low contrast in the shaded areas.",
    },
    "snow": {
        "light_cover": "Light snowy winter weather: overcast sky, visible snowflakes, light snow on horizontal "
                       "surfaces, wet dark road.",
        "heavy_cover": "Heavy snow cover: thick snow on the road edges, sidewalks, parked cars, roofs and ledges, "
                       "snow banks along the curb, white overcast sky, muted colours.",
        "snowfall": "Active snowfall reducing visibility: dense flakes in the air, grey sky, slushy tyre-tracked "
                    "road, snow collecting on surfaces.",
    },
    "overcast": {
        "flat_grey": "Flat overcast day: uniform grey sky, soft shadowless light, muted colours, slightly "
                     "underexposed image.",
    },
    "fog": {
        "dense_fog": "Dense fog: visibility drops quickly with distance, nearby facades and road stay readable, "
                     "distant buildings fade into grey, diffuse light.",
    },
}

# Calibrated on the 100-image pilot (salad/docs/weather_round.md): median SALAD
# descriptor drop night 0.090 (real SVOX night 0.088; v2 night 0.028), rainy
# night 0.070, snow 0.057, rain 0.041, sun 0.013, overcast ~0 (1/5 accepted).
# Overcast is kept for the verifier but no longer planned; sun stays small until
# its prompts produce a measurable shift.
DOMAIN_WEIGHTS = {"night": 0.44, "rainy_night": 0.14, "rain": 0.15, "sun": 0.08,
                  "snow": 0.14, "overcast": 0.0, "fog": 0.05}

# CLIP text per domain for the weather check. Each domain gets its own neutral
# text: "a clear sunny day" cannot be the opposite of "harsh sunlight".
CLIP_TEXT = {
    "night": "a street photo at night with artificial street lights",
    "rainy_night": "a street photo on a rainy night with wet reflective road",
    "rain": "a street photo in the rain with wet road",
    "sun": "a street photo in harsh direct sunlight with hard shadows and glare",
    "snow": "a street photo in snow",
    "overcast": "a street photo on an overcast cloudy day",
    "fog": "a street photo in thick fog",
}
CLIP_NEUTRAL = {domain: "a street photo on a clear sunny day" for domain in CLIP_TEXT}
CLIP_NEUTRAL["sun"] = "a street photo in soft even daylight"
LOW_LIGHT = frozenset({"night", "rainy_night"})

assert set(VARIANTS) == set(DOMAIN_WEIGHTS) == set(CLIP_TEXT)
assert abs(sum(DOMAIN_WEIGHTS.values()) - 1) < 1e-9


def prompt(domain: str, variant: str) -> str:
    return f"{HEADER}\n\n{VARIANTS[domain][variant]}\n\n{FOOTER}"


def choose_variant(domain: str, key: str) -> str:
    """Deterministic, uniform over a domain's variants for a stable job key."""
    names = sorted(VARIANTS[domain])
    digest = hashlib.sha256(f"{POLICY_VERSION}/{domain}/{key}".encode()).digest()
    return names[int.from_bytes(digest[:8], "big") % len(names)]


def definition() -> dict:
    return {"policy_version": POLICY_VERSION, "header": HEADER, "footer": FOOTER, "variants": VARIANTS,
            "domain_weights": DOMAIN_WEIGHTS, "clip_text": CLIP_TEXT, "clip_neutral": CLIP_NEUTRAL,
            "low_light_geometry": sorted(LOW_LIGHT)}
