"""Opt-in camera-response simulations for generated night training images.

These are bounded appearance perturbations, not a physical camera calibration.
In particular, gain/gamma cannot recover structure already clipped to black and
bloom can only spread highlights present in the input. No spatial coordinates
are changed. A local NumPy Generator makes draws independent of sampling,
DataLoader workers and Torch/CUDA model RNGs.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


_RANGES = {
    "gain_gamma": {"gain": (0., 4., False), "gamma": (.3, 3., False)},
    "bloom": {"threshold": (0., 1., False), "strength": (0., 1., False),
              "radius": (0., 32., False)},
    "motion_blur": {"length": (1, 31, True), "angle": (-180., 180., False)},
    "noise": {"shot_peak": (1., 1e6, False), "read_std": (0., .2, False)},
    "jpeg": {"quality": (1, 100, True)},
    "saturation": {"factor": (0., 1., False)},
    "vignette": {"strength": (0., 1., False)},
}


def camera_seed(*parts) -> int:
    """Private deterministic namespace; never consume any global RNG."""
    encoded = json.dumps(["synthetic_camera_v1", *parts], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _number(value, where):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number")
    return value


def _probability(value, where):
    if not 0 <= _number(value, where) <= 1:
        raise ValueError(f"{where} must be in [0,1]")


def validate_profile(profile):
    if not isinstance(profile, dict):
        raise ValueError("Camera profile must be a JSON object")
    unknown = set(profile) - {"version", "status", "conditions", "probability", "operations", "provenance"}
    if unknown:
        raise ValueError(f"Unknown camera profile fields: {sorted(unknown)}")
    if type(profile.get("version")) is not int or profile["version"] != 1:
        raise ValueError("Camera profile version must be 1")
    if not isinstance(profile.get("status"), str) or profile["status"] not in {"exploratory", "calibrated"}:
        raise ValueError("Camera profile status must be exploratory or calibrated")
    conditions = profile.get("conditions")
    if (not isinstance(conditions, list) or not conditions or
            any(not isinstance(c, str) or c not in {"night", "rainy_night"} for c in conditions)
            or len(set(conditions)) != len(conditions)):
        raise ValueError("Camera profile conditions must be unique night/rainy_night entries")
    _probability(profile.get("probability", 1.), "Camera profile probability")
    operations = profile.get("operations")
    if not isinstance(operations, dict) or not operations:
        raise ValueError("Camera profile operations must be a nonempty object")
    for name, operation in operations.items():
        if name not in _RANGES or not isinstance(operation, dict):
            raise ValueError(f"Unknown camera operation: {name}")
        parameters = _RANGES[name]
        if set(operation) - set(parameters) - {"probability"} or set(parameters) - set(operation):
            raise ValueError(f"Camera {name} must contain ranges {sorted(parameters)} and optional probability")
        _probability(operation.get("probability", 1.), f"Camera {name} probability")
        for parameter, (lo, hi, integer) in parameters.items():
            values = operation[parameter]
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError(f"Camera {name}.{parameter} must be a two-element range")
            for value in values:
                _number(value, f"Camera {name}.{parameter}")
                if integer and type(value) is not int:
                    raise ValueError(f"Camera {name}.{parameter} must contain integers")
            if not lo <= values[0] <= values[1] <= hi or (parameter == "gain" and values[0] == 0):
                raise ValueError(f"Camera {name}.{parameter} lies outside safe bounds ({lo},{hi})")
    provenance = profile.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Camera profile provenance must be an object")
    if profile["status"] == "calibrated":
        for field in ("calibration_manifest_sha256", "evaluation_exclusion_sha256"):
            if not isinstance(provenance.get(field), str) or not re.fullmatch("[a-fA-F0-9]{64}", provenance[field]):
                raise ValueError(f"A calibrated profile requires provenance.{field}")
    # Reject NaNs/unsupported metadata rather than write a nonportable checkpoint.
    try:
        json.dumps(profile, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Camera profile must contain finite JSON metadata") from exc


def _srgb_to_linear(x):
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    return np.where(x <= .0031308, x * 12.92, 1.055 * np.maximum(x, 0) ** (1 / 2.4) - .055)


def _pil(x):
    return Image.fromarray(np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8), "RGB")


def _directional_blur(x, length, angle):
    if length <= 1:
        return x
    # Integer offsets approximate a straight exposure trajectory. Reflecting
    # boundaries preserve frame size without wrapping opposite image borders.
    theta = math.radians(angle)
    trajectory = np.linspace(-(length - 1) / 2, (length - 1) / 2, length)
    offsets = set((round(float(t * math.sin(theta))), round(float(t * math.cos(theta)))) for t in trajectory)
    pad = length // 2 + 1
    extended = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    result = np.zeros_like(x)
    h, w = x.shape[:2]
    for dy, dx in offsets:
        result += extended[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
    return result / len(offsets)


class CameraAugmentation:
    def __init__(self, profile, *, source=None, sha256=None):
        validate_profile(profile)
        self.profile = copy.deepcopy(profile)
        self.source = str(source) if source is not None else None
        self.sha256 = sha256 or hashlib.sha256(json.dumps(profile, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    @classmethod
    def from_json(cls, path):
        path = Path(path).expanduser().resolve()
        contents = path.read_bytes()
        return cls(json.loads(contents), source=path, sha256=hashlib.sha256(contents).hexdigest())

    @property
    def provenance(self):
        return dict(version=1, profile_path=self.source, profile_sha256=self.sha256,
                    profile=copy.deepcopy(self.profile), rng="sha256_keyed_local_numpy_generator_v1",
                    order="after_resize_and_standard_augmentation_before_normalize",
                    input="synthetic_night_only_teacher_and_real_inputs_unchanged",
                    geometry="no_coordinate_warp", limitations="cannot_recover_clipped_structure_or_guarantee_domain_realism")

    def apply(self, image, *, seed: int, condition: str):
        if condition not in self.profile["conditions"] or self.profile.get("probability", 1.) == 0:
            return image
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Camera seed must be a nonnegative integer")
        generator = np.random.default_rng(seed)
        if generator.random() >= self.profile.get("probability", 1.):
            return image
        x = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.

        def enabled(name):
            op = self.profile["operations"].get(name)
            return op if op is not None and generator.random() < op.get("probability", 1.) else None

        def draw(op, name, *, integer=False):
            low, high = op[name]
            return int(generator.integers(low, high + 1)) if integer else float(generator.uniform(low, high))

        if (op := enabled("gain_gamma")) is not None:
            # Deliberate response simulation in normalized display values;
            # neither scene radiance nor the camera auto-exposure loop is known.
            x = np.clip(x * draw(op, "gain"), 0, 1) ** draw(op, "gamma")
        if (op := enabled("saturation")) is not None:
            x = np.asarray(ImageEnhance.Color(_pil(x)).enhance(draw(op, "factor")), dtype=np.float32) / 255.
        if (op := enabled("bloom")) is not None:
            threshold = draw(op, "threshold")
            highlight = np.maximum(x - threshold, 0) / max(1e-8, 1 - threshold)
            glow = np.asarray(_pil(highlight).filter(ImageFilter.GaussianBlur(draw(op, "radius"))),
                              dtype=np.float32) / 255.
            x = np.clip(x + draw(op, "strength") * glow, 0, 1)
        if (op := enabled("motion_blur")) is not None:
            x = _directional_blur(x, draw(op, "length", integer=True), draw(op, "angle"))
        if (op := enabled("vignette")) is not None:
            yy, xx = np.meshgrid(np.linspace(-1., 1., x.shape[0]), np.linspace(-1., 1., x.shape[1]), indexing="ij")
            x = x * (1 - draw(op, "strength") * (xx ** 2 + yy ** 2) / 2)[..., None]
        if (op := enabled("noise")) is not None:
            peak = draw(op, "shot_peak")
            linear = _srgb_to_linear(np.clip(x, 0, 1))
            noisy = generator.poisson(linear * peak) / peak
            noisy += generator.normal(0., draw(op, "read_std"), x.shape)
            x = _linear_to_srgb(np.clip(noisy, 0, 1))
        result = _pil(x)
        if (op := enabled("jpeg")) is not None:
            buffer = io.BytesIO()
            result.save(buffer, format="JPEG", quality=draw(op, "quality", integer=True))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                result = decoded.convert("RGB").copy()
        return result
