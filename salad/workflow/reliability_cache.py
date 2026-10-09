"""Compact, immutable weak targets from an offline frozen DINO teacher.

Only target/confidence grids are stored. Training needs no teacher model and
cannot silently substitute another source, resize, manifest, or modified image.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


TARGET_ALGORITHM = "local_structure_v1_radius3_tolerance1"
IMAGE_PREPROCESSING = "rgb_bilinear_imagenet_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_hash(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


class ReliabilityTargetCache:
    """Validate a completed cache and fetch exact paired grids on the CPU.

    Image bytes are hashed on first access, and again if file stat information
    changes. The small tensor file is always hashed when the cache is opened.
    """

    def __init__(self, path: Path, image_size: tuple[int, int]):
        self.path = Path(path).expanduser().resolve()
        image_size = tuple(image_size)
        if (len(image_size) != 2 or any(type(n) is not int or n <= 0 or n % 14
                                       for n in image_size)):
            raise ValueError("Cache image_size must contain two positive multiples of 14")
        index_path = self.path / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"No completed reliability cache index: {index_path}")
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        index = self.index
        if (not isinstance(index, dict) or type(index.get("format_version")) is not int
                or index.get("format_version") != 1
                or index.get("target_algorithm") != TARGET_ALGORITHM
                or index.get("image_preprocessing") != IMAGE_PREPROCESSING
                or type(index.get("patch_stride")) is not int or index.get("patch_stride") != 14):
            raise ValueError("Unsupported reliability cache format or target algorithm")
        if index.get("image_size") != list(image_size):
            raise ValueError("Reliability cache image_size differs from training image_size")
        for key in ("teacher_checkpoint_sha256", "manifest_sha256", "tensor_sha256"):
            if not _valid_hash(index.get(key)):
                raise ValueError(f"Reliability cache has invalid {key}")
        if index.get("tensor_file") != "targets.pt":
            raise ValueError("Reliability cache tensor_file must be targets.pt")
        tensor_path = self.path / "targets.pt"
        if file_sha256(tensor_path) != index["tensor_sha256"]:
            raise ValueError("Reliability cache tensor hash mismatch")
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {"targets", "confidence"}:
            raise ValueError("Reliability cache must store only targets and confidence")
        entries = index.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Reliability cache entries must be a nonempty list")
        shape = (len(entries), 1, image_size[0] // 14, image_size[1] // 14)
        for key in ("targets", "confidence"):
            tensor = payload[key]
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != shape or not torch.isfinite(tensor).all()
                    or (tensor < 0).any() or (tensor > 1).any()):
                raise ValueError(f"Invalid reliability cache {key} grid/range")
        if ((payload["targets"] == 0.5) & (payload["confidence"] != 0)).any():
            raise ValueError("Unknown reliability targets must have zero confidence")
        self._entries = {}
        for position, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError("Reliability cache entry must be an object")
            for key in ("source_path", "output_path"):
                value = entry.get(key)
                if (not isinstance(value, str) or not Path(value).is_absolute()
                        or str(Path(value).resolve()) != value):
                    raise ValueError(f"Reliability cache {key} must be a canonical absolute path")
            if (not _valid_hash(entry.get("source_sha256"))
                    or not _valid_hash(entry.get("output_sha256"))):
                raise ValueError("Reliability cache entry has invalid image hashes")
            if entry["source_path"] == entry["output_path"]:
                raise ValueError("Reliability cache source and output must differ")
            output = entry["output_path"]
            if output in self._entries:
                raise ValueError("Reliability cache has duplicate output paths")
            self._entries[output] = (position, entry)
        self._targets, self._confidence = payload["targets"], payload["confidence"]
        canonical = json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.integrity_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._verified_images = {}

    def _verify_image(self, path: Path, expected_hash: str):
        stat = path.stat()
        fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        key = str(path)
        if self._verified_images.get(key) != (fingerprint, expected_hash):
            if file_sha256(path) != expected_hash:
                raise ValueError(f"Reliability cache image content changed: {path}")
            self._verified_images[key] = (fingerprint, expected_hash)

    def fetch(self, output_path: Path, source_path: Path):
        output = Path(output_path).expanduser().resolve()
        source = Path(source_path).expanduser().resolve()
        record = self._entries.get(str(output))
        if record is None:
            raise ValueError(f"Generated image has no cached reliability targets: {output}")
        position, entry = record
        if str(source) != entry["source_path"]:
            raise ValueError(f"Reliability cache exact source mismatch: {output}")
        self._verify_image(source, entry["source_sha256"])
        self._verify_image(output, entry["output_sha256"])
        # A caller's augmentation/in-place update must never modify the cache.
        return self._targets[position].clone(), self._confidence[position].clone()


def load_teacher_image(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """Exactly the non-augmented training image preprocessing."""
    import numpy as np
    from PIL import Image

    height, width = image_size
    with Image.open(path) as original:
        image = original.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std
