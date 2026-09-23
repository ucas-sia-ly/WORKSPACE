"""Prepare BF16 Qwen blocks with the pinned Lightning LoRA merged on CPU.

Only one block is materialized at a time. Runtime validation reads headers, not
tensor payloads; conversion/resume verifies the completed files with SHA-256.
"""

from collections import defaultdict
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import struct


MANIFEST = "adaptvpr_disk_weights.json"
FORMAT_VERSION = 1


def read_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length <= 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header: {path}")
        raw = stream.read(length)
    header = json.loads(raw)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    expected = 8 + length + max((v["data_offsets"][1] for v in tensors.values()), default=0)
    if path.stat().st_size != expected:
        raise ValueError(f"Truncated/invalid safetensors file: {path}")
    return tensors, hashlib.sha256(raw).hexdigest()


def signature(path):
    path = Path(path).resolve()
    _, digest = read_header(path)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "header_sha256": digest}


def block_file(key):
    match = re.match(r"^transformer_blocks\.(\d+)\.", key)
    return f"block_{int(match[1])}.safetensors" if match else "non_block.safetensors"


def build_plan(model_path, lora_path):
    source = Path(model_path) / "transformer"
    config = json.loads((source / "config.json").read_text())
    tensors, sources = {}, {}
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No transformer weights found in {source}")
    for path in files:
        header, _ = read_header(path)
        for key, spec in header.items():
            if key in tensors:
                raise ValueError(f"Duplicate tensor: {key}")
            if spec["dtype"] != "BF16":
                raise ValueError(f"Expected BF16 base weights, found {key}: {spec['dtype']}")
            tensors[key], sources[key] = spec, path
    groups = defaultdict(list)
    for key in sorted(tensors):
        groups[block_file(key)].append(key)
    expected = {f"block_{i}.safetensors" for i in range(config["num_layers"])} | {"non_block.safetensors"}
    if set(groups) != expected:
        raise ValueError("Transformer block files do not match num_layers in config.json")
    lora_header, _ = read_header(lora_path)
    targets = {}
    consumed = set()
    for key in lora_header:
        if not key.endswith(".lora_up.weight"):
            continue
        stem = key.removesuffix(".lora_up.weight")
        down, alpha = stem + ".lora_down.weight", stem + ".alpha"
        target = stem + ".weight"
        if down not in lora_header or target not in tensors:
            raise ValueError(f"Unmatched Lightning LoRA tensor: {key}")
        up_shape, down_shape = lora_header[key]["shape"], lora_header[down]["shape"]
        if (len(up_shape) != 2 or len(down_shape) != 2 or up_shape[1] != down_shape[0]
                or tensors[target]["shape"] != [up_shape[0], down_shape[1]]):
            raise ValueError(f"LoRA shape mismatch: {key}")
        names = [key, down] + ([alpha] if alpha in lora_header else [])
        consumed.update(names)
        targets[target] = names
    if consumed != set(lora_header) or not targets:
        raise ValueError("Unsupported or unused LoRA keys; expected the pinned Lightning up/down/alpha format")
    identity = {
        "format_version": FORMAT_VERSION, "dtype": "BF16", "lora_merged": True,
        "lora_strength": 1.0, "num_layers": config["num_layers"],
        "sources": [signature(path) for path in files], "lora": signature(lora_path),
        "config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
    }
    return identity, tensors, sources, dict(groups), targets


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _write_manifest(output, manifest):
    temporary = output / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / MANIFEST)


def prepare_weights(model_path, lora_path, output):
    import fcntl
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    identity, tensors, sources, groups, targets = build_plan(model_path, lora_path)
    output = Path(output).resolve()
    source = (Path(model_path) / "transformer").resolve()
    if output == source or source in output.parents:
        raise ValueError("Prepared weights must use a separate directory outside the original transformer")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / MANIFEST
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("identity") != identity:
                raise ValueError("Existing preparation uses different input files; choose a new output directory")
        else:
            if any(p.name != ".prepare.lock" for p in output.iterdir()):
                raise ValueError(f"Refusing to overwrite nonempty directory: {output}")
            manifest = {"identity": identity, "complete": False, "files": {}}
        manifest["complete"] = False
        _write_manifest(output, manifest)
        for index, (filename, keys) in enumerate(groups.items(), start=1):
            destination = output / filename
            previous = manifest["files"].get(filename)
            if previous and destination.is_file() and _digest(destination) == previous["sha256"]:
                print(f"[{index}/{len(groups)}] verified {filename}; reusing", flush=True)
                continue
            print(f"[{index}/{len(groups)}] merging/saving {filename}", flush=True)
            weights = {}
            with ExitStack() as stack, torch.no_grad():
                readers = {path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                           for path in {sources[key] for key in keys}}
                lora = stack.enter_context(safe_open(lora_path, framework="pt", device="cpu"))
                for key in keys:
                    # Clone before adding deltas: original mmap files remain untouched.
                    weight = readers[sources[key]].get_tensor(key).clone()
                    if key in targets:
                        names = targets[key]
                        up = lora.get_tensor(names[0]).to(torch.bfloat16)
                        down = lora.get_tensor(names[1]).to(torch.bfloat16)
                        alpha = lora.get_tensor(names[2]).to(torch.bfloat16).item() if len(names) == 3 else None
                        scale = alpha / down.shape[0] if alpha else 1
                        # Match the pinned LoRALoader CPU/BF16 operation order.
                        delta = torch.mm(up, down) * scale
                        delta = delta * 1.0
                        weight.add_(delta)
                        del up, down, delta
                    weights[key] = weight
                del weight
            temporary = output / (filename + ".tmp")
            save_file(weights, str(temporary), metadata={"format": "pt", "adaptvpr_lora_merged": "true"})
            del weights
            temporary.replace(destination)
            header, header_digest = read_header(destination)
            manifest["files"][filename] = {
                "size": destination.stat().st_size, "sha256": _digest(destination),
                "header_sha256": header_digest, "tensor_count": len(header),
            }
            _write_manifest(output, manifest)
        manifest["complete"] = True
        _write_manifest(output, manifest)
    return validate_prepared_weights(model_path, lora_path, output)


def validate_prepared_weights(model_path, lora_path, output):
    """Validate all block headers and source identity without reading 38 GiB."""
    output = Path(output)
    try:
        manifest = json.loads((output / MANIFEST).read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Disk weights not prepared: run python scripts/prepare_qwen_disk_weights.py ({output})") from exc
    identity, tensors, _, groups, _ = build_plan(model_path, lora_path)
    if not manifest.get("complete") or manifest.get("identity") != identity:
        raise RuntimeError("Disk weights are incomplete or stale; rerun scripts/prepare_qwen_disk_weights.py")
    if set(manifest["files"]) != set(groups):
        raise RuntimeError("Prepared block manifest does not match the model")
    for filename, keys in groups.items():
        path = output / filename
        header, digest = read_header(path)
        record = manifest["files"][filename]
        if path.stat().st_size != record["size"] or digest != record["header_sha256"] or set(header) != set(keys):
            raise RuntimeError(f"Invalid prepared block: {path}")
        if any(header[key]["shape"] != tensors[key]["shape"] or header[key]["dtype"] != "BF16" for key in keys):
            raise RuntimeError(f"Wrong tensor shapes/dtypes in {path}")
    return manifest
