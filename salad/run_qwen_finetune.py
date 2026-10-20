#!/usr/bin/env python3
"""Freeze the current Qwen accepted pool, then reuse the CCR experiment runner.

New output directory = latest reviewed images. Existing directory = the same
frozen inputs, even while generation continues. No model is started by default.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import run_ccr_feasibility as ccr
from workflow.training_data import _resolve_manifest_file


# Only fields consumed by training/local supervision belong in the hot-path
# manifest. The complete original audit records are retained in a gzip archive.
TRAINING_FIELDS = (
    'sample_id', 'city', 'place_id', 'condition', 'source_path', 'output_path',
    'source_sha256', 'output_sha256', 'passed', 'eligible_for_training',
    'plausible', 'weather_ok', 'geometry_ok',
    *(name + suffix for name in ('edit_mask', 'occlusion_mask', 'destruction_mask')
      for suffix in ('_path', '_sha256')),
)
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}


def accepted_images(root):
    """Read aliases, never infer an exact source from filenames or place IDs."""
    directory = root / 'images/accepted'
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing Qwen accepted directory: {directory}')
    train = (root / 'images/train').resolve()
    result = {}
    for alias in sorted(directory.iterdir()):
        if alias.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        target = alias.resolve(strict=True)
        if not target.is_file() or not target.is_relative_to(train):
            raise ValueError(f'Accepted alias must resolve inside images/train: {alias}')
        if str(target) in result:
            raise ValueError(f'Duplicate accepted aliases for {target}')
        result[str(target)] = str(alias)
    return result


def checked_file(path, expected_hash, inventory):
    """Hash once per unique image/mask; keep memory bounded to one hash block."""
    key = str(path)
    if key not in inventory:
        before = path.stat()
        sha = ccr.digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f'Input changed while snapshotting: {path}; retry')
        inventory[key] = dict(size=after.st_size, mtime_ns=after.st_mtime_ns, sha256=sha)
    if expected_hash and inventory[key]['sha256'] != expected_hash:
        raise ValueError(f'Image/mask SHA256 differs from manifest: {path}')
    return inventory[key]['sha256']


def verify_snapshot(inputs, request):
    saved_request = json.loads((inputs / 'request.json').read_text())
    if saved_request != request:
        raise ValueError('Frozen experiment configuration changed; use a new --output-dir')
    receipt = json.loads((inputs / 'receipt.json').read_text())
    if receipt['entrypoint_sha256'] != ccr.digest(__file__):
        raise ValueError('Qwen entrypoint changed since preparation; use a new --output-dir')
    for name, sha in receipt['files_sha256'].items():
        if ccr.digest(inputs / name) != sha:
            raise ValueError(f'Frozen snapshot changed: {inputs / name}')
    inventory = json.loads((inputs / 'image_inventory.json').read_text())
    for name, record in inventory.items():
        path = Path(name)
        stat = path.stat()
        # Rehash only changed files on resume; adding unrelated images or
        # accepted aliases never changes the already frozen training cohort.
        if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns']):
            if ccr.digest(path) != record['sha256']:
                raise ValueError(f'Frozen image/mask content changed: {path}')
    return receipt


def prepare_inputs(config, out):
    """Atomically freeze a coherent intersection of manifest and accepted links."""
    inputs = out / 'inputs'
    request = dict(config)
    if inputs.exists():
        return verify_snapshot(inputs, request)
    if any((out / name).exists() for name in ('plan.json', 'runs', 'initial')):
        raise ValueError('Existing legacy experiment directory; use a new --output-dir')
    manifest = Path(config['synthetic_manifest'])
    if manifest.name != 'training_manifest.jsonl' or manifest.parent.name != 'state':
        raise ValueError('Use <qwen_curriculum>/state/training_manifest.jsonl in the Qwen config')
    root = manifest.parent.parent
    aliases = accepted_images(root)
    inventory, outputs, sample_ids = {}, {}, set()
    counts, conditions, cities = Counter(), Counter(), Counter()
    source_hash = hashlib.sha256()
    out.mkdir(parents=True, exist_ok=True)
    # Publish the directory only after every record, image and provenance file
    # has passed validation. Failed/in-flight snapshots leave no partial input.
    with tempfile.TemporaryDirectory(prefix='.qwen-inputs-', dir=out) as temporary:
        stage = Path(temporary) / 'inputs'
        stage.mkdir()
        with manifest.open('rb') as source, gzip.open(stage / 'audit_manifest.jsonl.gz', 'wb') as archive, \
                (stage / 'accepted_manifest.jsonl').open('w', encoding='utf-8') as target:
            for line_number, raw in enumerate(source, 1):
                source_hash.update(raw)
                archive.write(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError(f'{manifest}:{line_number}: expected a JSON object')
                counts['manifest_rows'] += 1
                if (row.get('passed') is not True or row.get('eligible_for_training') is not True
                        or any(key in row and row[key] is not True for key in ('plausible', 'weather_ok'))):
                    counts['not_eligible'] += 1
                    continue
                resolve = lambda key: _resolve_manifest_file(row.get(key), manifest=manifest,
                    line_number=line_number, field_name=key, real_data=Path(config['real_data']))
                output = resolve('output_path')
                if str(output) not in aliases:
                    counts['not_in_accepted'] += 1
                    continue
                original = resolve('source_path')
                if original == output or not original.is_relative_to(Path(config['real_data']) / 'Images'):
                    raise ValueError(f'{manifest}:{line_number}: invalid exact real source')
                if not isinstance(row.get('condition'), str) or not row['condition'].strip():
                    raise ValueError(f'{manifest}:{line_number}: missing weather condition')
                compact = {key: row[key] for key in TRAINING_FIELDS if key in row}
                compact.update(source_path=str(original), output_path=str(output))
                for path, key in ((original, 'source_sha256'), (output, 'output_sha256')):
                    compact[key] = checked_file(path, row.get(key), inventory)
                for name in ('edit_mask', 'occlusion_mask', 'destruction_mask'):
                    if row.get(name + '_path'):
                        path = resolve(name + '_path')
                        compact[name + '_path'] = str(path)
                        compact[name + '_sha256'] = checked_file(path, row.get(name + '_sha256'), inventory)
                identity = json.dumps(compact, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
                if str(output) in outputs:
                    if outputs[str(output)] != identity:
                        raise ValueError(f'Conflicting exact source/label/review for {output}')
                    counts['duplicate_rows'] += 1
                    continue
                sample_id = row.get('sample_id')
                if sample_id and sample_id in sample_ids:
                    raise ValueError(f'Duplicate Qwen sample_id: {sample_id}')
                if sample_id:
                    sample_ids.add(sample_id)
                outputs[str(output)] = identity
                target.write(identity + '\n')
                conditions[row['condition'].lower()] += 1
                cities[str(row.get('city', 'unknown'))] += 1
        if not outputs:
            raise ValueError('No reviewed images in both the manifest and images/accepted')
        if ccr.digest(manifest) != source_hash.hexdigest() or accepted_images(root) != aliases:
            raise ValueError('Qwen manifest/accepted links changed during snapshot; retry after publication')
        frozen = {**config, 'synthetic_manifest': str(inputs / 'accepted_manifest.jsonl')}
        ccr.write_json(stage / 'config.json', frozen)
        ccr.write_json(stage / 'request.json', request)
        ccr.write_json(stage / 'image_inventory.json', inventory)
        receipt = dict(version=1, curriculum_root=str(root), source_manifest=str(manifest),
            source_manifest_sha256=source_hash.hexdigest(), accepted_images=len(outputs),
            accepted_aliases=len(aliases), aliases_without_eligible_record=len(set(aliases) - outputs.keys()),
            counts=dict(counts), conditions=dict(conditions), cities=dict(cities),
            entrypoint_sha256=ccr.digest(__file__),
            files_sha256={path.name: ccr.digest(path) for path in sorted(stage.iterdir())},
            policy='New directory snapshots current accepted images; resume never refreshes the cohort.')
        ccr.write_json(stage / 'receipt.json', receipt)
        stage.rename(inputs)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ccr.ROOT / 'configs/ccr_qwen.json')
    parser.add_argument('--output-dir', type=Path, help='New directory to include newly accepted images')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--execute', action='store_true', help='Launch training and SVOX evaluation explicitly')
    action.add_argument('--snapshot-only', action='store_true', help='Freeze inputs only; no model or dataset scan')
    action.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args(argv)
    config = ccr.load_config(args.config.expanduser().resolve())
    if args.output_dir:
        config['output_dir'] = str(args.output_dir.expanduser().resolve())
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.qwen.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another Qwen experiment process owns this output directory')
        if args.summarize_only:
            ccr.main(['--config', str(out / 'inputs/config.json'), '--summarize-only'])
            return
        receipt = prepare_inputs(config, out)
        print(f"Frozen accepted images: {receipt['accepted_images']}; weather: {receipt['conditions']}", flush=True)
        print(f"Inputs: {out / 'inputs'}; use a new --output-dir to include later images.", flush=True)
        if not args.snapshot_only:
            ccr.main(['--config', str(out / 'inputs/config.json')] + (['--execute'] if args.execute else []))


if __name__ == '__main__':
    main()
