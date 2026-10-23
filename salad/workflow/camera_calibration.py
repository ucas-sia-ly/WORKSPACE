"""Metadata isolation and photometric diagnostics for camera augmentation.

Unpaired brightness statistics cannot identify a camera pipeline. A fit of three
tone parameters therefore remains exploratory; noise/blur/bloom are not claimed
to be calibrated by matching brightness or saturation histograms.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _path(value, owner, root=None):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else (root or owner.parent) / path).resolve()


def independent_real_images(manifest, exclusions):
    """Validate declared calibration identities against a required eval inventory.

    This checks recorded use, not undocumented past use. group_id must name the
    physical scene/trajectory consistently in both manifests, not each filename.
    """
    manifest, exclusions = Path(manifest).resolve(), Path(exclusions).resolve()
    data, inventory = json.loads(manifest.read_text()), json.loads(exclusions.read_text())
    if (data.get('version') != 1 or data.get('role') != 'camera_calibration'
            or data.get('split') != 'calibration'
            or data.get('prior_use') != 'never_evaluated_or_used_for_selection'):
        raise ValueError('Require a declared independent calibration split and prior-use declaration')
    dataset = data.get('dataset', '')
    if not isinstance(dataset, str) or not dataset or any(s in dataset.lower() for s in ('svox', 'robotcar')):
        raise ValueError('SVOX/RobotCar cannot calibrate this camera profile')
    if inventory.get('version') != 1 or inventory.get('role') != 'evaluation_exclusion':
        raise ValueError('Require an explicit evaluation_exclusion inventory')
    blocked, blocked_groups, blocked_hashes = set(), set(), set()
    rows = inventory.get('images')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Evaluation exclusion inventory must be nonempty')
    for row in rows:
        if not row.get('group_id') or not row.get('path'):
            raise ValueError('Evaluation inventory needs path and stable scene/trajectory group_id')
        blocked.add(_path(row['path'], exclusions))
        blocked_groups.add(row['group_id'])
        if row.get('sha256'):
            blocked_hashes.add(row['sha256'])
    images, seen, hashes = [], set(), set()
    rows = data.get('images')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Calibration images must be a nonempty list')
    for row in rows:
        if not isinstance(row, dict) or not row.get('path') or not row.get('group_id'):
            raise ValueError('Calibration rows need path and stable scene/trajectory group_id')
        if row.get('condition', data.get('condition')) not in ('night', 'rainy_night'):
            raise ValueError('Calibration images need an explicit night/rainy_night condition')
        path = _path(row['path'], manifest)
        if any(s in str(path).lower() for s in ('svox', 'robotcar')):
            raise ValueError('SVOX/RobotCar paths are excluded from camera calibration')
        checksum = digest(path)
        if (path in blocked or row['group_id'] in blocked_groups or checksum in blocked_hashes):
            raise ValueError('Calibration overlaps recorded evaluation images or scene/trajectory groups')
        if path in seen or checksum in hashes:
            raise ValueError('Duplicate calibration image identity/content')
        if row.get('sha256') and row['sha256'] != checksum:
            raise ValueError('Calibration image checksum changed')
        seen.add(path); hashes.add(checksum)
        images.append(dict(path=str(path), group_id=row['group_id'], sha256=checksum))
    return images, dict(calibration_manifest_sha256=digest(manifest),
                        evaluation_exclusion_sha256=digest(exclusions), dataset=dataset,
                        prior_use='declaration plus recorded path/hash/group isolation; not proof of undocumented use')


def photometry(image):
    import numpy as np
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.
    luminance = rgb @ np.array([.299, .587, .114], dtype=np.float32)
    maximum, minimum = rgb.max(axis=-1), rgb.min(axis=-1)
    saturation = np.divide(maximum-minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    return dict(mean_luminance=float(luminance.mean()), black_fraction=float((luminance < 15/255).mean()),
                highlight_fraction=float((luminance > 245/255).mean()), saturation=float(saturation.mean()))


def describe(rows):
    import numpy as np
    if not rows:
        raise ValueError('No photometric samples')
    return {key: dict(median=float(np.median([r[key] for r in rows])),
                      q25=float(np.quantile([r[key] for r in rows], .25)),
                      q75=float(np.quantile([r[key] for r in rows], .75))) for key in rows[0]}


def tone_distance(observed, target):
    # 1/255 is the RGB quantization resolution, a numerical scale floor, not an
    # acceptance threshold. Values are descriptive and never decide eligibility.
    return sum(abs(observed[k]['median']-target[k]['median']) /
               max(target[k]['q75']-target[k]['q25'], 1/255) for k in target) / len(target)


def eligible_synthetic_images(path):
    path = Path(path).resolve()
    images, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if (row.get('passed') is not True or row.get('eligible_for_training') is not True
                or row.get('condition') not in ('night', 'rainy_night')
                or row.get('prompt_variant') == 'shopfronts'):
            continue
        output = _path(row['output_path'], path)
        if output in seen: raise ValueError('Duplicate synthetic path in calibration input')
        seen.add(output)
        checksum = digest(output)
        if row.get('output_sha256') and row['output_sha256'] != checksum:
            raise ValueError('Synthetic image checksum changed')
        images.append(dict(path=str(output), sha256=checksum, condition=row['condition']))
    if not images: raise ValueError('No eligible low-light synthetic images')
    return images
