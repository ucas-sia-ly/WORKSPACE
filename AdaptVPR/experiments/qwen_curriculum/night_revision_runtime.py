"""Bind explicit text-planner repairs to an unchanged frozen cohort.

The original configuration, jobs and dependency bytes remain available for
audit. This supplement permits a narrowly scoped runtime repair; it cannot
change the scene policy or authorize image generation.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile

from . import campaign, common

VERSION = 'night_revision_runtime_repair_v1'
CHAIN_VERSION = 'night_revision_runtime_repair_v2'
ARCHIVE_VERSION = 'night_revision_runtime_archive_v1'
ORIGINAL_VERSION = 'night_revision_original_files_v1'
MODULE_DIR = Path(__file__).resolve().parent
ORIGINAL_NAMES = frozenset((
    'night_revision.py', 'night_diversity_policy_v2.py',
    'night_style_catalog_v2.json', 'night_revision_planner.py',
    'scene_prompting.py', 'common.py', 'run.py', 'campaign.py',
    'night_cluster_worker.py',
))
RUNTIME_NAME = 'night_revision_runtime.py'


def _paths(root):
    state = Path(root).resolve() / 'state'
    return state / 'revision_config.json', state / 'runtime_patches' / 'original_files', state / 'runtime_patch.json'


def _parent(root):
    config, _, _ = _paths(root)
    cfg = campaign.config_file(config)
    expected = {str((MODULE_DIR / name).resolve()) for name in ORIGINAL_NAMES}
    with_runtime = expected | {str((MODULE_DIR / RUNTIME_NAME).resolve())}
    if (cfg.get('version') != 'night_revision_v2'
            or cfg.get('condition') != 'night'
            or set(cfg.get('files_sha256', {})) not in (expected, with_runtime)):
        raise ValueError('Unexpected frozen revision dependency inventory')
    for name, digest in cfg['files_sha256'].items():
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('Invalid frozen revision dependency digest: ' + name)
    return cfg


def _archive_manifest(root, cfg):
    config, archive, _ = _paths(root)
    expected = dict(version=ORIGINAL_VERSION, parent_fingerprint=cfg['fingerprint'],
                    parent_config_sha256=common.file_sha256(config), files_sha256=cfg['files_sha256'])
    manifest = campaign.config_file(archive / 'manifest.json')
    if {k: v for k, v in manifest.items() if k != 'fingerprint'} != expected:
        raise ValueError('Original dependency archive belongs to a different revision')
    for digest in cfg['files_sha256'].values():
        path = archive / digest
        if not path.is_file() or common.file_sha256(path) != digest:
            raise ValueError('Original dependency archive missing or changed: ' + str(path))
    return manifest


def capture_original(root):
    """Archive verified original bytes before editing any frozen dependency."""
    cfg = _parent(root)
    config, archive, _ = _paths(root)
    if (archive / 'manifest.json').exists():
        return _archive_manifest(root, cfg)
    # Read and validate everything before writing a receipt. No changed runtime
    # can be relabelled as the original frozen code.
    originals = {}
    for name, digest in cfg['files_sha256'].items():
        payload = Path(name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('Original revision dependency already changed: ' + name)
        originals[digest] = payload
    archive.mkdir(parents=True, exist_ok=True)
    for digest, payload in originals.items():
        target = archive / digest
        if target.exists():
            if common.file_sha256(target) != digest:
                raise ValueError('Original dependency archive changed: ' + str(target))
            continue
        with tempfile.NamedTemporaryFile(dir=archive, prefix='.original-', delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    campaign.frozen(archive / 'manifest.json', dict(
        version=ORIGINAL_VERSION, parent_fingerprint=cfg['fingerprint'],
        parent_config_sha256=common.file_sha256(config), files_sha256=cfg['files_sha256']))
    return _archive_manifest(root, cfg)


def _mapping(dependencies, *, patched):
    expected = {str((MODULE_DIR / name).resolve()) for name in ORIGINAL_NAMES}
    if patched:
        expected.add(str((MODULE_DIR / RUNTIME_NAME).resolve()))
    if not isinstance(dependencies, dict) or set(dependencies) != expected:
        raise ValueError('Runtime repair dependency inventory changed')
    for name, digest in dependencies.items():
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('Invalid runtime repair dependency digest: ' + name)
    return dict(dependencies)


def _current(dependencies, *, patched):
    effective = _mapping(dependencies, patched=patched)
    for name, digest in effective.items():
        if common.file_sha256(Path(name)) != digest:
            raise ValueError('Runtime repair dependency changed: ' + name)
    return effective


def _same_policy(cfg, policy_definition):
    if policy_definition != cfg.get('policy'):
        raise ValueError('Runtime repair cannot change the frozen scene policy')


def _payload(root, cfg, effective, previous=None):
    config, archive, _ = _paths(root)
    before = previous[1]['effective_files_sha256'] if previous else cfg['files_sha256']
    value = dict(
        version=VERSION, parent_fingerprint=cfg['fingerprint'],
        parent_config_sha256=common.file_sha256(config),
        original_manifest_sha256=common.file_sha256(archive / 'manifest.json'),
        before_files_sha256=before, effective_files_sha256=effective,
        policy_sha256=common.fingerprint(cfg['policy']), policy_unchanged=True,
        changed_files=sorted(name for name, digest in effective.items()
                             if before.get(name) != digest),
        scope='text response validation and failed-prompt recovery only',
        generation_barrier=cfg['generation_barrier'], image_generation_authorized=False)
    if previous:
        path, patch = previous
        _runtime_archive(root, path, patch)
        archive_path = _runtime_archive_path(root, path) / 'manifest.json'
        value.update(version=CHAIN_VERSION, previous_patch_path=str(path),
                     previous_patch_sha256=common.file_sha256(path),
                     previous_archive_sha256=common.file_sha256(archive_path))
    return value


def _patch_paths(root):
    _, _, first = _paths(root)
    candidates = list(first.parent.glob('runtime_patch*.json'))
    if not candidates:
        return []
    numbered = {}
    for path in candidates:
        if path == first:
            number = 1
        else:
            matched = re.fullmatch(r'runtime_patch_(\d{6})\.json', path.name)
            if matched is None or int(matched[1]) < 2:
                raise ValueError('Unexpected runtime repair supplement name: ' + path.name)
            number = int(matched[1])
        if number in numbered:
            raise ValueError('Duplicate runtime repair supplement')
        numbered[number] = path
    if set(numbered) != set(range(1, len(numbered) + 1)):
        raise ValueError('Runtime repair supplement chain has a missing parent')
    return [numbered[number] for number in sorted(numbered)]


def _runtime_archive_path(root, patch_path):
    return Path(root).resolve() / 'state/runtime_patches' / common.file_sha256(patch_path)


def _runtime_archive(root, patch_path, patch):
    directory = _runtime_archive_path(root, patch_path)
    digest = common.file_sha256(patch_path)
    expected = dict(version=ARCHIVE_VERSION, parent_fingerprint=patch['parent_fingerprint'],
                    execution_patch_sha256=digest,
                    files_sha256={**patch['effective_files_sha256'], str(patch_path): digest})
    manifest = campaign.config_file(directory / 'manifest.json')
    if {k: v for k, v in manifest.items() if k != 'fingerprint'} != expected:
        raise ValueError('Historical runtime archive belongs to a different execution')
    for sha in expected['files_sha256'].values():
        path = directory / 'files' / sha
        if not path.is_file() or common.file_sha256(path) != sha:
            raise ValueError('Historical runtime archive missing or changed: ' + str(path))
    return manifest


def _chain(root, cfg):
    values = []
    for path in _patch_paths(root):
        patch = campaign.config_file(path)
        effective = _mapping(patch.get('effective_files_sha256'), patched=True)
        expected = _payload(root, cfg, effective, values[-1] if values else None)
        if patch != {**expected, 'fingerprint': common.fingerprint(expected)}:
            raise ValueError('Runtime repair supplement changed or belongs to a different revision')
        values.append((path, patch))
    return values


def archive_current(root):
    """Capture the current verified runtime before editing it for another repair."""
    cfg = _parent(root)
    _archive_manifest(root, cfg)
    chain = _chain(root, cfg)
    if not chain:
        raise ValueError('No installed runtime supplement to archive')
    patch_path, patch = chain[-1]
    effective = _current(patch['effective_files_sha256'], patched=True)
    directory = _runtime_archive_path(root, patch_path)
    files = {**effective, str(patch_path): common.file_sha256(patch_path)}
    payloads = {}
    for name, sha in files.items():
        payload = Path(name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('Runtime changed while archiving: ' + name)
        payloads[sha] = payload
    for sha, payload in payloads.items():
        target = directory / 'files' / sha
        if target.exists():
            if common.file_sha256(target) != sha:
                raise ValueError('Historical runtime archive changed: ' + str(target))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.runtime-', delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    campaign.frozen(directory / 'manifest.json', dict(
        version=ARCHIVE_VERSION, parent_fingerprint=cfg['fingerprint'],
        execution_patch_sha256=common.file_sha256(patch_path), files_sha256=files))
    return _runtime_archive(root, patch_path, patch)


def install(root, dependencies, policy_definition):
    """Append an immutable repair, requiring an archive of its predecessor.

    Repeating the same installation returns the existing latest supplement.
    Changed code needs a new numbered supplement and can never overwrite the
    predecessor, its ready results, or the original cohort configuration.
    """
    cfg = _parent(root)
    _same_policy(cfg, policy_definition)
    _archive_manifest(root, cfg)
    effective = _current(dependencies, patched=True)
    chain = _chain(root, cfg)
    if chain and chain[-1][1]['effective_files_sha256'] == effective:
        return chain[-1][1]
    _, _, first = _paths(root)
    path = first if not chain else first.with_name(f'runtime_patch_{len(chain) + 1:06d}.json')
    return campaign.frozen(path, _payload(root, cfg, effective, chain[-1] if chain else None))


def verify(root, cfg, current_dependencies, policy_definition):
    """Return execution provenance without changing parent identity.

    ``execution_patch_sha256`` identifies the execution contract: the explicit
    supplement for repaired legacy cohorts, or the configuration itself when a
    new cohort originally froze this runtime. ``execution_identity_kind`` makes
    that distinction explicit for journals and bounded recovery.
    """
    parent = _parent(root)
    if cfg != parent:
        raise ValueError('Runtime repair received a changed parent configuration')
    _same_policy(parent, policy_definition)
    config, _, _ = _paths(root)
    paths = _patch_paths(root)
    if not paths:
        native_runtime = str((MODULE_DIR / RUNTIME_NAME).resolve()) in parent['files_sha256']
        if not native_runtime and set(current_dependencies) != set(parent['files_sha256']):
            raise ValueError('Frozen revision dependency inventory changed; explicit runtime repair installation required')
        effective = _current(current_dependencies, patched=native_runtime)
        if effective != parent['files_sha256']:
            raise ValueError('Frozen revision dependency changed; explicit runtime repair required')
        if native_runtime:
            return {**cfg, 'execution_patch_sha256': common.file_sha256(config),
                    'execution_identity_kind': 'native_configuration',
                    'execution_patch_history': [common.file_sha256(config)],
                    'effective_files_sha256': effective}
        return {**cfg, 'effective_files_sha256': effective}
    _archive_manifest(root, parent)
    chain = _chain(root, parent)
    patch_path, patch = chain[-1]
    effective = _current(current_dependencies, patched=True)
    if effective != patch['effective_files_sha256']:
        raise ValueError('Runtime repair supplement changed or does not match current code')
    digest = common.file_sha256(patch_path)
    return {**cfg, 'execution_patch_sha256': digest, 'execution_identity_kind': 'runtime_patch',
            'execution_patch_history': [common.file_sha256(path) for path, _ in chain],
            'effective_files_sha256': {**effective, **{str(path): common.file_sha256(path) for path, _ in chain}}}
