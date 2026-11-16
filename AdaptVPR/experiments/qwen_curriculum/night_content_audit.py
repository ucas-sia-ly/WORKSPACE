#!/usr/bin/env python3
"""Resumable night-cohort DINO + local-structure safety audit (no training).

Existing geometry/weather evidence is required and bound to exact image hashes.
The original dataset and frozen v4.x implementation are never modified. Outputs
are safety-screen results, not unconditional training approval.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import content_audit_v4 as legacy
from experiments.qwen_curriculum import content_audit_v42 as audit


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def index_rows(path):
    result = {}
    for row in read_rows(path):
        key = row['sample_id']
        if key in result:
            raise ValueError('Duplicate evidence sample_id: ' + key)
        result[key] = row
    return result


def validate_calibration(calibration):
    if (calibration.get('version') != audit.VERSION or calibration.get('usable') is not True or
            calibration.get('artifact_sha256') != legacy.digest({k: v for k, v in calibration.items()
                                                                if k != 'artifact_sha256'}) or
            calibration.get('decision_implementation_sha256') != legacy.sha256(audit.__file__) or
            calibration.get('frozen_fit_rules_sha256') != legacy.digest(calibration.get('rules'))):
        raise ValueError('Missing, changed or unvalidated v4.2 calibration; recalibrate independent controls')
    audit.compatible_feature_pipeline(calibration['feature_pipeline'])


def attach_base(row, evidence):
    """Require identity/hashes; never transfer PASS from a different pair."""
    if evidence is None:
        raise ValueError('Missing geometry/weather evidence')
    for key in ('sample_id', 'source_sha256', 'output_sha256'):
        if not row.get(key) or evidence.get(key) != row[key]:
            raise ValueError('Geometry/weather evidence differs: ' + key)
    base = evidence.get('current_rule', evidence)
    return {**row, 'base_quality_passed': base.get('passed') is True and
            base.get('geometry_ok') is True and base.get('weather_ok') is True,
            'estimated_H': base.get('estimated_H'), 'matcher_image_size': base.get('matcher_image_size')}


def peak_region(signals, calibration):
    """Diagnostic maximum window; not a semantic defect box."""
    import numpy as np
    dino, structure, valid, structural_valid = audit._validated_signal(signals)
    joint = np.where(valid & structural_valid, np.minimum(dino, structure), 0.)
    summary = audit.local_joint_summary(signals, calibration['rules']['window_fractions'])
    peaks = []
    height, width = joint.shape
    for size in summary['window_sizes']:
        means = audit._window_mean(joint, size)
        y, x = np.unravel_index(means.argmax(), means.shape)
        peaks.append({'score': float(means[y, x]), 'window_patches': size,
                      'generated_box_xyxy_normalized': [x / width, y / height,
                                                        (x + size) / width, (y + size) / height]})
    return max(peaks, key=lambda value: value['score'])


def evaluate(row, evidence, signals, calibration, provider=None):
    """Fail closed; report static quarantine separately from image evidence."""
    bound = row
    try:
        for path, key in [('source_path', 'source_sha256'), ('output_path', 'output_sha256')]:
            if not row.get(key) or legacy.sha256(row[path]) != row[key]:
                raise ValueError('Image integrity failed: ' + path)
        bound = attach_base(row, evidence)
        if provider is not None and bound['base_quality_passed'] and row.get('prompt_variant') != 'shopfronts':
            pipeline = calibration['feature_pipeline']
            signals = legacy.pair_signals(bound, provider, pipeline['side'], pipeline['search_radius'])
        result = audit.audit_pair_v42(bound, signals, calibration)
        peak = None
        # Static quarantine is not a localized detection. Only valid, matching
        # feature caches can produce a diagnostic box, separately labelled.
        if signals is not None:
            audit._signal_matches_row(bound, signals)
            if signals['pipeline_sha256'] != calibration['feature_pipeline_sha256']:
                raise ValueError('Feature cache pipeline differs from calibration')
            peak = peak_region(signals, calibration)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result = {'decision': 'unknown', 'eligible_for_training': False,
                  'reasons': ['invalid_required_evidence: ' + str(error)]}
        peak = None
    reasons = result['reasons']
    return {'sample_id': row['sample_id'], 'review_id': row.get('review_id'),
            'city': row.get('city'), 'place_id': row.get('place_id'),
            'input_row_sha256': legacy.digest(row),
            'content_audit': result, 'safety_screen_passed': result['decision'] == 'pass',
            'decision_source': ('prompt_policy' if 'prompt_variant_quarantined_by_policy' in reasons
                                else 'local_content' if 'diagnostics' in result else 'missing_or_invalid_evidence'),
            'peak_region': peak, 'peak_region_is_semantic_defect': False,
            'eligible_for_training': False,
            'training_status': 'requires_full_dataset_quality_protocol'}


def load_provider(args, calibration):
    if not args.dino_repo or not args.dino_checkpoint:
        raise ValueError('Online inference requires --dino-repo and --dino-checkpoint (local files only)')
    import torch
    torch.set_num_threads(args.threads)
    expected = calibration['feature_pipeline']['provider']
    provider = legacy.FrozenDINOProvider(args.dino_repo, args.dino_checkpoint,
                                        expected['architecture'], args.device)
    actual = provider.definition
    # Absolute repository paths are deployment metadata; every model/code,
    # normalization and framework field must still match exactly.
    if {k: v for k, v in actual.items() if k != 'repository'} != {
            k: v for k, v in expected.items() if k != 'repository'}:
        raise ValueError('Model/code/torch preprocessing pipeline differs from calibration; '
                         'use the calibrated environment or independently recalibrate')
    provider.definition = dict(expected)
    return provider


def run(args):
    calibration = json.loads(Path(args.calibration).read_text())
    validate_calibration(calibration)
    protocol = {'version': 'night_content_audit_v1',
                'inputs_sha256': {name: legacy.sha256(getattr(args, name)) for name in
                                  ('manifest', 'base_measurements', 'calibration')},
                'signals_sha256': legacy.sha256(args.signals) if args.signals else None,
                'implementation_sha256': legacy.sha256(__file__),
                'feature_implementation_sha256': legacy.sha256(legacy.__file__),
                'device': args.device if not args.signals else 'cached_signals_no_model',
                'calibration_sha256': calibration['artifact_sha256'],
                'quarantined_variants': ['shopfronts'], 'unknown_policy': 'excluded',
                'training': False, 'generation': False, 'original_dataset_modified': False}
    if not args.execute:
        print(json.dumps({'status': 'prepared_not_executed', 'protocol': protocol}, indent=2))
        return
    out = Path(args.output_dir)
    if out.exists() and not args.resume:
        raise FileExistsError('Use a NEW output directory, or --resume for an identical audit')
    if not out.exists():
        out.mkdir(parents=True)
        (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    elif json.loads((out / 'protocol.json').read_text()) != protocol:
        raise ValueError('Resume protocol differs from frozen audit')
    base = index_rows(args.base_measurements)
    caches = index_rows(args.signals) if args.signals else {}
    journal = out / 'audited_manifest.jsonl'
    # Only a partial final line from this audit's interrupted write is removed.
    if journal.exists():
        with journal.open('rb+') as stream:
            end = 0
            while line := stream.readline():
                if not line.endswith(b'\n'):
                    stream.truncate(end)
                    break
                json.loads(line)
                end = stream.tell()
    previous = iter(read_rows(journal)) if journal.exists() else iter(())
    old = next(previous, None)
    provider = None
    counts = collections.Counter()
    seen = set()
    with journal.open('a') as log, (out / 'safety_pass_manifest.jsonl').open('w') as accepted:
        for row in read_rows(args.manifest):
            key = row['sample_id']
            if key in seen:
                raise ValueError('Duplicate input sample_id: ' + key)
            seen.add(key)
            if old is not None:
                if old['sample_id'] != key or old['input_row_sha256'] != legacy.digest(row):
                    raise ValueError('Resume journal is not an exact input prefix')
                # Recheck bytes on resume even for completed entries.
                for field, sha in [('source_path', 'source_sha256'), ('output_path', 'output_sha256')]:
                    if legacy.sha256(row[field]) != row[sha]:
                        raise ValueError('Previously audited image changed: ' + key)
                result = old
                old = next(previous, None)
            else:
                if not args.signals and provider is None:
                    provider = load_provider(args, calibration)
                result = evaluate(row, base.get(key), caches.get(key), calibration, provider)
                log.write(json.dumps(result, allow_nan=False) + '\n')
                log.flush()
                os.fsync(log.fileno())
            counts[result['content_audit']['decision']] += 1
            if result['safety_screen_passed']:
                accepted.write(json.dumps({**row, 'content_safety': result,
                                           'eligible_for_training': False}, allow_nan=False) + '\n')
            if len(seen) % 100 == 0:
                print(f'Audited {len(seen)}: {dict(counts)}', flush=True)
        if old is not None:
            raise ValueError('Resume journal longer than input')
    report = {'counts': dict(counts), 'total': len(seen), 'complete': True,
              'calibration_sha256': calibration['artifact_sha256'],
              'interpretation': 'Control-calibrated safety screen, not semantic correctness',
              'training_approval': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='outputs/night_dataset_v1/manifest.jsonl')
    parser.add_argument('--base-measurements', required=True, help='Exact pair-bound geometry/weather JSONL')
    parser.add_argument('--calibration', required=True, help='Validated frozen v4.2 control artifact')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--signals', help='Existing exact v4.1 signal JSONL; no model loaded')
    parser.add_argument('--dino-repo')
    parser.add_argument('--dino-checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Actually run; default only prechecks files/calibration')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    run(args)


if __name__ == '__main__':
    main()
