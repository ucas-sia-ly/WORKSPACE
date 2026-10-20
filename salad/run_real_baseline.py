#!/usr/bin/env python3
"""Prepare, screen and confirm a pure-real SALAD baseline. Execution is opt-in.

One full epoch screens P/LR/blocks on SVOX val. Independent seeds confirm the
chosen setting over the fixed full schedule. Test weather never selects it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import sys

import run_ccr_feasibility as ccr
from workflow.training_data import CoveragePlaceBatchSampler, MixedGSVCitiesDataset


def load_config(path):
    config = json.loads(path.read_text())
    for key in ('output_dir', 'real_data', 'checkpoint', 'backbone_repo', 'reserved_places', 'svox_root'):
        config[key] = str((path.parent / Path(config[key]).expanduser()).resolve())
    if config['precision'] not in ('bf16', '16', '32') or type(config['epochs']) is not int or config['epochs'] < 2:
        raise ValueError('Use explicit precision and at least two total epochs')
    for key in ('images_per_place', 'num_workers', 'prefetch_factor', 'torch_threads', 'eval_batch_size'):
        if type(config[key]) is not int or config[key] < (0 if key == 'num_workers' else 1):
            raise ValueError(f'Invalid {key}')
    if config['images_per_place'] < 2:
        raise ValueError('At least two real views per place are required')
    seeds = config['confirmation_seeds']
    if (len(seeds) < 3 or len(set(seeds)) != len(seeds) or config['screen_seed'] in seeds
            or any(type(seed) is not int or seed < 0 for seed in [config['screen_seed'], *seeds])):
        raise ValueError('Use at least three distinct confirmation seeds, independent of the screening seed')
    candidates = config['candidates']
    if not candidates or len({item['name'] for item in candidates}) != len(candidates):
        raise ValueError('Candidate names must be nonempty and unique')
    for item in candidates:
        if (not item['name'].replace('_', '').isalnum() or type(item['batch_size']) is not int
                or item['batch_size'] < 2 or type(item['num_trainable_blocks']) is not int
                or item['num_trainable_blocks'] < 1 or not math.isfinite(item['learning_rate'])
                or item['learning_rate'] <= 0):
            raise ValueError(f'Invalid P/LR/block candidate: {item}')
    gates = config['acceptance']
    if not (0 < gates['min_active_anchor_fraction'] <= 1 and 0 <= gates['max_empty_mining_fraction'] < 1
            and all(math.isfinite(gates[key]) and gates[key] >= 0 for key in ('max_val_drop_pp', 'max_seed_std_pp'))):
        raise ValueError('Invalid prespecified acceptance criteria')
    if (set(config['resource_limits']) != {'cuda_budget_gib', 'cuda_reserve_gib', 'host_rss_limit_gib', 'host_available_reserve_gib'}
            or any(not math.isfinite(value) or value <= 0 for value in config['resource_limits'].values())):
        raise ValueError('Resource limits must be finite and positive')
    return config


def reserved_identities(payload):
    if payload.get('version') != 1 or set(payload.get('groups', {})) != {'validation', 'calibration'}:
        raise ValueError('Reserved split requires version 1 and validation/calibration groups')
    reasons = {}
    for reason, rows in payload['groups'].items():
        if not isinstance(rows, list):
            raise ValueError('Reserved groups must contain lists')
        for row in rows:
            if (not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
                    or not row[0] or type(row[1]) is not int):
                raise ValueError('Reserved identities must be [city, integer place_id]')
            key = tuple(row)
            if key in reasons:
                raise ValueError(f'Duplicate or overlapping reserved identity: {key}')
            reasons[key] = reason
    return reasons


def validation_inventory(config):
    from workflow.evaluation import load_evaluation_set
    validation = load_evaluation_set('SVOX', Path(config['svox_root']), split='val', query_subdirs=['queries'])
    inventory = [(item.id, str(item.path), item.path.stat().st_size, item.path.stat().st_mtime_ns)
                 for item in validation.references + validation.queries]
    return {**validation.summary(), 'image_inventory_sha256': hashlib.sha256(ccr.encoded(inventory).encode()).hexdigest()}


def prepare(config, out):
    """Full real metadata validation; no synthetic manifest or teacher is read."""
    out.mkdir(parents=True, exist_ok=True)
    reasons = reserved_identities(json.loads(Path(config['reserved_places']).read_text()))
    dataset = MixedGSVCitiesDataset(Path(config['real_data']), synthetic_manifest=None,
        images_per_place=config['images_per_place'], min_images_per_place=config['images_per_place'],
        synthetic_fraction=0, real_sampling='coverage_pk')
    known = set(dataset.source_index.values())
    if missing := set(reasons) - known:
        raise ValueError(f'Reserved identities missing from GSV metadata: {sorted(missing)}')
    excluded = out / 'excluded_places.json'
    ccr.immutable(excluded, ccr.encoded([list(key) for key in sorted(reasons)]))
    counts = Counter(dataset.source_index.values())
    records = [dict(city=city, place_id=place_id, reason=reasons[city, place_id],
                    real_images=counts[city, place_id]) for city, place_id in sorted(reasons)]
    ccr.immutable(out / 'exclusion_report.json', ccr.encoded(dict(records=records,
        counts_by_reason=dict(Counter(reasons.values())), all_views_excluded=True,
        source_split_sha256=ccr.digest(config['reserved_places']))))
    train = [p for p in dataset.places if (p.city, p.place_id) not in reasons]
    if len(train) < 2:
        raise ValueError('Fewer than two training places remain')
    budgets = {item['name']: len(CoveragePlaceBatchSampler(len(train), item['batch_size'], 0))
               for item in config['candidates']}
    # The standard Coverage sampler visits each place exactly once, including
    # the last partial batch. Different P values imply different update counts.
    counts_by_city = Counter(p.city for p in train)
    validation_summary = validation_inventory(config)
    sources = [Path(__file__), ccr.ROOT / 'train_salad.py', ccr.ROOT / 'evaluate_salad.py', Path(ccr.__file__)]
    sources += sorted((ccr.ROOT / 'workflow').glob('*.py')) + sorted((ccr.ROOT / 'models/aggregators').glob('*.py'))
    inputs = [Path(config['checkpoint']), Path(config['reserved_places']), excluded,
              *sorted((Path(config['real_data']) / 'Dataframes').glob('*.csv'))]
    plan = dict(version=1, config=config, excluded_places=str(excluded), train_places=len(train),
        svox_validation=validation_summary,
        train_real_images=sum(len(p.real_paths) for p in train), cities=dict(sorted(counts_by_city.items())),
        reserved_places=len(reasons), steps_per_epoch=budgets,
        real_exposures_per_epoch=len(train) * config['images_per_place'],
        input_hashes={str(path): ccr.digest(path) for path in inputs},
        source_hashes={str(path): ccr.digest(path) for path in sources},
        selection_policy='One full epoch on screen_seed; highest eligible SVOX val R@1, ties use candidate order.',
        confirmation_policy='Independent seeds; fixed final epoch; all seeds must pass operational criteria.',
        limitations=['Operational signal/recall thresholds are engineering guardrails, not statistical guarantees.',
                     'Equal real image exposure across P; optimizer update counts and negative pools differ.',
                     'Reservations do not establish pretraining or cross-dataset geographic disjointness.',
                     'No synthetic data, teacher, distillation or Reliability is used. Original MS/miner is unchanged.'])
    ccr.immutable(out / 'plan.json', ccr.encoded(plan))
    return plan


def validate_plan(plan):
    for group in ('input_hashes', 'source_hashes'):
        for name, expected in plan[group].items():
            if ccr.digest(name) != expected:
                raise ValueError(f'Frozen {group} changed: {name}; use a new output directory')
    if validation_inventory(plan['config']) != plan['svox_validation']:
        raise ValueError('Frozen SVOX validation inventory changed')


def training_command(plan, candidate, seed, directory, stop_epoch, resume=False):
    c = plan['config']
    return [sys.executable, str(ccr.ROOT / 'train_salad.py'),
        '--real-data', c['real_data'], '--output-dir', str(directory),
        '--exclude-places', plan['excluded_places'], '--synthetic-fraction', '0',
        '--real-sampling', 'coverage_pk', '--images-per-place', str(c['images_per_place']),
        '--min-images-per-place', str(c['images_per_place']), '--batch-size', str(candidate['batch_size']),
        '--learning-rate', str(candidate['learning_rate']), '--num-trainable-blocks', str(candidate['num_trainable_blocks']),
        '--weight-decay', '0', '--miner-margin', '0.1', '--seed', str(seed),
        '--epochs', str(c['epochs']), '--stop-after-epoch', str(stop_epoch), '--save-every', str(c['epochs']),
        '--num-workers', str(c['num_workers']), '--prefetch-factor', str(c['prefetch_factor']),
        '--image-size', '224', '224', '--precision', c['precision'], '--device', c['device'],
        '--backbone-repo', c['backbone_repo'], '--log-mined-pairs', '--profile-training',
        '--resume' if resume else '--init-checkpoint', str(directory / 'checkpoint.pt') if resume else c['checkpoint']]


def evaluation_plan(plan, test=False):
    conditions = ('default', 'night', 'rain', 'snow', 'sun', 'overcast') if test else ('val',)
    evaluations = [dict(name='svox_' + condition, dataset='SVOX', root=plan['config']['svox_root'],
        split='test' if test else 'val', role='test' if test else 'validation', cadence='final' if test else 'every',
        kind='real' if condition in ('val', 'default') else 'natural_weather',
        query_subdirs=['queries' if condition in ('val', 'default') else 'queries_' + condition]) for condition in conditions]
    return dict(config=plan['config'], evaluations=evaluations)


def checked_state(saved, plan, candidate, seed):
    if saved.get('format_version') != 1 or 'optimizer_state_dict' not in saved:
        raise ValueError('Baseline resume requires a complete workflow checkpoint')
    c, training = plan['config'], saved['training_config']
    expected = dict(synthetic_manifest=None, synthetic_fraction=0, synthetic_places_only=False,
        batch_size=candidate['batch_size'], learning_rate=candidate['learning_rate'],
        weight_decay=0, miner_margin=.1, real_sampling='coverage_pk', max_batches_per_epoch=0,
        seed=seed, augment=True, images_per_place=c['images_per_place'], real_data=c['real_data'])
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError('Checkpoint is not the expected clean full-real run')
    if (any(key in training for key in ('teacher', 'synthetic_aux', 'local_distillation', 'real_descriptor_retention', 'reliability'))
            or training['model_config']['backbone_config']['num_trainable_blocks'] != candidate['num_trainable_blocks']
            or training['init_checkpoint_sha256'] != plan['input_hashes'][c['checkpoint']]
            or training['model_config']['agg_config'].get('reliability_ot', False)
            or training['excluded_places']['sha256'] != plan['input_hashes'][plan['excluded_places']]
            or saved['total_epochs'] != c['epochs'] or saved['precision'] != c['precision']
            or saved['global_step'] != saved['epoch'] * plan['steps_per_epoch'][candidate['name']]):
        raise ValueError('Checkpoint objective, initialization, exclusions or schedule differs from the plan')


def run_candidate(plan, candidate, seed, directory, target_epoch):
    from workflow.model import read_checkpoint
    c = plan['config']
    checkpoint = directory / 'checkpoint.pt'
    previous_epoch = -1
    while True:
        epoch = 0
        if checkpoint.exists():
            saved = read_checkpoint(checkpoint)
            checked_state(saved, plan, candidate, seed)
            epoch = saved['epoch']
            if epoch <= previous_epoch:
                raise RuntimeError('Training did not advance to the next full epoch')
            if epoch > target_epoch:
                raise ValueError('Checkpoint advanced past the requested fixed comparison epoch')
            ccr.immutable(directory / f'metrics_epoch_{epoch:03d}.json', ccr.encoded(saved['metrics']))
            del saved
            ccr.evaluate_checkpoint(evaluation_plan(plan), checkpoint, directory / 'evaluations' / str(epoch))
        if epoch == target_epoch:
            return
        previous_epoch = epoch
        ccr.run_child(training_command(plan, candidate, seed, directory, epoch + 1, checkpoint.exists()),
                      directory / 'train.log', c['torch_threads'], limits=c['resource_limits'])
        if not checkpoint.is_file():
            raise RuntimeError('Training returned without a checkpoint')


def signal_failures(metrics, plan):
    gates = plan['config']['acceptance']
    mining = metrics.get('mining', {})
    failures = []
    if (metrics.get('synthetic_exposure') != 0 or metrics.get('real_exposure') != plan['real_exposures_per_epoch']
            or any(key in metrics for key in ('synthetic_aux', 'local_distillation', 'real_descriptor_retention', 'reliability'))):
        failures.append('not_clean_full_real_exposure')
    loss = metrics.get('loss', float('nan'))
    if not math.isfinite(loss) or loss <= 0:
        failures.append('nonfinite_or_zero_MS_loss')
    active, empty = mining.get('active_anchor_fraction', -1), mining.get('no_mined_pair_batch_fraction', 2)
    if not math.isfinite(active) or active < gates['min_active_anchor_fraction']:
        failures.append('insufficient_active_anchors')
    if not math.isfinite(empty) or empty > gates['max_empty_mining_fraction']:
        failures.append('too_many_empty_mining_batches')
    if mining.get('real_real_mined_positive', 0) <= 0 or mining.get('real_real_mined_negative', 0) <= 0:
        failures.append('missing_real_positive_or_negative_pairs')
    return failures


def verified_evaluation(path, checkpoint_sha256, protocol):
    receipt_path = path.with_suffix('.receipt.json')
    if not path.exists() or not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('result_sha256') != ccr.digest(path)
            or receipt.get('checkpoint_sha256') != checkpoint_sha256 or receipt.get('protocol') != protocol):
        raise ValueError(f'Evaluation receipt/checkpoint mismatch: {path}')
    return json.loads(path.read_text())


def validation_result(directory, epoch, initial, plan):
    metrics_path = directory / f'metrics_epoch_{epoch:03d}.json'
    result_path = directory / 'evaluations' / str(epoch) / 'svox_val.json'
    if not metrics_path.exists() or not result_path.exists():
        return None
    result = verified_evaluation(result_path, ccr.digest(directory / 'checkpoint.pt'), evaluation_plan(plan)['evaluations'][0])
    if result is None:
        return None
    metrics = json.loads(metrics_path.read_text())
    value = result['recall']['R@1']
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid SVOX validation Recall')
    delta = 100 * (value - initial['recall']['R@1'])
    failures = signal_failures(metrics, plan)
    if delta < -plan['config']['acceptance']['max_val_drop_pp']:
        failures.append('validation_recall_drop')
    return dict(epoch=epoch, recall=result['recall'], delta_pp=delta, failures=failures,
                paired_queries=ccr.paired_changes(initial, result), metrics=metrics)


def select_candidate(plan, rows):
    if len(rows) != len(plan['config']['candidates']):
        return None
    eligible = [row for row in rows if not row['failures']]
    if not eligible:
        return None
    # Python max preserves input order for an exact recall tie.
    return max(eligible, key=lambda row: row['recall']['R@1'])['candidate']


def summarize(plan, out):
    c = plan['config']
    initial_path = out / 'initial/svox_val.json'
    report = dict(status='prepared_not_trained', screening=[], confirmation=[], selected=None,
                  acceptance=c['acceptance'], limitations=plan['limitations'])
    initial = verified_evaluation(initial_path, plan['input_hashes'][c['checkpoint']], evaluation_plan(plan)['evaluations'][0])
    if initial is not None:
        report['initial_recall'] = initial['recall']
        for candidate in c['candidates']:
            row = validation_result(out / 'screen' / candidate['name'], 1, initial, plan)
            if row is not None:
                report['screening'].append(dict(candidate=candidate['name'], **row))
        selected = select_candidate(plan, report['screening'])
        report.update(selected=selected, status='screening_incomplete' if len(report['screening']) < len(c['candidates'])
                      else 'no_candidate_passed' if selected is None else 'confirmation_required')
        if selected:
            for seed in c['confirmation_seeds']:
                directory = out / 'confirm' / selected / str(seed)
                row = validation_result(directory, c['epochs'], initial, plan)
                if row is not None:
                    for epoch in range(1, c['epochs']):
                        path = directory / f'metrics_epoch_{epoch:03d}.json'
                        if not path.exists():
                            row['failures'].append(f'missing_epoch_{epoch}_metrics')
                        else:
                            row['failures'].extend(f'epoch_{epoch}:{reason}' for reason in signal_failures(json.loads(path.read_text()), plan))
                    report['confirmation'].append(dict(seed=seed, **row))
            if len(report['confirmation']) == len(c['confirmation_seeds']):
                values = [100 * row['recall']['R@1'] for row in report['confirmation']]
                std = statistics.stdev(values)
                report.update(mean_R1_percent=statistics.mean(values), seed_std_pp=std,
                    status='passed_operational_checks' if std <= c['acceptance']['max_seed_std_pp']
                    and not any(row['failures'] for row in report['confirmation']) else 'baseline_not_stable')
    # Test is descriptive only and is read after every selection/stability
    # decision. It can never rescue a candidate that failed validation.
    report['weather_test'] = []
    if report['selected']:
        for seed in ('initial', *c['confirmation_seeds']):
            checkpoint = Path(c['checkpoint']) if seed == 'initial' else out / 'confirm' / report['selected'] / str(seed) / 'checkpoint.pt'
            if not checkpoint.exists():
                continue
            checksum = ccr.digest(checkpoint)
            for protocol in evaluation_plan(plan, test=True)['evaluations']:
                result = verified_evaluation(out / 'test' / str(seed) / (protocol['name'] + '.json'), checksum, protocol)
                if result is not None:
                    report['weather_test'].append(dict(seed=seed, domain=protocol['name'], recall=result['recall']))
    ccr.write_json(out / 'report.json', report)
    lines = ['# Stage 2 pure-real baseline', '', f"Status: {report['status']}", '',
             '| Candidate / seed | Val R@1 | Δ pp | Active anchors | Empty batches | Failures |',
             '|---|---:|---:|---:|---:|---|']
    for row in report['screening'] + report['confirmation']:
        mining = row['metrics']['mining']
        lines.append(f"| {row.get('candidate', row.get('seed'))} | {100*row['recall']['R@1']:.4f} | {row['delta_pp']:+.4f} | "
                     f"{mining['active_anchor_fraction']:.2%} | {mining['no_mined_pair_batch_fraction']:.2%} | {', '.join(row['failures']) or 'none'} |")
    lines += ['', 'Selection uses SVOX val only. All criteria are prespecified engineering checks, not a statistical proof.',
              'No stable baseline is claimed before all independent confirmation seeds finish.', '', *plan['limitations']]
    if report['weather_test']:
        lines += ['', '| Seed | Test category | R@1 | R@5 | R@10 |', '|---|---|---:|---:|---:|']
        for row in report['weather_test']:
            recall = row['recall']
            lines.append(f"| {row['seed']} | {row['domain']} | {100*recall['R@1']:.3f}% | {100*recall['R@5']:.3f}% | {100*recall['R@10']:.3f}% |")
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    if report['status'] == 'passed_operational_checks':
        checkpoints = {str(seed): out / 'confirm' / report['selected'] / str(seed) / 'checkpoint.pt'
                       for seed in c['confirmation_seeds']}
        ccr.immutable(out / 'baseline.json', ccr.encoded(dict(
            status=report['status'], candidate=report['selected'],
            checkpoint_by_seed={seed: dict(path=str(path), sha256=ccr.digest(path)) for seed, path in checkpoints.items()},
            excluded_places=plan['excluded_places'], exclusions_sha256=plan['input_hashes'][plan['excluded_places']],
            initial_checkpoint_sha256=plan['input_hashes'][c['checkpoint']],
            selection='SVOX val only; fixed final epoch, no best-test seed/checkpoint selection',
            mean_R1_percent=report['mean_R1_percent'], seed_std_pp=report['seed_std_pp'])))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ccr.ROOT / 'configs/real_baseline.json')
    parser.add_argument('--phase', choices=('prepare', 'screen', 'confirm', 'test', 'report'), default='prepare')
    parser.add_argument('--candidate', help='Optionally run one screening candidate; selection waits for all candidates')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    c = load_config(args.config.resolve())
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'plan.json').exists():
            plan = json.loads((out / 'plan.json').read_text())
            if plan['config'] != c:
                raise ValueError('Frozen baseline configuration changed; use a new output directory')
            if args.phase != 'report':
                validate_plan(plan)
        else:
            plan = prepare(c, out)
        commands = [shlex.join(training_command(plan, candidate, c['screen_seed'], out / 'screen' / candidate['name'], 1))
                    for candidate in c['candidates']]
        (out / 'screen_commands.txt').write_text('\n'.join(commands) + '\n')
        report = summarize(plan, out)
        if args.candidate and args.phase != 'screen':
            raise ValueError('--candidate is only supported during screening')
        print(f"Plan: {out / 'plan.json'}; real places={plan['train_places']}; excluded={plan['reserved_places']}", flush=True)
        if not args.execute or args.phase in ('prepare', 'report'):
            print(f"Status: {report['status']}; no GPU/training launched. Use --phase screen --execute explicitly.")
            return
        if args.phase == 'screen':
            candidates = [item for item in c['candidates'] if not args.candidate or item['name'] == args.candidate]
            if not candidates:
                raise ValueError('Unknown screening candidate')
            ccr.evaluate_checkpoint(evaluation_plan(plan), Path(c['checkpoint']), out / 'initial')
            for candidate in candidates:
                run_candidate(plan, candidate, c['screen_seed'], out / 'screen' / candidate['name'], 1)
                summarize(plan, out)
        else:
            if report['selected'] is None:
                raise ValueError('Complete all screening candidates and require one eligible candidate first')
            candidate = next(item for item in c['candidates'] if item['name'] == report['selected'])
            selection = dict(candidate=candidate, confirmation_seeds=c['confirmation_seeds'], plan_sha256=ccr.digest(out / 'plan.json'))
            ccr.immutable(out / 'selection.json', ccr.encoded(selection))
            if args.phase == 'confirm':
                for seed in c['confirmation_seeds']:
                    run_candidate(plan, candidate, seed, out / 'confirm' / candidate['name'] / str(seed), c['epochs'])
                    summarize(plan, out)
            else:
                if report['status'] != 'passed_operational_checks':
                    raise ValueError('All confirmation seeds must pass before final weather test reporting')
                test_plan = evaluation_plan(plan, test=True)
                ccr.evaluate_checkpoint(test_plan, Path(c['checkpoint']), out / 'test/initial')
                for seed in c['confirmation_seeds']:
                    checkpoint = out / 'confirm' / candidate['name'] / str(seed) / 'checkpoint.pt'
                    ccr.evaluate_checkpoint(test_plan, checkpoint, out / 'test' / str(seed))
        report = summarize(plan, out)
        print(f"Status: {report['status']}; report: {out / 'report.md'}", flush=True)


if __name__ == '__main__':
    main()
