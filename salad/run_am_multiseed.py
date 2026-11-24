#!/usr/bin/env python3
"""Sequential A-only selection, then frozen independent A/M seeds. Opt-in GPU."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import sys

import run_real_baseline as baseline
import run_qwen_finetune as qwen
from workflow.training_data import MixedGSVCitiesDataset, CoveragePlaceBatchSampler
from workflow.am_statistics import paired, holm, query_ranks

ccr = baseline.ccr
STAGES = ('batch_size', 'learning_rate', 'num_trainable_blocks')
ALLOWED = dict(batch_size={8, 16, 32}, learning_rate={1e-6, 3e-6, 1e-5}, num_trainable_blocks={1, 2, 4})


def immutable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    ccr.immutable(path, content)


def load_config(path):
    c = baseline.load_config(path)
    for key in ('synthetic_manifest', 'seed_protocol', 'group_manifest'):
        if c.get(key):
            c[key] = str((path.parent / Path(c[key]).expanduser()).resolve())
    parent = baseline.load_config(Path(c['seed_protocol']))
    if c['confirmation_seeds'] != parent['confirmation_seeds'] or c['screen_seed'] != parent['screen_seed']:
        raise ValueError('Keep the Stage 2 seed protocol; screening and formal seeds must be distinct')
    if c['reserved_places'] != parent['reserved_places'] or c['checkpoint'] != parent['checkpoint']:
        raise ValueError('Keep Stage 2 reservations and initial checkpoint')
    s = c['screening']
    if s['order'] != list(STAGES) or set(s['center']) != set(STAGES) or set(s['values']) != set(STAGES):
        raise ValueError('Selection order must be P, LR, blocks')
    for key in STAGES:
        values = s['values'][key]
        if (not values or len(set(values)) != len(values) or not set(values) <= ALLOWED[key]
                or s['center'][key] not in values):
            raise ValueError(f'Invalid finite candidates for {key}')
    if c['candidates'] != [candidate({**s['center'], 'batch_size': p}) for p in s['values']['batch_size']]:
        raise ValueError('Initial candidate metadata must match the P stage')
    if (c['methods'] != {'A': 0., 'M25': .25, 'M50': .5} or c['primary_method'] not in ('M25', 'M50')
            or c['images_per_place'] != 4 or c['precision'] not in ('bf16', '32')):
        raise ValueError('Protocol requires A/M25/M50, K=4 and BF16/FP32 (no FP16 skipped-step ambiguity)')
    if (not 0 < c['statistics']['alpha'] < 1 or type(c['statistics']['bootstrap_samples']) is not int
            or c['statistics']['bootstrap_samples'] < 100):
        raise ValueError('Invalid prespecified statistics')
    return c


def candidate(values):
    name = f"p{values['batch_size']}_lr{values['learning_rate']:.0e}_b{values['num_trainable_blocks']}".replace('-', 'm')
    return dict(name=name, **values)


def validation_inventory(c):
    """Never inspect test directories; natural-weather labels require actual val folders."""
    from workflow.evaluation import load_evaluation_set
    root = Path(c['svox_root'])
    val = (root / 'images/val').resolve()
    test = (root / 'images/test').resolve()
    if val == test or val.is_relative_to(test):
        raise ValueError('Validation resolves to test')
    protocols, summaries, missing = [], {}, []
    for domain in ('val', 'night', 'rain', 'snow', 'sun', 'overcast'):
        folder = 'queries' if domain == 'val' else 'queries_' + domain
        if domain != 'val' and not (val / folder).is_dir():
            missing.append(domain)
            continue
        p = dict(name='svox_val' if domain == 'val' else 'svox_val_' + domain,
                 dataset='SVOX', root=str(root), split='val', role='validation', cadence='every',
                 kind='real' if domain == 'val' else 'natural_weather', query_subdirs=[folder])
        data = load_evaluation_set('SVOX', root, split='val', query_subdirs=[folder])
        identities = []
        for item in data.references + data.queries:
            actual = item.path.resolve()
            if not actual.is_relative_to(val) or actual.is_relative_to(test):
                raise ValueError(f'Validation image outside independent val tree: {actual}')
            identities.append((item.id, str(actual), actual.stat().st_size, actual.stat().st_mtime_ns))
        summaries[p['name']] = {**data.summary(), 'query_ids': [q.id for q in data.queries],
            'inventory_sha256': hashlib.sha256(ccr.encoded(identities).encode()).hexdigest()}
        protocols.append(p)
    return dict(evaluations=protocols, summaries=summaries, missing_natural_weather_validation=missing)


def prepare(c, out):
    receipt = qwen.prepare_inputs(c, out)
    # Reuse Stage 2's exclusion checks and full-real metadata audit, untouched.
    pool = baseline.prepare(c, out / 'real_pool')
    excluded = {tuple(row) for row in json.loads(Path(pool['excluded_places']).read_text())}
    snapshot = out / 'inputs/accepted_manifest.jsonl'
    dataset = MixedGSVCitiesDataset(Path(c['real_data']), synthetic_manifest=snapshot,
        images_per_place=4, min_images_per_place=4, synthetic_fraction=.5, real_sampling='coverage_pk')
    places = [p for p in dataset.places if (p.city, p.place_id) not in excluded]
    if len(places) != pool['train_places'] or sum(len(p.real_paths) for p in places) != pool['train_real_images']:
        raise ValueError('A/M real pools differ')
    keys = {(p.city, p.place_id) for p in places}
    filtered, counts, dropped = [], Counter(), Counter()
    for line in snapshot.read_text().splitlines():
        row = json.loads(line)
        key = dataset.source_index[Path(row['source_path'])]
        if key not in keys:
            dropped['reserved' if key in excluded else 'ineligible_real_pool'] += 1
            continue
        filtered.append(line)
        counts[row['condition'].lower()] += 1
    if not filtered:
        raise ValueError('No generated images remain outside reservations')
    manifest = out / 'train_manifest.jsonl'
    immutable(manifest, '\n'.join(filtered) + '\n')
    validation = validation_inventory(c)
    grouping = None
    if c.get('group_manifest'):
        grouping = json.loads(Path(c['group_manifest']).read_text())
        if not grouping.get('provenance') or not isinstance(grouping.get('domains'), dict):
            raise ValueError('Group manifest needs provenance and explicit domain/query/group mappings')
        if not set(grouping['domains']) <= set(validation['summaries']):
            raise ValueError('Groups reference unavailable validation domains')
        for domain, groups in grouping['domains'].items():
            if set(groups) != set(validation['summaries'][domain]['query_ids']) or any(not isinstance(g, str) or not g for g in groups.values()):
                raise ValueError('Incomplete/invalid query grouping')
    plan = {**pool, **validation, 'train_manifest': str(manifest), 'grouping': grouping,
        'synthetic_snapshot': dict(accepted=receipt['accepted_images'], train_images=len(filtered),
            excluded=dict(dropped), weather=dict(counts), places_with_synthetic=sum(bool(p.synthetic_paths) for p in places)),
        'expected_exposure_per_epoch': {
            method: dict(synthetic=sum(min(math.floor(4*fraction),len(p.synthetic_paths),3) for p in places),
                         total=4*len(places), nominal_fraction=fraction)
            for method, fraction in c['methods'].items()},
        'real_pool_sha256': hashlib.sha256(ccr.encoded([(p.city, p.place_id, [str(v) for v in p.real_paths]) for p in places]).encode()).hexdigest(),
        'selection_policy': 'A only; P then LR then blocks; one full epoch each under fixed total cosine schedule; training gates then highest SVOX val R@1; exact ties preserve candidate order',
        'formal_policy': 'Independent initialization per method/seed; identical final full-epoch optimizer budget; no test access',
        'limitations': ['Validation has been used for selection; formal seed replication is not an unseen-domain test.',
            'No natural-weather validation claim is possible for missing domains. Synthetic weather is never natural validation.',
            'MS mining activity is a signal indicator, not a proof of nonzero gradients for every parameter.',
            'M replaces real slots: equal optimizer budget and total images, not equal real image exposure.',
            'Reservations govern fine-tuning only; pretraining/geographic overlap is not certified.',
            'McNemar assumes independent queries; group bootstrap is available only with actual supplied group metadata.',
            'No pooled-query p-value across seeds: trained seeds share evaluation queries.']}
    for path in (manifest, Path(c['seed_protocol']), *( [Path(c['group_manifest'])] if c.get('group_manifest') else [])):
        plan['input_hashes'][str(path)] = ccr.digest(path)
    for path in (Path(__file__), ccr.ROOT / 'train_am.py', Path(baseline.__file__), Path(qwen.__file__)):
        plan['source_hashes'][str(path)] = ccr.digest(path)
    immutable(out / 'plan.json', ccr.encoded(plan))
    return plan


def validate_plan(plan, out):
    baseline.validate_plan(plan)
    qwen.verify_snapshot(out / 'inputs', plan['config'])
    current = validation_inventory(plan['config'])
    if any(current[key] != plan[key] for key in current):
        raise ValueError('Frozen validation protocol/inventory changed')


def ep_plan(plan):
    return dict(config=plan['config'], evaluations=plan['evaluations'])


def run_plan(plan, spec):
    return {**plan, 'steps_per_epoch': {spec['name']: len(CoveragePlaceBatchSampler(plan['train_places'], spec['batch_size'], 0))}}


def command(plan, spec, seed, method, directory, stop_epoch, resume=False):
    cmd = baseline.training_command(plan, spec, seed, directory, stop_epoch, resume)
    cmd[1] = str(ccr.ROOT / 'train_am.py')
    cmd[cmd.index('--synthetic-fraction') + 1] = str(plan['config']['methods'][method])
    if method != 'A':
        cmd += ['--synthetic-manifest', plan['train_manifest']]
    return cmd


def metadata_digest(c, manifest=None):
    digest = hashlib.sha256()
    paths = sorted((Path(c['real_data']) / 'Dataframes').glob('*.csv'))
    if manifest:
        paths.append(Path(manifest))
    for path in paths:
        digest.update(str(path).encode() + b'\0')
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def checked_state(saved, plan, spec, seed, method):
    c, training = plan['config'], saved['training_config']
    if (training.get('synthetic_fraction') != c['methods'][method]
            or training.get('synthetic_manifest') != (None if method == 'A' else plan['train_manifest'])
            or training.get('synthetic_mode') != 'mix' or training.get('num_workers') != c['num_workers']
            or training.get('metadata_sha256') != metadata_digest(c, None if method == 'A' else plan['train_manifest'])
            or training.get('max_steps', 0) != 0 or training.get('cities') is not None
            or training.get('min_images_per_place') != c['images_per_place']):
        raise ValueError('Wrong method/manifest/workers in resumed checkpoint')
    if not all(key in saved for key in ('scheduler_state_dict', 'rng_state', 'sampling_coverage', 'scaler_state_dict')):
        raise ValueError('Resume lacks scheduler/RNG/sampling/scaler state')
    scheduler = saved['scheduler_state_dict']
    expected_lr = spec['learning_rate'] * (1 + math.cos(math.pi*saved['epoch']/c['epochs']))/2
    if (scheduler.get('T_max') != c['epochs'] or scheduler.get('last_epoch') != saved['epoch']
            or scheduler.get('base_lrs') != [spec['learning_rate']]
            or not math.isclose(saved['optimizer_state_dict']['param_groups'][0]['lr'], expected_lr, abs_tol=1e-15)
            or not {'python', 'torch', 'numpy', 'loader', 'cuda'} <= saved['rng_state'].keys()):
        raise ValueError('Scheduler/LR/RNG state differs from the frozen schedule')
    # Reuse pure-real validation after checking the ONLY allowed differences.
    shadow = {**saved, 'training_config': {**training, 'synthetic_fraction': 0., 'synthetic_manifest': None}}
    baseline.checked_state(shadow, run_plan(plan, spec), spec, seed)


def validate_metrics(metrics, plan, spec, method):
    real, syn = metrics['real_exposure'], metrics['synthetic_exposure']
    if real + syn != plan['real_exposures_per_epoch'] or (method == 'A' and syn):
        raise ValueError('Incomplete/wrong A/M exposure budget')
    if (metrics['batches'] != run_plan(plan, spec)['steps_per_epoch'][spec['name']]
            or sum(metrics['sampling']['synthetic_weather_exposures'].values()) != syn):
        raise ValueError('Wrong update budget or weather accounting')
    if not math.isfinite(metrics['loss']):
        raise ValueError('Nonfinite MS loss')


def read_result(plan, directory, checkpoint):
    results = {}
    checksum = ccr.digest(checkpoint)
    for protocol in plan['evaluations']:
        result = baseline.verified_evaluation(directory / (protocol['name'] + '.json'), checksum, protocol)
        if result is None:
            return None
        query_ranks(result)
        results[protocol['name']] = result
    return results


def run_one(plan, spec, seed, method, directory, target_epoch):
    from workflow.model import read_checkpoint
    directory.mkdir(parents=True, exist_ok=True)
    c = plan['config']
    contract = dict(spec=spec, seed=seed, method=method, plan_sha256=ccr.digest(Path(c['output_dir']) / 'plan.json'),
        init_checkpoint_sha256=plan['input_hashes'][c['checkpoint']], total_epochs=c['epochs'],
        total_updates=c['epochs'] * run_plan(plan, spec)['steps_per_epoch'][spec['name']])
    immutable(directory / 'contract.json', ccr.encoded(contract))
    checkpoint, previous = directory / 'checkpoint.pt', -1
    while True:
        epoch = 0
        if checkpoint.exists():
            saved = read_checkpoint(checkpoint)
            checked_state(saved, plan, spec, seed, method)
            epoch = saved['epoch']
            if epoch <= previous or epoch > target_epoch:
                raise ValueError('Unexpected epoch progress/budget')
            metrics = saved['metrics']
            validate_metrics(metrics, plan, spec, method)
            metrics = {**metrics, 'learning_rate_used': spec['learning_rate'] * (1 + math.cos(math.pi*(epoch-1)/c['epochs']))/2}
            immutable(directory / f'metrics_epoch_{epoch:03d}.json', ccr.encoded(metrics))
            del saved
        if epoch == target_epoch:
            ccr.evaluate_checkpoint(ep_plan(plan), checkpoint, directory / 'final_validation')
            ccr.run_child([sys.executable, str(Path(__file__).resolve()), '--audit-run', str(directory),
                           '--config', str(Path(c['output_dir']) / 'resolved_config.json')],
                          directory / 'audit.log', c['torch_threads'], limits=c['resource_limits'])
            return
        previous = epoch
        ccr.run_child(command(plan, spec, seed, method, directory, epoch+1, checkpoint.exists()),
                      directory / 'train.log', c['torch_threads'], limits=c['resource_limits'])
        if not checkpoint.exists():
            raise RuntimeError('Trainer returned without checkpoint')


def audit_run(plan, directory):
    """CPU-only checkpoint displacement; one model tensor difference at a time."""
    import torch
    from workflow.model import read_checkpoint, checkpoint_state_and_config
    checkpoint = directory / 'checkpoint.pt'
    checksum = ccr.digest(checkpoint)
    dest = directory / 'audit.json'
    if dest.exists() and json.loads(dest.read_text()).get('checkpoint_sha256') == checksum:
        return
    contract = json.loads((directory / 'contract.json').read_text())
    saved = read_checkpoint(checkpoint)
    checked_state(saved, plan, contract['spec'], contract['seed'], contract['method'])
    initial, _ = checkpoint_state_and_config(read_checkpoint(plan['config']['checkpoint']))
    state = saved['state_dict']
    if initial.keys() != state.keys():
        raise ValueError('Initial/final model keys differ')
    displacement = {}
    with torch.no_grad():
        for branch in ('backbone', 'aggregator'):
            square = norm = 0.
            for name, tensor in state.items():
                if name.startswith(branch + '.') and tensor.is_floating_point():
                    a, b = tensor.double(), initial[name].double()
                    square += float((a-b).square().sum())
                    norm += float(b.square().sum())
            displacement[branch] = dict(l2=math.sqrt(square), initial_l2=math.sqrt(norm),
                                        relative_l2=math.sqrt(square/norm) if norm else None)
    rows = [json.loads((directory / f'metrics_epoch_{epoch:03d}.json').read_text()) for epoch in range(1, saved['epoch']+1)]
    resources = []
    for line in (directory / 'train.log').read_text().splitlines():
        if line.startswith('RESOURCE '):
            resources.append(json.loads(line[9:]))
    ccr.write_json(dest, dict(checkpoint_sha256=checksum, initial_sha256=plan['input_hashes'][plan['config']['checkpoint']],
        optimizer_updates=saved['global_step'], actual_epochs=saved['epoch'], parameter_displacement=displacement,
        lr_curve=[dict(epoch=r['epoch'], used=r['learning_rate_used'], next=r['learning_rate_next_epoch']) for r in rows],
        real_exposures=sum(r['real_exposure'] for r in rows), synthetic_exposures=sum(r['synthetic_exposure'] for r in rows),
        weather=dict(sum((Counter(r['sampling']['synthetic_weather_exposures']) for r in rows), Counter())),
        training_seconds_completed_epochs=sum(r['seconds'] for r in rows),
        resource_attempts=resources, resource_scope='process-tree RSS is sampled, shared pages may be counted twice; interrupted attempts retained',
        dataset_summary=saved['dataset_summary'], training_config=saved['training_config'], epoch_metrics=rows))


def screening_candidates(plan, stage, prior):
    if stage not in STAGES or len(prior) != STAGES.index(stage):
        raise ValueError('Finish P, then LR, then blocks; no skipping stages')
    center = dict(plan['config']['screening']['center'])
    for record in prior:
        center = {key: record['selected'][key] for key in STAGES}
    return [candidate({**center, stage: value}) for value in plan['config']['screening']['values'][stage]]


def selection_record(plan, out, stage, prior):
    initial = read_result(plan, out / 'initial', Path(plan['config']['checkpoint']))
    if initial is None:
        return None
    rows = []
    for spec in screening_candidates(plan, stage, prior):
        directory = out / 'screen' / spec['name']
        metrics_path = directory / 'metrics_epoch_001.json'
        if not metrics_path.exists():
            return None
        results = read_result(plan, directory / 'final_validation', directory / 'checkpoint.pt')
        if results is None:
            return None
        metrics = json.loads(metrics_path.read_text())
        failures = baseline.signal_failures(metrics, plan)
        delta = 100 * (results['svox_val']['recall']['R@1'] - initial['svox_val']['recall']['R@1'])
        if delta < -plan['config']['acceptance']['max_val_drop_pp']:
            failures.append('validation_recall_drop')
        rows.append(dict(spec=spec, failures=failures, delta_initial_pp=delta,
                         validation_recall=results['svox_val']['recall'], metrics=metrics))
    eligible = [row for row in rows if not row['failures']]
    return dict(stage=stage, rows=rows, selected=max(eligible, key=lambda row: row['validation_recall']['R@1'])['spec'] if eligible else None,
                rule=plan['selection_policy'], screening_seed=plan['config']['screen_seed'])


def frozen_selection(plan, out):
    prior = []
    for stage in STAGES:
        record = selection_record(plan, out, stage, prior)
        if record is None or record['selected'] is None:
            raise ValueError(f'Stage {stage} incomplete or no eligible candidate')
        immutable(out / 'selection' / (stage+'.json'), ccr.encoded(record))
        prior.append(record)
    spec = prior[-1]['selected']
    formal = dict(spec=spec, seeds=plan['config']['confirmation_seeds'], methods=plan['config']['methods'],
        total_epochs=plan['config']['epochs'], updates_per_epoch=run_plan(plan, spec)['steps_per_epoch'][spec['name']],
        plan_sha256=ccr.digest(out / 'plan.json'), policy=plan['formal_policy'],
        cosine='CosineAnnealingLR T_max=total_epochs, one step per completed full epoch; identical in A/M',
        primary_method=plan['config']['primary_method'])
    formal['optimizer_updates'] = formal['total_epochs'] * formal['updates_per_epoch']
    immutable(out / 'formal_protocol.json', ccr.encoded(formal))
    commands = [shlex.join([sys.executable, str(Path(__file__).resolve()), '--config', str(out/'resolved_config.json'),
                '--phase', 'train', '--method', method, '--seed', str(seed), '--execute'])
                for seed in formal['seeds'] for method in formal['methods']]
    immutable(out / 'formal_commands.txt', '\n'.join(commands)+'\n')
    return formal


def report(plan, out):
    c = plan['config']
    result = dict(status='prepared_not_trained', screening=[], runs=[], paired_statistics=[], seed_distributions=[],
                  missing_natural_weather_validation=plan['missing_natural_weather_validation'], limitations=plan['limitations'])
    prior = []
    for stage in STAGES:
        record = selection_record(plan, out, stage, prior)
        if record is None:
            break
        result['screening'].append(record)
        if record['selected'] is None:
            result['status'] = 'screening_failed'
            break
        prior.append(record)
    formal_path = out / 'formal_protocol.json'
    if formal_path.exists():
        formal = json.loads(formal_path.read_text())
        if len(prior) != 3 or formal['spec'] != prior[-1]['selected'] or formal['plan_sha256'] != ccr.digest(out / 'plan.json'):
            raise ValueError('Formal selection differs from frozen validation decisions')
        result['formal_protocol'] = formal
        initial = read_result(plan, out / 'initial', Path(c['checkpoint']))
        lookup = {}
        for seed in c['confirmation_seeds']:
            for method in c['methods']:
                directory = out / 'runs' / str(seed) / method
                path = directory / 'audit.json'
                if not path.exists():
                    continue
                audit = json.loads(path.read_text())
                if audit['checkpoint_sha256'] != ccr.digest(directory / 'checkpoint.pt'):
                    raise ValueError('Audit checkpoint changed')
                if audit['optimizer_updates'] != formal['optimizer_updates'] or audit['actual_epochs'] != formal['total_epochs']:
                    raise ValueError('Formal run has wrong final budget')
                evaluations = read_result(plan, directory / 'final_validation', directory / 'checkpoint.pt')
                if evaluations is None:
                    continue
                lookup[seed, method] = evaluations
                real, syn = audit['real_exposures'], audit['synthetic_exposures']
                failures = [f"epoch_{r['epoch']}:{why}" for r in audit['epoch_metrics'] for why in baseline.signal_failures(r, plan)] if method == 'A' else []
                result['runs'].append(dict(seed=seed, method=method, audit=str(path),
                    checkpoint_sha256=audit['checkpoint_sha256'], optimizer_updates=audit['optimizer_updates'],
                    actual_epochs=audit['actual_epochs'], real_exposures=real, synthetic_exposures=syn,
                    nominal_fraction=c['methods'][method], actual_fraction=syn/(real+syn), weather=audit['weather'],
                    signal_failures=failures, recalls={d:r['recall'] for d,r in evaluations.items()}))
        for seed in c['confirmation_seeds']:
            for method in c['methods']:
                if (seed, method) not in lookup:
                    continue
                comparisons = [('initial', initial)]
                if method != 'A' and (seed, 'A') in lookup:
                    comparisons.append(('A', lookup[seed, 'A']))
                for reference, reference_results in comparisons:
                    for domain, trial in lookup[seed, method].items():
                        groups = (plan.get('grouping') or {}).get('domains', {}).get(domain)
                        for k in (1, 5, 10):
                            result['paired_statistics'].append(dict(seed=seed, method=method, reference=reference, domain=domain, recall_k=k,
                                **paired(reference_results[domain], trial, k, groups,
                                    c['statistics']['bootstrap_samples'], c['statistics']['bootstrap_seed'])))
        # One prespecified family: every seed/domain/recall cutoff, all methods
        # vs initial plus M variants vs A. Missing planned tests remain p=1.
        family_size = len(c['confirmation_seeds']) * len(plan['evaluations']) * 3 * (2*len(c['methods'])-1)
        adjusted = holm([r['p_exact'] for r in result['paired_statistics']], family_size)
        for row, value in zip(result['paired_statistics'], adjusted):
            row.update(p_holm=value, significant_holm=value < c['statistics']['alpha'])
        result['holm_family_size'] = family_size
        for method in c['methods']:
            if method == 'A':
                continue
            for domain in plan['summaries']:
                for k in (1, 5, 10):
                    rows = [r for r in result['paired_statistics'] if r['method']==method and r['reference']=='A' and r['domain']==domain and r['recall_k']==k]
                    if rows:
                        values = [r['delta_pp'] for r in rows]
                        result['seed_distributions'].append(dict(method=method, domain=domain, recall_k=k,
                            deltas_by_seed={str(r['seed']):r['delta_pp'] for r in rows}, mean_pp=statistics.mean(values),
                            std_pp=statistics.stdev(values) if len(values)>1 else None, min_pp=min(values), max_pp=max(values),
                            positive_seeds=sum(v>0 for v in values), complete=len(rows)==len(c['confirmation_seeds']),
                            all_seeds_positive=len(rows)==len(c['confirmation_seeds']) and min(values)>0))
        result['status'] = 'formal_complete_descriptive' if len(result['runs']) == len(c['confirmation_seeds'])*len(c['methods']) else 'formal_incomplete'
        a_rows = [r for r in result['runs'] if r['method'] == 'A']
        if a_rows:
            values = [100*r['recalls']['svox_val']['R@1'] for r in a_rows]
            deltas = [v - 100*initial['svox_val']['recall']['R@1'] for v in values]
            std = statistics.stdev(values) if len(values)>1 else None
            result['A_assessment'] = dict(complete=len(a_rows)==len(c['confirmation_seeds']),
                mean_R1_percent=statistics.mean(values), seed_std_pp=std,
                min_delta_initial_pp=min(deltas), signal_failures_by_seed={str(r['seed']):r['signal_failures'] for r in a_rows},
                passes_engineering_gates=len(a_rows)==len(c['confirmation_seeds'])
                    and not any(r['signal_failures'] for r in a_rows)
                    and std <= c['acceptance']['max_seed_std_pp']
                    and min(deltas) >= -c['acceptance']['max_val_drop_pp'])
    elif len(prior) == 3:
        result['status'] = 'ready_to_freeze'
    elif prior:
        result['status'] = 'screening_incomplete'
    ccr.write_json(out / 'report.json', result)
    for name, rows in (('paired_statistics', result['paired_statistics']), ('seed_metrics', result['runs'])):
        if rows:
            fields = list(rows[0])
            with (out / (name+'.csv')).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    lines = ['# CCR stage 3 A/M multiseed', '', 'Status: '+result['status'], '',
        'No SVOX test is loaded. Missing natural-weather validation: '+', '.join(result['missing_natural_weather_validation']), '',
        '| Seed | Method | Actual synthetic | Val R@1 | A signal failures |', '|---|---|---:|---:|---|']
    for row in result['runs']:
        lines.append(f"| {row['seed']} | {row['method']} | {row['actual_fraction']:.2%} | {row['recalls']['svox_val']['R@1']:.4%} | {', '.join(row['signal_failures']) or 'none'} |")
    lines += ['', '| Method vs A | Domain | Recall | Mean Δ pp | Seed std pp | Positive seeds |', '|---|---|---|---:|---:|---:|']
    for row in result['seed_distributions']:
        lines.append(f"| {row['method']} | {row['domain']} | R@{row['recall_k']} | {row['mean_pp']:+.4f} | {row['std_pp']} | {row['positive_seeds']} |")
    lines += ['', 'Completion is not evidence of benefit. Inspect A signal gates, paired seed deltas and Holm-adjusted tests.',
              'Detailed selection decisions, query counts, paired changes and conditional group intervals: report.json / paired_statistics.csv.', '', *plan['limitations']]
    (out / 'report.md').write_text('\n'.join(lines)+'\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ccr.ROOT / 'configs/am_multiseed.json')
    p.add_argument('--phase', choices=('prepare', 'screen-p', 'screen-lr', 'screen-blocks', 'freeze', 'train', 'report'), default='prepare')
    p.add_argument('--method', choices=('A', 'M25', 'M50'))
    p.add_argument('--seed', type=int)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--audit-run', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    c = load_config(args.config.resolve())
    out = Path(c['output_dir'])
    if args.audit_run:
        audit_run(json.loads((out / 'plan.json').read_text()), args.audit_run)
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'plan.json').exists():
            plan = json.loads((out / 'plan.json').read_text())
            if c != plan['config']:
                raise ValueError('Configuration changed; use a new output_dir')
            if args.phase != 'report':
                validate_plan(plan, out)
        else:
            plan = prepare(c, out)
        immutable(out / 'resolved_config.json', ccr.encoded(c))
        if (args.method or args.seed is not None) and args.phase != 'train':
            raise ValueError('--method/--seed only select formal runs')
        if args.seed is not None and args.seed not in c['confirmation_seeds']:
            raise ValueError('Seed not in frozen formal protocol')
        if args.phase == 'freeze':
            frozen_selection(plan, out)
        elif args.execute and args.phase.startswith('screen-'):
            stage = {'screen-p':STAGES[0], 'screen-lr':STAGES[1], 'screen-blocks':STAGES[2]}[args.phase]
            prior = []
            for earlier in STAGES[:STAGES.index(stage)]:
                record = selection_record(plan, out, earlier, prior)
                if record is None or record['selected'] is None:
                    raise ValueError(f'Complete eligible {earlier} stage first')
                immutable(out / 'selection' / (earlier+'.json'), ccr.encoded(record))
                prior.append(record)
            ccr.evaluate_checkpoint(ep_plan(plan), Path(c['checkpoint']), out / 'initial')
            for spec in screening_candidates(plan, stage, prior):
                run_one(plan, spec, c['screen_seed'], 'A', out / 'screen' / spec['name'], 1)
            record = selection_record(plan, out, stage, prior)
            immutable(out / 'selection' / (stage+'.json'), ccr.encoded(record))
        elif args.execute and args.phase == 'train':
            formal = frozen_selection(plan, out)
            for seed in c['confirmation_seeds']:
                if args.seed is not None and seed != args.seed:
                    continue
                for method in c['methods']:
                    if args.method and method != args.method:
                        continue
                    run_one(plan, formal['spec'], seed, method, out / 'runs' / str(seed) / method, c['epochs'])
                    report(plan, out)
        summary = report(plan, out)
        print(f"Status: {summary['status']}; plan={out / 'plan.json'}; report={out / 'report.md'}")
        if not args.execute:
            print('No GPU execution requested; prepare/freeze/report only.')


if __name__ == '__main__':
    main()
