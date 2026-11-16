"""Replace retired mercury sources with the original 29-style night policy.

Saved source evidence is reused; only the assigned mechanism and prose change.
Text requests and per-source commits survive interruption. Historical files are
never rewritten, and no source is silently skipped after an invalid response.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path
import signal
import sys
import threading
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import (
    common, run, campaign, balanced_night, night_diversity_policy as policy,
    night_place_cohort as cohort, night_place_cohort_v4 as original,
    night_revision_planner as planner)

ROOT = common.WORKSPACE_ROOT/'outputs/night_mercury_replacement_v1'
SNAPSHOT = common.WORKSPACE_ROOT/'outputs/night_revision_v2/state'
OLD = common.WORKSPACE_ROOT/'outputs/night_original_no_mercury_v1'
VERSION = 'night_mercury_replacement_v1'


def dependencies():
    modules = (sys.modules[__name__], cohort, original, planner, balanced_night, policy)
    paths = [Path(m.__file__).resolve() for m in modules]+[policy.CATALOG_PATH]
    return {str(p): common.file_sha256(p) for p in paths}


def source_evidence(receipt):
    # V4 used a deterministic full-frame preservation envelope. It is not
    # semantic evidence, a landmark detection, or correspondence supervision.
    return original.source_evidence(copy.deepcopy(receipt['plan']['source_evidence']))


def allocate(tasks, receipts, coverage, seed=42):
    assigned = Counter(); seen = set(); records = []
    for task, receipt in zip(tasks, receipts, strict=True):
        source = task['source']; key = source['city'], int(source['place_id'])
        if (key in seen or task['task_reason'] != 'retired_mercury_style'
                or task['style_id'] != 'mercury_vapor' or receipt['source'] != source
                or receipt['job'] != task['original_job'] or receipt['status'] != 'ready'):
            raise ValueError('Replacement must preserve each unique retired mercury source')
        seen.add(key); evidence = source_evidence(receipt)
        eligible, excluded = policy.eligible_styles(evidence)
        eligible = [s for s in eligible if s != 'mercury_vapor']
        style = policy.select_styles(eligible, 1, coverage, list(assigned.elements()),
                                     seed, source['sample_id'])[0]
        assigned[style] += 1
        records.append(run.sealed(dict(source=source, evidence=evidence, style_id=style,
            predecessor_sample_id=task['sample_id'],
            original_record_sha256=task['original_record_sha256'],
            original_receipt_sha256=receipt['result_sha256'], excluded_styles=excluded)))
    return records


def prepare(root=ROOT):
    state = root/'state'; state.mkdir(parents=True, exist_ok=True)
    if (state/'config.json').exists():
        return load(root)
    inventory = cohort.sealed_read(SNAPSHOT/'inventory.json')
    if common.file_sha256(SNAPSHOT/'tasks.jsonl') != inventory['hashes']['tasks.jsonl']:
        raise ValueError('Retirement inventory changed')
    tasks = [t for t in cohort.rows(SNAPSHOT/'tasks.jsonl')
             if t['task_reason'] == 'retired_mercury_style']
    receipts = [cohort.sealed_read(common.WORKSPACE_ROOT/'outputs/night_place_fast_v1/state/prompts'/
                                 (t['source']['sample_id']+'.json')) for t in tasks]
    counts = Counter(r['style_id'] for r in cohort.rows(SNAPSHOT/'retained_manifest.jsonl'))
    # Account for the complete frozen pending queue, not just today's progress.
    feed = campaign.config_file(OLD/'feed.json')
    for shard in feed['entries']:
        counts.update(j['style_id'] for j in cohort.rows(Path(shard['original_state'])/'plan.jsonl'))
    counts.pop('mercury_vapor', None)
    coverage = {s: dict(saved=counts[s], pending=0, mismatch=0) for s in policy.STYLES}
    records = allocate(tasks, receipts, coverage)
    if len(records) != 2066:
        raise ValueError('Unexpected retired mercury inventory size')
    common.write_jsonl(state/'sources.jsonl', records)
    campaign.frozen(state/'coverage.json', dict(counts=coverage))
    model = Path('/home/admin123/github/checkpoints/Qwen3-VL-4B-Instruct')
    cfg = campaign.frozen(state/'config.json', dict(version=VERSION, seed=42,
        num_places=len(records), batch_size=64, compose_tokens=768,
        model_path=str(model), model_config_sha256=common.file_sha256(model/'config.json'),
        sources_sha256=common.file_sha256(state/'sources.jsonl'),
        coverage_sha256=common.file_sha256(state/'coverage.json'),
        files_sha256=dependencies(), policy=policy.definition(),
        inference='original BF16 Qwen3-VL; cached old source evidence; original V4 composer',
        recovery='reuse committed prose; retry interrupted text inference; never skip invalid sources',
        memory=dict(allocator_gib=42, min_host_available_gib=4, min_free_disk_gib=8)))
    common.write_json(state/'progress.json', dict(state='prepared', total=len(records), ready=0))
    return cfg


def load(root):
    state = root/'state'; cfg = campaign.config_file(state/'config.json')
    if cfg['version'] != VERSION:
        raise ValueError('Wrong replacement protocol')
    for name, digest in cfg['files_sha256'].items():
        if common.file_sha256(name) != digest:
            raise ValueError('Frozen prompt dependency changed: '+name)
    for name, field in [('sources.jsonl', 'sources_sha256'), ('coverage.json', 'coverage_sha256')]:
        if common.file_sha256(state/name) != cfg[field]:
            raise ValueError('Frozen replacement input changed')
    if common.file_sha256(Path(cfg['model_path'])/'config.json') != cfg['model_config_sha256']:
        raise ValueError('VLM model config changed')
    return cfg


def validate_receipt(record, item, cfg):
    job = record.get('job', {}); plan = record.get('plan', {})
    if (record != run.sealed(record) or record.get('status') != 'ready'
            or record.get('config_fingerprint') != cfg['fingerprint']
            or record.get('input_sha256') != item['result_sha256']
            or record.get('source') != item['source']
            or record.get('predecessor_sample_id') != item['predecessor_sample_id']):
        raise ValueError('Replacement receipt differs from its frozen input')
    policy.validate_plan(plan, 1)
    expected = balanced_night.make_jobs([item['source']], [plan], False, cfg['seed'])[0]
    if (job != expected or job['style_id'] != item['style_id']
            or job['style_id'] == 'mercury_vapor' or job['sample_id'] == item['predecessor_sample_id']):
        raise ValueError('Replacement job changed source, style, prompt or seed')
    return record


def receipt_path(state, item):
    return state/'prompts'/(item['source']['sample_id']+'.json')


def plan(root):
    state = root/'state'; stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    with run.run_lock(state):
        cfg = load(root); reader = planner.Planner(cfg['model_path'], policy=policy)
        items = common.read_jsonl(state/'sources.jsonl'); ready = 0; started = time.monotonic()
        try:
            for batch in cohort.chunks(items, cfg['batch_size']):
                pending = []
                for item in batch:
                    path = receipt_path(state, item)
                    if path.exists():
                        validate_receipt(cohort.sealed_read(path), item, cfg); ready += 1
                    else:
                        pending.append(item)
                feedback = {}
                for attempt in range(6):
                    if not pending or stopped.is_set() or (state/'pause_requested').exists():
                        break
                    # No image backend may share the primary GPU with this VLM.
                    import subprocess
                    pid = subprocess.check_output(['systemctl', '--user', 'show',
                        'ccr-night-local-backend.service', '-p', 'MainPID', '--value'], text=True).strip()
                    if int(pid or 0):
                        raise RuntimeError('Pause primary image generation before prompting')
                    from experiments.qwen_curriculum.night_cluster_worker import resource_sample
                    resources = resource_sample(0, [state])
                    if resources['available_gib'] < cfg['memory']['min_host_available_gib']:
                        raise RuntimeError('Prompt inference paused for insufficient available RAM')
                    requests = [original.compose_request(i['evidence'], i['style_id'],
                        feedback.get(i['source']['sample_id'], '')) for i in pending]
                    replies = reader.ask(state, [i['source'] for i in pending], requests, 'compose', attempt, cfg)
                    retry = []
                    for item, raw in zip(pending, replies, strict=True):
                        try:
                            value = cohort.parse_plan(raw, item['evidence'], item['style_id'])
                        except (ValueError, KeyError, TypeError) as exc:
                            feedback[item['source']['sample_id']] = 'Correct the complete JSON: '+str(exc)
                            retry.append(item); continue
                        job = balanced_night.make_jobs([item['source']], [value], False, cfg['seed'])[0]
                        record = run.sealed(dict(status='ready', config_fingerprint=cfg['fingerprint'],
                            input_sha256=item['result_sha256'], source=item['source'], plan=value, job=job,
                            predecessor_sample_id=item['predecessor_sample_id']))
                        validate_receipt(record, item, cfg)
                        common.write_json(receipt_path(state, item), record); ready += 1
                    pending = retry
                common.write_json(state/'progress.json', dict(state='paused' if stopped.is_set() else 'prompting',
                    total=len(items), ready=ready, elapsed_seconds=time.monotonic()-started))
                print(f'Original 29-style replacement prompts: {ready}/{len(items)}', flush=True)
                if stopped.is_set() or (state/'pause_requested').exists():
                    return
                if pending:
                    raise RuntimeError('Invalid prompts remain; saved successes retained, resume retries missing sources')
            records = [validate_receipt(cohort.sealed_read(receipt_path(state, i)), i, cfg) for i in items]
            common.write_jsonl(state/'ready.jsonl', records)
            common.write_json(state/'prompts_complete.json', run.sealed(dict(config_fingerprint=cfg['fingerprint'],
                ready=len(records), ready_sha256=common.file_sha256(state/'ready.jsonl'))))
            common.write_json(state/'progress.json', dict(state='complete', total=len(items), ready=len(records),
                elapsed_seconds=time.monotonic()-started))
        finally:
            reader.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'plan'))
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    if args.command == 'prepare':
        print(prepare(args.root.resolve()))
    else:
        plan(args.root.resolve())


if __name__ == '__main__':
    main()
