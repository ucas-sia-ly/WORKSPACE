"""Execute replacement prompts with the unchanged original image engine."""
from __future__ import annotations

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import night_original_images as original
from experiments.qwen_curriculum import night_mercury_prompts as prompts

base = original.base
ROOT = prompts.ROOT
VERSION = prompts.VERSION


def validate_proof(job, proof):
    cfg = proof.get('prompt_config', {})
    prompts.validate_receipt(proof, proof['input'], cfg)
    if proof['job'] != job:
        raise ValueError('Image job differs from its prompt receipt')


def prepare(root=ROOT):
    cfg = prompts.load(root); state = root/'state'
    complete = prompts.cohort.sealed_read(state/'prompts_complete.json')
    if (complete['config_fingerprint'] != cfg['fingerprint'] or complete['ready'] != cfg['num_places']
            or base.common.file_sha256(state/'ready.jsonl') != complete['ready_sha256']):
        raise ValueError('Finish and validate all replacement prompts first')
    (root/'deployment').mkdir(parents=True, exist_ok=True)
    files = {str(p): base.common.file_sha256(p) for p in base.code_paths()}
    files.update(cfg['files_sha256'])
    files[str(Path(__file__).resolve())] = base.common.file_sha256(__file__)
    definition = base.cluster.freeze(root/'cluster.json', dict(version=VERSION,
        source_total=cfg['num_places'], retained_old_images=0,
        files_sha256=files, source_fingerprint=cfg['fingerprint'],
        local_gate='complete original-policy prompt barrier and VLM process exit',
        storage='all images on primary; remote copies pruned only after SHA acknowledgement',
        primary_disk_reserve_gib=8, verification='deferred; never approved automatically'))
    if not (root/'ownership.json').exists():
        base.cluster.save(root/'ownership.json', dict(version=1, cluster_fingerprint=definition['fingerprint'], owners={}))
    if not (root/'feed.json').exists():
        base.cluster.save(root/'feed.json', dict(version=1, entries=[], predecessors={}))
    for node in ('local', 'gpu2'):
        directory = root/'nodes'/node; directory.mkdir(parents=True, exist_ok=True)
        previous = base.worker.read_sealed(prompts.OLD/'nodes'/node/'node.json')
        value = {k: v for k, v in previous.items() if k != 'fingerprint'}
        value['files_sha256'] = {**value['files_sha256'], **files}
        node_cfg = base.cluster.freeze(directory/'node.json', value)
        previous = base.worker.read_sealed(prompts.OLD/'nodes'/node/'resource_policy.json')
        value = {k: v for k, v in previous.items() if k != 'fingerprint'}
        value['node_fingerprint'] = node_cfg['fingerprint']
        base.cluster.freeze(directory/'resource_policy.json', value)
        if not (directory/'queue.json').exists():
            base.cluster.save(directory/'queue.json', dict(version=1, node=node, entries=[]))
        # Stay paused until the preceding queue has released this node's backend.
        if not (root/'prepared.json').exists():
            (directory/'pause_requested').touch()
    inputs = {i['source']['sample_id']: i for i in base.common.read_jsonl(state/'sources.jsonl')}
    feed = base.worker.read_sealed(root/'feed.json')
    for index, group in enumerate(prompts.cohort.chunks(prompts.cohort.rows(state/'ready.jsonl'), 32)):
        directory = root/'shards'/f's{index:05d}'/'state'; directory.mkdir(parents=True, exist_ok=True)
        proofs = []
        for record in group:
            item = inputs[record['source']['sample_id']]
            prompts.validate_receipt(record, item, cfg)
            proofs.append(base.run.sealed({**record, 'input': item, 'prompt_config': cfg}))
        jobs = [r['job'] for r in proofs]
        for name, rows in [('plan.jsonl', jobs), ('prompt_receipts.jsonl', proofs),
                           ('source_snapshot.jsonl', [p['source'] for p in proofs])]:
            path = directory/name
            if path.exists() and base.common.read_jsonl(path) != rows:
                raise ValueError('Frozen replacement shard changed')
            if not path.exists():
                base.common.write_jsonl(path, rows)
        base.campaign.frozen(directory/'planner_config.json', dict(version=VERSION, source_fingerprint=cfg['fingerprint']))
        base.campaign.frozen(directory/'coverage_snapshot.json', dict(excluded_style='mercury_vapor'))
        base.campaign.frozen(directory/'plan_config.json', dict(version=1, stage=VERSION,
            num_jobs=len(jobs), request_budget=len(jobs), domain_counts={'night': len(jobs)},
            image_root=str(root/'images'), quality=base.quality(),
            service=base.worker.read_sealed(root/'nodes/local/node.json')['service_identity'],
            plan_sha256=base.common.file_sha256(directory/'plan.jsonl'), files_sha256={**files,
                str(root/'cluster.json'): base.common.file_sha256(root/'cluster.json'),
                str(directory/'prompt_receipts.jsonl'): base.common.file_sha256(directory/'prompt_receipts.jsonl')}))
        descriptor = dict(original_state=str(directory),
            original_config_sha256=base.common.file_sha256(directory/'plan_config.json'), jobs=len(jobs), saved_at_creation=0)
        if index < len(feed['entries']):
            if feed['entries'][index] != descriptor:
                raise ValueError('Published replacement feed changed')
        else:
            feed['entries'].append(descriptor)
            feed['predecessors'].update({r['predecessor_sample_id']: r['job']['record_sha256'] for r in proofs})
            base.cluster.save(root/'feed.json', feed)
    if len(feed['predecessors']) != cfg['num_places']:
        raise ValueError('Incomplete replacement feed')
    base.cluster.freeze(root/'prepared.json', dict(total=cfg['num_places'], prompt_fingerprint=cfg['fingerprint']))
    return definition


def load_execution_plan(directory):
    directory = Path(directory); cfg = base.campaign.config_file(directory/'plan_config.json')
    if cfg.get('stage') != VERSION or cfg['quality'] != base.quality():
        raise ValueError('Wrong replacement execution policy')
    for name, digest in cfg['files_sha256'].items():
        if base.common.file_sha256(name) != digest:
            raise ValueError('Frozen execution dependency changed: '+name)
    if base.common.file_sha256(directory/'plan.jsonl') != cfg['plan_sha256']:
        raise ValueError('Replacement plan changed')
    jobs = base.common.read_jsonl(directory/'plan.jsonl')
    proofs = base.common.read_jsonl(directory/'prompt_receipts.jsonl')
    if (len(jobs) != cfg['num_jobs'] or len(proofs) != len(jobs)
            or len({j['sample_id'] for j in jobs}) != len(jobs) or cfg['domain_counts'] != {'night': len(jobs)}):
        raise ValueError('Replacement image budget changed')
    for job, proof in zip(jobs, proofs, strict=True):
        validate_proof(job, proof)
    return cfg, jobs


def configure():
    base.ROOT = ROOT; base.VERSION = VERSION; base.__file__ = __file__
    base.CONTROLLER = 'ccr-night-completion.service'
    base.PLANNER = 'ccr-night-mercury-prompts.service'
    base.unit = lambda node: f'ccr-night-mercury-{node}-worker.service'
    base.prepare = prepare; base.load_execution_plan = load_execution_plan
    base.prune_remote = original.prune_remote


def main():
    configure(); base.main()


if __name__ == '__main__':
    main()
