"""Repair lost remote queue publication without reassigning any image jobs.

The frozen controller reserves ownership and then publishes the queue. If that
last transfer fails, its outstanding-work limit prevents a new grant and never
retries the publication. Reconcile the remote prefix before each remote grant.
Frozen plans, models, worker code, existing ownership and POST journals stay intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import night_original_images as original

base=original.base
cluster=base.cluster


def remote_queue(root):
    script='import sys;from pathlib import Path;p=Path(sys.argv[1]);print(p.read_text() if p.exists() else "null")'
    raw=cluster.remote(cluster.REMOTE_PY,'-c',script,str(root/'nodes/gpu2/queue.json'))
    value=json.loads(raw)
    if value is None:return None
    if value!=cluster.sealed(value) or value.get('node')!='gpu2' or not isinstance(value.get('entries'),list):
        raise ValueError('Remote queue seal or node identity differs')
    return value


def validate_prefix(local,remote):
    observed=[] if remote is None else remote['entries']
    if local.get('node')!='gpu2' or observed!=local['entries'][:len(observed)]:
        raise ValueError('Remote queue is not an exact prefix; refusing overwrite or reassignment')
    return local['entries'][len(observed):]


def verify_staged(entries):
    files={}
    for entry in entries:
        state=Path(entry['state'])
        if cluster.common.file_sha256(state/'plan_config.json')!=entry['config_sha256']:
            raise ValueError('Assigned local config changed')
        for name in ('plan_config.json','plan.jsonl','planner_config.json','coverage_snapshot.json',
                     'source_snapshot.jsonl','prompt_receipts.jsonl'):
            path=state/name;files[str(path)]=cluster.common.file_sha256(path)
        for job in cluster.common.read_jsonl(state/'plan.jsonl'):
            files[job['source_path']]=job['source_sha256']
    script=('import json,hashlib,sys;from pathlib import Path;values=json.loads(sys.argv[1]);'
            'bad=[p for p,h in values.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];'
            'print(json.dumps({"checked":len(values),"bad":bad}));sys.exit(bool(bad))')
    cluster.remote(cluster.REMOTE_PY,'-c',script,json.dumps(files),timeout=120)


def reconcile(root):
    local=cluster.queue(root,'gpu2');remote=remote_queue(root)
    missing=validate_prefix(local,remote)
    if not missing:return 0
    # Missing staged inputs must be investigated before the worker can see a
    # new queue. Never publish a task with absent or changed source images.
    verify_staged(missing)
    cluster.rsync_files(root,[root/'nodes/gpu2/queue.json'])
    delivered=remote_queue(root)
    if delivered!=local:
        raise RuntimeError('Remote queue publication not confirmed; keep ownership and retry next cycle')
    receipt=cluster.sealed(dict(event='reconciled_remote_queue_prefix',timestamp=datetime.now(timezone.utc).isoformat(),
        queue_fingerprint=local['fingerprint'],previous_entries=len(local['entries'])-len(missing),
        entries=len(local['entries']),published_entries=missing,
        adapter_sha256=cluster.common.file_sha256(__file__),ownership_unchanged=True))
    cluster.common.write_json(root/'deployment/queue_deliveries'/f'{time.time_ns()}.json',receipt)
    print(json.dumps(dict(queue_delivery_repaired=len(missing),remote_entries=len(local['entries']))),flush=True)
    return len(missing)


def install(root):
    original.configure()
    cfg=base.load(root)
    cluster.freeze(root/'queue_delivery_policy.json',dict(version='queue_prefix_reconciliation_v1',
        cluster_fingerprint=cfg['fingerprint'],adapter_sha256=cluster.common.file_sha256(__file__),
        policy='Retry publication of assigned queue prefix; verify staged inputs; never reassign or resubmit images'))
    grant=base.grant
    def reliable_grant(root,node):
        if node=='gpu2':reconcile(root)
        return grant(root,node)
    base.grant=reliable_grant


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root',type=Path,default=original.ROOT)
    args,_=parser.parse_known_args()
    install(args.root.resolve())
    original.main()


if __name__=='__main__':main()
