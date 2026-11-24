"""Measured batch-size successor; import committed prompts without regeneration.

Preserves v4 sources, prompt policy, precision and successful prose. New batch
membership changes future sampling draws, explicitly frozen in a new protocol.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import night_place_cohort_v4 as policy

engine=policy.engine
ROOT=engine.common.WORKSPACE_ROOT/'outputs/night_place_fast_v1'
VERSION='night_place_fast_v1'
_chunks=engine.chunks
_dependencies=policy.dependencies
_prefix=0
_prefix_batch=4


def source_groups(values,count,prefix,prefix_batch):
    """A sealed old prefix keeps its original groups; new work uses measured size."""
    iterator=iter(values)
    prefix_items=itertools.islice(iterator,prefix)
    yield from _chunks(prefix_items,prefix_batch)
    yield from _chunks(iterator,count)


def chunks(values,count):
    return source_groups(values,count,_prefix,_prefix_batch)


def dependencies(model):
    return {**_dependencies(model),str(Path(__file__).resolve()):engine.common.file_sha256(__file__)}


def configure(directory):
    global _prefix,_prefix_batch
    policy.configure()
    engine.ROOT,engine.VERSION=ROOT,VERSION
    engine.dependencies=dependencies
    cfg=engine.campaign.config_file(Path(directory)/'cohort_config.json')
    _prefix,_prefix_batch=cfg['imported_prefix'],cfg['imported_batch_size']
    engine.chunks=chunks


def choose_profile(report):
    if report.get('error') or report.get('guard_events'):
        raise ValueError('Profile failed; review it before choosing a batch size')
    def fits(r):
        return (r['cuda_peak_reserved_gib']<40 and r['resources']['gpu_used_gib']<42 and
                r['resources']['available_gib']>=6 and all(v['valid']==v['total'] for v in r['phases'].values()))
    stress=report.get('stress_validation',{})
    def stress_passed(r):
        if r['batch_size']<64: return True
        return (stress.get('largest_input_stress') is True and not stress.get('error') and
                not stress.get('guard_events') and
                any(v['batch_size']==r['batch_size'] and fits(v) for v in stress.get('records',[])))
    safe=[r for r in report['records'] if fits(r) and stress_passed(r)]
    if not safe: raise ValueError('No validated safe inference batch profile')
    return max(safe,key=lambda r:r['valid_complete_per_second'])


def migrate(parent,directory,profile_path):
    policy.configure()
    with engine.run.run_lock(parent),engine.run.run_lock(directory):
        old=engine.load(parent)
        report=json.loads(profile_path.read_text())
        if report['parent_fingerprint']!=old['fingerprint']:
            raise ValueError('Profile belongs to another cohort')
        selected=choose_profile(report)
        groups=[];prefix=0
        for group in _chunks(engine.rows(parent/'sources.jsonl'),old['batch_size']):
            path=parent/'groups'/f"{engine.common.fingerprint([s['sample_id'] for s in group])}.json"
            if not path.exists(): break
            value=engine.sealed_read(path)
            if value['cohort_fingerprint']!=old['fingerprint'] or value['source_ids']!=[s['sample_id'] for s in group]:
                raise ValueError('Parent committed batch changed')
            if len(value['records'])!=len(group): raise ValueError('Incomplete parent commit')
            for source,record in zip(group,value['records'],strict=True):
                if record!=engine.run.sealed(record) or record['source']!=source:
                    raise ValueError('Parent receipt differs from its source')
            groups.append((path,value));prefix+=len(group)
        for name in ('sources.jsonl','coverage_snapshot.json'):
            target=directory/name
            if target.exists() and engine.common.file_sha256(target)!=engine.common.file_sha256(parent/name):
                raise ValueError('Successor inputs differ')
            if not target.exists(): shutil.copyfile(parent/name,target)
        cfg={k:v for k,v in old.items() if k!='fingerprint'}
        cfg.update(version=VERSION,batch_size=selected['batch_size'],image_root=str(directory.parent/'images'),
            imported_prefix=prefix,imported_batch_size=old['batch_size'],
            parent_state=str(parent),parent_fingerprint=old['fingerprint'],
            profile=str(profile_path),selected_profile=selected,
            sampling_migration='Committed prompts unchanged; future RNG keyed to new batch membership and successor fingerprint.',
            memory_policy=dict(allocator_max_gib=42,guard_gpu_gib=44,host_reserve_gib=4,
                               cgroup_high_gib=12,cgroup_max_gib=16,cgroup_swap_max=0),
            files_sha256={**dependencies(Path(old['model_path'])),
                **{name:digest for name,digest in old['files_sha256'].items()
                   if 'Dataframes' in Path(name).parts or name==old['exclude_places']},
                str(profile_path):engine.common.file_sha256(profile_path),
                str(parent/'cohort_config.json'):engine.common.file_sha256(parent/'cohort_config.json'),
                **{str(path):engine.common.file_sha256(path) for path,_ in groups}})
        cfg=engine.campaign.frozen(directory/'cohort_config.json',cfg)
        ready=failed=0
        for path,group in groups:
            receipts=[]
            for value in group['records']:
                value=engine.run.sealed({**value,'cohort_fingerprint':cfg['fingerprint'],
                    'imported_from':dict(parent_fingerprint=old['fingerprint'],
                                         original_receipt_sha256=value['result_sha256'])})
                receipts.append(value)
                ready+=value['status']=='ready';failed+=value['status']!='ready'
            out=engine.run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],source_ids=group['source_ids'],records=receipts))
            engine.common.write_json(directory/'groups'/path.name,out)
            for value in receipts:
                engine.common.write_json(directory/'prompts'/f"{value['source']['sample_id']}.json",value)
        unresolved=[str(path) for path in (parent/'batches').glob('*.request.json')
                    if not path.with_name(path.name.replace('.request.json','.json')).exists()]
        marker=engine.run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],imported=prefix,ready=ready,failed=failed,
            uncommitted_parent_calls=unresolved,
            interrupted_work_policy='Uncommitted old work remains archived; new execution batches may regenerate those source prompts.'))
        engine.common.write_json(directory/'migration_complete.json',marker)
        if not (directory/'progress.json').exists():
            engine.common.write_json(directory/'progress.json',dict(state='migrated_not_started',total=cfg['num_places'],
                processed=prefix,ready=ready,failed=failed,generation_started=False))
        return marker


class Planner(policy.Planner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # A hard PyTorch allocator cap supplements the sampled 44GiB whole-GPU guard.
        self.torch.cuda.set_per_process_memory_fraction(
            42*1024**3/self.torch.cuda.get_device_properties(0).total_memory)


def generation_resource_plan(directory):
    configure(directory)
    return engine.generation_resource_plan(directory)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='migrate':
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('command');p.add_argument('--parent',type=Path,default=policy.ROOT/'state')
        p.add_argument('--output-dir',type=Path,default=ROOT/'state')
        p.add_argument('--profile',type=Path,required=True)
        a=p.parse_args()
        print(json.dumps(migrate(a.parent.resolve(),a.output_dir.resolve(),a.profile.resolve()),indent=2))
    else:
        # Preserve the shared CLI, but require the committed import barrier first.
        p=argparse.ArgumentParser(add_help=False);p.add_argument('--output-dir',type=Path,default=ROOT/'state')
        a,_=p.parse_known_args()
        marker=engine.sealed_read(a.output_dir/'migration_complete.json')
        configure(a.output_dir)
        if marker['cohort_fingerprint']!=engine.campaign.config_file(a.output_dir/'cohort_config.json')['fingerprint']:
            raise ValueError('Migration barrier differs')
        engine.BatchedPlanner=Planner
        engine.main()


if __name__=='__main__': main()
