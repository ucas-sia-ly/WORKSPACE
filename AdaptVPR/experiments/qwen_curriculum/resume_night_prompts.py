"""Recover local Prompt inference after interruption, without skipping sources.

This adapter preserves the frozen cohort and all completed outputs. Image-service
POST journals retain their separate no-duplicate policy; this applies ONLY to
local VLM Prompt preparation. Saved responses are replayed. Missing responses
are regenerated with the same batch membership, inputs and seed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import night_place_fast as fast

engine=fast.engine
BASE=fast.policy._planner
ORIGINAL_ASK=BASE.ask
POLICY_VERSION='local_prompt_recovery_v1'


def recover_call(directory,sources,requests,phase,attempt,cfg,invoke):
    key=engine.common.fingerprint([cfg['fingerprint'],phase,attempt,
                                   [s['sample_id'] for s in sources],requests])
    path=directory/'batches'/f'{key}.json'
    reserve=path.with_suffix('.request.json')
    seed=engine.common.candidate_seed(cfg['seed'],key,attempt)
    try:
        if path.exists():
            saved=engine.sealed_read(path)
            if (saved.get('key')!=key or saved.get('source_ids')!=[s['sample_id'] for s in sources] or
                saved.get('phase')!=phase or saved.get('seed')!=seed or
                not isinstance(saved.get('responses'),list) or len(saved['responses'])!=len(sources) or
                any(not isinstance(value,str) for value in saved['responses'])):
                raise ValueError('Saved response does not match the exact batch inputs')
            return saved['responses']
        if reserve.exists():
            expected=dict(cohort=cfg['fingerprint'],source_ids=[s['sample_id'] for s in sources],
                          phase=phase,attempt=attempt,seed=seed,logical_inputs=len(sources))
            original=json.loads(reserve.read_text())
            if original!=expected:
                raise ValueError('Interrupted request does not match the exact batch inputs')
            archive_dir=directory/'prompt_recovery'/key
            archive_dir.mkdir(parents=True,exist_ok=True)
            existing=[int(p.name.split('.')[0]) for p in archive_dir.glob('*.request.json')]
            sequence=max(existing,default=0)+1
            archive=archive_dir/f'{sequence:06d}.request.json'
            event=engine.run.sealed(dict(policy=POLICY_VERSION,cohort_fingerprint=cfg['fingerprint'],
                key=key,phase=phase,attempt=attempt,seed=seed,source_ids=expected['source_ids'],
                event='retry_missing_local_response',timestamp=datetime.now(timezone.utc).isoformat(),
                original_request=original,original_request_sha256=engine.common.file_sha256(reserve),
                archived_request=str(archive),adapter_sha256=engine.common.file_sha256(__file__)))
            # Record intent, then move the original atomically. A crash at either
            # point retains evidence and cannot create a false failed-place label.
            engine.common.write_json(archive.with_suffix('.event.json'),event)
            os.replace(reserve,archive)
        return invoke()
    except ValueError as error:
        # The old driver treats ValueError as a failed batch. Corrupt recovery
        # data should instead pause for inspection, preserving all source work.
        raise RuntimeError('Prompt recovery paused: '+str(error)) from error


def recovered_ask(self,directory,sources,requests,phase,attempt,cfg):
    def invoke():
        self.ensure_model()
        return ORIGINAL_ASK(self,directory,sources,requests,phase,attempt,cfg)
    return recover_call(directory,sources,requests,phase,attempt,cfg,invoke)


class Planner(fast.Planner):
    """Complete durable cached work without loading the GPU model first."""
    def __init__(self,*args,**kwargs):
        self._model_args=args
        self._model_kwargs=kwargs
        self._loaded=False

    def ensure_model(self):
        if not self._loaded:
            super().__init__(*self._model_args,**self._model_kwargs)
            self._loaded=True

    def close(self):
        if self._loaded:
            super().close()
            self._loaded=False


def bind(directory):
    with engine.run.run_lock(directory):
        fast.configure(directory)
        cfg=engine.load(directory)
        marker=engine.sealed_read(directory/'migration_complete.json')
        if marker['cohort_fingerprint']!=cfg['fingerprint']:
            raise ValueError('Migration barrier differs')
        return engine.campaign.frozen(directory/'prompt_recovery_policy.json',dict(
            version=POLICY_VERSION,cohort_fingerprint=cfg['fingerprint'],
            cohort_config_sha256=engine.common.file_sha256(directory/'cohort_config.json'),
            adapter_sha256=engine.common.file_sha256(__file__),
            scope='local Prompt preparation only; no image generation or training',
            cached_response='validate and reuse without model loading',
            missing_response='archive interrupted request and retry the same inputs and seed',
            corruption='pause instead of marking source places failed',
            completed_prompts='reuse unchanged',batch_size=cfg['batch_size']))


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--output-dir',type=Path,default=fast.ROOT/'state')
    p.add_argument('--install-only',action='store_true')
    args,_=p.parse_known_args()
    directory=args.output_dir.resolve()
    policy=bind(directory)
    if args.install_only:
        print(json.dumps(policy,ensure_ascii=False,indent=2))
        return
    if len(sys.argv)<2 or sys.argv[1] not in ('prepare','check','report'):
        raise ValueError('This recovery adapter only supports local Prompt preparation')
    BASE.ask=recovered_ask
    engine.BatchedPlanner=Planner
    engine.main()


if __name__=='__main__': main()
