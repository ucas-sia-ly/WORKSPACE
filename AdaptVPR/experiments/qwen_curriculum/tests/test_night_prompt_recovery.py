import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.qwen_curriculum import resume_night_prompts as recovery
from experiments.qwen_curriculum.tests.test_night_place_cohort import fixture,FakePlanner


class PromptRecoveryTests(unittest.TestCase):
    def batch(self,directory,phase='compose'):
        sources=[dict(sample_id='a'),dict(sample_id='b')];requests=['first','second']
        cfg=dict(fingerprint='cohort',seed=42)
        key=recovery.engine.common.fingerprint([cfg['fingerprint'],phase,0,['a','b'],requests])
        seed=recovery.engine.common.candidate_seed(42,key,0)
        response=directory/'batches'/f'{key}.json'
        reserve=response.with_suffix('.request.json')
        value=dict(cohort=cfg['fingerprint'],source_ids=['a','b'],phase=phase,attempt=0,
                   seed=seed,logical_inputs=2)
        return sources,requests,cfg,key,seed,response,reserve,value

    def test_saved_response_reused_without_model_or_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s,r,c,key,seed,response,reserve,value=self.batch(root)
            recovery.engine.common.write_json(response,recovery.engine.run.sealed(dict(
                key=key,phase='compose',source_ids=['a','b'],seed=seed,responses=['saved1','saved2'])))
            before=response.read_bytes()
            def forbidden(): self.fail('Saved response must not invoke the model')
            self.assertEqual(recovery.recover_call(root,s,r,'compose',0,c,forbidden),['saved1','saved2'])
            self.assertEqual(response.read_bytes(),before)
            self.assertFalse((root/'prompt_recovery').exists())

    def test_interrupted_reservation_archived_and_regenerated_once(self):
        for phase in ('analysis','compose'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);s,r,c,key,seed,response,reserve,value=self.batch(root,phase)
                recovery.engine.common.write_json(reserve,value);original=reserve.read_bytes();calls=[]
                def generate():
                    self.assertFalse(reserve.exists())
                    calls.append((copy.deepcopy(s),list(r),seed))
                    recovery.engine.common.write_json(reserve,value)
                    recovery.engine.common.write_json(response,recovery.engine.run.sealed(dict(
                        key=key,phase=phase,source_ids=['a','b'],seed=seed,responses=['new1','new2'])))
                    return ['new1','new2']
                out=recovery.recover_call(root,s,r,phase,0,c,generate)
                self.assertEqual(out,['new1','new2'])
                archives=list((root/'prompt_recovery'/key).glob('*.request.json'))
                self.assertEqual(len(archives),1);self.assertEqual(archives[0].read_bytes(),original)
                self.assertEqual(recovery.recover_call(root,s,r,phase,0,c,generate),out)
                self.assertEqual(calls,[(s,r,seed)])

    def test_repeated_interruptions_do_not_become_failed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s,r,c,key,seed,response,reserve,value=self.batch(root)
            recovery.engine.common.write_json(reserve,value)
            def interrupt():
                recovery.engine.common.write_json(reserve,value)
                raise KeyboardInterrupt('another interruption')
            for _ in range(2):
                with self.assertRaises(KeyboardInterrupt):
                    recovery.recover_call(root,s,r,'compose',0,c,interrupt)
            self.assertEqual(len(list((root/'prompt_recovery'/key).glob('*.request.json'))),2)
            self.assertFalse((root/'prompts').exists())

    def test_corrupt_or_mismatched_data_pauses_instead_of_skipping(self):
        for kind in ('request','response'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);s,r,c,key,seed,response,reserve,value=self.batch(root)
                if kind=='request':
                    recovery.engine.common.write_json(reserve,{**value,'seed':seed+1})
                else:
                    recovery.engine.common.write_json(response,recovery.engine.run.sealed(dict(
                        key=key,phase='compose',source_ids=['b','a'],seed=seed,responses=['x','y'])))
                with self.assertRaisesRegex(RuntimeError,'recovery paused'):
                    recovery.recover_call(root,s,r,'compose',0,c,lambda:self.fail('must not infer'))
                self.assertFalse((root/'prompt_recovery').exists())

    def test_lazy_planner_does_not_load_gpu_until_cache_miss(self):
        planner=recovery.Planner(Path('/unused/model'))
        planner.close()
        with patch.object(recovery.fast.Planner,'__init__',return_value=None) as initialize:
            planner.ensure_model();planner.ensure_model()
        self.assertEqual(initialize.call_count,1)

    def test_production_planner_wiring_uses_cache_then_initializes_only_missing_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s,r,c,key,seed,response,reserve,value=self.batch(root)
            planner=recovery.Planner(Path('/unused/model'))
            recovery.engine.common.write_json(response,recovery.engine.run.sealed(dict(
                key=key,phase='compose',source_ids=['a','b'],seed=seed,responses=['saved1','saved2'])))
            with patch.object(recovery.BASE,'ask',recovery.recovered_ask),patch.object(planner,'ensure_model') as initialize:
                self.assertEqual(planner.ask(root,s,r,'compose',0,c),['saved1','saved2'])
                initialize.assert_not_called()
                response.unlink()
                recovery.engine.common.write_json(reserve,value)
                with patch.object(recovery,'ORIGINAL_ASK',return_value=['new1','new2']) as infer:
                    self.assertEqual(planner.ask(root,s,r,'compose',0,c),['new1','new2'])
                initialize.assert_called_once();infer.assert_called_once()

    def test_saved_batches_finish_group_commit_without_another_forward(self):
        e=recovery.engine
        with tempfile.TemporaryDirectory() as tmp:
            args,cfg,sources=fixture(Path(tmp),4);calls=[]
            class CachedFake(FakePlanner):
                def ask(self,directory,sources,requests,phase,attempt,cfg):
                    def infer():
                        calls.append(phase)
                        key=e.common.fingerprint([cfg['fingerprint'],phase,attempt,[s['sample_id'] for s in sources],requests])
                        seed=e.common.candidate_seed(cfg['seed'],key,attempt)
                        path=directory/'batches'/f'{key}.json'
                        e.common.write_json(path.with_suffix('.request.json'),dict(cohort=cfg['fingerprint'],
                            source_ids=[s['sample_id'] for s in sources],phase=phase,attempt=attempt,seed=seed,logical_inputs=len(sources)))
                        if phase=='compose' and calls.count('compose')==1:
                            raise KeyboardInterrupt('before response save')
                        replies=super(CachedFake,self).ask(directory,sources,requests,phase,attempt,cfg)
                        e.common.write_json(path,e.run.sealed(dict(key=key,phase=phase,source_ids=[s['sample_id'] for s in sources],seed=seed,responses=replies)))
                        if phase=='compose': raise KeyboardInterrupt('response saved but group not committed')
                        return replies
                    return recovery.recover_call(directory,sources,requests,phase,attempt,cfg,infer)
            with patch.object(e,'BatchedPlanner',CachedFake),patch.object(e.campaign,'ensure_stopped'):
                with self.assertRaises(KeyboardInterrupt): e.prompt_places(args)
                with self.assertRaises(KeyboardInterrupt): e.prompt_places(args)
                result=e.prompt_places(args)
                before={p.name:p.read_bytes() for p in (args.output_dir/'prompts').glob('*.json')}
                e.prompt_places(args)
            self.assertEqual(calls,['analysis','compose','compose'])
            self.assertEqual((result['ready'],result['failed']),(4,0))
            self.assertEqual(before,{p.name:p.read_bytes() for p in (args.output_dir/'prompts').glob('*.json')})


if __name__=='__main__': unittest.main()
