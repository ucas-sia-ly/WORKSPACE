import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.qwen_curriculum import night_place_fast as fast
from experiments.qwen_curriculum.tests.test_night_place_cohort import fixture,FakePlanner


def record(batch,speed,reserved=20):
    return dict(batch_size=batch,valid_complete_per_second=speed,cuda_peak_reserved_gib=reserved,
        resources=dict(gpu_used_gib=reserved+1,available_gib=12),
        phases={phase:dict(valid=batch,total=batch) for phase in ('analysis','compose')})


class FastCohortTests(unittest.TestCase):
    def test_source_groups_preserve_imported_prefix_and_tail(self):
        groups=list(fast.source_groups(range(29),16,8,4))
        self.assertEqual([len(g) for g in groups],[4,4,16,5])
        self.assertEqual([item for g in groups for item in g],list(range(29)))

    def test_select_measured_throughput_under_memory_and_validity_limits(self):
        report=dict(records=[record(4,.2),record(16,.4),record(32,.6,41)])
        self.assertEqual(fast.choose_profile(report)['batch_size'],16)
        report['records'][1]['phases']['compose']['valid']=15
        self.assertEqual(fast.choose_profile(report)['batch_size'],4)
        with self.assertRaises(ValueError): fast.choose_profile({**report,'guard_events':[{}]})

    def test_large_batch_requires_largest_input_stress_and_memory_margin(self):
        report=dict(records=[record(32,.5),record(64,.55,36)])
        self.assertEqual(fast.choose_profile(report)['batch_size'],32)
        report['stress_validation']=dict(largest_input_stress=True,records=[record(64,.54,37)])
        self.assertEqual(fast.choose_profile(report)['batch_size'],64)
        report['stress_validation']['records'][0]['resources']['gpu_used_gib']=43
        self.assertEqual(fast.choose_profile(report)['batch_size'],32)

    def test_import_is_idempotent_and_preserves_exact_source_plan_job(self):
        e=fast.engine
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args,cfg,sources=fixture(root/'parent')
            with patch.object(e,'BatchedPlanner',FakePlanner),patch.object(e.campaign,'ensure_stopped'):
                e.prompt_places(args)
            profile=root/'profile.json'
            profile.write_text(json.dumps(dict(parent_fingerprint=cfg['fingerprint'],records=[record(16,.4)])))
            target=root/'fast/state'
            original={s['sample_id']:e.sealed_read(args.output_dir/'prompts'/f"{s['sample_id']}.json") for s in sources}
            with patch.object(fast.policy,'configure'),patch.object(fast,'dependencies',return_value={}):
                first=fast.migrate(args.output_dir,target,profile)
                second=fast.migrate(args.output_dir,target,profile)
            self.assertEqual(first,second)
            self.assertEqual(first['imported'],8)
            new_cfg=e.campaign.config_file(target/'cohort_config.json')
            self.assertEqual(new_cfg['batch_size'],16)
            for key,old in original.items():
                new=e.sealed_read(target/'prompts'/f'{key}.json')
                for field in ('source','plan','job','status'):
                    self.assertEqual(new[field],old[field])
                self.assertEqual(new['cohort_fingerprint'],new_cfg['fingerprint'])
                self.assertEqual(new['imported_from']['original_receipt_sha256'],old['result_sha256'])

    def test_incomplete_prefix_is_not_silently_imported(self):
        e=fast.engine
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args,cfg,sources=fixture(root/'parent')
            with patch.object(e,'BatchedPlanner',FakePlanner),patch.object(e.campaign,'ensure_stopped'):
                e.prompt_places(args)
            path=next((args.output_dir/'groups').glob('*.json'))
            value=e.sealed_read(path);value['records'].pop()
            e.common.write_json(path,e.run.sealed(value))
            profile=root/'profile.json'
            profile.write_text(json.dumps(dict(parent_fingerprint=cfg['fingerprint'],records=[record(8,.3)])))
            with patch.object(fast.policy,'configure'),patch.object(fast,'dependencies',return_value={}):
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    fast.migrate(args.output_dir,root/'fast/state',profile)


if __name__=='__main__': unittest.main()
