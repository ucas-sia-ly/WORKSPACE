"""Scheduling changes must preserve fixed jobs and durable resume semantics."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.qwen_curriculum import common, night_first_v4 as v4, weather_v4_priority as priority
import test_weather_v4 as v4_tests
from test_night_first_v3 import HEALTH, FakeVerifier, Session


class PriorityTests(unittest.TestCase):
    setUp = v4_tests.PlanTests.setUp

    def test_partition_uses_city_and_place_and_keeps_base_order(self):
        jobs = [{'city':c, 'place_id':p} for c,p in [('A',1),('B',1),('A',2),('A',3)]]
        old = list(jobs)
        self.assertEqual(priority.prioritize(jobs,{('A',1),('A',2)}),[jobs[1],jobs[3],jobs[0],jobs[2]])
        self.assertEqual(jobs,old)

    def test_snapshot_resume_and_saved_image_never_reposts(self):
        class Verifier(FakeVerifier):
            definition = v4.quality_definition_v4(clip_device='cpu')
            def evaluate(self,*args):
                row=super().evaluate(*args)
                return {**row,'base_quality_passed':row['passed'],'eligible_for_training':False}
        engine=v4._engine(); legacy=self.root/'legacy'; directory=self.root/'v4/state'
        common.write_json(legacy/'state/all_city_campaign_config.json',{'reference_execution':{'service':v4.run.service_identity(HEALTH)}})
        args=SimpleNamespace(output_dir=directory,image_root=self.root/'v4',real_data=self.root/'gsv',
            seed=42,min_images=4,workers=1,domain='night',device='cpu',qwen_url='http://local/generate',request_timeout=20,max_calls=1)
        service=self.root/'service'; service.mkdir();session=Session(directory,service)
        with ExitStack() as stack:
            stack.enter_context(patch.object(engine,'LEGACY_ROOT',legacy))
            stack.enter_context(patch.object(v4.source_screen_v3,'screen_sources',return_value=self.sources))
            stack.enter_context(patch.object(v4.source_reliability_v3,'refine_sources',return_value=(self.sources,{})))
            v4.prepare(args)
            base=(directory/'plan.jsonl').read_bytes()
            _,jobs=v4.load_plan(directory)
            night=[j for j in jobs if j['condition']=='night']
            seen={priority.place(night[0])}
            stack.enter_context(patch.object(priority,'historical_places',return_value=(seen,[])))
            priority.prepare_priority(directory)
            frozen=(directory/'priority_config.json').read_bytes()
            priority.prepare_priority(directory)
            self.assertEqual(frozen,(directory/'priority_config.json').read_bytes())
            ordered=[j for j in priority.load_priority(directory)[1] if j['condition']=='night']
            self.assertEqual(ordered[-1],night[0]); self.assertEqual(ordered[0],night[1])
            stack.enter_context(patch.object(engine,'load_plan',side_effect=priority.load_priority))
            stack.enter_context(patch.object(engine.requests,'Session',return_value=session))
            stack.enter_context(patch.object(engine.run,'service_output_directory',return_value=service))
            stack.enter_context(patch.object(engine,'QwenQualityVerifierV3',return_value=Verifier()))
            stack.enter_context(redirect_stdout(io.StringIO()))
            engine.generate(args)
            saved=common.read_jsonl(directory/'generated.jsonl')
            self.assertEqual(saved[0]['sample_id'],night[1]['sample_id'])
            args.max_calls=0; engine.generate(args);engine.generate(args)
            self.assertEqual(len(session.posts),len(night))
            self.assertEqual(base,(directory/'plan.jsonl').read_bytes())
            snapshot=directory/'previously_generated_places.jsonl';snapshot.write_text('')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):priority.load_priority(directory)

if __name__=='__main__':unittest.main()
