import copy
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from experiments.qwen_curriculum import campaign, common, run, night_diversity_policy as policy
from experiments.qwen_curriculum import night_place_cohort as cohort


def evidence(name):
    present = {'outdoors','drivable_road','paved_ground','windows'}
    return dict(coordinate_system='0_1000', scene_summary='brick facade '+name,
        protected_regions=[dict(description='brick window facade',box=[100,100,800,500])],
        features={k:dict(status='present' if k in present else 'absent',evidence='visible source evidence') for k in policy.FEATURES})


class FakePlanner:
    calls = []
    def __init__(self, *_): pass
    def close(self): pass
    def ask(self, directory, sources, requests, phase, attempt, cfg):
        self.calls.append((phase,[s['sample_id'] for s in sources]))
        if phase == 'analysis': return [json.dumps(evidence(s['sample_id'])) for s in sources]
        output = []
        for request in requests:
            style = re.search(r'"style_id":"([^"]+)"', request).group(1)
            output.append(json.dumps(dict(variants=[dict(style_id=style,style_name='local night',weather='night',
                surface_state='wet',lighting='existing local night illumination',exposure='natural nighttime falloff',
                appearance_prompt='Relight this exact brick facade and the existing paved street for a plausible night photograph. '
                    'Keep the same window and doorway positions and the original road topology. '
                    'Make the pavement wet and allow plausible passing vehicles while keeping permanent place evidence readable.')])) )
        return output


def fixture(root, count=8):
    directory = root/'state'; sources=[]
    for place in range(count):
        paths=[]
        for view in range(4):
            path=root/'real/Images/City'/f'City_{place:07d}_{view}.jpg'
            path.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(24,20),(place*10,view*20,10)).save(path); paths.append(path)
        source,_=cohort.select_source((('City',place),paths),42,4); sources.append(source)
    common.write_jsonl(directory/'sources.jsonl',sources)
    snapshot=campaign.frozen(directory/'coverage_snapshot.json',policy.coverage_snapshot([]))
    cfg=campaign.frozen(directory/'cohort_config.json',dict(version=cohort.VERSION,condition='night',seed=42,
        num_places=count,batch_size=4,shard_size=2,model_path=str(root/'model'),files_sha256={},
        source_manifest_sha256=common.file_sha256(directory/'sources.jsonl'),coverage_fingerprint=snapshot['fingerprint'],
        image_root=str(root/'images'),compose_drafts=2,analysis_tokens=1400,compose_tokens=900,
        service=dict(performance=dict(resident_dit_blocks=24))))
    return SimpleNamespace(output_dir=directory),cfg,sources


class CohortTests(unittest.TestCase):
    def setUp(self): FakePlanner.calls=[]

    def test_source_requires_four_distinct_existing_images_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); args,cfg,sources=fixture(root,1)
            paths=list((root/'real/Images/City').glob('*.jpg'))
            first,_=cohort.select_source((('City',0),paths),42,4)
            second,_=cohort.select_source((('City',0),list(reversed(paths))),42,4)
            self.assertEqual(first,second)
            self.assertEqual(first['num_real_images'],4)
            self.assertIsNone(cohort.select_source((('City',0),paths[:3]+paths[:1]),42,4)[0])
            paths[0].unlink()
            self.assertIsNone(cohort.select_source((('City',0),paths),42,4)[0])

    def test_one_prompt_per_place_all_night_and_generation_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,cfg,sources=fixture(Path(tmp))
            with self.assertRaises(ValueError): cohort.generation_resource_plan(args.output_dir)
            with patch.object(cohort,'BatchedPlanner',FakePlanner),patch.object(campaign,'ensure_stopped'):
                result=cohort.prompt_places(args)
                cohort.prompt_places(args)
            self.assertEqual((result['ready'],result['failed']),(8,0))
            self.assertEqual(len(FakePlanner.calls),4)  # Two phases per fixed 4-image batch.
            jobs=list(cohort.rows(args.output_dir/'night_prompts.jsonl'))
            self.assertEqual(len({(j['city'],j['place_id']) for j in jobs}),8)
            self.assertTrue(all(j['condition']=='night' for j in jobs))
            self.assertEqual({j['source_path'] for j in jobs},{s['source_path'] for s in sources})
            self.assertEqual(cohort.generation_resource_plan(args.output_dir)[2],8)
            self.assertFalse((args.output_dir.parent/'images').exists())

    def test_atomic_batch_commit_repairs_partial_receipts_without_model_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,cfg,sources=fixture(Path(tmp))
            original=common.write_json; writes=0
            def interrupt(path,value):
                nonlocal writes
                if Path(path).parent.name=='prompts':
                    writes+=1
                    if writes==2: raise KeyboardInterrupt('test interruption after batch commit')
                original(path,value)
            with patch.object(cohort,'BatchedPlanner',FakePlanner),patch.object(campaign,'ensure_stopped'):
                with patch.object(common,'write_json',side_effect=interrupt),self.assertRaises(KeyboardInterrupt):
                    cohort.prompt_places(args)
                result=cohort.prompt_places(args)
            self.assertEqual(result['ready'],8)
            self.assertEqual(len(FakePlanner.calls),4)

    def test_failed_source_analysis_is_reported_not_fabricated(self):
        class InvalidOne(FakePlanner):
            def ask(self,*args):
                result=super().ask(*args)
                if args[3]=='analysis': result[0]='{}'
                return result
        with tempfile.TemporaryDirectory() as tmp:
            args,_,_=fixture(Path(tmp),4)
            with patch.object(cohort,'BatchedPlanner',InvalidOne),patch.object(campaign,'ensure_stopped'):
                result=cohort.prompt_places(args)
            self.assertEqual((result['ready'],result['failed']),(3,1))
            self.assertEqual(len(list(cohort.rows(args.output_dir/'night_prompts.jsonl'))),3)

    def test_exported_shards_load_with_existing_balanced_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,_,_=fixture(Path(tmp),4)
            with patch.object(cohort,'BatchedPlanner',FakePlanner),patch.object(campaign,'ensure_stopped'):
                result=cohort.prompt_places(args)
            module=cohort.balanced_night.runtime()
            for shard in result['shards']:
                cfg,jobs=module.load_plan(Path(shard['state']))
                self.assertEqual(cfg['num_jobs'],2)
                self.assertEqual(len(jobs),2)


if __name__=='__main__': unittest.main()
