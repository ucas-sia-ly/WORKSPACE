import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from experiments.qwen_curriculum import common,run,night_revision as revision
from experiments.qwen_curriculum import night_diversity_policy_v2 as policy
from test_night_diversity_policy_v2 import source,variant,plan


class RevisionTests(unittest.TestCase):
    def fixture(self):
        s=run.sealed(dict(sample_id='source',city='London',place_id=1,source_path='/source.jpg',
            source_sha256='a'*64,source_dimensions=[640,480],num_real_images=5,real_pool_sha256='b'*64))
        task=dict(sample_id='old-mercury',original_record_sha256='c'*64,source=s,task_reason='retired_mercury_style')
        cfg=dict(fingerprint='new-cohort',seed=42,analysis_drafts=2,compose_drafts=3,batch_size=32,
            coverage_counts={k:dict(saved=0 if k=='sodium_lamps' else 100,pending=0,mismatch=0) for k in policy.STYLES})
        return task,cfg

    def test_new_job_retains_source_place_and_links_replaced_job(self):
        task,cfg=self.fixture();p=plan(source('outdoors','drivable_road'))
        job=revision.make_job(task,p,cfg)
        self.assertEqual(job['source_path'],task['source']['source_path'])
        self.assertEqual(job['source_sha256'],task['source']['source_sha256'])
        self.assertEqual(job['supersedes_sample_id'],task['sample_id'])
        self.assertNotEqual(job['sample_id'],task['sample_id'])
        self.assertNotEqual(job['style_id'],'mercury_vapor')
        self.assertEqual(job['record_sha256'],common.fingerprint({k:v for k,v in job.items() if k!='record_sha256'}))
        self.assertIn('MUST PRESERVE',job['prompt']);self.assertIn('MAY CHANGE',job['prompt'])

    def test_group_has_separate_scene_analysis_and_original_composition(self):
        task,cfg=self.fixture();evidence=source('outdoors','drivable_road')
        reader=SimpleNamespace(ask=Mock(side_effect=[[json.dumps(evidence)],[json.dumps(dict(variants=[variant()]))]]))
        records=revision.plan_group(reader,Path('/unused'),[task],cfg,{})
        self.assertEqual(records[0]['status'],'ready')
        self.assertEqual(records[0]['job']['style_id'],'sodium_lamps')
        self.assertEqual(reader.ask.call_count,2)
        self.assertEqual(reader.ask.call_args_list[0].args[3],'analysis')
        self.assertEqual(reader.ask.call_args_list[1].args[3],'compose')

    def test_analysis_repair_does_not_force_unproven_category(self):
        task,cfg=self.fixture()
        reader=SimpleNamespace(ask=Mock(side_effect=[['{}'],[json.dumps(source('outdoors','drivable_road'))],
                    [json.dumps(dict(variants=[variant()]))]]))
        records=revision.plan_group(reader,Path('/unused'),[task],cfg,{})
        self.assertEqual(records[0]['status'],'ready');self.assertEqual(reader.ask.call_count,3)

    def test_no_valid_source_yields_terminal_failure_without_composer(self):
        task,cfg=self.fixture();reader=SimpleNamespace(ask=Mock(return_value=['{}']))
        records=revision.plan_group(reader,Path('/unused'),[task],cfg,{})
        self.assertEqual(records[0]['status'],'failed');self.assertEqual(reader.ask.call_count,2)

    def test_corrupt_model_receipt_propagates_instead_of_becoming_source_failure(self):
        task,cfg=self.fixture();reader=SimpleNamespace(ask=Mock(side_effect=ValueError('checksum changed')))
        with self.assertRaisesRegex(ValueError,'checksum'):revision.plan_group(reader,Path('/unused'),[task],cfg,{})

    def test_group_integrity_is_bound_to_each_source_and_new_prompt(self):
        task,cfg=self.fixture();reader=SimpleNamespace(ask=Mock(side_effect=[
            [json.dumps(source('outdoors','drivable_road'))],[json.dumps(dict(variants=[variant()]))]]))
        records=revision.plan_group(reader,Path('/unused'),[task],cfg,{})
        group=run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],predecessor_ids=[task['sample_id']],records=records))
        revision.check_group(group,[task],cfg)
        bad=copy.deepcopy(group);bad['records'][0]['job']['prompt']='changed'
        with self.assertRaises(ValueError):revision.check_group(bad,[task],cfg)

    def test_generation_requires_a_new_explicit_launcher(self):
        with self.assertRaises(SystemExit):revision.main(['generate'])
        with self.assertRaisesRegex(ValueError,'--execute'):revision.main(['plan'])

    def test_old_cluster_cannot_resume_after_revision_request(self):
        from experiments.qwen_curriculum import night_cluster
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'prompt_revision_requested.json').write_text('{}')
            with patch.object(night_cluster,'command_node') as command, \
                    self.assertRaisesRegex(ValueError,'retired for prompt revision'):
                night_cluster.controls(root,'resume','all')
            command.assert_not_called()

    def test_atomic_export_requires_terminal_receipt_for_every_source(self):
        task,cfg=self.fixture();cfg['num_places']=1;cfg['retained_images']=0;cfg['files_sha256']={}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'state';state.mkdir()
            common.write_jsonl(state/'tasks.jsonl',[task])
            (state/'revision_config.json').write_text('{}')
            with self.assertRaises(FileNotFoundError):revision.export(root,cfg)
            self.assertFalse((state/'prompts_complete.json').exists())

    def test_standalone_export_cannot_bypass_group_source_validation(self):
        task,cfg=self.fixture();cfg.update(num_places=1,retained_images=0,files_sha256={})
        reader=SimpleNamespace(ask=Mock(side_effect=[[json.dumps(source('outdoors','drivable_road'))],
                    [json.dumps(dict(variants=[variant()]))]]))
        records=revision.plan_group(reader,Path('/unused'),[task],cfg,{})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'state';state.mkdir()
            common.write_jsonl(state/'tasks.jsonl',[task]);(state/'revision_config.json').write_text('{}')
            key=common.fingerprint([task['sample_id']])
            common.write_json(state/'groups'/(key+'.json'),run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],
                predecessor_ids=[task['sample_id']],records=records)))
            altered=copy.deepcopy(records[0]);altered['job']['source_path']='/wrong.jpg';altered=run.sealed(altered)
            common.write_json(state/'prompts'/(task['sample_id']+'.json'),altered)
            with self.assertRaisesRegex(ValueError,'atomic'):revision.export(root,cfg)
            self.assertFalse((state/'prompts_complete.json').exists())

    def test_foreign_completion_marker_cannot_skip_work(self):
        task,cfg=self.fixture();cfg['num_places']=1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'state';state.mkdir()
            common.write_json(state/'prompts_complete.json',run.sealed(dict(cohort_fingerprint='other',selected=1,ready=1,failed=0)))
            with self.assertRaisesRegex(ValueError,'completion'):revision.validate_completion(root,cfg)

    def test_failed_group_recovery_keeps_ready_record_and_never_repeats_on_resume(self):
        task,cfg=self.fixture();cfg['execution_patch_sha256']='patch-sha'
        other={**copy.deepcopy(task),'sample_id':'second-old'}
        reader=SimpleNamespace(ask=Mock(side_effect=[[json.dumps(source('outdoors','drivable_road'))],
                    [json.dumps(dict(variants=[variant()]))]]))
        ready=revision.plan_group(reader,Path('/unused'),[task],cfg,{})[0]
        failed=run.sealed(dict(status='failed',reason='invalid_source_analysis',source=other['source'],
            predecessor_sample_id=other['sample_id'],cohort_fingerprint=cfg['fingerprint']))
        tasks=[task,other]
        base=run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],
            predecessor_ids=[t['sample_id'] for t in tasks],records=[ready,failed]))
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);key=common.fingerprint(base['predecessor_ids'])
            common.write_json(state/'groups'/(key+'.json'),base)
            original_hash=common.file_sha256(state/'groups'/(key+'.json'))
            reader.ask=Mock(side_effect=[[json.dumps(source('outdoors','drivable_road'))],
                    [json.dumps(dict(variants=[variant()]))]])
            value=revision.repair_group(reader,state,tasks,cfg,{},base,base)
            self.assertEqual(value['records'][0],ready)
            self.assertEqual(value['recovered'],1)
            self.assertEqual(value['records'][1]['status'],'ready')
            loaded,parent=revision.read_group(state,tasks,cfg)
            self.assertEqual(loaded,value);self.assertEqual(parent,base)
            reader.ask=Mock(side_effect=AssertionError('Must not repeat committed recovery'))
            self.assertEqual(revision.repair_group(reader,state,tasks,cfg,{},value,base),value)
            self.assertEqual(common.file_sha256(state/'groups'/(key+'.json')),original_hash)

    def test_recovery_overlay_cannot_change_ready_record_even_if_resealed(self):
        task,cfg=self.fixture();cfg['execution_patch_sha256']='patch-sha'
        reader=SimpleNamespace(ask=Mock(side_effect=[[json.dumps(source('outdoors','drivable_road'))],
                    [json.dumps(dict(variants=[variant()]))]]))
        ready=revision.plan_group(reader,Path('/unused'),[task],cfg,{})[0]
        base=run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],predecessor_ids=[task['sample_id']],records=[ready]))
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);key=common.fingerprint(base['predecessor_ids'])
            common.write_json(state/'groups'/(key+'.json'),base)
            altered=run.sealed({**ready,'attempt_errors':[dict(error='altered')]})
            common.write_json(state/'repairs'/(key+'.json'),run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],
                predecessor_ids=base['predecessor_ids'],records=[altered],parent_group_sha256=base['result_sha256'],
                repair_round=1,execution_patch_sha256='patch-sha')))
            with self.assertRaisesRegex(ValueError,'already ready'):revision.read_group(state,[task],cfg)

    def test_index_recovers_crash_after_overlay_but_cannot_overwrite_ready(self):
        task,cfg=self.fixture()
        old=run.sealed(dict(status='failed',source=task['source']))
        new=run.sealed(dict(status='ready',source=task['source']))
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);path=state/'prompts'/(task['sample_id']+'.json')
            common.write_json(path,old)
            revision.sync_prompt(state,task,new,old)
            self.assertEqual(revision.sealed_read(path),new)
            revision.sync_prompt(state,task,new,old)
            with self.assertRaisesRegex(ValueError,'lineage'):revision.sync_prompt(state,task,old,new)

    def test_composer_failure_recovery_reuses_valid_source_and_same_assigned_style(self):
        task,cfg=self.fixture();evidence=policy.source_evidence(source('outdoors','drivable_road'))
        prior={task['sample_id']:dict(source_evidence=evidence,selected_style='sodium_lamps')}
        reader=SimpleNamespace(ask=Mock(return_value=[json.dumps(dict(variants=[variant()]))]))
        values=revision.plan_group(reader,Path('/unused'),[task],cfg,{},prior=prior)
        self.assertEqual(values[0]['status'],'ready')
        self.assertEqual(reader.ask.call_count,1)
        self.assertEqual(reader.ask.call_args.args[3],'compose')

    def test_source_retry_explicitly_corrects_language_and_json_format(self):
        task,cfg=self.fixture();reader=SimpleNamespace(ask=Mock(side_effect=[['{}'],
            [json.dumps(source('outdoors','drivable_road'))],[json.dumps(dict(variants=[variant()]))]]))
        records=revision.plan_group(reader,Path('/unused'),[task],cfg,{})
        request=reader.ask.call_args_list[1].args[2][0]
        self.assertIn('ENGLISH',request);self.assertIn('paraphrase signs',request)
        self.assertEqual(len(records[0]['attempt_errors']),1)

    def test_later_repair_preserves_earlier_recovery_success_and_failure_history(self):
        task,cfg=self.fixture();cfg.update(execution_patch_sha256='patch-two',execution_patch_history=['patch-one','patch-two'])
        reader=SimpleNamespace(ask=Mock(side_effect=[[json.dumps(source('outdoors','drivable_road'))],
                    [json.dumps(dict(variants=[variant()]))]]))
        ready=revision.plan_group(reader,Path('/unused'),[task],cfg,{})[0]
        failed=run.sealed(dict(status='failed',reason='invalid_source_analysis',source=task['source'],
            predecessor_sample_id=task['sample_id'],cohort_fingerprint=cfg['fingerprint']))
        base=run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],predecessor_ids=[task['sample_id']],records=[failed]))
        earlier=run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],predecessor_ids=base['predecessor_ids'],
            records=[ready],parent_group_sha256=base['result_sha256'],repair_round=1,execution_patch_sha256='patch-one'))
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);key=common.fingerprint(base['predecessor_ids'])
            common.write_json(state/'groups'/(key+'.json'),base)
            common.write_json(state/'repairs'/(key+'.json'),earlier)
            value,parent=revision.read_group(state,[task],cfg)
            reader.ask=Mock(side_effect=AssertionError('Ready source must never be inferred again'))
            self.assertEqual(revision.repair_group(reader,state,[task],cfg,{},value,parent),earlier)
            self.assertEqual(revision.read_group(state,[task],cfg)[0]['records'][0],ready)

    def test_cached_analysis_format_recovery_does_not_rerun_source_vlm(self):
        task,cfg=self.fixture();cfg['execution_patch_sha256']='patch-two'
        failed=run.sealed(dict(status='failed',reason='invalid_source_analysis',source=task['source'],
            predecessor_sample_id=task['sample_id'],cohort_fingerprint=cfg['fingerprint']))
        base=run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],predecessor_ids=[task['sample_id']],records=[failed]))
        evidence=source('outdoors','drivable_road')
        evidence['features']['existing_signs']['evidence']='Existing sign with "PICK UP" lettering'
        raw=json.dumps(evidence,indent=2).replace('\\"PICK UP\\"','"PICK UP"')
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);common.write_json(state/'batches'/'old.json',run.sealed(dict(
                identity=dict(cohort_fingerprint=cfg['fingerprint'],phase='analysis',attempt=0,
                    source_inputs=[task['source']]),responses=[raw])))
            reader=SimpleNamespace(ask=Mock(return_value=[json.dumps(dict(variants=[variant()]))]))
            result=revision.repair_group(reader,state,[task],cfg,{},base,base)
            record=result['records'][0]
            self.assertEqual(record['status'],'ready')
            self.assertEqual(reader.ask.call_count,1);self.assertEqual(reader.ask.call_args.args[3],'compose')
            self.assertTrue(record['source_analysis_reuse']['format_repairs'])


if __name__=='__main__':unittest.main()
