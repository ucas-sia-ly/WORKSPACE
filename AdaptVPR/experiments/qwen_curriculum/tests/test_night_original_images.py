import copy
from pathlib import Path
import tempfile
import unittest

from experiments.qwen_curriculum import common,campaign,run
from experiments.qwen_curriculum import night_original_images as original


class OriginalPromptContinuationTests(unittest.TestCase):
    def fixture(self,root):
        state=root/'old/state';state.mkdir(parents=True)
        job=dict(sample_id='qnight_'+'a'*24,condition='night',style_id='sodium_lamps',
            prompt='The exact archived source-specific night prompt.',seed=12345,
            source_path='/unchanged/source.jpg',source_sha256='b'*64)
        job['record_sha256']=common.fingerprint(job)
        common.write_jsonl(state/'plan.jsonl',[job])
        campaign.frozen(state/'plan_config.json',dict(plan_sha256=common.file_sha256(state/'plan.jsonl')))
        task=dict(style_id=job['style_id'],task_reason='not_committed',committed=False,attempts=0,
            original_state=str(state),original_config_sha256=common.file_sha256(state/'plan_config.json'),
            sample_id=job['sample_id'],original_job=job,original_record_sha256=job['record_sha256'])
        snapshot=root/'snapshot';snapshot.mkdir()
        return job,task,snapshot

    def test_original_prompt_seed_and_job_identity_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            job,task,snapshot=self.fixture(Path(tmp));common.write_jsonl(snapshot/'tasks.jsonl',[task])
            values=list(original.pending_jobs(snapshot))
            self.assertEqual(values[0][1],job)
            self.assertEqual(values[0][1]['prompt'],job['prompt']);self.assertEqual(values[0][1]['seed'],12345)

    def test_all_mercury_sources_are_excluded_without_reassignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            job,task,snapshot=self.fixture(Path(tmp))
            common.write_jsonl(snapshot/'tasks.jsonl',[dict(style_id='mercury_vapor'),task])
            self.assertEqual([j['sample_id'] for _,j in original.pending_jobs(snapshot)],[job['sample_id']])

    def test_committed_or_reserved_job_cannot_receive_another_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,task,snapshot=self.fixture(Path(tmp))
            for change in (dict(committed=True),dict(attempts=1),dict(task_reason='retained')):
                common.write_jsonl(snapshot/'tasks.jsonl',[{**task,**change}])
                with self.assertRaisesRegex(ValueError,'never-reserved'):list(original.pending_jobs(snapshot))

    def test_changed_prompt_is_not_accepted_even_with_a_new_job_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,task,snapshot=self.fixture(Path(tmp));task=copy.deepcopy(task)
            task['original_job']['prompt']='Rewritten prompt is forbidden'
            task['original_job']['record_sha256']=common.fingerprint({k:v for k,v in task['original_job'].items() if k!='record_sha256'})
            task['original_record_sha256']=task['original_job']['record_sha256']
            common.write_jsonl(snapshot/'tasks.jsonl',[task])
            with self.assertRaisesRegex(ValueError,'differs'):list(original.pending_jobs(snapshot))

    def test_execution_proof_rejects_mercury_and_revised_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            job,_,_=self.fixture(Path(tmp))
            def proof(value):return run.sealed(dict(status='original_frozen_pending',job=value,
                original_record_sha256=value['record_sha256']))
            original.validate_proof(job,proof(job))
            for updates in (dict(style_id='mercury_vapor'),dict(sample_id='qnight2_'+'a'*24)):
                changed={**job,**updates};changed['record_sha256']=common.fingerprint({k:v for k,v in changed.items() if k!='record_sha256'})
                with self.assertRaisesRegex(ValueError,'unchanged original'):original.validate_proof(changed,proof(changed))


if __name__=='__main__':unittest.main()
