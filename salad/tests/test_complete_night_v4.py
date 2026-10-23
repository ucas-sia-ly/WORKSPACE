import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import complete_night_v4 as workflow


class ContinuationTests(unittest.TestCase):
    def test_cache_receipt_and_identity_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.jsonl';cache=root/'signals.jsonl'
            manifest.write_text(json.dumps({'sample_id':'one'})+'\n')
            cache.write_text(json.dumps({'sample_id':'one','error':'missing geometry'})+'\n')
            workflow.save(Path(str(cache)+'.receipt.json'),
                dict(input_manifest_sha256=workflow.sha(manifest),signals_sha256=workflow.sha(cache)))
            self.assertEqual(workflow.verify_cache(manifest,cache)['errors'],1)
            manifest.write_text(json.dumps({'sample_id':'two'})+'\n')
            with self.assertRaisesRegex(ValueError,'receipt'):workflow.verify_cache(manifest,cache)

    def test_atomic_stage_resume_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def fake(command,**kwargs):
                Path(command[0]).write_text('complete')
            with patch.object(workflow,'monitored_run',side_effect=fake) as called:
                stage=workflow.stage(root,'sample',lambda path:[str(path/'result.txt')])
                workflow.stage(root,'sample',lambda path:self.fail('completed stage reran'))
                self.assertEqual(called.call_count,1)
            (stage/'result.txt').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'Changed completed'):workflow.committed(root,'sample')

    def test_failed_stage_never_becomes_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(workflow,'monitored_run',side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):workflow.stage(root,'sample',lambda path:['fake'])
            self.assertIsNone(workflow.committed(root,'sample'))

    def test_data_check_cannot_launch_a_training_command(self):
        unsafe={'training_preflight':{'data_check_commands':[
            {'method':'A','argv':['python','train_am.py']}]}}
        with patch.object(workflow,'report',return_value=unsafe), \
             patch.object(workflow,'stage') as called:
            with self.assertRaisesRegex(ValueError,'never trains'):
                workflow.run_data_checks({},Path('/unused'))
            called.assert_not_called()

    def test_data_check_requires_filtered_synthetic_and_other_places(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'execution.log'
            def write(synthetic,total):
                log.write_text(json.dumps(dict(places_with_synthetic=synthetic,num_places=total),indent=2)+
                    '\nData check passed; no model loaded and no training started.\n')
            command=['python','--check-data','--synthetic-place-slots','8']
            write(8,20)
            self.assertEqual(workflow.validate_data_check(log,command)['num_places'],20)
            for synthetic,total in ((7,20),(8,8)):
                write(synthetic,total)
                with self.assertRaisesRegex(ValueError,'Filtered'):
                    workflow.validate_data_check(log,command)

    def test_combination_preserves_rows_prefers_v4_and_excludes_heldout_places(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cal=root/'calibration.json';cal.write_text('{}')
            exclude=root/'exclude.json';exclude.write_text(json.dumps([['City',3]]))
            a=dict(city='City',place_id=1,condition='night',sample_id='new',audit_record_sha256='unchanged')
            old={**a,'sample_id':'old'}
            extra={**a,'place_id':2,'sample_id':'extra'}
            heldout={**a,'place_id':3,'sample_id':'heldout'}
            snow={**a,'place_id':4,'condition':'snow','sample_id':'snow'}
            for version,values in [('v4_audit',[a,heldout]),('v3_audit',[old,extra,snow])]:
                directory=root/version/'export';directory.mkdir(parents=True)
                (directory/'training_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
            abandoned=root/'attempts'/'combined_001';abandoned.mkdir(parents=True)
            (abandoned/'partial').write_text('interrupted')
            output=workflow.combine(root,cal,exclude)
            self.assertEqual(list(workflow.rows(output/'training_manifest.jsonl')),[a,extra])
            self.assertEqual(json.loads((output/'report.json').read_text())['counts']['pass'],2)
            self.assertEqual(workflow.combine(root,cal,exclude),output)
            self.assertTrue((abandoned/'partial').exists())


if __name__=='__main__':unittest.main()
