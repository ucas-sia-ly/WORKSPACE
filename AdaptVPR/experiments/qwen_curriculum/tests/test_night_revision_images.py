import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.qwen_curriculum import common,campaign,run,night_revision as revision
from experiments.qwen_curriculum import night_revision_images as images
import test_night_revision as revision_tests
from test_night_diversity_policy_v2 import source,plan


class RevisedImagesTests(unittest.TestCase):
    def record(self):
        task,cfg=revision_tests.RevisionTests().fixture()
        value=plan(source('outdoors','drivable_road'))
        record=run.sealed(dict(status='ready',source=task['source'],plan=value,
            predecessor_sample_id=task['sample_id'],cohort_fingerprint=cfg['fingerprint'],
            job=revision.make_job(task,value,cfg)))
        return task,cfg,record

    def test_local_never_starts_before_planner_exits(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(revision,'validate_completion') as validate:
            root=Path(tmp);(root/'state').mkdir();(root/'state/prompts_complete.json').write_text('{}')
            self.assertFalse(images.planner_finished(root,{},1234));validate.assert_not_called()

    def test_local_requires_valid_complete_marker_not_just_stopped_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'state').mkdir()
            self.assertFalse(images.planner_finished(root,{},0))
            (root/'state/prompts_complete.json').write_text('{}')
            with patch.object(revision,'validate_completion',side_effect=ValueError('bad barrier')):
                with self.assertRaisesRegex(ValueError,'bad barrier'):images.planner_finished(root,{},0)
            with patch.object(revision,'validate_completion',return_value={}) as validate:
                self.assertTrue(images.planner_finished(root,{},0));validate.assert_called_once()

    def test_execution_loader_keeps_exact_validated_prompt_and_deferred_quality(self):
        _,_,record=self.record()
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);common.write_jsonl(state/'plan.jsonl',[record['job']])
            common.write_jsonl(state/'prompt_receipts.jsonl',[record])
            cfg=dict(stage=images.VERSION,quality=images.quality(),files_sha256={},num_jobs=1,
                domain_counts={'night':1},plan_sha256=common.file_sha256(state/'plan.jsonl'))
            campaign.frozen(state/'plan_config.json',cfg)
            _,jobs=images.load_execution_plan(state)
            self.assertEqual(jobs,[record['job']])
            altered=copy.deepcopy(record);altered['job']['prompt']='changed'
            common.write_jsonl(state/'prompt_receipts.jsonl',[run.sealed(altered)])
            with self.assertRaises(ValueError):images.load_execution_plan(state)

    def test_stream_only_publishes_complete_ready_receipts_and_is_idempotent(self):
        task,cfg,record=self.record();cfg.update(batch_size=1,coverage_counts={'x':1})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'images';state=Path(tmp)/'source/state';state.mkdir(parents=True);root.mkdir()
            common.write_jsonl(state/'tasks.jsonl',[task])
            key=common.fingerprint([task['sample_id']])
            common.write_json(state/'groups'/(key+'.json'),run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],
                predecessor_ids=[task['sample_id']],records=[record])))
            images.cluster.save(root/'feed.json',dict(version=1,entries=[],predecessors={}))
            settings=dict(source_root=str(state.parent),files_sha256={})
            images.cluster.freeze(root/'cluster.json',settings)
            images.cluster.freeze(root/'nodes/local/node.json',dict(service_identity={}))
            # Group alone is not sufficient while the per-source commit index is pending.
            self.assertEqual(images.publish_ready(root,settings,cfg),0)
            common.write_json(state/'prompts'/(task['sample_id']+'.json'),record)
            before=common.file_sha256(state/'groups'/(key+'.json'))
            self.assertEqual(images.publish_ready(root,settings,cfg),1)
            self.assertEqual(images.publish_ready(root,settings,cfg),0)
            feed=images.worker.read_sealed(root/'feed.json')
            self.assertEqual(len(feed['entries']),1)
            self.assertEqual(common.file_sha256(state/'groups'/(key+'.json')),before)
            self.assertFalse((root/'images').exists())

    def test_failed_source_is_not_upgraded_or_executed(self):
        task,cfg,record=self.record();cfg.update(batch_size=1,coverage_counts={})
        failed=run.sealed({k:v for k,v in record.items() if k not in ('job','plan','result_sha256')})
        failed=run.sealed({**failed,'status':'failed'})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'source/state';state.mkdir(parents=True)
            common.write_jsonl(state/'tasks.jsonl',[task]);key=common.fingerprint([task['sample_id']])
            common.write_json(state/'groups'/(key+'.json'),run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],
                predecessor_ids=[task['sample_id']],records=[failed])))
            common.write_json(state/'prompts'/(task['sample_id']+'.json'),failed)
            images.cluster.save(root/'feed.json',dict(version=1,entries=[],predecessors={}))
            self.assertEqual(images.publish_ready(root,dict(source_root=str(state.parent)),cfg),0)

    def test_durable_ownership_is_not_stolen_after_transfer_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);descriptor=dict(original_state='/source/state',original_config_sha256='a'*64,jobs=1)
            images.cluster.save(root/'feed.json',dict(entries=[descriptor],predecessors={}))
            images.cluster.save(root/'ownership.json',dict(cluster_fingerprint='cohort',owners={}))
            for node in ('local','gpu2'):images.cluster.save(root/'nodes'/node/'queue.json',dict(node=node,entries=[]))
            with patch.object(images,'load',return_value={'fingerprint':'cohort'}),\
                    patch.object(images.cluster,'stage_assignment',side_effect=RuntimeError('transfer failed')):
                with self.assertRaisesRegex(RuntimeError,'transfer failed'):images.grant(root,'local')
                self.assertIsNone(images.grant(root,'gpu2'))
            self.assertEqual(images.worker.read_sealed(root/'ownership.json')['owners'],{'/source/state':'local'})

    def test_prune_requires_completed_shard_and_exact_owned_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'nodes/gpu2';entry={'state':str(directory/'shards/s00000/state')}
            images.cluster.save(directory/'completed.json',dict(node_fingerprint='node',entries=[entry]))
            images.cluster.save(directory/'queue.json',dict(node='gpu2',entries=[entry]))
            outside=root/'do-not-delete.png';outside.write_bytes(b'keep')
            row=dict(sample_id='qnight2_'+'a'*24,output_path=str(outside),raw_output_path=str(outside),
                output_sha256=common.file_sha256(outside),raw_output_sha256=common.file_sha256(outside),metadata_sha256='b'*64)
            common.write_jsonl(Path(entry['state'])/'results.jsonl',[row])
            ack=root/'ack.json';images.cluster.freeze(ack,dict(node_fingerprint='node',entry=entry,
                results_fingerprint=common.fingerprint([row]),files=[]))
            with patch.object(images.worker,'load_node',return_value={'node':'gpu2','fingerprint':'node'}):
                with self.assertRaisesRegex(ValueError,'exact owned'):images.prune_remote(root,ack)
            self.assertTrue(outside.exists())

    def test_prune_only_removes_verified_remote_copies_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'nodes/gpu2';entry={'state':str(directory/'shards/s00000/state')}
            images.cluster.save(directory/'completed.json',dict(node_fingerprint='node',entries=[entry]))
            images.cluster.save(directory/'queue.json',dict(node='gpu2',entries=[entry]))
            sample='qnight2_'+'a'*24
            output=directory/'images/night'/(sample+'.png');raw=directory/'images/raw/night'/(sample+'.png')
            sidecar=raw.with_suffix('.json')
            for p in (output,raw,sidecar):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'confirmed bytes')
            primary=root/'primary-preserved.png';primary.write_bytes(output.read_bytes())
            row=dict(sample_id=sample,output_path=str(output),output_sha256=common.file_sha256(output),
                raw_output_path=str(raw),raw_output_sha256=common.file_sha256(raw),metadata_sha256=common.file_sha256(sidecar))
            common.write_jsonl(Path(entry['state'])/'results.jsonl',[row]);ack=root/'ack.json'
            images.cluster.freeze(ack,dict(node_fingerprint='node',entry=entry,results_fingerprint=common.fingerprint([row]),
                files=[dict(path=str(p),sha256=common.file_sha256(p)) for p in (output,raw,sidecar)]))
            with patch.object(images.worker,'load_node',return_value={'node':'gpu2','fingerprint':'node'}):
                self.assertTrue(images.prune_remote(root,ack)['remote_copies_removed'])
                images.prune_remote(root,ack)
            self.assertTrue(primary.exists());self.assertTrue((Path(entry['state'])/'results.jsonl').exists())
            for p in (output,raw,sidecar):self.assertFalse(p.exists())


if __name__=='__main__':unittest.main()
