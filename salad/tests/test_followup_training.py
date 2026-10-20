"""Integration contracts for opt-in retention, local eval, monitoring and resume."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from train_salad import train, auxiliary_model_rng,parse_args
from workflow.model import SALADModel,read_checkpoint
import test_frozen_teacher as fixture
import test_local_distillation as local_fixture


class FollowupTrainingTests(unittest.TestCase):
    setUp=fixture.FrozenTeacherTests.setUp
    arguments=fixture.FrozenTeacherTests.arguments

    def args(self,out,*extra):
        return self.arguments(out,'--init-checkpoint',str(self.teacher_path),'--teacher-checkpoint',str(self.teacher_path),
            '--num-trainable-blocks','1','--real-sampling','coverage_pk','--synthetic-aux',
            '--synthetic-exposure-prob','1','--batch-size','3',*extra)

    def run_train(self,args):
        with contextlib.redirect_stdout(io.StringIO()):train(args)
        return read_checkpoint(args.output_dir/'checkpoint.pt')

    def test_retention_and_mining_accumulate_before_one_step_and_teacher_no_probe(self):
        args=self.args(self.root/'combined','--epochs','1','--real-global-distill-weight','.2',
                       '--real-global-distill-batch-size','2','--log-mined-pairs','--isolate-auxiliary-rng')
        events=[];backward=torch.Tensor.backward;step=torch.optim.AdamW.step
        def bp(value,*a,**kw):events.append('backward');return backward(value,*a,**kw)
        def opt(value,*a,**kw):events.append('step');return step(value,*a,**kw)
        with patch.object(torch.Tensor,'backward',bp),patch.object(torch.optim.AdamW,'step',opt), \
             patch('workflow.model.profile_teacher_targets',side_effect=AssertionError('interface-only probe must be skipped')):
            actual=self.run_train(args)
        self.assertEqual(events.count('step'),1);self.assertEqual(events[-1],'step')
        self.assertEqual(events.count('backward'),5)
        metrics=actual['metrics'];self.assertEqual(metrics['real_descriptor_retention']['images'],6)
        self.assertEqual(metrics['real_descriptor_retention']['loss'],0.)
        self.assertIn('no_mined_pair_batch_fraction',metrics['mining'])
        self.assertEqual(metrics['mining']['real_synthetic_mined_positive'],0)
        self.assertEqual(actual['training_config']['real_descriptor_retention']['student_mode'],'eval_with_grad_fp32')

    def test_retention_resume_exact_and_weight_change_rejected(self):
        args=self.args(self.root/'full','--epochs','2','--real-global-distill-weight','.2','--isolate-auxiliary-rng')
        expected=self.run_train(args)
        args.output_dir=self.root/'resumed';args.init_checkpoint=None;args.resume=self.root/'full/checkpoint_epoch_001.pt'
        actual=self.run_train(args)
        for key,value in expected['state_dict'].items():torch.testing.assert_close(value,actual['state_dict'][key],rtol=0,atol=0)
        args.output_dir=self.root/'changed';args.real_global_distill_weight=.3
        with self.assertRaisesRegex(ValueError,'training_config'):self.run_train(args)

    def test_place_exclusion_removes_real_and_synthetic_and_hashes_file(self):
        excluded=self.root/'exclude.json';excluded.write_text(json.dumps([['Test',1]]))
        args=self.args(self.root/'exclude','--epochs','1','--exclude-places',str(excluded))
        actual=self.run_train(args)
        self.assertEqual(actual['dataset_summary']['num_places'],2)
        self.assertEqual(actual['dataset_summary']['num_real_images'],4)
        self.assertEqual(actual['dataset_summary']['num_synthetic_images'],1)
        self.assertEqual(actual['dataset_summary']['excluded_by_place_file'],1)
        self.assertIn('sha256',actual['training_config']['excluded_places'])

    def test_auxiliary_model_rng_does_not_advance_main_torch_stream(self):
        args=self.args(self.root/'unused','--isolate-auxiliary-rng')
        before=torch.get_rng_state()
        with auxiliary_model_rng(args,torch.device('cpu'),3,0):first=torch.rand(5)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        with auxiliary_model_rng(args,torch.device('cpu'),3,0):second=torch.rand(5)
        torch.testing.assert_close(first,second,rtol=0,atol=0)


class LocalEvalIntegrationTests(unittest.TestCase):
    setUp=local_fixture.LocalWorkflowTests.setUp
    arguments=local_fixture.LocalWorkflowTests.arguments
    args=local_fixture.LocalWorkflowTests.args
    run_training=local_fixture.LocalWorkflowTests.run_training

    def test_local_student_eval_retains_gradients_and_restores_main_train(self):
        args=self.args(self.root/'local_eval','--epochs','1','--local-distill-eval')
        forward=SALADModel.forward;local_modes=[]
        def capture(model,images,*a,**kw):
            if kw.get('return_aux') and any(p.requires_grad for p in model.parameters()):
                local_modes.append((model.training,torch.is_grad_enabled()))
            return forward(model,images,*a,**kw)
        with patch.object(SALADModel,'forward',capture):actual=self.run_training(args)
        self.assertTrue(local_modes);self.assertTrue(all(not mode and grad for mode,grad in local_modes))
        self.assertEqual(actual['training_config']['local_distillation']['student_local_mode'],'eval_with_grad')
        self.assertEqual(actual['global_step'],1)


class FollowupCommandTests(unittest.TestCase):
    def test_full_real_commands_keep_explicit_exclusion_and_feature_only(self):
        from run_ccr_feasibility import load_config,training_command
        root=Path(__file__).resolve().parents[1]
        for name,method in [('am_fullreal','A'),('b0_fullreal','B0'),
                            ('b0_retention_fullreal','B0'),('feature_retention_fullreal','B1')]:
            c=load_config(root/'configs'/f'ccr_followup_{name}.json')
            plan=dict(config=c,train_manifest='/manifest',training_exclude_places='/heldout',
                      epochs=3,steps_per_epoch=1950,calibration_path='/calibration')
            cmd=training_command(plan,method,42,Path('/run'),1,False)
            self.assertNotIn('--synthetic-places-only',cmd)
            self.assertEqual(cmd[cmd.index('--exclude-places')+1],'/heldout')
            args=parse_args(cmd[2:])
            self.assertEqual(args.batch_size,32);self.assertTrue(args.log_mined_pairs)
            if method=='B1':
                self.assertEqual(args.ot_distill_weight,0)
                self.assertEqual(args.score_distill_weight,0)
                self.assertGreater(args.feature_distill_weight,0)
                self.assertTrue(args.local_distill_eval)
            if 'retention' in name:self.assertEqual(args.real_global_distill_weight,1.)


if __name__=='__main__':unittest.main()
