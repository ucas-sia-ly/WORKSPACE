"""CCR 4 CPU contracts and tiny real backward/resume; optional tiny CUDA RNG smoke."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

import train_ccr4
import run_ccr_stage4 as runner
from workflow.ccr4_diagnostics import auxiliary_rng,local_mode,confidence_pair,support_digest,local_patch_record
from workflow.synthetic_losses import feature_distillation_loss,conditional_ot_distillation_loss
from workflow.model import read_checkpoint
import test_pretrained_initialization as fixture
import test_local_distillation as local_fixture


class RNGTests(unittest.TestCase):
    def test_cpu_isolation_changes_masks_per_step_and_reproduces_resume(self):
        torch.manual_seed(20);state=torch.get_rng_state()
        masks=[]
        for step in range(4):
            before=torch.get_rng_state()
            with auxiliary_rng(43,step,0,'cpu'):
                masks.append(torch.nn.functional.dropout(torch.ones(256),.5,True))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertFalse(torch.equal(masks[0],masks[1]))
        with auxiliary_rng(43,2,0,'cpu'): resumed=torch.nn.functional.dropout(torch.ones(256),.5,True)
        torch.testing.assert_close(masks[2],resumed,rtol=0,atol=0)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        with auxiliary_rng(43,0,0,'cpu',False): torch.rand(1)
        self.assertFalse(torch.equal(state,torch.get_rng_state()))

    @unittest.skipUnless(os.environ.get('CCR4_CUDA_SMOKE')=='1','opt-in tiny CUDA smoke')
    def test_cuda_cpu_states_and_two_backward_one_step(self):
        torch.manual_seed(7);torch.cuda.manual_seed(8)
        layer=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.Dropout(.5)).cuda().train()
        optimizer=torch.optim.AdamW(layer.parameters(),lr=1e-4)
        images=torch.ones(4,8,device='cuda')
        outputs=[]
        for step in range(3):
            optimizer.zero_grad();real=layer(images);real.square().mean().backward()
            cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
            with auxiliary_rng(43,step,0,'cuda'):
                syn=layer(images);syn.square().mean().backward()
                outputs.append(syn.detach().clone())
            self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
            self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in layer.parameters()))
            optimizer.step()
        self.assertFalse(torch.equal(outputs[0],outputs[1]))


class LocalTests(unittest.TestCase):
    def test_common_A_support_q_detach_no_unknown_promotion(self):
        q=torch.tensor([[.2,.6,.8,0]],requires_grad=True);a=torch.tensor([[True,True,False,True]])
        idx=torch.arange(4)[None]
        uniform,verified=confidence_pair(a,q,idx)
        self.assertTrue(torch.equal(uniform>0,verified>0))
        self.assertEqual(support_digest(idx,uniform),support_digest(idx,verified))
        s=torch.randn(1,4,4,requires_grad=True);t=torch.randn(1,4,4,requires_grad=True)
        loss=feature_distillation_loss(s,t,idx,verified);loss.backward()
        self.assertIsNone(t.grad);self.assertIsNone(q.grad)
        self.assertGreater(float(s.grad.abs().sum()),0)
        zero=feature_distillation_loss(s,t,torch.full_like(idx,-1),torch.zeros_like(q))
        self.assertEqual(float(zero.detach()),0.)

    def test_salad_only_dropout_off_keeps_autograd_and_backbone_mode(self):
        from types import SimpleNamespace
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.backbone=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Dropout(.5))
                self.aggregator=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Dropout(.5))
            def forward(self,x): return self.aggregator(self.backbone(x))
        model=Model().train();x=torch.randn(3,4)
        with local_mode(model,'salad_dropout_off'):
            self.assertTrue(model.backbone[1].training);self.assertFalse(model.aggregator[1].training)
            model(x).sum().backward()
        self.assertTrue(model.aggregator[1].training)
        self.assertGreater(float(model.backbone[0].weight.grad.abs().sum()),0)
        with local_mode(model,'eval'): self.assertFalse(model.backbone[1].training)
        self.assertTrue(model.backbone[1].training)

    def test_feature_ot_independent_numeric_and_weighted_diagnostics(self):
        idx=torch.arange(6)[None];q=torch.tensor([[0,.1,.2,.5,.8,1.]])
        for dtype in (torch.float32,torch.bfloat16,torch.float16):
            s=torch.randn(1,8,6,dtype=dtype,requires_grad=True);t=torch.randn_like(s,requires_grad=True)
            logits=torch.randn(1,5,6,dtype=dtype,requires_grad=True);target=torch.randn_like(logits,requires_grad=True)
            feat=feature_distillation_loss(s,t,idx,q);ot=conditional_ot_distillation_loss(logits,target,idx,q)
            feat.backward();self.assertIsNone(logits.grad)
            ot.backward();self.assertIsNone(target.grad);self.assertIsNone(t.grad)
            self.assertTrue(torch.isfinite(feat+ot))
            self.assertTrue((logits.grad[:,-1]==0).all())
        row=local_patch_record(dict(backbone_local_features=s,log_assignment=logits),dict(backbone_local_features=t,log_assignment=target),idx,q)
        self.assertAlmostEqual(row['effective_sample_size'],float(q.sum().square()/q.square().sum()),places=5)
        self.assertAlmostEqual(sum(b['mass'] for b in row['confidence_bands']),float(q.sum()),places=5)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        fixture.PretrainedInitializationTests.setUp(self)
        self.initial=self.root/'initial.pt';torch.save(dict(state_dict=self.state,model_config=self.config),self.initial)

    def args(self,directory,aux=False,weight=.1,rng='isolated',resume=None,stop=None):
        args=fixture.PretrainedInitializationTests.arguments(self,directory,'--real-sampling','coverage_pk',
            '--synthetic-fraction','0','--synthetic-exposure-prob','1','--synthetic-global-weight',str(weight),
            '--log-mined-pairs','--init-checkpoint',str(self.initial))
        args.synthetic_aux=aux;args.stage4_rng=rng;args.stage4_local_mode='salad_dropout_off';args.stage4_gradient_interval=1
        if not aux: args.synthetic_manifest=None
        if resume: args.resume=resume;args.init_checkpoint=None
        if stop: args.stop_after_epoch=stop
        return args

    def run_train(self,*a,**kw):
        with contextlib.redirect_stdout(io.StringIO()): train_ccr4.train(self.args(*a,**kw))
        return read_checkpoint(a[0]/'checkpoint.pt')

    def test_zero_isolated_equals_A_original_zero_changes_trajectory(self):
        a=self.run_train(self.root/'a');z=self.run_train(self.root/'z',True,0)
        original=self.run_train(self.root/'original',True,0,'original')
        for name,value in a['state_dict'].items(): torch.testing.assert_close(value,z['state_dict'][name],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,original['state_dict'][n]) for n,v in a['state_dict'].items()))
        self.assertEqual(z['training_config']['stage4']['rng'],'isolated')
        rows=[json.loads(r) for r in (self.root/'z/attribution.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['gradients']['groups']['all']['losses'].get('syn_weighted',{}).get('raw_norm',0)==0 for r in rows))

    def test_full_resume_and_diagnostics_are_identical(self):
        full=self.run_train(self.root/'full',True)
        self.run_train(self.root/'split',True,stop=1)
        split=self.run_train(self.root/'split',True,resume=self.root/'split/checkpoint.pt')
        for name,value in full['state_dict'].items(): torch.testing.assert_close(value,split['state_dict'][name],rtol=0,atol=0)
        for name,state in full['optimizer_state_dict']['state'].items():
            for key,value in state.items(): torch.testing.assert_close(value,split['optimizer_state_dict']['state'][name][key],rtol=0,atol=0)
        self.assertEqual(full['scheduler_state_dict'],split['scheduler_state_dict'])
        self.assertEqual((self.root/'full/attribution.jsonl').read_text(),(self.root/'split/attribution.jsonl').read_text())
        self.assertEqual(full['rng_state']['synthetic_aux'],split['rng_state']['synthetic_aux'])

    def test_runner_complete_cpu_epoch_audit_and_immutable_contract(self):
        from run_am_multiseed import load_config,candidate,ccr
        c=load_config(ccr.ROOT/'configs/am_multiseed.json')
        out=self.root/'stage4';out.mkdir()
        excluded=out/'excluded.json';excluded.write_text('[]')
        c.update(real_data=str(self.data),checkpoint=str(self.initial),backbone_repo=str(self.repo),device='cpu',precision='32',
                 images_per_place=2,num_workers=0,epochs=2,output_dir=str(out))
        config=runner.load_config(ccr.ROOT/'configs/ccr_stage4.json');config.update(output_dir=str(out),gradient_interval=1)
        plan=dict(config=config,parent=dict(config=c,train_places=3,train_manifest=str(self.manifest),excluded_places=str(excluded),
            evaluations=[],input_hashes={str(self.initial):ccr.digest(self.initial),str(excluded):ccr.digest(excluded)}),
            probe_pairs=[json.loads(self.manifest.read_text().splitlines()[0])])
        protocol=dict(spec=candidate(dict(batch_size=2,learning_rate=1e-5,num_trainable_blocks=1)),epochs=2,updates=2,seeds=[43,44,45])
        ccr.write_json(out/'plan.json',plan)
        def child(cmd,log,*args,**kw):
            if '--worker' in cmd:
                runner.audit(plan,Path(cmd[cmd.index('--run-dir')+1]));return
            log.parent.mkdir(parents=True,exist_ok=True);log.touch()
            parsed=train_ccr4.parse_args(cmd[2:]);parsed.image_size=[28,28]
            train_ccr4.train(parsed)
        with contextlib.redirect_stdout(io.StringIO()),patch.object(ccr,'run_child',side_effect=child),patch.object(ccr,'evaluate_checkpoint'):
            runner.execute_one(plan,out,protocol,'B0',43,'formal',2,.03)
            runner.execute_one(plan,out,protocol,'B0',43,'formal',2,.03)
        directory=out/'formal/43/B0';audit=json.loads((directory/'audit.json').read_text())
        self.assertEqual(audit['optimizer_updates'],2)
        self.assertEqual(audit['actual_epochs'],2)
        self.assertIn('descriptor_displacement',audit)
        self.assertIn('attribution_summary',audit)
        self.assertEqual(sum(v['queries'] for v in audit['synthetic_weather'].values()),audit['auxiliary_queries'])
        with self.assertRaises(ValueError): runner.execute_one(plan,out,protocol,'B0',43,'formal',2,.1)


class LocalWorkflowTests(local_fixture.LocalWorkflowTests):
    # Only new adapter-specific methods; inherited legacy cases are removed below.
    def adapted(self,path,feature,ot,mode):
        args=self.args(path)
        args.synthetic_fraction=0;args.feature_distill_weight=feature;args.ot_distill_weight=ot
        args.stage4_rng='isolated';args.stage4_local_mode='salad_dropout_off';args.stage4_gradient_interval=1
        args.pair_confidence_mode=mode;args.log_mined_pairs=True
        from workflow.model import FrozenSALADTeacher
        instances=[];original=FrozenSALADTeacher.forward
        def forward(teacher,*a,**kw):
            instances.append(teacher)
            return original(teacher,*a,**kw)
        with contextlib.redirect_stdout(io.StringIO()),patch.object(FrozenSALADTeacher,'forward',forward): train_ccr4.train(args)
        self.assertTrue(instances)
        self.assertTrue(all(not p.requires_grad and p.grad is None for teacher in instances for p in teacher.parameters()))
        rows=[json.loads(r) for r in (path/'attribution.jsonl').read_text().splitlines()]
        return read_checkpoint(path/'checkpoint.pt'),rows

    def test_adapter_B4_B5_identical_support_and_independent_losses(self):
        a,ra=self.adapted(self.root/'b4',.1,.1,'uniform')
        b,rb=self.adapted(self.root/'b5',.1,.1,'verified')
        support=lambda rows:[(r['step'],v['source_path'],v['generated_path'],v['support_sha256']) for r in rows for v in r['local_pairs']]
        self.assertEqual(support(ra),support(rb))
        self.assertTrue(support(ra))
        f,rf=self.adapted(self.root/'feature',.1,0,'uniform')
        o,ro=self.adapted(self.root/'ot',0,.1,'uniform')
        self.assertTrue(any('feature_weighted' in r['gradients']['weights'] for r in rf))
        self.assertTrue(all('ot_weighted' not in r['gradients']['weights'] for r in rf))
        self.assertTrue(any('ot_weighted' in r['gradients']['weights'] for r in ro))
        self.assertTrue(all('feature_weighted' not in r['gradients']['weights'] for r in ro))

    def test_local_resume_teacher_frozen_and_q_control(self):
        full,rows=self.adapted(self.root/'full',.1,.1,'verified')
        path=self.root/'resumed';args=self.args(path)
        args.synthetic_fraction=0;args.ot_distill_weight=.1;args.pair_confidence_mode='verified';args.log_mined_pairs=True
        args.stage4_rng='isolated';args.stage4_local_mode='salad_dropout_off';args.stage4_gradient_interval=1
        args.stop_after_epoch=1
        with contextlib.redirect_stdout(io.StringIO()): train_ccr4.train(args)
        args.stop_after_epoch=None;args.resume=path/'checkpoint.pt';args.init_checkpoint=None
        with contextlib.redirect_stdout(io.StringIO()): train_ccr4.train(args)
        resumed=read_checkpoint(path/'checkpoint.pt')
        for n,v in full['state_dict'].items(): torch.testing.assert_close(v,resumed['state_dict'][n],rtol=0,atol=0)
        self.assertTrue(any('q_gradient_control' in p for r in rows for p in r['local_pairs']))

    @unittest.skipUnless(os.environ.get('CCR4_CUDA_SMOKE')=='1','opt-in tiny SALAD CUDA smoke')
    def test_cuda_bf16_local_backward_chain(self):
        import time
        path=self.root/'gpu_smoke';args=self.args(path)
        args.synthetic_fraction=0;args.device='cuda';args.precision='bf16';args.epochs=1
        args.stage4_rng='isolated';args.stage4_local_mode='salad_dropout_off';args.stage4_gradient_interval=1
        args.log_mined_pairs=True;args.profile_training=True
        before=runner.ccr.digest(self.teacher_path);started=time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()): train_ccr4.train(args)
        saved=read_checkpoint(path/'checkpoint.pt')
        self.assertEqual(saved['global_step'],1)
        self.assertTrue(all(torch.isfinite(v).all() for v in saved['state_dict'].values() if v.is_floating_point()))
        self.assertEqual(before,runner.ccr.digest(self.teacher_path))
        print(json.dumps(dict(scope='tiny fixture SALAD BF16, not real-data performance',optimizer_updates=1,
              seconds=time.monotonic()-started,runtime=saved['metrics']['runtime'])))

# Avoid rerunning inherited training cases under an unrelated class name.
for _name in dir(local_fixture.LocalWorkflowTests):
    if _name.startswith('test_'): setattr(LocalWorkflowTests,_name,None)


class ProtocolTests(unittest.TestCase):
    def test_declarations_weights_rng_local_only_and_gate(self):
        c=runner.load_config(runner.ccr.ROOT/'configs/ccr_stage4.json');p={'config':c}
        self.assertEqual(runner.specification(p,'B0-zero')['rng'],'original')
        self.assertEqual(runner.specification(p,'B0-zero-isolated')['global_weight'],0)
        for w in c['global_weights']: self.assertEqual(runner.specification(p,runner.weight_name(w))['global_weight'],w)
        a=runner.specification(p,'B4',.03);b=runner.specification(p,'B5',.03)
        self.assertEqual({k:v for k,v in a.items() if k!='confidence'},{k:v for k,v in b.items() if k!='confidence'})
        self.assertEqual(runner.specification(p,'B-Feature',.03)['ot'],0)
        self.assertEqual(runner.specification(p,'B-OT',.03)['feature'],0)
        self.assertIn('B4',runner.REFERENCES['B5'])
        self.assertIn('B-OT',runner.REFERENCES['B-OT-train'])
        self.assertEqual(sum(map(len,runner.REFERENCES.values()))*3*3,198)
        with tempfile.TemporaryDirectory() as tmp:
            p['config']['stage3_dir']=tmp
            with self.assertRaisesRegex(ValueError,'formal_protocol.json missing'): runner.freeze(p,Path(tmp))

    def test_local_gate_cannot_pass_partial_or_regressed_B0(self):
        c=runner.load_config(runner.ccr.ROOT/'configs/ccr_stage4.json')
        plan=dict(config=c,parent=dict(config=dict(confirmation_seeds=[43,44,45],epochs=3,checkpoint='/initial')))
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner.am,'read_result',return_value=None):
            self.assertFalse(runner.b0_gate(plan,Path(tmp))['passed'])

    def test_local_gate_checks_every_seed_and_recall_regression(self):
        c=runner.load_config(runner.ccr.ROOT/'configs/ccr_stage4.json')
        config=dict(confirmation_seeds=[43,44,45],epochs=3,checkpoint='/initial',
            acceptance=dict(min_active_anchor_fraction=.05,max_empty_mining_fraction=.1,max_val_drop_pp=.1,max_seed_std_pp=.1))
        plan=dict(config=c,parent=dict(config=config,real_exposures_per_epoch=8))
        good=dict(recall={'R@1':.96});initial={'svox_val':good}
        metrics=dict(real_exposure=8,synthetic_exposure=0,loss=.1,
            mining=dict(active_anchor_fraction=.2,no_mined_pair_batch_fraction=0,real_real_mined_positive=2,real_real_mined_negative=4))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);runner.ccr.write_json(out/'protocol.json',dict(updates=3))
            for seed in (43,44,45):
                for method in ('A','B0'):
                    directory=out/'formal'/str(seed)/method;directory.mkdir(parents=True)
                    cp=directory/'checkpoint.pt';cp.write_bytes(b'fixture')
                    runner.ccr.write_json(directory/'audit.json',dict(actual_epochs=3,optimizer_updates=3,
                        checkpoint_sha256=runner.ccr.digest(cp),epoch_metrics=[metrics]*3))
            with patch.object(runner.am,'read_result',return_value=initial),patch.object(runner,'result',return_value=initial):
                self.assertTrue(runner.b0_gate(plan,out)['passed'])
            def regressed(p,directory): return {'svox_val':dict(recall={'R@1':.95 if directory.name=='B0' else .96})}
            with patch.object(runner.am,'read_result',return_value=initial),patch.object(runner,'result',side_effect=regressed):
                self.assertFalse(runner.b0_gate(plan,out)['passed'])


if __name__=='__main__': unittest.main()
