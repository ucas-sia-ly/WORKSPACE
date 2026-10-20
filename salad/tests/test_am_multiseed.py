"""CPU contracts, staged selection, statistics and real adapter resume tests."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import run_am_multiseed as am
import train_am
from train_salad import parse_args
from workflow.am_statistics import exact_mcnemar, holm, paired, query_ranks
import test_real_baseline as pool_fixture
import test_pretrained_initialization as model_fixture


def result(ranks):
    valid = [v for v in ranks if v is not None]
    return dict(recall={f'R@{k}':sum(v<=k for v in valid)/len(valid) for k in (1,5,10)},
                query_results=[dict(query_id=str(i),rank=v) for i,v in enumerate(ranks)])


class StatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_known_values_symmetry_and_large_n(self):
        self.assertEqual(exact_mcnemar(0,0),1)
        self.assertAlmostEqual(exact_mcnemar(5,0),.0625)
        self.assertAlmostEqual(exact_mcnemar(23,10),exact_mcnemar(10,23))
        self.assertAlmostEqual(exact_mcnemar(1,1),1)
        self.assertTrue(0<=exact_mcnemar(10000,9990)<=1)
        with self.assertRaises(ValueError): exact_mcnemar(-1,0)

    def test_holm_fixed_family_not_shrunk_by_incomplete_runs(self):
        self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])
        self.assertEqual(holm([.01],45),[.45])
        with self.assertRaises(ValueError): holm([float('nan')])

    def test_pairs_recall_denominators_and_group_bootstrap(self):
        a,b=result([1,2,12,None]),result([2,1,1,None])
        row=paired(a,b,groups={str(i):str(i//2) for i in range(4)},bootstrap_samples=100)
        self.assertEqual((row['improved_to_correct'],row['regressed_to_wrong'],row['evaluated_queries']),(2,1,3))
        self.assertAlmostEqual(row['delta_pp'],100/3)
        self.assertEqual(row,paired(a,b,groups={str(i):str(i//2) for i in range(4)},bootstrap_samples=100))
        self.assertEqual(row['grouped_bootstrap']['groups'],2)
        self.assertIsNone(paired(a,b)['grouped_bootstrap'])
        with self.assertRaises(ValueError): paired(a,b,groups={'0':'g'})
        wrong=copy.deepcopy(b);wrong['query_results'][-1]['rank']=1
        with self.assertRaises(ValueError): paired(a,wrong)
        wrong=copy.deepcopy(b);wrong['recall']['R@1']=.99
        with self.assertRaises(ValueError): query_ranks(wrong)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.c=am.load_config(am.ccr.ROOT/'configs/am_multiseed.json')
        self.plan=dict(config=self.c,train_places=62374,excluded_places='/fixed/excluded.json',train_manifest='/fixed/train.jsonl')

    def test_seed_protocol_and_finite_order(self):
        self.assertEqual(self.c['confirmation_seeds'],[43,44,45])
        self.assertNotIn(self.c['screen_seed'],self.c['confirmation_seeds'])
        p=am.screening_candidates(self.plan,'batch_size',[])
        self.assertEqual([v['batch_size'] for v in p],[8,16,32])
        self.assertEqual({v['learning_rate'] for v in p},{3e-6})
        lr=am.screening_candidates(self.plan,'learning_rate',[{'selected':p[0]}])
        self.assertEqual({v['batch_size'] for v in lr},{8})
        blocks=am.screening_candidates(self.plan,'num_trainable_blocks',[{'selected':p[0]},{'selected':lr[1]}])
        self.assertEqual({v['learning_rate'] for v in blocks},{3e-6})
        with self.assertRaises(ValueError): am.screening_candidates(self.plan,'learning_rate',[])

    def test_equal_formal_budget_and_independent_initialization(self):
        spec=am.candidate(self.c['screening']['center'])
        seen=set()
        for seed in self.c['confirmation_seeds']:
            for method in self.c['methods']:
                directory=Path('/new')/str(seed)/method
                args=parse_args(am.command(self.plan,spec,seed,method,directory,3)[2:])
                self.assertEqual(str(args.init_checkpoint),self.c['checkpoint'])
                self.assertIsNone(args.resume)
                self.assertFalse(args.synthetic_aux or args.teacher_checkpoint or args.reliability_ot or args.synthetic_places_only)
                self.assertEqual((args.batch_size,args.epochs,args.learning_rate,args.num_trainable_blocks),(16,3,3e-6,2))
                self.assertEqual((args.max_steps,args.max_batches_per_epoch),(0,0))
                self.assertEqual(args.synthetic_fraction,self.c['methods'][method])
                seen.add(str(args.output_dir))
                resumed=parse_args(am.command(self.plan,spec,seed,method,directory,3,True)[2:])
                self.assertEqual(str(resumed.resume),str(directory/'checkpoint.pt'))
                self.assertEqual(resumed.epochs,3)
        self.assertEqual(len(seen),9)
        self.assertEqual(am.run_plan(self.plan,spec)['steps_per_epoch'][spec['name']],3899)

    def test_config_rejects_seed_drift_grid_and_precision(self):
        original=am.ccr.ROOT/'configs/am_multiseed.json'
        with tempfile.TemporaryDirectory() as tmp:
            # Resolved paths keep validation independent of the temporary cwd.
            for edit in ({'confirmation_seeds':[42,43,44]}, {'precision':'16'},
                         {'screening':{**self.c['screening'],'order':list(reversed(am.STAGES))}}):
                c={**self.c,**edit};path=Path(tmp)/'config.json';path.write_text(json.dumps(c))
                with self.assertRaises(ValueError): am.load_config(path)

    def test_test_symlink_cannot_be_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'images/test').mkdir(parents=True)
            (root/'images/val').symlink_to('test',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'resolves to test'):
                am.validation_inventory({'svox_root':str(root)})

    def test_only_existing_val_domains_are_discovered(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);val=root/'images/val';(val/'queries').mkdir(parents=True)
            (root/'images/test/queries_night').mkdir(parents=True)
            image=val/'q.jpg';image.write_bytes(b'fixture')
            data=SimpleNamespace(references=[],queries=[SimpleNamespace(id='q',path=image)],summary=lambda:{'num_queries':1})
            with patch('workflow.evaluation.load_evaluation_set',return_value=data) as loader:
                found=am.validation_inventory({'svox_root':str(root)})
            self.assertEqual([e['name'] for e in found['evaluations']],['svox_val'])
            self.assertIn('night',found['missing_natural_weather_validation'])
            self.assertEqual(loader.call_args.kwargs['split'],'val')


class SelectionTests(unittest.TestCase):
    _write_csv=pool_fixture.RealBaselineTests._write_csv
    def setUp(self):
        pool_fixture.RealBaselineTests.setUp(self)
        c=am.load_config(am.ccr.ROOT/'configs/am_multiseed.json')
        self.plan['config'].update(screening=c['screening'],methods=c['methods'],statistics=c['statistics'],primary_method='M50')
        self.plan.update(evaluations=[dict(name='svox_val',split='val',role='validation')],
                         summaries={'svox_val':{}},missing_natural_weather_validation=['night','rain'],grouping=None,
                         formal_policy='fixture independent equal budgets')
        self.initial=result([1]*8+[2]*2)
        self.write_evaluation(self.out/'initial',self.checkpoint,self.initial)

    def write_evaluation(self,directory,checkpoint,value):
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/'svox_val.json';am.ccr.write_json(path,value)
        am.ccr.write_json(path.with_suffix('.receipt.json'),dict(checkpoint_sha256=am.ccr.digest(checkpoint),
            result_sha256=am.ccr.digest(path),protocol=self.plan['evaluations'][0]))

    def test_selection_waits_all_uses_signal_and_val_not_test(self):
        specs=am.screening_candidates(self.plan,'batch_size',[])
        self.assertIsNone(am.selection_record(self.plan,self.out,'batch_size',[]))
        for i,spec in enumerate(specs):
            directory=self.out/'screen'/spec['name'];directory.mkdir(parents=True)
            checkpoint=directory/'checkpoint.pt';checkpoint.write_bytes(b'model')
            metrics=pool_fixture.RealBaselineTests.metrics(self)
            if i==2: metrics['mining']['active_anchor_fraction']=0
            am.ccr.write_json(directory/'metrics_epoch_001.json',metrics)
            self.write_evaluation(directory/'final_validation',checkpoint,result([1]*(8+i)+[2]*(2-i)))
        record=am.selection_record(self.plan,self.out,'batch_size',[])
        self.assertEqual(record['selected']['batch_size'],16)
        # A test file, however high, is neither parsed nor used.
        (self.out/'test').mkdir();(self.out/'test/svox_night.json').write_text('invalid test data')
        self.assertEqual(am.selection_record(self.plan,self.out,'batch_size',[]),record)

    def test_formal_statistics_keep_seeds_separate_and_reject_wrong_budget(self):
        prior=[]
        for stage in am.STAGES:
            for spec in am.screening_candidates(self.plan,stage,prior):
                directory=self.out/'screen'/spec['name'];directory.mkdir(parents=True,exist_ok=True)
                checkpoint=directory/'checkpoint.pt';checkpoint.write_bytes(b'model')
                am.ccr.write_json(directory/'metrics_epoch_001.json',pool_fixture.RealBaselineTests.metrics(self))
                self.write_evaluation(directory/'final_validation',checkpoint,self.initial)
            prior.append(am.selection_record(self.plan,self.out,stage,prior))
        formal=am.frozen_selection(self.plan,self.out)
        self.assertEqual(formal['optimizer_updates'],3)
        for seed in self.plan['config']['confirmation_seeds']:
            for method in self.plan['config']['methods']:
                directory=self.out/'runs'/str(seed)/method;directory.mkdir(parents=True)
                checkpoint=directory/'checkpoint.pt';checkpoint.write_bytes(b'model')
                value=self.initial if method=='A' else result([1]*9+[2])
                self.write_evaluation(directory/'final_validation',checkpoint,value)
                am.ccr.write_json(directory/'audit.json',dict(checkpoint_sha256=am.ccr.digest(checkpoint),
                    optimizer_updates=3,actual_epochs=3,real_exposures=24 if method=='A' else 20,
                    synthetic_exposures=0 if method=='A' else 4,weather={} if method=='A' else {'night':4},
                    epoch_metrics=[pool_fixture.RealBaselineTests.metrics(self,epoch=e) for e in (1,2,3)]))
        report=am.report(self.plan,self.out)
        self.assertEqual(report['status'],'formal_complete_descriptive')
        self.assertEqual(report['holm_family_size'],45)
        self.assertEqual(len(report['paired_statistics']),45)
        self.assertTrue(all(r['complete'] for r in report['seed_distributions']))
        self.assertTrue(all(set(r['deltas_by_seed'])=={'43','44','45'} for r in report['seed_distributions']))
        path=self.out/'runs/43/A/audit.json';bad=json.loads(path.read_text());bad['optimizer_updates']=2
        am.ccr.write_json(path,bad)
        with self.assertRaisesRegex(ValueError,'wrong final budget'): am.report(self.plan,self.out)

    def test_prepare_filters_both_reserved_groups_and_retains_full_real_pool(self):
        from PIL import Image
        c=am.load_config(am.ccr.ROOT/'configs/am_multiseed.json')
        out=self.root/'new_am'
        c.update(real_data=str(self.gsv),checkpoint=str(self.checkpoint),reserved_places=str(self.split),output_dir=str(out))
        curriculum=self.root/'qwen_curriculum'
        for name in ('state','images/accepted','images/train'): (curriculum/name).mkdir(parents=True)
        rows=[]
        for i,(city,place) in enumerate((('Alpha',1),('Beta',1),('Alpha',2))):
            image=curriculum/f'images/train/{i}.png';Image.new('RGB',(12,12)).save(image)
            (curriculum/f'images/accepted/{i}.png').symlink_to(f'../train/{i}.png')
            rows.append(dict(source_path=str(self.paths[city,place,0]),output_path=str(image),condition='rain',
                             passed=True,eligible_for_training=True))
        manifest=curriculum/'state/training_manifest.jsonl'
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        c['synthetic_manifest']=str(manifest)
        with patch.object(am.baseline,'validation_inventory',return_value={'fixture':True}),patch.object(am,'validation_inventory',return_value=dict(evaluations=[],summaries={},missing_natural_weather_validation=['night'])):
            plan=am.prepare(c,out)
        self.assertEqual(plan['train_places'],2)
        self.assertEqual(plan['train_real_images'],8)
        self.assertEqual(plan['synthetic_snapshot']['train_images'],1)
        self.assertEqual(plan['synthetic_snapshot']['excluded'],{'reserved':2})
        self.assertEqual(plan['expected_exposure_per_epoch']['M50']['synthetic'],1)
        self.assertEqual(plan['expected_exposure_per_epoch']['M50']['total'],8)
        self.assertEqual(len(Path(plan['train_manifest']).read_text().splitlines()),1)


class AdapterResumeTests(unittest.TestCase):
    def setUp(self):
        model_fixture.PretrainedInitializationTests.setUp(self)
        import torch
        self.initial=self.root/'initial.pt'
        torch.save({'state_dict':self.state,'model_config':self.config},self.initial)
        rows=[json.loads(line) for line in self.manifest.read_text().splitlines()]
        for row in rows: row['condition']='night'
        self.manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def args(self,directory,*extra):
        return model_fixture.PretrainedInitializationTests.arguments(self,directory,
            '--real-sampling','coverage_pk','--log-mined-pairs','--seed','43',*extra)

    def test_m_resume_weights_rng_optimizer_schedule_and_weather_match(self):
        import torch
        from workflow.model import read_checkpoint
        with contextlib.redirect_stdout(io.StringIO()):
            train_am.train(self.args(self.root/'full','--init-checkpoint',str(self.initial)))
            train_am.train(self.args(self.root/'split','--init-checkpoint',str(self.initial),'--stop-after-epoch','1'))
            train_am.train(self.args(self.root/'split','--resume',str(self.root/'split/checkpoint.pt')))
        full=read_checkpoint(self.root/'full/checkpoint.pt');split=read_checkpoint(self.root/'split/checkpoint.pt')
        for key,value in full['state_dict'].items(): torch.testing.assert_close(value,split['state_dict'][key],rtol=0,atol=0)
        self.assertEqual(full['global_step'],split['global_step'])
        self.assertEqual(full['scheduler_state_dict'],split['scheduler_state_dict'])
        self.assertEqual(full['rng_state']['python'],split['rng_state']['python'])
        torch.testing.assert_close(full['rng_state']['torch'],split['rng_state']['torch'],rtol=0,atol=0)
        self.assertEqual(full['metrics']['sampling']['synthetic_weather_exposures'],{'night':2})
        self.assertEqual(full['metrics']['synthetic_exposure'],2)
        for key,value in full['optimizer_state_dict']['state'].items():
            for name,tensor in value.items(): torch.testing.assert_close(tensor,split['optimizer_state_dict']['state'][key][name],rtol=0,atol=0)
        bad=self.args(self.root/'bad','--synthetic-aux')
        with self.assertRaises(ValueError): train_am.train(bad)

    def test_weather_counter_does_not_change_training_rng_or_weights(self):
        import torch
        import train_salad
        from workflow.model import read_checkpoint
        with contextlib.redirect_stdout(io.StringIO()):
            train_am.train(self.args(self.root/'wrapped','--init-checkpoint',str(self.initial)))
            train_salad.train(self.args(self.root/'plain','--init-checkpoint',str(self.initial)))
        a,b=(read_checkpoint(self.root/p/'checkpoint.pt') for p in ('wrapped','plain'))
        for key,value in a['state_dict'].items(): torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)
        torch.testing.assert_close(a['rng_state']['torch'],b['rng_state']['torch'],rtol=0,atol=0)

    def test_weather_counts_workers_0_2_4(self):
        from torch.utils.data import DataLoader
        from workflow.training_data import MixedGSVCitiesDataset,CoveragePlaceBatchSampler
        for workers in (0,2,4):
            dataset=MixedGSVCitiesDataset(self.data,self.manifest,images_per_place=2,min_images_per_place=2,
                synthetic_fraction=.5,image_size=(28,28),augment=False,real_sampling='coverage_pk',return_sampling_metadata=True)
            dataset.set_epoch(1)
            sampler=CoveragePlaceBatchSampler(len(dataset),2,42);sampler.set_epoch(1)
            coverage=train_am.WeatherCoverage(dataset)
            for batch in DataLoader(dataset,batch_sampler=sampler,num_workers=workers): coverage.update(batch[-1])
            self.assertEqual(coverage.summary()['synthetic_weather_exposures'],{'night':2})

    def test_runner_epoch_recovery_audit_and_contract(self):
        c=am.load_config(am.ccr.ROOT/'configs/am_multiseed.json')
        out=self.root/'runner';out.mkdir()
        excluded=out/'excluded.json';excluded.write_text('[]')
        c.update(real_data=str(self.data),checkpoint=str(self.initial),backbone_repo=str(self.repo),
                 output_dir=str(out),device='cpu',precision='32',num_workers=0,images_per_place=2,epochs=2)
        spec=am.candidate(dict(batch_size=2,learning_rate=1e-5,num_trainable_blocks=1))
        plan=dict(config=c,train_places=3,real_exposures_per_epoch=6,excluded_places=str(excluded),
                  train_manifest=str(self.manifest),evaluations=[],
                  input_hashes={str(self.initial):am.ccr.digest(self.initial),str(excluded):am.ccr.digest(excluded)})
        am.ccr.write_json(out/'plan.json',plan)
        directory=out/'runs/43/M50'
        calls=[]
        def child(cmd,log,*args,**kwargs):
            if '--audit-run' in cmd:
                am.audit_run(plan,Path(cmd[cmd.index('--audit-run')+1]));return
            log.parent.mkdir(parents=True,exist_ok=True);log.touch()
            args=parse_args(cmd[2:]);args.image_size=[28,28]
            calls.append(args)
            train_am.train(args)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(am.ccr,'run_child',side_effect=child),patch.object(am.ccr,'evaluate_checkpoint'):
            am.run_one(plan,spec,43,'M50',directory,1)
            am.run_one(plan,spec,43,'M50',directory,2)
        self.assertEqual(len(calls),2)
        self.assertIsNone(calls[0].resume)
        self.assertEqual(calls[1].resume,directory/'checkpoint.pt')
        audit=json.loads((directory/'audit.json').read_text())
        self.assertEqual(audit['optimizer_updates'],2)
        self.assertEqual(audit['weather'],{'night':4})
        self.assertEqual(set(audit['parameter_displacement']),{'backbone','aggregator'})
        self.assertEqual(audit['lr_curve'][1]['used'],5e-6)
        from workflow.model import read_checkpoint
        state=read_checkpoint(directory/'checkpoint.pt')
        with self.assertRaises(ValueError): am.checked_state(state,plan,spec,44,'M50')
        with self.assertRaises(ValueError): am.checked_state(state,plan,spec,43,'A')


if __name__=='__main__': unittest.main()
