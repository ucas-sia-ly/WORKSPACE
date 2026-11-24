"""Fixed update budgets, exact epoch-boundary resume and orchestration contracts."""
import contextlib
import csv
import shutil
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import test_synthetic_aux as fixtures
from train_salad import train
from workflow.model import read_checkpoint
from run_ccr_feasibility import training_command, paired_changes, immutable, summarize, execute


class BudgetTests(unittest.TestCase):
    setUp = fixtures.SyntheticAuxWorkflowTests.setUp
    arguments = fixtures.SyntheticAuxWorkflowTests.arguments

    def test_partial_final_epoch_and_pause_keep_full_schedule_and_exact_resume(self):
        self.initial=self.root/'initial.pt'
        torch.save({'state_dict':self.state,'model_config':self.config},self.initial)
        csv_path=self.data/'Dataframes/Test.csv'
        with csv_path.open() as handle: rows=list(csv.reader(handle))
        extra=[]
        for row in rows[1:]:
            if row[0]=='2':
                added=list(row);added[0]='3';extra.append(added)
        with csv_path.open('w',newline='') as handle: csv.writer(handle).writerows(rows+extra)
        for image in (self.data/'Images/Test').glob('Test_0000002_*.jpg'):
            shutil.copyfile(image,image.with_name(image.name.replace('_0000002_','_0000003_')))
        args=self.arguments(self.root/'continuous','--init-checkpoint',str(self.initial),
            '--epochs','3','--max-steps','5','--batch-size','2','--num-trainable-blocks','1')
        with contextlib.redirect_stdout(io.StringIO()): train(args)
        complete=read_checkpoint(args.output_dir/'checkpoint.pt')
        self.assertEqual(complete['global_step'],5)
        self.assertEqual(complete['epoch'],3)
        args.output_dir=self.root/'resumed'; args.stop_after_epoch=1
        with contextlib.redirect_stdout(io.StringIO()): train(args)
        first=read_checkpoint(args.output_dir/'checkpoint.pt')
        self.assertEqual(first['global_step'],2)
        self.assertEqual(first['total_epochs'],3)
        args.resume=args.output_dir/'checkpoint.pt';args.init_checkpoint=None;args.stop_after_epoch=3
        with contextlib.redirect_stdout(io.StringIO()): train(args)
        resumed=read_checkpoint(args.output_dir/'checkpoint.pt')
        for key,value in complete['state_dict'].items():
            torch.testing.assert_close(value,resumed['state_dict'][key],rtol=0,atol=0)
        self.assertEqual(complete['scheduler_state_dict'],resumed['scheduler_state_dict'])
        self.assertEqual(complete['training_config'],resumed['training_config'])


class RunnerTests(unittest.TestCase):
    def plan(self):
        config=dict(real_data='/data',cities=['City'],backbone_repo='/dino',steps=500,batch_size=8,
            images_per_place=4,num_workers=0,num_trainable_blocks=1,learning_rate=1e-6,precision='16',
            device='cuda',checkpoint='/student.pt',teacher_checkpoint='/teacher.pt',augment=True,
            mix_fraction=.5,exposure_probability=.5,global_weight=.1,feature_weight=.1,ot_weight=.1,
            methods=['A','B5'],seeds=[42],max_real_drop_pp=0,minimum_weather_gain_pp=0)
        return dict(config=config,train_manifest='/train.jsonl',epochs=5,steps_per_epoch=100,
                    calibration_path='/calibration.json',evaluation_steps=[100,500],
                    evaluations=[dict(name='real',kind='real'),dict(name='rain',kind='synthetic_proxy')],limitations=[])

    def test_all_methods_same_budget_and_required_switches(self):
        plan=self.plan()
        commands={m:training_command(plan,m,42,Path('/run'),2,False) for m in ('A','M','B0','B4','B5')}
        for cmd in commands.values():
            self.assertEqual(cmd[cmd.index('--max-steps')+1],'500')
            self.assertEqual(cmd[cmd.index('--epochs')+1],'5')
            self.assertEqual(cmd[cmd.index('--stop-after-epoch')+1],'2')
            self.assertNotIn('--reliability-ot',cmd)
        self.assertNotIn('--synthetic-aux',commands['A'])
        self.assertEqual(commands['A'][commands['A'].index('--synthetic-fraction')+1],'0')
        self.assertIn('--synthetic-aux',commands['B0'])
        self.assertNotIn('--feature-distill-weight',commands['B0'])
        self.assertEqual(commands['B4'][-1],'uniform')
        self.assertEqual(commands['B5'][-1],'verified')
        resumed=training_command(plan,'A',42,Path('/run'),3,True)
        self.assertIn('--resume',resumed);self.assertNotIn('--init-checkpoint',resumed)

    def test_pairwise_results_and_frozen_inputs(self):
        a={'query_results':[dict(query_id='a',rank=1),dict(query_id='b',rank=3)]}
        b={'query_results':[dict(query_id='a',rank=2),dict(query_id='b',rank=1)]}
        self.assertEqual(paired_changes(a,b),dict(improved_to_correct=1,regressed_to_wrong=1,evaluated_queries=2))
        with self.assertRaises(ValueError): paired_changes(a,dict(query_results=[]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';immutable(path,'a');immutable(path,'a')
            with self.assertRaises(ValueError): immutable(path,'b')

    def test_final_budget_summary_does_not_select_intermediate_peak(self):
        plan=self.plan()
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for domain in ('real','rain'):
                initial={'recall':{'R@1':.5,'R@5':1.,'R@10':1.},
                    'query_results':[dict(query_id='a',rank=1),dict(query_id='b',rank=2)]}
                path=out/'initial'/f'{domain}.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(initial))
                for method in ('A','B5'):
                    for step in (100,500):
                        result=initial
                        if method=='B5' and step==100:
                            result={**initial,'recall':{'R@1':1.,'R@5':1.,'R@10':1.},
                                    'query_results':[dict(query_id='a',rank=1),dict(query_id='b',rank=1)]}
                        path=out/'runs'/method/'42'/'evaluations'/str(step)/f'{domain}.json'
                        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result))
            (out/'runs/B5/42/metrics_step_500.json').write_text(json.dumps({
                'local_distillation':{'feature_loss':.2,'feature_fraction':.7},
                'runtime':{'step_seconds':.3}}))
            summarize(plan,out)
            report=json.loads((out/'report.json').read_text())
            self.assertTrue(all(row['step']==500 for row in report['final']))
            comparison=report['comparisons'][-1]
            self.assertEqual(comparison['domains']['rain']['mean_pp'],0)
            self.assertIn('no_prespecified',comparison['verdict'])
            with (out/'training_metrics.csv').open() as handle: metrics=list(csv.DictReader(handle))
            self.assertEqual(metrics[0]['metric.local_distillation.feature_fraction'],'0.7')
            self.assertEqual(metrics[0]['metric.runtime.step_seconds'],'0.3')

    def test_orchestrator_restarts_only_unfinished_epoch(self):
        plan=self.plan();plan.update(epochs=3,steps_per_epoch=2,evaluation_steps=[2,4,5])
        plan['config'].update(steps=5,methods=['A'],torch_threads=1)
        calls=[];failed=[False]
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            def child(command,log,threads,**kwargs):
                if '--calibrate-plan' in command: return
                epoch=int(command[command.index('--stop-after-epoch')+1]);calls.append(command)
                if epoch==2 and not failed[0]:
                    failed[0]=True;raise RuntimeError('simulated interruption')
                destination=Path(command[command.index('--output-dir')+1]);destination.mkdir(parents=True,exist_ok=True)
                torch.save(dict(epoch=epoch,global_step=min(epoch*2,5),training_config={'max_steps':5},
                                metrics={'loss':1.}),destination/'checkpoint.pt')
            def evaluate(plan,checkpoint,destination,**kwargs):
                destination.mkdir(parents=True,exist_ok=True)
                for e in plan['evaluations']:
                    (destination/(e['name']+'.json')).write_text(json.dumps(dict(
                        recall={'R@1':1.,'R@5':1.,'R@10':1.},query_results=[dict(query_id='q',rank=1)])))
            with patch('run_ccr_feasibility.run_child',child),patch('run_ccr_feasibility.evaluate_checkpoint',evaluate):
                with self.assertRaisesRegex(RuntimeError,'simulated'): execute(plan,out)
                execute(plan,out)
            self.assertEqual([cmd[cmd.index('--stop-after-epoch')+1] for cmd in calls],['1','2','2','3'])
            self.assertNotIn('--resume',calls[0])
            self.assertTrue(all('--resume' in cmd for cmd in calls[1:]))
            result=read_checkpoint(out/'runs/A/42/checkpoint.pt')
            self.assertEqual(result['global_step'],5)


if __name__=='__main__': unittest.main()
