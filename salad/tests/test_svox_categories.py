"""Category isolation, shared gallery and final-only test reporting; no training."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from evaluate_salad import parse_args, evaluate
from run_ccr_feasibility import load_config, evaluate_checkpoint, summarize
from workflow.evaluation import EvaluationImage, EvaluationSet


class SVOXCategoryTests(unittest.TestCase):
    def test_category_evaluation_extracts_gallery_once_and_keeps_ranks_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);groups=out/'groups.json'
            groups.write_text(json.dumps(dict(svox_night=['queries_night'],svox_rain=['queries_rain'])))
            args=parse_args(['--checkpoint',str(out/'checkpoint.pt'),'--dataset','SVOX',
                '--dataset-root',str(out),'--output',str(out/'unused.json'),'--split','test',
                '--device','cpu','--svox-query-groups',str(groups),'--save-query-results'])
            gallery=[EvaluationImage('r0',out/'r0.jpg'),EvaluationImage('r1',out/'r1.jpg')]
            def load(*a,**kw):
                name=kw['query_subdirs'][0]
                return EvaluationSet(name,gallery,[EvaluationImage(name,out/(name+'.jpg'))],[[0]],
                                     dict(split='test',query_subdirs=[name]))
            descriptors=[torch.eye(2),torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]])]
            model=Mock(image_size=(224,224),config={'agg_config':{'num_clusters':64}})
            with patch('workflow.evaluation.load_evaluation_set',side_effect=load), \
                 patch('workflow.model.load_checkpoint_model',return_value=model), \
                 patch('workflow.evaluation.extract_descriptors',side_effect=descriptors) as extract, \
                 contextlib.redirect_stdout(io.StringIO()):
                evaluate(args)
            self.assertEqual(extract.call_count,3)
            night=json.loads((out/'svox_night.json').read_text())
            rain=json.loads((out/'svox_rain.json').read_text())
            self.assertEqual(night['recall']['R@1'],1.)
            self.assertEqual(rain['recall']['R@1'],0.)
            self.assertEqual(night['query_results'][0]['rank'],1)
            self.assertEqual(rain['query_results'][0]['rank'],2)
            self.assertFalse((out/'unused.json').exists())

    def test_config_is_svox_only_and_test_is_final(self):
        config=Path(__file__).resolve().parents[1]/'configs/ccr_accepted_13710_4090_48g.json'
        c=load_config(config)
        self.assertFalse(c['include_proxy_evaluations'])
        self.assertEqual({e['name'] for e in c['external_validations']},
                         {'svox_val','svox_default','svox_night','svox_rain','svox_snow','svox_sun','svox_overcast'})
        self.assertTrue(all(e['dataset']=='SVOX' for e in c['external_validations']))
        raw=json.loads(config.read_text());raw['external_validations'][1]['cadence']='every'
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.json';path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'final cadence'): load_config(path)

    def test_runner_groups_test_folders_and_skips_intermediate_test(self):
        c=load_config(Path(__file__).resolve().parents[1]/'configs/ccr_accepted_13710_4090_48g.json')
        plan=dict(config=c,evaluations=c['external_validations'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'checkpoint';checkpoint.write_bytes(b'weights')
            calls=[]
            def child(cmd,log,*a,**kw):
                calls.append(cmd);dest=Path(cmd[cmd.index('--output')+1]);dest.parent.mkdir(parents=True,exist_ok=True)
                if '--svox-query-groups' in cmd:
                    names=json.loads(Path(cmd[cmd.index('--svox-query-groups')+1]).read_text())
                else: names=[dest.stem]
                for name in names:
                    (dest.parent/(name+'.json')).write_text(json.dumps(dict(query_results=[dict(query_id='q',rank=1)])))
            with patch('run_ccr_feasibility.run_child',side_effect=child):
                evaluate_checkpoint(plan,checkpoint,root/'intermediate',final=False)
                self.assertEqual(len(calls),1);self.assertIn('val',calls[0])
                evaluate_checkpoint(plan,checkpoint,root/'final',final=True)
                self.assertEqual(len(calls),3)
                self.assertIn('--svox-query-groups',calls[-1]);self.assertIn('test',calls[-1])
                evaluate_checkpoint(plan,checkpoint,root/'final',final=True)
                self.assertEqual(len(calls),3)

    def test_no_proxy_summary_is_valid_and_test_gain_does_not_drive_verdict(self):
        c=load_config(Path(__file__).resolve().parents[1]/'configs/ccr_accepted_13710_4090_48g.json')
        c.update(methods=['A','B5'],steps=1)
        plan=dict(config=c,evaluations=c['external_validations'],evaluation_steps=[1],limitations=[])
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for e in plan['evaluations']:
                for method in ['initial','A','B5']:
                    path=(out/'initial' if method=='initial' else out/'runs'/method/'42/evaluations/1')/(e['name']+'.json')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    good=method=='B5' and e['role']=='test'
                    path.write_text(json.dumps(dict(recall={'R@1':float(good),'R@5':1.,'R@10':1.},
                        query_results=[dict(query_id='q',rank=1 if good else 2)])))
            summarize(plan,out)
            report=json.loads((out/'report.json').read_text())
            comparison=report['comparisons'][-1]
            self.assertEqual(comparison['verdict'],'descriptive_only_no_proxy_validation')
            self.assertEqual(comparison['domains']['svox_night']['role'],'test')
            self.assertIn('svox_night',(out/'report.md').read_text())


if __name__=='__main__': unittest.main()
