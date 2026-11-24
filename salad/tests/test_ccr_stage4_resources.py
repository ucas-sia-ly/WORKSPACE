"""Low-memory runtime keeps inherited scientific settings and archives intact."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_ccr_stage4 as runner


class ResourceTests(unittest.TestCase):
    def test_runtime_override_does_not_mutate_parent_or_budget(self):
        plan={'parent':{'config':{'num_workers':4,'epochs':3,'seed':43},
                        'train_manifest':'frozen.jsonl'},
              'config':{'runtime_num_workers':0}}
        before=copy.deepcopy(plan)
        runtime=runner.runtime_parent(plan)
        self.assertEqual(runtime['config'],{'num_workers':0,'epochs':3,'seed':43})
        self.assertEqual(plan,before)
        self.assertEqual(runtime['train_manifest'],plan['parent']['train_manifest'])
        del plan['config']['runtime_num_workers']
        self.assertIs(runner.runtime_parent(plan),plan['parent'])

    def test_command_uses_runtime_workers_for_all_methods(self):
        plan={'parent':{'config':{'num_workers':4},'train_manifest':'frozen.jsonl'},
              'config':{'runtime_num_workers':0,'local_mode':'salad_dropout_off',
                        'gradient_interval':100,'exposure_probability':.5,'temperature':.1}}
        def training(parent,*args):
            return ['python','train.py','--num-workers',str(parent['config']['num_workers'])]
        with patch.object(runner.am.baseline,'training_command',side_effect=training):
            for method in ('A','B0-zero','B0-isolated'):
                cmd=runner.command(plan,{'spec':{}},method,42,Path('unused'),1,False)
                self.assertEqual(cmd[cmd.index('--num-workers')+1],'0')
        self.assertEqual(plan['parent']['config']['num_workers'],4)

    def test_resource_stop_not_reported_as_never_trained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(runner.pilot_progress(root)['status'],'not_started')
            resource=dict(minimum_system_available_gib=2.95,peak_tree_rss_gib=8.2,
                          limits=dict(host_available_reserve_gib=3,host_rss_limit_gib=20))
            (root/'train.log').write_text('RESOURCE '+json.dumps(resource)+'\n')
            self.assertEqual(runner.pilot_progress(root)['status'],'resource_stopped')
            (root/'checkpoint.pt').touch()
            self.assertTrue(runner.pilot_progress(root)['completed_epoch_checkpoint'])
            (root/'audit.json').write_text('{}')
            self.assertEqual(runner.pilot_progress(root)['status'],'audited')

    def test_runtime_config_validation(self):
        template=json.loads((runner.ccr.ROOT/'configs/ccr_stage4_lowmem.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            for value in (-1,True,1.5):
                path.write_text(json.dumps({**template,'runtime_num_workers':value}))
                with self.assertRaisesRegex(ValueError,'runtime_num_workers'):
                    runner.load_config(path)


if __name__=='__main__':
    unittest.main()
