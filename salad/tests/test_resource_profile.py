"""Resource budgets and numerically equivalent acceleration paths."""
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from workflow.evaluation import exact_retrieval
from workflow.reliability_cache import load_teacher_image, preprocess_teacher_image
from workflow.resource_limits import (GIB, configure_cuda_budget, memory_violation,
                                     monitored_run)
from run_ccr_feasibility import load_config, training_command


class ResourceProfileTests(unittest.TestCase):
    def test_bounded_reference_cache_preserves_exact_ranks_ties_and_missing_positives(self):
        torch.manual_seed(12)
        refs=torch.randn(19,13);refs[3]=refs[2]
        queries=torch.cat([refs[2:3],torch.randn(6,13)])
        positives=[[3],[1,2],[],[17],[0,15],[12],[18]]
        baseline=exact_retrieval(refs,queries,positives,query_chunk_size=3,reference_chunk_size=5)
        for budget in (1,100000):
            cached=exact_retrieval(refs,queries,positives,query_chunk_size=3,
                                   reference_chunk_size=5,reference_cache_bytes=budget)
            self.assertEqual(cached,baseline)

    def test_predecoded_pair_resize_is_identical(self):
        im=Image.fromarray(np.random.default_rng(1).integers(0,256,(71,93,3),dtype=np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'image.jpg';im.save(path)
            with Image.open(path) as source:
                decoded=source.convert('RGB')
            torch.testing.assert_close(load_teacher_image(path,(28,42)),
                                       preprocess_teacher_image(decoded,(28,42)),rtol=0,atol=0)

    def test_cuda_budget_uses_current_free_memory_and_reserve(self):
        with patch.dict(os.environ,{'CCR_CUDA_BUDGET_GIB':'40','CCR_CUDA_RESERVE_GIB':'4'}), \
             patch('torch.cuda.current_device',return_value=0), \
             patch('torch.cuda.mem_get_info',return_value=(12*GIB,48*GIB)), \
             patch('torch.cuda.set_per_process_memory_fraction') as setter:
            result=configure_cuda_budget('cuda:0')
            setter.assert_called_once_with(8/48,torch.device('cuda:0'))
            self.assertEqual(result['allocator_budget_gib'],8)
            configure_cuda_budget('cuda')
            self.assertEqual(setter.call_args.args[1],torch.device('cuda:0'))
        with patch.dict(os.environ,{'CCR_CUDA_BUDGET_GIB':'40','CCR_CUDA_RESERVE_GIB':'4'}), \
             patch('torch.cuda.current_device',return_value=0), \
             patch('torch.cuda.mem_get_info',return_value=(4*GIB,48*GIB)):
            with self.assertRaisesRegex(RuntimeError,'Insufficient'): configure_cuda_budget('cuda')

    def test_host_pressure_stops_only_owned_process_group(self):
        limits=dict(host_rss_limit_gib=20,host_available_reserve_gib=3)
        normal=dict(rss_bytes=GIB,available_bytes=10*GIB)
        pressure=dict(rss_bytes=21*GIB,available_bytes=10*GIB)
        self.assertIsNone(memory_violation(normal,limits))
        self.assertIn('MemAvailable',memory_violation(dict(rss_bytes=GIB,available_bytes=GIB),limits))
        child=Mock(pid=98765)
        with patch('workflow.resource_limits.memory_snapshot',side_effect=[normal,pressure]), \
             patch('subprocess.Popen',return_value=child) as spawn, \
             patch('workflow.resource_limits.stop_process_group') as stop:
            with self.assertRaisesRegex(RuntimeError,'Resource guard stopped'):
                monitored_run(['fake'],handle=io.StringIO(),env={},limits=limits)
            self.assertTrue(spawn.call_args.kwargs['start_new_session'])
            stop.assert_called_once_with(child)

    def test_profile_keeps_pk_and_loss_choices_and_bounds_prefetch(self):
        config=load_config(Path(__file__).resolve().parents[1]/'configs/ccr_feasibility_4090_48g.json')
        plan=dict(config=config,train_manifest='/train',epochs=5,steps_per_epoch=100,calibration_path='/cal')
        command=training_command(plan,'B4',42,Path('/run'),1,False)
        for key,value in [('--batch-size','8'),('--images-per-place','4'),('--precision','bf16'),
                          ('--prefetch-factor','1'),('--teacher-batch-size','2'),('--score-distill-weight','0')]:
            self.assertEqual(command[command.index(key)+1],value)
        self.assertEqual(config['resource_limits']['host_rss_limit_gib'],20)


if __name__=='__main__': unittest.main()
