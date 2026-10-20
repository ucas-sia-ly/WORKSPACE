"""CPU tests for the isolated augmentation/retention experiment."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import train_real_recovery as training
import run_real_recovery as runner
import test_pretrained_initialization as fixture
from workflow.model import read_checkpoint
from workflow.training_data import CoveragePlaceBatchSampler


class WindowTests(unittest.TestCase):
    def test_windows_equal_continuous_stream_including_cycle_boundary(self):
        base = CoveragePlaceBatchSampler(13, 4, 43)
        expected = []
        for epoch in range(3):
            base.set_epoch(epoch)
            expected.extend(list(base))
        windows = training.WindowSampler(13, 4, 43, 3)
        actual = []
        for epoch in range(4):
            windows.set_epoch(epoch)
            actual.extend(list(windows))
        self.assertEqual(actual, expected)
        resumed = training.WindowSampler(13, 4, 43, 3)
        resumed.set_epoch(2)
        self.assertEqual(list(resumed), expected[6:9])

    def test_config_cannot_sneak_in_test_or_other_losses(self):
        path = runner.ccr.ROOT / "configs/real_recovery.json"
        config = json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "config.json"
            dest.write_text(json.dumps({**config, "test_evaluations": []}))
            with self.assertRaises(ValueError):
                runner.load_config(dest)
            dest.write_text(json.dumps({**config, "budgets": [501, 1000]}))
            with self.assertRaises(ValueError):
                runner.load_config(dest)

    def test_command_factorial_controls_do_not_change_lr_or_budget(self):
        pc = dict(epochs=3, checkpoint="/initial", num_workers=4)
        plan = dict(config=dict(window_steps=100, budgets=[500,1000], num_workers=0,
                                retention_weight=.1,retention_batch_size=8),
                    parent=dict(config=pc), formal=dict(spec={}))
        def base(parent, spec, seed, directory, until, resume):
            self.assertEqual(parent["config"]["epochs"],10)
            self.assertEqual(parent["config"]["num_workers"],0)
            self.assertEqual(until,5)
            return ["python","train.py","--save-every","10"]
        with patch.object(runner.am.baseline,"training_command",side_effect=base):
            for case in runner.CASES:
                cmd = runner.command(plan,case,42,Path("/unused"),500)
                self.assertEqual(cmd[cmd.index("--max-steps")+1],"1000")
                self.assertEqual("--real-global-distill-weight" in cmd,case.endswith("retention"))
        self.assertEqual(pc["num_workers"],4)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.PretrainedInitializationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.initial = f.root / "initial.pt"
        torch.save(dict(state_dict=f.state, model_config=f.config, global_step=100), self.initial)

    def args(self, directory, policy="weak", retention=False, extra=()):
        f = self.fixture
        cmd = ["--real-data",str(f.data),"--output-dir",str(directory),
               "--epochs","3","--window-steps","1","--max-steps","3",
               "--batch-size","2","--images-per-place","2","--min-images-per-place","2",
               "--image-size","28","28","--num-workers","0","--precision","32","--device","cpu",
               "--synthetic-fraction","0","--real-sampling","coverage_pk","--seed","43",
               "--backbone-repo",str(f.repo),"--augmentation-policy",policy,"--log-mined-pairs"]
        if "--resume" not in extra:
            cmd += ["--init-checkpoint",str(self.initial)]
        if retention:
            cmd += ["--teacher-checkpoint",str(self.initial),"--real-global-distill-weight",".1",
                    "--real-global-distill-batch-size","2"]
        return training.parse_args(cmd + list(extra))

    def run_training(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            training.train(args)
        return read_checkpoint(args.output_dir / "checkpoint.pt")

    def dataset(self, policy):
        return training.RecoveryDataset(self.fixture.data, augmentation_policy=policy, synthetic_fraction=0,
            images_per_place=2,min_images_per_place=2,image_size=(28,28),real_sampling="coverage_pk",
            seed=43,return_sampling_metadata=True)

    def test_augmentation_matches_reference_transform_without_rng_leak(self):
        from PIL import Image
        for policy in ("weak","randaugment"):
            dataset = self.dataset(policy)
            python_state, torch_state = random.getstate(), torch.get_rng_state()
            first = dataset[(0,0)]
            second = dataset[(0,0)]
            torch.testing.assert_close(first[0],second[0],rtol=0,atol=0)
            self.assertEqual(random.getstate(),python_state)
            self.assertTrue(torch.equal(torch.get_rng_state(),torch_state))
            self.assertFalse(first[2].any())
            self.assertEqual(first[0].shape,(2,3,28,28))
        # Constructor exactly declares the repository's RandAugment parameters.
        transform = self.dataset("randaugment").transform.transforms[1]
        self.assertEqual(transform.num_ops,3)
        self.assertEqual(transform.magnitude,9)
        dataset = self.dataset("randaugment")
        place = dataset.places[0]
        seed = int.from_bytes(hashlib.sha256(
            f"recovery:{dataset.seed}:0:{place.city}:{place.place_id}".encode()
        ).digest()[:8],"big") % (2**63)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            reference = []
            for path, _ in dataset.sample_paths(0,epoch=0):
                with Image.open(path) as image:
                    reference.append(dataset.transform(image.convert("RGB")))
        torch.testing.assert_close(dataset[(0,0)][0],torch.stack(reference),rtol=0,atol=0)

    def test_workers_have_identical_augmented_inputs(self):
        ds = self.dataset("randaugment")
        outputs = []
        for workers in (0,2,4):
            loader = DataLoader(ds,batch_sampler=training.WindowSampler(len(ds),2,43,2),
                                num_workers=workers)
            outputs.append([batch[0] for batch in loader])
        for other in outputs[1:]:
            for left,right in zip(outputs[0],other):
                torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_resume_randaugment_with_retention_is_exact_and_policy_checked(self):
        f = self.fixture
        full = self.run_training(self.args(f.root/"full","randaugment",True))
        self.run_training(self.args(f.root/"partial","randaugment",True,("--stop-after-epoch","1")))
        resumed = self.run_training(self.args(f.root/"partial","randaugment",True,
                    ("--resume",str(f.root/"partial/checkpoint.pt"))))
        self.assertEqual(full["global_step"],3)
        self.assertEqual(full["scheduler_state_dict"],resumed["scheduler_state_dict"])
        for name,value in full["state_dict"].items():
            torch.testing.assert_close(value,resumed["state_dict"][name],rtol=0,atol=0)
        self.assertTrue(torch.equal(full["rng_state"]["torch"],resumed["rng_state"]["torch"]))
        with self.assertRaisesRegex(ValueError,"Resume data/model/optimizer"):
            self.run_training(self.args(f.root/"wrong","weak",True,
                ("--resume",str(f.root/"partial/checkpoint_epoch_001.pt"))))

    def test_teacher_construction_does_not_shift_first_main_update(self):
        f = self.fixture
        a = self.run_training(self.args(f.root/"A","randaugment",False,("--stop-after-epoch","1")))
        r = self.run_training(self.args(f.root/"R","randaugment",True,("--stop-after-epoch","1")))
        # At identical initial weights, same-input eval retention is exactly zero.
        self.assertEqual(r["metrics"]["real_descriptor_retention"]["loss"],0)
        self.assertEqual(a["metrics"]["mining"],r["metrics"]["mining"])
        for name,value in a["state_dict"].items():
            torch.testing.assert_close(value,r["state_dict"][name],rtol=0,atol=0)
        self.assertTrue(torch.equal(a["rng_state"]["torch"],r["rng_state"]["torch"]))

    def test_synthetic_and_auxiliary_modes_rejected(self):
        args = self.args(self.fixture.root/"invalid")
        args.synthetic_fraction = .5
        with self.assertRaisesRegex(ValueError,"real MS"):
            training.validate_args(args)


if __name__ == "__main__":
    unittest.main()
