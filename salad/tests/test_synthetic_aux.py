"""CCR-VPR Stage 1 unit/integration checks using a tiny offline backbone."""

import contextlib
import io
import json
import math
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import test_pretrained_initialization as fixture
from train_salad import train
from workflow.model import read_checkpoint, SALADModel
from workflow.synthetic_losses import asymmetric_multi_positive_infonce as infonce
from workflow.training_data import MixedGSVCitiesDataset, SyntheticAuxSampler


class SyntheticGlobalLossTests(unittest.TestCase):
    def test_all_same_place_positives_match_analytic_probability(self):
        q = torch.tensor([[1., 0.]], requires_grad=True)
        r = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], requires_grad=True)
        loss = infonce(q, torch.tensor([4]), r, torch.tensor([4, 4, 9]), temperature=0.5)
        expected = math.log(math.exp(2) + 1 + math.exp(-2)) - math.log(math.exp(2) + 1)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertIsNone(r.grad)
        self.assertGreater(q.grad.abs().sum().item(), 0)

    def test_student_query_branch_updates_shared_parameters_only_via_queries(self):
        torch.manual_seed(3)
        student = nn.Linear(3, 2, bias=False)
        real = student(torch.randn(4, 3))
        real.retain_grad()
        query = student(torch.randn(2, 3))
        loss = infonce(query, torch.tensor([0, 1]), real, torch.tensor([0, 0, 1, 1]))
        loss.backward()
        self.assertIsNone(real.grad)
        self.assertGreater(student.weight.grad.abs().sum().item(), 0)

    def test_empty_invalid_and_unmatched_queries_return_finite_zero(self):
        for count in (0, 2):
            q = torch.ones(count, 3, requires_grad=True)
            r = torch.ones(4, 3, requires_grad=True)
            loss = infonce(q, torch.ones(count, dtype=torch.long), r, torch.zeros(4, dtype=torch.long))
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertIsNone(r.grad)
        q = torch.randn(2, 3, requires_grad=True)
        loss = infonce(q, torch.zeros(2, dtype=torch.long), torch.randn(4, 3),
                       torch.zeros(4, dtype=torch.long), pair_valid=torch.zeros(2, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)

    def test_fp16_bf16_fp32_inputs_stay_finite_and_compute_fp32(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            q = torch.tensor([[1., 1., 0.], [-1., 1., 0.]], dtype=dtype, requires_grad=True)
            r = torch.tensor([[1., 0., 1.], [0., 1., 1.], [-1., 0., 1.],
                              [0., -1., 1.]], dtype=dtype, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = infonce(q, torch.tensor([0, 1]), r, torch.tensor([0, 0, 1, 1]), temperature=0.001)
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(q.grad).all())
            self.assertIsNone(r.grad)

    def test_nan_and_invalid_temperature_rejected(self):
        q, r = torch.ones(1, 2), torch.ones(2, 2)
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                infonce(q, torch.tensor([0]), r, torch.tensor([0, 1]), temperature)
        q[0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            infonce(q, torch.tensor([0]), r, torch.tensor([0, 1]))


class SyntheticAuxWorkflowTests(unittest.TestCase):
    # Reuse only the fixture helpers, without inheriting unrelated test cases.
    setUp = fixture.PretrainedInitializationTests.setUp
    arguments = fixture.PretrainedInitializationTests.arguments

    def dataset(self, augment=False):
        entries = [json.loads(line) for line in self.manifest.read_text().splitlines()]
        for entry, condition in zip(entries, ("rain", "night")):
            entry["condition"] = condition
        self.manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        return MixedGSVCitiesDataset(self.data, self.manifest, images_per_place=2,
                                     min_images_per_place=2, synthetic_fraction=0, augment=augment,
                                     image_size=(28, 28), return_sampling_metadata=True)

    def aux_arguments(self, output, *extra):
        initial = self.root / "initial.pt"
        if not initial.exists():
            torch.save({"state_dict": self.state, "model_config": self.config}, initial)
        return self.arguments(output, "--init-checkpoint", str(initial), "--batch-size", "3",
                              "--num-trainable-blocks", "1", "--synthetic-aux",
                              "--synthetic-exposure-prob", "1", "--real-sampling", "coverage_pk", *extra)

    def run_training(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        return read_checkpoint(args.output_dir / "checkpoint.pt")

    def test_sampling_exact_sources_one_per_place_conditions_and_independent_rng(self):
        dataset = self.dataset(augment=True)
        batch = next(iter(DataLoader(dataset, batch_size=3)))
        metadata = batch[-1]
        sampler = SyntheticAuxSampler(dataset, 1, seed=5)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        selected = sampler.sample(metadata)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(selected["synthetic_images"].shape, (2, 3, 28, 28))
        self.assertEqual(sorted(selected["synthetic_labels"].tolist()), [1, 2])
        self.assertEqual(selected["synthetic_labels"].unique().numel(), 2)
        self.assertTrue(selected["pair_valid"].all())
        for i, flat_index in enumerate(selected["source_indices"].tolist()):
            row, slot = divmod(flat_index, 2)
            place = dataset.places[metadata["place_index"][row]]
            source = place.real_paths[metadata["real_indices"][row, slot]]
            self.assertEqual(str(source), selected["source_paths"][i])
            self.assertEqual(batch[1].flatten()[flat_index], selected["synthetic_labels"][i])
            self.assertIn(selected["synthetic_paths"][i], [str(p) for p in place.synthetic_paths
                                                         if place.source_by_synthetic_path[p] == source])
        stats = sampler.summary()
        self.assertEqual(stats["actual_trigger_rate"], 1)
        self.assertEqual(stats["available_source_fraction"], 2 / 6)
        self.assertEqual(stats["condition_exposure"], {"rain": 1, "night": 1})
        self.assertEqual(stats["synthetic_fraction_observed"], 2 / 8)

    def test_no_out_of_batch_source_fallback_and_zero_probability_no_decode(self):
        dataset = self.dataset()
        # Both places have accepted variants, but only for view 0; main views are view 1.
        metadata = {"place_index": torch.tensor([1, 2]), "real_indices": torch.tensor([[1], [1]])}
        sampler = SyntheticAuxSampler(dataset, 1, seed=5)
        with patch.object(sampler, "_read_image", side_effect=AssertionError("must not decode")):
            self.assertEqual(len(sampler.sample(metadata)["synthetic_images"]), 0)
        self.assertEqual(sampler.summary()["places_with_synthetic"], 2)
        self.assertEqual(sampler.summary()["eligible_places"], 0)
        sampler = SyntheticAuxSampler(dataset, 0, seed=5)
        metadata["real_indices"] = torch.tensor([[0], [0]])
        before = sampler.state_dict()
        with patch.object(sampler, "_read_image", side_effect=AssertionError("must not decode")):
            self.assertEqual(len(sampler.sample(metadata)["synthetic_images"]), 0)
        self.assertEqual(before, sampler.state_dict())

    def test_zero_probability_matches_real_baseline_with_augmentation(self):
        baseline_args = self.aux_arguments(self.root / "baseline", "--synthetic-fraction", "0")
        baseline_args.synthetic_aux = False
        baseline_args.no_augment = False
        baseline = self.run_training(baseline_args)
        args = self.aux_arguments(self.root / "p0", "--synthetic-exposure-prob", "0")
        args.no_augment = False
        actual = self.run_training(args)
        for name, value in baseline["state_dict"].items():
            torch.testing.assert_close(value, actual["state_dict"][name], rtol=0, atol=0)
        self.assertEqual(actual["rng_state"]["python"], baseline["rng_state"]["python"])
        self.assertTrue(torch.equal(actual["rng_state"]["torch"], baseline["rng_state"]["torch"]))
        self.assertEqual(actual["sampling_coverage"], baseline["sampling_coverage"])
        self.assertEqual(actual["metrics"]["synthetic_aux"]["triggered_places"], 0)

    def test_no_manifest_trains_real_only(self):
        args = self.aux_arguments(self.root / "no_manifest")
        args.synthetic_manifest = None
        actual = self.run_training(args)
        self.assertEqual(actual["global_step"], 2)
        self.assertEqual(actual["metrics"]["real_exposure"], 6)
        self.assertEqual(actual["metrics"]["synthetic_aux"]["nonempty_batches"], 0)

    def test_real_backward_then_query_backward_then_one_step(self):
        args = self.aux_arguments(self.root / "ordering", "--epochs", "1")
        events = []
        query_backbone_gradients = []
        original_forward, original_backward, original_step = SALADModel.forward, torch.Tensor.backward, torch.optim.AdamW.step
        def forward(model, images, *a, **kw):
            events.append(("forward", len(images)))
            if len(images) == 2:
                # Installed only after the real backward, so this records the
                # synthetic branch's contribution to the actual student backbone.
                parameter = next(p for p in model.backbone.parameters() if p.requires_grad)
                parameter.register_hook(lambda grad: query_backbone_gradients.append(float(grad.abs().sum())))
            return original_forward(model, images, *a, **kw)
        def backward(tensor, *a, **kw):
            events.append(("backward",))
            return original_backward(tensor, *a, **kw)
        def step(optimizer, *a, **kw):
            events.append(("step",))
            return original_step(optimizer, *a, **kw)
        with patch.object(SALADModel, "forward", forward), patch.object(torch.Tensor, "backward", backward), \
                patch.object(torch.optim.AdamW, "step", step):
            actual = self.run_training(args)
        self.assertEqual(events, [("forward", 6), ("backward",), ("forward", 2), ("backward",), ("step",)])
        self.assertEqual(len(query_backbone_gradients), 1)
        self.assertGreater(query_backbone_gradients[0], 0)
        self.assertEqual(actual["metrics"]["synthetic_exposure"], 0)
        self.assertEqual(actual["metrics"]["synthetic_aux"]["triggered_places"], 2)
        metrics = actual["metrics"]
        self.assertAlmostEqual(metrics["loss"], metrics["synthetic_aux"]["loss_real"]
                               + 0.1 * metrics["synthetic_aux"]["loss_syn_per_step"], places=6)

    def test_resume_exact_and_auxiliary_configuration_checked(self):
        args = self.aux_arguments(self.root / "full", "--synthetic-exposure-prob", "0.7")
        args.no_augment = False
        full = self.run_training(args)
        resume_args = self.aux_arguments(self.root / "resumed", "--synthetic-exposure-prob", "0.7")
        resume_args.init_checkpoint = None
        resume_args.resume = args.output_dir / "checkpoint_epoch_001.pt"
        resume_args.no_augment = False
        resumed = self.run_training(resume_args)
        for name, value in full["state_dict"].items():
            torch.testing.assert_close(value, resumed["state_dict"][name], rtol=0, atol=0)
        self.assertEqual(full["rng_state"]["synthetic_aux"], resumed["rng_state"]["synthetic_aux"])
        self.assertEqual(full["metrics"]["synthetic_aux"], resumed["metrics"]["synthetic_aux"])
        self.assertEqual(full["sampling_coverage"], resumed["sampling_coverage"])
        for name, value in (("synthetic_exposure_prob", 0.5), ("synthetic_global_weight", 0.2),
                            ("synthetic_global_temperature", 0.2), ("synthetic_aux", False)):
            old = getattr(resume_args, name)
            setattr(resume_args, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "training_config"), \
                    contextlib.redirect_stdout(io.StringIO()):
                train(resume_args)
            setattr(resume_args, name, old)

    def test_nonfinite_auxiliary_loss_never_steps_optimizer(self):
        args = self.aux_arguments(self.root / "nan", "--epochs", "1")
        with patch("workflow.synthetic_losses.asymmetric_multi_positive_infonce",
                   side_effect=lambda q, *a, **kw: q.sum() * float("nan")), \
                patch.object(torch.optim.AdamW, "step", autospec=True) as step, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Nonfinite synthetic"):
            train(args)
        step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
