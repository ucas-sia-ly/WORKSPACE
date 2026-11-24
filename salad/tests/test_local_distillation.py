"""Independent losses, exact patch reduction, fixed geometry and training resume."""
import contextlib
import hashlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import test_frozen_teacher as teacher_fixture
from train_salad import train
from workflow.model import SALADModel, read_checkpoint
from workflow.pair_correspondence import CorrespondenceCalibration
from workflow.reliability_cache import load_teacher_image
from workflow.synthetic_losses import (feature_distillation_loss, score_distillation_loss,
    conditional_ot_distillation_loss, local_distillation_losses, local_target_weights)


class LocalLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        self.index = torch.tensor([[[5, 2, 8], [0, 1, 7]], [[3, 4, 2], [8, 6, 0]]])
        self.q = torch.tensor([[[1., .4, 0], [.2, .8, 0]], [[1., 1, 1], [0, .3, .7]]], requires_grad=True)

    def aux(self, n, requires_grad=False):
        return {"backbone_local_features": torch.randn(2, 8, n, requires_grad=requires_grad),
                "score_logits": torch.randn(2, 4, n, requires_grad=requires_grad),
                "log_assignment": torch.randn(2, 5, n, requires_grad=requires_grad)}

    def test_feature_exact_gather_and_weighted_formula_rectangular(self):
        s = torch.randn(2, 8, 2, 3, requires_grad=True)
        t = torch.randn(2, 8, 3, 3, requires_grad=True)
        gathered = t.detach().flatten(2).gather(2, self.index.flatten(1)[:, None].expand(-1, 8, -1))
        cosine = F.cosine_similarity(s.flatten(2), gathered, dim=1)
        expected = ((1-cosine) * self.q.detach().flatten(1)).sum() / (self.q.detach().sum()+1e-8)
        actual = feature_distillation_loss(s, t, self.index, self.q)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(float(s.grad.abs().sum()), 0)
        self.assertIsNone(t.grad); self.assertIsNone(self.q.grad)

    def test_identical_kl_zero_and_uniform_mean(self):
        logits = torch.randn(1, 64, 12, requires_grad=True)
        idx, q = torch.arange(12)[None], torch.ones(1, 12)
        self.assertLess(float(score_distillation_loss(logits, logits, idx, q)), 1e-7)
        ot = torch.randn(1, 65, 12, requires_grad=True)
        self.assertLess(float(conditional_ot_distillation_loss(ot, ot, idx, q)), 1e-7)
        teacher = torch.randn_like(logits)
        expected = F.kl_div(logits.log_softmax(1), teacher.log_softmax(1), log_target=True,
                            reduction="none").sum(1).mean()
        torch.testing.assert_close(score_distillation_loss(logits, teacher, idx, q), expected)

    def test_kl_direction_teacher_and_q_detached_student_nonzero(self):
        for function, channels in ((score_distillation_loss, 4), (conditional_ot_distillation_loss, 5)):
            s = torch.randn(2, channels, 6, requires_grad=True)
            t = torch.randn(2, channels, 9, requires_grad=True)
            loss = function(s, t, self.index, self.q)
            loss.backward()
            self.assertIsNone(t.grad); self.assertIsNone(self.q.grad)
            self.assertGreater(float(s.grad.abs().sum()), 0)
            if channels == 5:
                self.assertTrue((s.grad[:, -1] == 0).all())

    def test_zero_confidence_and_unknown_indices_have_zero_loss_and_gradient(self):
        s, t = self.aux(6, True), self.aux(9, True)
        q, idx = torch.zeros_like(self.q), torch.full_like(self.index, -1)
        losses = local_distillation_losses(s, t, idx, q)
        total = sum(losses[k] for k in ("feature", "score", "ot"))
        self.assertEqual(float(total), 0.)
        total.backward()
        for value in s.values():
            self.assertTrue((value.grad == 0).all())
        self.assertTrue(all(value.grad is None for value in t.values()))

    def test_tiny_teacher_keep_masks_only_ot_student_keep_cannot_evade(self):
        s = torch.tensor([[[1.], [-1.], [1000.]]], requires_grad=True)
        t = torch.tensor([[[-1000.], [-1001.], [0.]]], requires_grad=True)
        idx, q = torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1)
        loss = conditional_ot_distillation_loss(s, t, idx, q)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue((s.grad == 0).all())
        s.grad = None
        t = torch.tensor([[[0.], [1.], [-1000.]]], requires_grad=True)
        loss = conditional_ot_distillation_loss(s, t, idx, q)
        self.assertGreater(float(loss), 0.)
        loss.backward()
        self.assertGreater(float(s.grad[:, :-1].abs().sum()), 0.)
        self.assertEqual(float(s.grad[:, -1].abs().sum()), 0.)

    def test_fp16_bf16_fp32_extreme_logits_are_stable_under_amp(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            s = {k: v.to(dtype).detach().requires_grad_() for k, v in self.aux(6).items()}
            t = {k: v.to(dtype).detach().requires_grad_() for k, v in self.aux(9).items()}
            s["score_logits"] = (s["score_logits"].detach()*1000).requires_grad_()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = local_distillation_losses(s, t, self.index, self.q)
            total = sum(loss[k] for k in ("feature", "score", "ot"))
            self.assertEqual(total.dtype, torch.float32)
            self.assertTrue(torch.isfinite(total))
            total.backward()
            self.assertTrue(all(torch.isfinite(v.grad).all() for v in s.values()))
            self.assertTrue(all(v.grad is None for v in t.values()))

    def test_microbatch_weighting_matches_whole_batch_gradients(self):
        teacher = self.aux(9)
        full = self.aux(6, True)
        split = {k: v.detach().clone().requires_grad_() for k, v in full.items()}
        weights = dict(feature=.2, score=.3, ot=.4)
        combined = local_distillation_losses(full, teacher, self.index, self.q)
        expected = sum(weights[k]*combined[k] for k in weights)
        expected.backward()
        masses = {k: float(v.sum()) for k, v in local_target_weights(teacher, self.index, self.q).items()}
        actual = 0.
        for i in range(2):
            losses = local_distillation_losses({k:v[i:i+1] for k,v in split.items()},
                {k:v[i:i+1] for k,v in teacher.items()}, self.index[i:i+1], self.q[i:i+1])
            loss = sum(weights[k]*losses[k]*(losses[k+"_mass"]+1e-8)/(masses[k]+1e-8) for k in weights)
            actual += float(loss.detach()); loss.backward()
        self.assertAlmostEqual(actual, float(expected), places=6)
        for k in full:
            torch.testing.assert_close(full[k].grad, split[k].grad)

    def test_invalid_indices_confidence_and_nan_rejected(self):
        s,t = self.aux(6),self.aux(9)
        with self.assertRaises(ValueError):
            local_distillation_losses(s,t,self.index+99,self.q)
        with self.assertRaises(ValueError):
            local_distillation_losses(s,t,self.index,self.q*float("nan"))
        for k in s:
            bad = {**s, k: s[k]*float("nan")}
            with self.assertRaises(RuntimeError):
                local_distillation_losses(bad,t,self.index,self.q)


class LocalWorkflowTests(unittest.TestCase):
    arguments = teacher_fixture.FrozenTeacherTests.arguments

    def setUp(self):
        teacher_fixture.FrozenTeacherTests.setUp(self)
        # Spatial content is necessary: Stage 2B deliberately rejects flat or
        # exactly repeated pixels, regardless of a teacher's position encoding.
        for i,path in enumerate(sorted((self.data / "Images/Test").glob("*.jpg"))):
            image = np.random.default_rng(i).integers(20,230,(28,28,3),dtype=np.uint8)
            Image.fromarray(image).save(path)
        for row in map(json.loads,self.manifest.read_text().splitlines()):
            with Image.open(row["source_path"]) as im:
                image = (np.asarray(im).astype(float)*.7).astype(np.uint8)
            Image.fromarray(image).save(row["output_path"])
        self.calibration_path = self.root / "calibration.json"
        # Permissive construction-fixture calibration, never a production default.
        calibration = CorrespondenceCalibration(-1.,0.,2.,.01,100,1,dict(teacher=dict(
            checkpoint_sha256=hashlib.sha256(self.teacher_path.read_bytes()).hexdigest(),
            image_preprocessing=self.payload["image_preprocessing"])))
        self.calibration_path.write_text(json.dumps(calibration.to_dict()))

    def args(self, output, *extra):
        return self.arguments(output,"--init-checkpoint",str(self.teacher_path),"--batch-size","3",
            "--num-trainable-blocks","1","--synthetic-aux","--synthetic-exposure-prob","1",
            "--real-sampling","coverage_pk","--teacher-checkpoint",str(self.teacher_path),
            "--pair-calibration",str(self.calibration_path),"--feature-distill-weight",".1",
            "--ot-distill-weight",".1",*extra)

    def run_training(self,args):
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        return read_checkpoint(args.output_dir / "checkpoint.pt")

    def test_fixed_resize_student_views_and_one_step_after_all_backwards(self):
        args = self.args(self.root/"ordering","--epochs","1")
        args.no_augment = False
        events, local_images, teacher_models = [], [], []
        forward, backward, step = SALADModel.forward, torch.Tensor.backward, torch.optim.AdamW.step
        def capture_forward(model, images, *a, **kw):
            if not model.training:
                teacher_models.append(model)
            elif kw.get("return_aux"):
                local_images.append(images.detach().cpu())
            return forward(model,images,*a,**kw)
        def capture_backward(value,*a,**kw):
            events.append("backward");return backward(value,*a,**kw)
        def capture_step(opt,*a,**kw):
            events.append("step");return step(opt,*a,**kw)
        with patch.object(SALADModel,"forward",capture_forward), patch.object(torch.Tensor,"backward",capture_backward), \
                patch.object(torch.optim.AdamW,"step",capture_step):
            actual = self.run_training(args)
        self.assertEqual(events,["backward"]*4+["step"])
        expected = [load_teacher_image(self.root/f"aug_{i}.jpg",(28,28))[None] for i in (1,2)]
        self.assertEqual(len(local_images),2)
        for image in local_images:
            self.assertTrue(any(torch.equal(image,x) for x in expected))
        self.assertTrue(all(p.grad is None for m in teacher_models for p in m.parameters()))
        local = actual["metrics"]["local_distillation"]
        self.assertGreater(local["positive_a_fraction"],0)
        aux = actual["metrics"]["synthetic_aux"]
        expected_total = aux["loss_real"] + aux["global_weight"]*aux["loss_syn_per_step"] + local["weighted_total"]
        self.assertAlmostEqual(actual["metrics"]["loss"],expected_total,places=6)

    def test_disabled_local_flags_exactly_match_stage1_and_do_not_load_calibration(self):
        args = self.args(self.root/"stage1")
        args.feature_distill_weight=args.ot_distill_weight=0
        args.teacher_checkpoint=None
        baseline = self.run_training(args)
        args.output_dir=self.root/"zeros"
        args.pair_confidence_mode="verified"
        args.pair_calibration=self.root/"does_not_exist.json"
        with patch("workflow.local_distillation.LocalDistillationSupervisor",side_effect=AssertionError("must stay off")):
            actual=self.run_training(args)
        for k,v in baseline["state_dict"].items():
            torch.testing.assert_close(v,actual["state_dict"][k],rtol=0,atol=0)
        self.assertEqual(baseline["training_config"],actual["training_config"])
        torch.testing.assert_close(baseline["rng_state"]["torch"],actual["rng_state"]["torch"])

    def test_resume_and_configuration_checked(self):
        args=self.args(self.root/"full","--pair-confidence-mode","verified")
        args.no_augment=False
        full=self.run_training(args)
        args.output_dir=self.root/"resume"
        args.init_checkpoint=None
        args.resume=self.root/"full/checkpoint_epoch_001.pt"
        resumed=self.run_training(args)
        for k,v in full["state_dict"].items():
            torch.testing.assert_close(v,resumed["state_dict"][k],rtol=0,atol=0)
        self.assertEqual(full["rng_state"]["synthetic_aux"],resumed["rng_state"]["synthetic_aux"])
        for key,value in (("pair_confidence_mode","uniform"),("score_distill_weight",.1),("ot_min_teacher_keep",.01)):
            old=getattr(args,key);setattr(args,key,value)
            with self.assertRaisesRegex(ValueError,"training_config"):
                self.run_training(args)
            setattr(args,key,old)

    def test_empty_auxiliary_batch_is_valid_and_has_zero_local_loss(self):
        actual=self.run_training(self.args(self.root/"empty","--synthetic-exposure-prob","0","--epochs","1"))
        local=actual["metrics"]["local_distillation"]
        self.assertEqual(local["pairs"],0)
        self.assertEqual(local["weighted_total"],0)

    def test_nonfinite_local_loss_never_steps(self):
        args=self.args(self.root/"nan","--epochs","1")
        with patch("workflow.local_distillation.local_distillation_losses",side_effect=RuntimeError("Nonfinite local")), \
                patch.object(torch.optim.AdamW,"step",autospec=True) as step:
            with self.assertRaisesRegex(RuntimeError,"Nonfinite local"):
                self.run_training(args)
        step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
