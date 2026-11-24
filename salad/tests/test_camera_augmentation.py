"""CPU-only contract checks for optional synthetic camera simulations."""
from __future__ import annotations

import hashlib
import json
import random
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

import test_training_data as data_fixture
import test_local_distillation as local_fixture
from workflow.camera_augmentation import CameraAugmentation, camera_seed, validate_profile
from workflow.training_data import MixedGSVCitiesDataset, SyntheticAuxSampler, CoveragePlaceBatchSampler


def profile():
    # Unit-test ranges exercise all operations; not a production calibration.
    return dict(version=1, status="exploratory", conditions=["night", "rainy_night"],
        probability=1., operations={
            "gain_gamma": dict(gain=[1.05, 1.4], gamma=[.8, 1.]),
            "saturation": dict(factor=[.65, .95]),
            "bloom": dict(threshold=[.7, .9], strength=[.1, .3], radius=[1, 3]),
            "motion_blur": dict(length=[1, 5], angle=[-30, 30]),
            "vignette": dict(strength=[.05, .2]),
            "noise": dict(shot_peak=[200, 1000], read_std=[.001, .004]),
            "jpeg": dict(quality=[70, 95])})


def _numpy_camera_collate(batch):
    # Plain pickle transport also works in sandboxes that prohibit the Unix
    # socket used by Torch shared-memory tensor descriptor transfer.
    return np.stack([row[0].numpy() for row in batch])


class CameraOperationTests(unittest.TestCase):
    def test_seed_replay_global_rng_and_non_night_identity(self):
        image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (31, 47, 3), dtype=np.uint8))
        aug = CameraAugmentation(profile())
        python, numpy, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        a = aug.apply(image, seed=camera_seed(5, 2, "place"), condition="night")
        b = aug.apply(image, seed=camera_seed(5, 2, "place"), condition="night")
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, image))
        self.assertFalse(np.array_equal(a, aug.apply(image, seed=camera_seed(5, 3, "place"), condition="night")))
        self.assertEqual(random.getstate(), python)
        np.testing.assert_equal(np.random.get_state(), numpy)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        for condition in ("rain", "snow", "day", "unknown"):
            self.assertIs(aug.apply(image, seed=7, condition=condition), image)
        disabled = profile(); disabled["probability"] = 0
        self.assertIs(CameraAugmentation(disabled).apply(image, seed=7, condition="night"), image)
        self.assertEqual(a.size, image.size)
        self.assertEqual(np.asarray(a).dtype, np.uint8)

    def test_validation_finite_bounded_known_fields_and_calibration_claim(self):
        validate_profile(profile())
        mutations = [dict(unknown="oops"), dict(version=True), dict(status="empirically_correct"), dict(status=[]),
                     dict(probability=float("nan")), dict(conditions=["sun"]), dict(status="calibrated")]
        for change in mutations:
            invalid = {**profile(), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_profile(invalid)
        for operation, parameter, value in (("noise", "shot_peak", [0, 100]),
                ("gain_gamma", "gamma", [.2, 1]), ("jpeg", "quality", [50., 100]),
                ("motion_blur", "length", [1, 100]), ("saturation", "factor", [1, 2]),
                ("vignette", "strength", [float("inf"), float("inf")])):
            invalid = profile(); invalid["operations"][operation][parameter] = value
            with self.assertRaises(ValueError):
                validate_profile(invalid)

    def test_all_operations_black_white_and_tiny_images_finite(self):
        aug = CameraAugmentation(profile())
        for shape in ((1, 1), (13, 7), (224, 224)):
            for value in (0, 255):
                image = Image.new("RGB", shape, (value,) * 3)
                actual = aug.apply(image, seed=7, condition="rainy_night")
                self.assertEqual(actual.size, shape)
                self.assertTrue(np.isfinite(actual).all())

    def test_bloom_does_not_invent_light_and_blur_is_directional(self):
        p = profile(); p["operations"] = {"bloom": dict(threshold=[.8, .8], strength=[.8, .8], radius=[3, 3])}
        black = Image.new("RGB", (31, 31), (0, 0, 0))
        aug = CameraAugmentation(p)
        np.testing.assert_array_equal(aug.apply(black, seed=8, condition="night"), black)
        impulse = np.zeros((31, 31, 3), dtype=np.uint8); impulse[15, 15] = 255
        p["operations"] = {"motion_blur": dict(length=[7, 7], angle=[0, 0])}
        blurred = np.asarray(CameraAugmentation(p).apply(Image.fromarray(impulse), seed=8, condition="night"))
        self.assertGreater(np.count_nonzero(blurred[15]), 3)
        self.assertEqual(np.count_nonzero(blurred[:, 15]), 3)
        self.assertEqual(np.count_nonzero(blurred[0]), 0)


class CameraDatasetTests(unittest.TestCase):
    setUp = data_fixture.TrainingDataTests.setUp
    tearDown = data_fixture.TrainingDataTests.tearDown
    _write_csv = data_fixture.TrainingDataTests._write_csv
    _candidate = data_fixture.TrainingDataTests._candidate
    _manifest = data_fixture.TrainingDataTests._manifest

    def dataset(self, enabled=True, mode="mix", augment=False):
        if not self.manifest.exists():
            self._manifest([{**self._candidate(f"night_{i}.jpg", (city, place, 0)), "condition": "night"}
                for i, (city, place) in enumerate((("Alpha", 1), ("Alpha", 2), ("Beta", 1), ("Beta", 2)))])
        path = self.root / "camera.json"
        path.write_text(json.dumps(profile()))
        return MixedGSVCitiesDataset(self.gsv, self.manifest, images_per_place=4,
            image_size=(28, 28), augment=augment, synthetic_fraction=.5,
            synthetic_mode=mode, real_sampling="coverage_pk", seed=42,
            return_sampling_metadata=True, synthetic_camera_profile=path if enabled else None)

    def test_main_mixed_and_replaced_preserve_real_images_and_selection(self):
        for mode in ("mix", "replace"):
            a, b = self.dataset(False, mode, True), self.dataset(True, mode, True)
            for epoch in (0, 1):
                for index in range(len(a)):
                    baseline, actual = a[(epoch, index)], b[(epoch, index)]
                    self.assertEqual(a.sample_paths(index, epoch=epoch), b.sample_paths(index, epoch=epoch))
                    torch.testing.assert_close(baseline[0][~baseline[2]], actual[0][~actual[2]], rtol=0, atol=0)
                    torch.testing.assert_close(baseline[1], actual[1], rtol=0, atol=0)
                    torch.testing.assert_close(baseline[2], actual[2], rtol=0, atol=0)
                    if bool(actual[2].any()):
                        self.assertFalse(torch.equal(baseline[0][baseline[2]], actual[0][actual[2]]))

    def test_workers_and_epoch_replay(self):
        dataset = self.dataset(True)
        expected = None
        for workers in (0, 2, 4):
            sampler = CoveragePlaceBatchSampler(len(dataset), 2, seed=42)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                                persistent_workers=bool(workers), collate_fn=_numpy_camera_collate,
                                timeout=20 if workers else 0)
            results = []
            for epoch in (0, 3):
                sampler.set_epoch(epoch)
                results.append(torch.from_numpy(np.concatenate(list(loader))))
            if expected is None:
                expected = results
            else:
                for a, b in zip(expected, results):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            del loader
        self.assertFalse(torch.equal(expected[0], expected[1]))

    def test_aux_source_stream_and_rng_unchanged_replay_camera_seed(self):
        a, b = self.dataset(False, augment=True), self.dataset(True, augment=True)
        # Explicit all-real main views ensure each exact synthetic source exists.
        metadata = {"place_index": torch.tensor([0, 1, 2, 3]),
                    "real_indices": torch.tensor([[0, 1, 2, 3]] * 4)}
        sa, sb = SyntheticAuxSampler(a, 1., 42), SyntheticAuxSampler(b, 1., 42)
        before = sb.state_dict()
        baseline, actual = sa.sample(metadata), sb.sample(metadata)
        self.assertEqual(sa.state_dict(), sb.state_dict())
        for key in ("synthetic_paths", "source_paths", "condition"):
            self.assertEqual(baseline[key], actual[key])
        self.assertFalse(torch.equal(baseline["synthetic_images"], actual["synthetic_images"]))
        sb.load_state_dict(before)
        replay = sb.sample(metadata)
        self.assertEqual(replay["camera_seeds"], actual["camera_seeds"])
        torch.testing.assert_close(replay["synthetic_images"], actual["synthetic_images"], rtol=0, atol=0)
        self.assertEqual(b.summary["synthetic_camera_augmentation"]["profile_sha256"],
                         hashlib.sha256((self.root / "camera.json").read_bytes()).hexdigest())
        self.assertNotIn("synthetic_camera_augmentation", a.summary)


class CameraLocalWorkflowTests(unittest.TestCase):
    setUp = local_fixture.LocalWorkflowTests.setUp
    arguments = local_fixture.LocalWorkflowTests.arguments
    args = local_fixture.LocalWorkflowTests.args
    run_training = local_fixture.LocalWorkflowTests.run_training

    def test_clean_teacher_and_correspondence_corrupted_student_and_profile_resume_check(self):
        # Tiny offline backbone fixture; no GPU or real training data involved.
        from workflow.model import SALADModel, FrozenSALADTeacher
        import workflow.local_distillation as local_module
        rows = [json.loads(line) for line in self.manifest.read_text().splitlines()]
        self.manifest.write_text("".join(json.dumps({**r, "condition": "night"}) + "\n" for r in rows))
        path = self.root / "camera.json"; path.write_text(json.dumps(profile()))
        args = self.args(self.root / "camera_local", "--synthetic-camera-profile", str(path))
        captured, teacher_inputs, support_inputs = [], [], []
        forward = SALADModel.forward
        targets, correspondence = FrozenSALADTeacher.iter_targets, local_module.build_pair_correspondence
        def capture(model, images, *pos, **kw):
            if model.training and kw.get("return_aux"):
                captured.append(images.detach().cpu())
            return forward(model, images, *pos, **kw)
        def capture_teacher(model, images, *pos, **kw):
            teacher_inputs.append(images.detach().cpu())
            return targets(model, images, *pos, **kw)
        def capture_support(source, generated, *pos, **kw):
            support_inputs.append((source.tobytes(), generated.tobytes()))
            return correspondence(source, generated, *pos, **kw)
        with patch.object(SALADModel, "forward", capture), \
                patch.object(FrozenSALADTeacher, "iter_targets", capture_teacher), \
                patch.object(local_module, "build_pair_correspondence", capture_support):
            actual = self.run_training(args)
        self.assertEqual(len(captured), 4)
        from workflow.reliability_cache import load_teacher_image
        clean = [load_teacher_image(self.root / f"aug_{i}.jpg", (28, 28))[None] for i in (1, 2)]
        self.assertTrue(all(not any(torch.equal(x, y) for y in clean) for x in captured))
        self.assertEqual(len(teacher_inputs), 4)
        for teacher_input in teacher_inputs:
            self.assertTrue(any(torch.equal(teacher_input[1:2], x) for x in clean))
        expected_support = []
        for row in rows:
            with Image.open(row["source_path"]) as im:
                source_bytes = im.convert("RGB").tobytes()
            with Image.open(row["output_path"]) as im:
                generated_bytes = im.convert("RGB").tobytes()
            expected_support.append((source_bytes, generated_bytes))
        self.assertTrue(all(pair in expected_support for pair in support_inputs))
        self.assertEqual(actual["training_config"]["synthetic_camera_augmentation"]["profile_sha256"],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        args.output_dir = self.root / "resume"; args.init_checkpoint = None
        args.resume = self.root / "camera_local" / "checkpoint_epoch_001.pt"
        replay = self.run_training(args)
        for name, value in actual["state_dict"].items():
            torch.testing.assert_close(value, replay["state_dict"][name], rtol=0, atol=0)
        altered_profile = profile(); altered_profile["operations"]["jpeg"]["quality"] = [80, 95]
        path.write_text(json.dumps(altered_profile))
        args.output_dir = self.root / "mismatch"
        with self.assertRaisesRegex(ValueError, "Resume data/model/optimizer options"):
            self.run_training(args)


if __name__ == "__main__":
    unittest.main()
