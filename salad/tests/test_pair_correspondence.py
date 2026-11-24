"""Small CPU contracts; construction-labelled controls, not weather truth."""
import unittest

from PIL import Image
import numpy as np
import torch

from workflow.pair_correspondence import (
    build_pair_correspondence, correspondence_measurements, fit_calibration,
    points_to_patch, _mask,
)
from workflow.reliability import build_local_targets


class PairCorrespondenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.rng = torch.Generator().manual_seed(19)
        nulls = [correspondence_measurements(cls.aux(cls.features()), cls.aux(cls.features()))
                 for _ in range(8)]
        cls.calibration = fit_calibration(nulls, tail_probability=.01, search_radius=1,
                                          provenance={"fixture": "independent random nulls"})

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @classmethod
    def features(cls, shape=(12, 16)):
        return torch.randn(1, 48, *shape, generator=cls.rng)

    @staticmethod
    def aux(f):
        return {"backbone_local_features": f}

    @staticmethod
    def image(size=(320, 240)):
        return Image.fromarray(np.random.default_rng(19).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))

    def build(self, s, g, **kwargs):
        return build_pair_correspondence(self.image(), self.image(), self.aux(s), self.aux(g),
                                         calibration=self.calibration, **kwargs)

    def test_same_copy_many_reliable_including_boundary(self):
        f = self.features()
        r = self.build(f, f)
        self.assertGreater(int((r["positive_conf"] > 0).sum()), 180)
        self.assertTrue((r["positive_conf"][[0, -1]] > 0).all())
        self.assertFalse(r["negative_conf"].any())
        torch.testing.assert_close(r["match_index"].flatten(), torch.arange(192))

    def test_translation_one_patch_and_no_wraparound(self):
        s = self.features()
        g = torch.zeros_like(s)
        g[..., 1:] = s[..., :-1]
        r = self.build(s, g)
        valid = r["positive_conf"] > 0
        expected = torch.arange(192).reshape(12, 16) - 1
        self.assertGreater(int(valid.sum()), 100)
        self.assertTrue((r["match_index"][valid] == expected[valid]).all())
        self.assertFalse(valid[:, 0].any())
        self.assertFalse(r["negative_conf"].any())

    def test_known_occlusion_and_edit_mask_are_distinct(self):
        f = self.features()
        mask = torch.zeros(240, 320)
        mask[80:160, 80:160] = 1
        r = self.build(f, f, occlusion_mask=mask)
        self.assertTrue((r["positive_conf"][4:8, 4:8] == 0).all())
        self.assertTrue((r["negative_conf"][4:8, 4:8] == 1).all())
        self.assertTrue((r["match_index"][4:8, 4:8] == -1).all())
        e = self.build(f, f, edit_mask=mask)
        self.assertFalse(e["negative_conf"].any())
        self.assertTrue(e["unknown_mask"][4:8, 4:8].all())

    def test_low_cosine_never_means_negative_and_geometry_can_rescue(self):
        s = self.features()
        no_geometry = self.build(s, -s)
        self.assertFalse(no_geometry["negative_conf"].any())
        points = [[10., 10.], [12., 12.], [30., 30.], [32., 32.]]
        geometry = dict(status="ok", coordinate_convention="xy_pixel_centers_full_frame_resize",
                        H=torch.eye(3).tolist(), ransac_reprojection_threshold=3.,
                        source_canvas_hw=[240, 320], generated_canvas_hw=[240, 320],
                        inlier_kpts0=points, inlier_kpts1=points)
        r = self.build(s, -s, geometry=geometry)
        self.assertGreater(r["match_diagnostics"]["positive_b_count"], 0)
        self.assertFalse(r["negative_conf"].any())

    def test_repeated_wall_not_resolved_by_identity_tie_preference(self):
        tile = self.features((1, 1))
        f = tile.expand(1, 48, 12, 16)
        r = self.build(f, f)
        self.assertTrue(r["unknown_mask"].all())
        self.assertFalse(r["positive_conf"].any())

    def test_unknown_without_calibration_or_geometry(self):
        f = self.features()
        r = build_pair_correspondence(Image.new("RGB", (320, 240)), Image.new("RGB", (128, 96)),
                                      self.aux(f), self.aux(f))
        self.assertTrue(r["unknown_mask"].all())
        self.assertTrue((r["match_index"] == -1).all())

    def test_canvas_scaling_and_boundary_coordinates(self):
        index, valid = points_to_patch([[0, 0], [511, 255], [256, 128], [-1, 0], [512, 0]],
                                       (256, 512), (8, 16))
        self.assertEqual(index.tolist(), [0, 127, 72, -1, -1])
        self.assertEqual(valid.tolist(), [True, True, True, False, False])
        f = self.features((7, 11))
        r = self.build(f, f)
        self.assertEqual(tuple(r["match_index"].shape), (7, 11))
        # Unequal grids and native image sizes must not index outside source.
        r = self.build(f, self.features((9, 13)))
        self.assertTrue((r["match_index"] < 77).all())

    def test_geometry_ambiguity_duplicates_and_failure_do_not_label_errors(self):
        s = self.features()
        geo = dict(status="ok", coordinate_convention="xy_pixel_centers_full_frame_resize",
                   H=torch.eye(3).tolist(), ransac_reprojection_threshold=3.,
                   source_canvas_hw=[240, 320], generated_canvas_hw=[240, 320],
                   inlier_kpts0=[[10, 10], [10, 10]], inlier_kpts1=[[10, 10], [10, 10]])
        r = self.build(s, -s, geometry=geo)
        self.assertEqual(r["match_diagnostics"]["positive_b_count"], 0)
        r = self.build(s, -s, geometry={"status": "matcher_failed"})
        self.assertFalse(r["negative_conf"].any())

    def test_quality_rejection_is_unknown_not_negative(self):
        f = self.features()
        r = self.build(f, f, quality={"geometry_ok": False})
        self.assertTrue(r["unknown_mask"].all())
        self.assertFalse(r["negative_conf"].any())

    def test_no_grad_and_amp_finite(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            f = self.features().to(dtype).requires_grad_()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                r = self.build(f, f)
            for key in ("positive_conf", "negative_conf"):
                self.assertEqual(r[key].dtype, torch.float32)
                self.assertTrue(torch.isfinite(r[key]).all())
                self.assertFalse(r[key].requires_grad)

    def test_radius_one_inclusive_legacy_semantics(self):
        f = self.features((20, 20))
        _, _, info = build_local_targets(f, f, search_radius=1, alignment_tolerance=1)
        self.assertGreater(int(info["positive_patches"]), 0)
        self.assertEqual(int(info["negative_patches"]), 0)
        for tolerance in (-1, 2, True):
            with self.assertRaises(ValueError):
                build_local_targets(f, f, search_radius=1, alignment_tolerance=tolerance)
        self.build(f, f, search_radius=1, alignment_tolerance=0)

    def test_invalid_inputs_and_tiny_grid(self):
        f = self.features()
        with self.assertRaises(ValueError):
            self.build(f, f * float("nan"))
        with self.assertRaises(ValueError):
            self.build(f, f, occlusion_mask=torch.ones(12, 16))
        with self.assertRaises(ValueError):
            self.build(f, f, search_radius=2)
        tiny = self.features((1, 1))
        self.assertTrue(self.build(tiny, tiny)["unknown_mask"].all())

    def test_exact_mask_area_with_uneven_resize_and_partial_patch(self):
        mask = torch.zeros(7, 11)
        mask[2:4, 3:6] = 1
        area = _mask(mask, (7, 11), (3, 4), torch.device("cpu"))
        self.assertAlmostEqual(float(area.mean()), 6 / 77, places=6)
        self.assertTrue(((area > 0) & (area < 1)).any())
        self.assertFalse(area.requires_grad)

    def test_bad_homography_is_unknown_not_negative(self):
        s = self.features()
        geometry = dict(status="ok", coordinate_convention="xy_pixel_centers_full_frame_resize",
                        H=torch.zeros(3, 3).tolist(), ransac_reprojection_threshold=3.,
                        inlier_kpts0=[[10, 10], [12, 12]], inlier_kpts1=[[10, 10], [12, 12]],
                        source_canvas_hw=[240, 320], generated_canvas_hw=[240, 320])
        r = self.build(s, -s, geometry=geometry)
        self.assertEqual(r["match_diagnostics"]["positive_b_count"], 0)
        self.assertFalse(r["negative_conf"].any())

    def test_independent_geometry_conflict_vetoes_a_positive(self):
        s = self.features()
        geometry = dict(status="ok", coordinate_convention="xy_pixel_centers_full_frame_resize",
                        H=[[1, 0, 20], [0, 1, 0], [0, 0, 1]], ransac_reprojection_threshold=3.,
                        inlier_kpts0=[[10, 10], [12, 12]], inlier_kpts1=[[30, 10], [32, 12]],
                        source_canvas_hw=[240, 320], generated_canvas_hw=[240, 320])
        r = self.build(s, s, geometry=geometry)
        self.assertTrue(r["match_diagnostics"]["geometry_conflict"][0, 1])
        self.assertTrue(r["unknown_mask"][0, 1])
        self.assertEqual(float(r["negative_conf"][0, 1]), 0.)

    def test_keep_and_score_are_diagnostics_only(self):
        f = self.features()
        first = self.aux(f)
        second = dict(first, score_logits=torch.ones(1, 64, 192) * 100,
                      log_assignment=torch.full((1, 65, 192), -torch.log(torch.tensor(65.))),
                      assignment_keep=torch.ones(1, 192))
        im = self.image()
        a = build_pair_correspondence(im, im, first, first, calibration=self.calibration)
        b = build_pair_correspondence(im, im, second, second, calibration=self.calibration)
        torch.testing.assert_close(a["positive_conf"], b["positive_conf"])
        torch.testing.assert_close(a["unknown_mask"], b["unknown_mask"])

    def test_repeated_pixels_veto_feature_position_bias(self):
        # DINO positional encoding can make identical tiled pixels look unique.
        # Arbitrarily distinctive features must not override image ambiguity.
        f = self.features()
        tile = self.image((14, 14))
        im = Image.new("RGB", (16 * 14, 12 * 14))
        for y in range(0, im.height, 14):
            for x in range(0, im.width, 14):
                im.paste(tile, (x, y))
        r = build_pair_correspondence(im, im, self.aux(f), self.aux(f), calibration=self.calibration)
        self.assertTrue(r["unknown_mask"].all())
        self.assertTrue(r["match_diagnostics"]["appearance_alias"].all())

    def test_copy_calibration_is_recorded_and_has_no_rng_side_effects(self):
        s, g = self.features(), self.features()
        null = correspondence_measurements(self.aux(s), self.aux(g))
        copy = correspondence_measurements(self.aux(s), self.aux(s))
        rng = torch.get_rng_state().clone()
        fitted = fit_calibration([null], tail_probability=.01, search_radius=1,
                                 provenance={"fixture": "control"}, positive_controls=[copy])
        self.assertEqual(fitted.provenance["margin_rule"], "known_copy_lower_tail")
        self.assertEqual(fitted.samples, 192)
        torch.testing.assert_close(rng, torch.get_rng_state())
        with self.assertRaises(ValueError):
            fit_calibration([null], tail_probability=.0001, search_radius=1, provenance={})


if __name__ == "__main__":
    unittest.main()
