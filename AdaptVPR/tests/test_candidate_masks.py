"""Geometry/coverage contracts with no model, VLM, or diffusion imports."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from PIL import Image

from targeted.candidate_masks import (
    CoverageContext,assess_thresholds,facade_alignment,generate_candidates,rasterize,
)
from targeted.family_constraints import load_constraints,validate_constraints
from targeted.render_mask import build_render_mask

ROOT = Path(__file__).resolve().parents[1]


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.config = load_constraints()
        self.roi = np.zeros((16,16),bool)
        self.roi[5:11,5:11] = True
        self.roi[4,5:7] = True
        self.weights = (np.arange(256,dtype=np.float64).reshape(16,16)+1)/256
        self.image = Image.new("RGB",(96,80),"gray")

    def test_distinct_family_priors_and_explicit_thresholds(self):
        self.assertEqual(self.config["tau_target_precision"],.7)
        self.assertEqual(self.config["weighted_coverage_thresholds"],[.3,.5,.7])
        self.assertEqual(len({tuple(s["area_range"]) for s in self.config["families"].values()}),7)
        for key,value in (("weighted_coverage_thresholds",[.5]),("weighted_coverage_scope","roi"),
                          ("weight_map","arbitrary"),("tau_target_precision",float("nan"))):
            invalid = copy.deepcopy(self.config)
            invalid[key] = value
            with self.assertRaises(ValueError): validate_constraints(invalid)

    def test_precision_binary_and_full_weight_coverage_are_distinct(self):
        roi = np.zeros((16,16),bool)
        roi[:4,:4] = True
        weights = np.ones((16,16),float)
        weights[8:,8:] = 10
        context = CoverageContext(roi,weights,(160,160))
        core = np.zeros(context.shape,bool)
        core[:40,:40] = True
        metrics,occupancy = context.measure(core)
        self.assertEqual(metrics["target_precision"],1)
        self.assertEqual(metrics["binary_roi_coverage"],1)
        self.assertEqual(metrics["roi_conditioned_weighted_coverage"],1)
        self.assertAlmostEqual(metrics["vulnerability_weighted_coverage"],16/weights.sum())
        self.assertEqual(metrics["num_components"],1)
        np.testing.assert_array_equal(occupancy,roi.astype(float))
        core[80:90,80:90] = True
        metrics,_ = context.measure(core)
        self.assertAlmostEqual(metrics["target_precision"],16/17)
        self.assertEqual(metrics["num_components"],2)

    def test_token_mass_preserved_for_unequal_native_cell_areas(self):
        weights = np.ones((16,16),dtype=np.float32)
        context = CoverageContext(np.ones((16,16),bool),weights,(19,23))
        core = context.token_ids==0
        metrics,occupancy = context.measure(core)
        self.assertAlmostEqual(metrics["vulnerability_weighted_coverage"],1/256)
        self.assertNotAlmostEqual(metrics["area_fraction"],1/256)
        self.assertEqual(occupancy[0,0],1)
        core[0,0] = False
        metrics,_ = context.measure(core)
        self.assertAlmostEqual(metrics["vulnerability_weighted_coverage"],.75/256)

    def test_zero_weight_is_undefined_never_automatic_pass(self):
        context = CoverageContext(self.roi,np.zeros((16,16),float),(80,96))
        metrics,_ = context.measure(context.roi_pixels)
        self.assertIsNone(metrics["vulnerability_weighted_coverage"])
        gates = assess_thresholds(metrics,dict(area_range=[0,1]),dict(clipped_pixels=0),self.config)
        self.assertTrue(all(not gate["passes"] for gate in gates.values()))
        self.assertIn("zero_total_weight_coverage_undefined",gates["0.3"]["rejection_reasons"])

    def test_three_threshold_results_never_pick_a_winner(self):
        metric = dict(area_fraction=.05,num_components=1,target_precision=.7,
                      weighted_coverage_defined=True,vulnerability_weighted_coverage=.5)
        gates = assess_thresholds(metric,dict(area_range=[.03,.1]),dict(clipped_pixels=0),self.config)
        self.assertEqual([gate["passes"] for gate in gates.values()],[True,True,False])
        metric["target_precision"] = .6999
        gates = assess_thresholds(metric,dict(area_range=[.03,.1]),dict(clipped_pixels=0),self.config)
        self.assertFalse(any(gate["passes"] for gate in gates.values()))

    def test_family_shapes_and_native_rasterization(self):
        for family,spec in self.config["families"].items():
            with self.subTest(family=family):
                area = sum(spec["area_range"])/2
                mask,geometry = rasterize(spec["shape"],(480,640),(320,240),area,spec["aspect_ratios"][0],0,.4)
                context = CoverageContext(self.roi,self.weights,(480,640))
                metric,_ = context.measure(mask)
                self.assertEqual(mask.shape,(480,640))
                self.assertEqual(metric["num_components"],1)
                self.assertLess(abs(metric["area_fraction"]-area),.001)
                self.assertEqual(geometry["clipped_pixels"],0)
                yy,xx = np.nonzero(mask)
                if spec["shape"]=="horizontal_compact": self.assertGreater(np.ptp(xx),np.ptp(yy))
                if spec["shape"] in ("vertical_rectangle","cone_triangle"): self.assertLess(np.ptp(xx),np.ptp(yy))
                if family=="vegetation":
                    self.assertLess(mask.sum(),(np.ptp(xx)+1)*(np.ptp(yy)+1))
                    self.assertFalse(np.array_equal(mask,np.fliplr(mask)))

    def test_determinism_including_failed_candidates_and_family_order(self):
        self.config["search"].update(center_offsets=[0.],roi_anchors=0)
        kwargs = dict(roi=self.roi,weights=self.weights,source=self.image,config=self.config,image_key="City/a.jpg",seed=17)
        a = list(generate_candidates(families=["temporary_sign","vegetation"],**kwargs))
        b = list(generate_candidates(families=["vegetation","temporary_sign"],**kwargs))
        self.assertGreater(len(a),0)
        for left,right in zip(a,b):
            self.assertEqual(left.diagnostic,right.diagnostic)
            np.testing.assert_array_equal(left.core_mask,right.core_mask)
            np.testing.assert_array_equal(left.render_mask,right.render_mask)
            np.testing.assert_array_equal(left.token_occupancy,right.token_occupancy)
            self.assertFalse(left.diagnostic["selected"])
            self.assertEqual(set(left.diagnostic["thresholds"]),{"0.3","0.5","0.7"})
        # Even impossible thresholds preserve every mask and all diagnostics.
        self.config["tau_target_precision"] = 1.
        failed = list(generate_candidates(families=["temporary_sign","vegetation"],**kwargs))
        self.assertEqual(len(a),len(failed))

    def test_facade_alignment_is_image_derived_and_reports_fallback(self):
        context = CoverageContext(self.roi,self.weights,(80,96))
        yy,xx = np.mgrid[:80,:96]
        source = Image.fromarray((127+100*np.sin((yy-.2*xx)*.4)).astype(np.uint8)).convert("RGB")
        frame = facade_alignment(source,context)
        self.assertEqual(frame["method"],"local_gradient_orthogonal_frame")
        self.assertAlmostEqual(frame["angle_degrees"],np.degrees(np.arctan(.2)),delta=1)
        self.assertFalse(frame["facade_plane_verified"])
        self.assertEqual(facade_alignment(self.image,context)["method"],"image_axis_fallback")

    def test_clipping_retained_and_rejected(self):
        core,geometry = rasterize("horizontal_compact",(80,96),(0,0),.06,2,0,0)
        self.assertGreater(geometry["clipped_pixels"],0)
        metric,_ = CoverageContext(self.roi,self.weights,(80,96)).measure(core)
        gates = assess_thresholds(metric,dict(area_range=[0,1]),geometry,self.config)
        self.assertTrue(all("core_footprint_clipped_at_image_boundary" in g["rejection_reasons"] for g in gates.values()))

    def test_render_bounded_contains_core_and_does_not_change_metrics(self):
        core = np.zeros((100,120),bool)
        core[20:50,40:70] = True
        before = core.copy()
        settings = dict(radius_fraction_of_short_side=.05,max_radius_pixels=5,max_added_area_ratio=.12,max_image_area_fraction=.2)
        render,diagnostic = build_render_mask(core,settings)
        np.testing.assert_array_equal(core,before)
        self.assertFalse(np.any(core & ~render))
        self.assertLessEqual(diagnostic["added_area"],int(core.sum()*.12))
        self.assertLess(diagnostic["effective_radius_pixels"],diagnostic["requested_radius_pixels"])
        from scipy.ndimage import distance_transform_edt
        self.assertTrue((distance_transform_edt(~core)[render] <= diagnostic["effective_radius_pixels"]).all())
        context = CoverageContext(self.roi,self.weights,core.shape)
        self.assertEqual(context.measure(before)[0],context.measure(core)[0])

    def test_render_empty_and_infeasible_cap_are_explicit(self):
        settings = self.config["render"]
        _,empty = build_render_mask(np.zeros((32,32),bool),settings)
        self.assertEqual(empty["status"],"empty_core")
        core = np.ones((32,32),bool)
        render,diagnostic = build_render_mask(core,settings)
        self.assertFalse(diagnostic["feasible"])
        np.testing.assert_array_equal(core,render)

    def test_import_and_cli_have_no_model_runtime(self):
        code = "import sys; import targeted.candidate_masks; import targeted.render_mask; assert not any(x in sys.modules for x in ['torch','diffusers','transformers'])"
        subprocess.run([sys.executable,"-c",code],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/"scripts/stage3_build_candidate_masks.py"),"--help"],check=True,capture_output=True)


if __name__=="__main__": unittest.main()
