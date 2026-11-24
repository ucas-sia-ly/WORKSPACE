import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from workflow.camera_calibration import independent_real_images, photometry


class CameraCalibrationTests(unittest.TestCase):
    def fixture(self, root):
        image = root/'calibration.png'
        Image.new('RGB',(8,8),(70,90,110)).save(image)
        data = dict(version=1,role='camera_calibration',split='calibration',dataset='independent-night',
                    condition='night',
                    prior_use='never_evaluated_or_used_for_selection',
                    images=[dict(path=str(image),group_id='independent-night/scene1')])
        blocked = dict(version=1,role='evaluation_exclusion',
                       images=[dict(path=str(root/'test.png'),group_id='independent-night/scene2')])
        return data,blocked

    def test_isolation_rejects_same_scene_even_different_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data,blocked=self.fixture(root)
            a,b=root/'real.json',root/'excluded.json'
            a.write_text(json.dumps(data));b.write_text(json.dumps(blocked))
            rows,proof=independent_real_images(a,b)
            self.assertEqual(len(rows),1)
            self.assertEqual(len(proof['calibration_manifest_sha256']),64)
            blocked['images'][0]['group_id']=data['images'][0]['group_id']
            b.write_text(json.dumps(blocked))
            with self.assertRaisesRegex(ValueError,'overlaps'): independent_real_images(a,b)

    def test_robotcar_and_test_declarations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data,blocked=self.fixture(root)
            a,b=root/'real.json',root/'excluded.json';b.write_text(json.dumps(blocked))
            for key,value in [('dataset','Oxford RobotCar'),('split','test')]:
                altered={**data,key:value};a.write_text(json.dumps(altered))
                with self.assertRaises(ValueError): independent_real_images(a,b)

    def test_photometry_measurement_definitions(self):
        pixels=np.array([[[0,0,0],[255,255,255]],[[255,0,0],[127,127,127]]],dtype=np.uint8)
        row=photometry(Image.fromarray(pixels))
        self.assertAlmostEqual(row['black_fraction'],.25)
        self.assertAlmostEqual(row['highlight_fraction'],.25)
        self.assertAlmostEqual(row['saturation'],.25)

    def test_cli_tone_fit_stays_exploratory_and_write_once(self):
        from calibrate_synthetic_camera import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data,blocked=self.fixture(root)
            a,b=root/'real.json',root/'excluded.json'
            a.write_text(json.dumps(data));b.write_text(json.dumps(blocked))
            synthetic=root/'synthetic.png';Image.new('RGB',(8,8),(25,30,35)).save(synthetic)
            manifest=root/'synthetic.jsonl'
            manifest.write_text(json.dumps(dict(output_path=str(synthetic),condition='night',
                                               passed=True,eligible_for_training=True))+'\n')
            args=['--real-manifest',str(a),'--evaluation-exclusion',str(b),
                  '--synthetic-manifest',str(manifest),'--output-dir',str(root/'result'),
                  '--image-size','16','--max-images','1']
            with contextlib.redirect_stdout(io.StringIO()): main(args)
            profile=json.loads((root/'result/profile.json').read_text())
            self.assertEqual(profile['status'],'exploratory')
            self.assertEqual(profile['provenance']['fitted_operations'],['gain_gamma','saturation'])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): main(args)


if __name__ == '__main__': unittest.main()
