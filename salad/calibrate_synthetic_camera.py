#!/usr/bin/env python3
"""Fit exploratory tone parameters on an explicitly independent real-night split.

CPU image statistics only. Never launches generation, training, model inference,
or retrieval. Does not certify night realism or synthesize missing image content.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from workflow.camera_calibration import (describe, digest, eligible_synthetic_images,
                                         independent_real_images, photometry, tone_distance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-manifest', type=Path, required=True)
    parser.add_argument('--evaluation-exclusion', type=Path, required=True)
    parser.add_argument('--synthetic-manifest', type=Path, required=True)
    parser.add_argument('--base-profile', type=Path, default=Path(__file__).parent/'configs/synthetic_camera_exploratory.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-images', type=int, default=128)
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args(argv)
    if args.max_images < 1 or args.image_size < 16:
        parser.error('Positive image budget and image size >=16 required')
    if args.output_dir.exists():
        parser.error('Use a fresh output directory; calibration artifacts are write-once')
    real, provenance = independent_real_images(args.real_manifest, args.evaluation_exclusion)
    synthetic = eligible_synthetic_images(args.synthetic_manifest)
    if {r['sha256'] for r in real} & {r['sha256'] for r in synthetic}:
        raise ValueError('Real calibration and synthetic training images overlap')
    # Fixed identity ordering, never sorted by model errors or retrieval scores.
    real = sorted(real, key=lambda r:r['sha256'])[:args.max_images]
    synthetic = sorted(synthetic, key=lambda r:r['sha256'])[:args.max_images]
    from PIL import Image
    from workflow.camera_augmentation import CameraAugmentation
    def read(row):
        with Image.open(row['path']) as image:
            return image.convert('RGB').resize((args.image_size,args.image_size),Image.Resampling.BILINEAR)
    target = describe([photometry(read(r)) for r in real])
    images = [read(r) for r in synthetic]
    before = describe([photometry(image) for image in images])
    candidates = []
    for gain, gamma, saturation in itertools.product((1.,1.25,1.5,2.),(.65,.8,1.),(.6,.8,1.)):
        profile = dict(version=1,status='exploratory',conditions=['night','rainy_night'],probability=1.,
                       operations={'gain_gamma':{'gain':[gain,gain],'gamma':[gamma,gamma]},
                                   'saturation':{'factor':[saturation,saturation]}})
        transform = CameraAugmentation(profile)
        observed = describe([photometry(transform.apply(image,seed=i,condition='night')) for i,image in enumerate(images)])
        candidates.append(dict(gain=gain,gamma=gamma,saturation=saturation,statistics=observed,
                               distance=tone_distance(observed,target)))
    best = min(candidates,key=lambda r:r['distance'])
    profile = json.loads(args.base_profile.read_text())
    # Other operations remain explicit priors, hence the overall status remains exploratory.
    profile['status'] = 'exploratory'
    profile['operations']['gain_gamma'] = dict(probability=1.,gain=[best['gain']]*2,gamma=[best['gamma']]*2)
    profile['operations']['saturation'] = dict(probability=1.,factor=[best['saturation']]*2)
    profile.setdefault('provenance',{}).update(provenance,
        synthetic_manifest_sha256=digest(args.synthetic_manifest),base_profile_sha256=digest(args.base_profile),
        fitted_operations=['gain_gamma','saturation'],unfitted_operations=['bloom','motion_blur','noise','jpeg','vignette'],
        fit_scope='Tone-only median/IQR comparison on unpaired resized images. Not a fitted physical camera pipeline.',
        selected_real=real,selected_synthetic=synthetic)
    CameraAugmentation(profile)  # Validate before creating artifacts.
    args.output_dir.mkdir(parents=True)
    report = dict(real=target,synthetic_before=before,best_tone_only=best,candidates=candidates,
                  image_size=args.image_size,real_count=len(real),synthetic_count=len(synthetic),
                  provenance=provenance,limitation='Report is tone-only; full stochastic profile also includes unfitted priors.')
    for name,data in [('profile.json',profile),('report.json',report)]:
        (args.output_dir/name).write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(args.output_dir/'profile.json')


if __name__ == '__main__': main()
