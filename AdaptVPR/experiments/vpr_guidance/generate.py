"""Run paired released IC-Light baseline and inference-time SALAD guidance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]


def released_global_negative_prompt():
    # AdaptVPR/prompts/__init__.py uses top-level imports intended for its scripts.
    # Read only the pure policy module without importing that package or planner.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        '_vpr_guidance_released_rules', ROOT / 'AdaptVPR/prompts/rules.py')
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    return rules.global_negative_prompt()


def read_prompts(path, conditions=(), limit=0):
    rows = []
    seen = set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('route') != 'global':
                continue
            if conditions and row.get('condition', '').lower() not in conditions:
                continue
            for key in ('sample_id', 'source_id', 'condition', 'prompt'):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f'{path}:{line_number}: missing string {key}')
            if row['sample_id'] in seen:
                raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
            seen.add(row['sample_id'])
            rows.append(row)
    if not rows:
        raise ValueError('No Global-route samples match the filters')
    return rows[:limit] if limit else rows


def geometry_score(source, generated):
    """Optional independent SIFT/homography score: inliers / source keypoints.

    At least 8 ratio-test matches are required; failed verification scores zero.
    This diagnostic measures retained corresponding features, not semantic layout.
    """
    import cv2
    import numpy as np
    detector = cv2.SIFT_create(nfeatures=2048)
    a = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.asarray(generated.resize(source.size)), cv2.COLOR_RGB2GRAY)
    ka, da = detector.detectAndCompute(a, None)
    kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None or len(db) < 2:
        return 0.0
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair]
            if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0.0
    cv2.setRNGSeed(0)
    _, mask = cv2.findHomography(
        np.float32([ka[m.queryIdx].pt for m in good]),
        np.float32([kb[m.trainIdx].pt for m in good]), cv2.RANSAC, 3.0)
    return float(mask.sum() / max(len(ka), 1)) if mask is not None else 0.0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts', type=Path, default=ROOT / 'AdaptVPR/tests/demo_10_prompts.jsonl')
    p.add_argument('--image-root', type=Path, default=ROOT / 'dataset/gsv-cities/Images')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--conditions', nargs='+', default=[], help='Exact condition names, e.g. snow night rain fog')
    p.add_argument('--guidance-scale', type=float, nargs='+', default=[0.01, 0.03, 0.1])
    p.add_argument('--guidance-every', type=int, default=5)
    p.add_argument('--guidance-last-n', type=int, default=0, help='Last N actual steps across both stages; 0=all')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--geometry', action='store_true', help='Independent SIFT homography diagnostic (requires OpenCV)')
    p.add_argument('--salad-repo', default='serizba/salad', help='Torch Hub repo[:ref]; log/pin ref for repeat runs')
    p.add_argument('--dry-run', action='store_true', help='Validate inputs without loading models')
    return p


def main():
    args = parser().parse_args()
    if (args.guidance_every < 1 or args.guidance_last_n < 0 or args.limit < 0
            or any(not 0 <= s < float('inf') for s in args.guidance_scale)):
        raise ValueError('Scales must be finite and nonnegative, every>=1, last-n/limit>=0')
    if len(set(args.guidance_scale)) != len(args.guidance_scale):
        raise ValueError('Duplicate guidance scales')
    from .gsv_pairs import image_index, parse_filename
    samples = read_prompts(args.prompts, [c.lower() for c in args.conditions], args.limit)
    images = image_index(args.image_root)
    for row in samples:
        row['source_path'] = images[parse_filename(row['source_id']).capture_id]
    if args.dry_run:
        print(json.dumps([dict(row, source_path=str(row['source_path'])) for row in samples], indent=2))
        return

    import torch
    from PIL import Image, ImageDraw
    from .ddim import Guidance, ICLightExperiment
    from .vpr import load_salad
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; generation requires the released CUDA IC-Light stack')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    experiment = ICLightExperiment()
    salad = load_salad(repo=args.salad_repo)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Fail rather than mix runs or accidentally append duplicate queries.
    records_path = output / 'records.jsonl'
    source_cache = {}
    default_negative = released_global_negative_prompt()
    with records_path.open('x') as records:
        for row in samples:
            safe_id = re.sub(r'[^A-Za-z0-9_.-]', '_', row['sample_id'])
            folder = output / safe_id
            folder.mkdir(exist_ok=False)
            with Image.open(row['source_path']) as opened:
                source = opened.convert('RGB')
            source_path = folder / 'source.png'
            source.save(source_path)
            if row['source_id'] not in source_cache:
                with torch.no_grad():
                    source_cache[row['source_id']] = salad.from_pil(source).detach()
            source_desc = source_cache[row['source_id']]
            # Save the cached normalized descriptor used as the fixed loss target.
            torch.save(source_desc.cpu(), folder / 'source_salad.pt')
            strength = (experiment.adapter.RAIN_HIGHRES_DENOISE if row['condition'].lower() == 'rain'
                        else experiment.adapter.DEFAULT_HIGHRES_DENOISE)
            negative = row.get('negative_prompt') or default_negative
            baseline, _ = experiment.generate(source, row['prompt'], negative, args.seed,
                                               strength, salad, source_desc, Guidance(scale=0))
            baseline_path = folder / 'baseline.png'
            baseline.save(baseline_path)
            with torch.no_grad():
                baseline_cosine = float((salad.from_pil(baseline) * source_desc).sum())
            baseline_geo = geometry_score(source, baseline) if args.geometry else None
            for scale in args.guidance_scale:
                guided, trace = experiment.generate(source, row['prompt'], negative, args.seed,
                                                     strength, salad, source_desc,
                                                     Guidance(scale, args.guidance_every, args.guidance_last_n))
                guided_path = folder / f'guided_{scale:g}.png'
                guided.save(guided_path)
                with torch.no_grad():
                    guided_cosine = float((salad.from_pil(guided) * source_desc).sum())
                guided_geo = geometry_score(source, guided) if args.geometry else None
                width, height = baseline.size
                panel = Image.new('RGB', (width * 3, height + 24), 'white')
                draw = ImageDraw.Draw(panel)
                for i, (label, image) in enumerate([('source', source), ('baseline', baseline),
                                                   (f'guided {scale:g}', guided)]):
                    panel.paste(image.resize((width, height)), (i * width, 24))
                    draw.text((i * width + 6, 4), label, fill='black')
                panel_path = folder / f'comparison_{scale:g}.png'
                panel.save(panel_path)
                record = {
                    'sample_id': row['sample_id'], 'source_id': row['source_id'],
                    'route': 'global', 'condition': row['condition'], 'seed': args.seed,
                    'guidance_scale': scale, 'guidance_every': args.guidance_every,
                    'guidance_last_n': args.guidance_last_n,
                    'salad_cosine_similarity': {'baseline': baseline_cosine, 'guided': guided_cosine},
                    'geometry_verification_score': {'baseline': baseline_geo, 'guided': guided_geo},
                    'geometry_method': 'sift_homography_inliers/source_keypoints' if args.geometry else None,
                    'output_paths': {'source': str(source_path), 'baseline': str(baseline_path),
                                     'guided': str(guided_path), 'comparison': str(panel_path)},
                    'guidance_trace': trace, 'prompt': row['prompt'], 'negative_prompt': negative,
                    'sampling': {'base_steps': experiment.adapter.DEFAULT_INFERENCE_STEPS,
                                 'highres_steps': experiment.adapter.DEFAULT_HIGHRES_STEPS,
                                 'highres_scale': experiment.adapter.DEFAULT_HIGHRES_SCALE,
                                 'highres_denoise': strength, 'cfg_scale': 7.5, 'eta': 0},
                    'models': {'salad_hub_repo': args.salad_repo,
                               'base_id': experiment.adapter.BASE_MODEL_ID,
                               'base_revision': experiment.adapter.BASE_MODEL_REVISION,
                               'iclight_revision': experiment.adapter.CHECKPOINT_REVISION},
                    'versions': {'torch': torch.__version__,
                                 'diffusers': __import__('diffusers').__version__},
                }
                records.write(json.dumps(record, allow_nan=False) + '\n')
                records.flush()
                print(f"{row['sample_id']} scale={scale:g}: SALAD {baseline_cosine:.4f} -> {guided_cosine:.4f}", flush=True)


if __name__ == '__main__':
    main()
