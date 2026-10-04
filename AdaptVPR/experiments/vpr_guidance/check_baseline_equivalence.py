"""Compare the untouched released adapter with the current zero-guidance sampler."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import statistics
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
import torch

from .ddim import Guidance, ICLightExperiment
from .generate import ROOT, read_prompts, released_global_negative_prompt
from .gsv_pairs import image_index, parse_filename
from .vpr import freeze, load_salad, pil_tensor


@contextmanager
def observe_sampling(experiment):
    """Record actual scheduler settings and UNet timesteps without altering tensors."""
    from diffusers import DDIMScheduler
    stages = []
    original_set = DDIMScheduler.set_timesteps
    unet = experiment.t2i.unet
    original_forward = unet.forward

    def set_timesteps(scheduler, *args, **kwargs):
        result = original_set(scheduler, *args, **kwargs)
        stages.append({
            'scheduler': type(scheduler).__name__,
            'scheduler_config': {k: v for k, v in dict(scheduler.config).items() if not k.startswith('_')},
            'scheduled_timesteps': scheduler.timesteps.cpu().tolist(),
            'used_timesteps': [],
        })
        return result

    def forward(sample, timestep, *args, **kwargs):
        stages[-1]['used_timesteps'].append(int(timestep))
        return original_forward(sample, timestep, *args, **kwargs)

    if experiment.i2i.unet is not unet:
        raise RuntimeError('Released stack must share the same UNet between stages')
    with patch.object(DDIMScheduler, 'set_timesteps', new=set_timesteps), \
            patch.object(unet, 'forward', new=forward):
        yield stages


def released_generation(experiment, source_path, prompt, negative_prompt, seed, strength, output_dir):
    """Call the existing adapter.generate function directly, including its pipeline calls.

    No copied sampling implementation and no HTTP transport/JPEG re-encoding.
    Temporary adapter state/environment bindings are restored even on failure.
    """
    adapter = experiment.adapter
    request = adapter.GenerateRequest(image_path=str(source_path), prompt=prompt,
                                      negative_prompt=negative_prompt, seed=seed,
                                      highres_denoise=strength)
    with patch.object(adapter.state, 'pipe_t2i', experiment.t2i), \
            patch.object(adapter.state, 'pipe_i2i', experiment.i2i), \
            patch.object(adapter.state, 'vae', experiment.vae), \
            patch.object(experiment.i2i, 'scheduler', experiment.t2i.scheduler), \
            patch.dict(os.environ, {'ICLIGHT_OUTPUT_DIR': str(output_dir)}):
        response = adapter.generate(request)
    return Path(response['result_path'])


def sampling_configuration(experiment, source, source_path, prompt, negative_prompt, seed,
                           strength, stages, released=False):
    adapter = experiment.adapter
    width, height = adapter._valid_size(source)
    # Observe released CFG/eta defaults rather than pass overrides to the adapter.
    import inspect
    cfg = [inspect.signature(type(pipe).__call__).parameters['guidance_scale'].default
           for pipe in (experiment.t2i, experiment.i2i)] if released else [7.5, 7.5]
    eta = [inspect.signature(type(pipe).__call__).parameters['eta'].default
           for pipe in (experiment.t2i, experiment.i2i)] if released else [0.0, 0.0]
    return {
        'source_path': str(source_path), 'prompt': prompt, 'negative_prompt': negative_prompt,
        'seed': seed, 'base_resolution': [width, height],
        'refinement_resolution': [int(width * adapter.DEFAULT_HIGHRES_SCALE) // 8 * 8,
                                  int(height * adapter.DEFAULT_HIGHRES_SCALE) // 8 * 8],
        'base_steps': adapter.DEFAULT_INFERENCE_STEPS,
        'highres_scale': adapter.DEFAULT_HIGHRES_SCALE, 'highres_denoise': strength,
        'highres_steps': adapter.DEFAULT_HIGHRES_STEPS,
        'refinement_scheduler_steps': max(1, int(adapter.DEFAULT_HIGHRES_STEPS / strength)),
        'cfg_scale_by_stage': cfg, 'eta_by_stage': eta,
        'base_model_path': str(Path(os.environ['ICLIGHT_BASE_MODEL_PATH']).resolve()),
        'iclight_checkpoint_path': str(Path(os.environ['ICLIGHT_MODEL_PATH']).resolve()),
        'dtype': str(experiment.dtype), 'stages': stages,
        'interstage_image': 'PIL RGB 8-bit; resize then stochastic VAE posterior sampling',
    }


def pixel_metrics(released, explicit):
    if released.size != explicit.size:
        raise ValueError(f'Output resolutions differ: {released.size} vs {explicit.size}')
    difference = np.asarray(released, dtype=np.float64) - np.asarray(explicit, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    return {'pixel_mae': float(np.mean(np.abs(difference))), 'pixel_rmse': rmse,
            'psnr': 20 * math.log10(255 / rmse) if rmse > 0 else None,
            'psnr_is_infinite': rmse == 0}


def comparison_metrics(released, explicit, salad, lpips_model, device, with_ssim=False):
    metrics = pixel_metrics(released, explicit)
    with torch.no_grad():
        a, b = pil_tensor(released).to(device), pil_tensor(explicit).to(device)
        metrics['lpips'] = float(lpips_model(a * 2 - 1, b * 2 - 1).mean())
        metrics['salad_cosine_between_outputs'] = float(
            (salad.from_pil(released) * salad.from_pil(explicit)).sum())
    if with_ssim:
        from skimage.metrics import structural_similarity
        metrics['ssim'] = float(structural_similarity(np.asarray(released), np.asarray(explicit),
                                                     channel_axis=-1, data_range=255))
    if any(value is not None and not math.isfinite(value)
           for key, value in metrics.items() if key != 'psnr_is_infinite'):
        raise FloatingPointError('Non-finite baseline-equivalence metric')
    return metrics


def diagnostic_warnings(metrics, salad_threshold=.98, lpips_threshold=.10,
                        mae_threshold=None, psnr_threshold=None):
    warnings = []
    if metrics['salad_cosine_between_outputs'] < salad_threshold:
        warnings.append(f'SALAD cosine below {salad_threshold:g}')
    if metrics['lpips'] > lpips_threshold:
        warnings.append(f'LPIPS above {lpips_threshold:g}')
    if mae_threshold is not None and metrics['pixel_mae'] > mae_threshold:
        warnings.append(f'Pixel MAE above {mae_threshold:g}')
    if psnr_threshold is not None and metrics['psnr'] is not None and metrics['psnr'] < psnr_threshold:
        warnings.append(f'PSNR below {psnr_threshold:g} dB')
    return warnings


def mean_metrics(records):
    keys = ['pixel_mae', 'pixel_rmse', 'lpips', 'salad_cosine_between_outputs']
    if 'ssim' in records[0]:
        keys.append('ssim')
    means = {key: statistics.mean(record[key] for record in records) for key in keys}
    infinite = any(record['psnr_is_infinite'] for record in records)
    means.update(psnr=None if infinite else statistics.mean(record['psnr'] for record in records),
                 psnr_is_infinite=infinite)
    return means


def print_metrics(label, metrics):
    psnr = 'inf' if metrics['psnr_is_infinite'] else f"{metrics['psnr']:.2f}"
    print(f"{label}: MAE={metrics['pixel_mae']:.4f} RMSE={metrics['pixel_rmse']:.4f} "
          f"PSNR={psnr} dB LPIPS={metrics['lpips']:.5f} "
          f"SALAD cosine={metrics['salad_cosine_between_outputs']:.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, default=ROOT / 'AdaptVPR/tests/demo_10_prompts.jsonl')
    parser.add_argument('--image-root', type=Path, default=ROOT / 'dataset/gsv-cities/Images')
    parser.add_argument('--conditions', nargs='+', default=['snow'])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--salad-repo', default='serizba/salad')
    parser.add_argument('--metric-device', default='cuda', help='LPIPS/SALAD device; generation always uses CUDA')
    parser.add_argument('--warn-salad-cosine', type=float, default=.98)
    parser.add_argument('--warn-lpips', type=float, default=.10)
    parser.add_argument('--warn-pixel-mae', type=float)
    parser.add_argument('--warn-psnr', type=float)
    parser.add_argument('--ssim', action='store_true', help='Optional SSIM, requires scikit-image')
    parser.add_argument('--dry-run', action='store_true', help='Resolve samples without model loads')
    args = parser.parse_args()
    thresholds = [args.warn_salad_cosine, args.warn_lpips, args.warn_pixel_mae, args.warn_psnr]
    if args.limit < 0 or any(v is not None and not math.isfinite(v) for v in thresholds):
        parser.error('Limit must be nonnegative and warning thresholds must be finite')
    rows = read_prompts(args.prompts, [c.lower() for c in args.conditions], args.limit)
    images = image_index(args.image_root)
    for row in rows:
        row['source_path'] = str(images[parse_filename(row['source_id']).capture_id])
    if args.dry_run:
        print(json.dumps(rows, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; baseline equivalence needs the released CUDA IC-Light stack')
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError('Install the experiment requirements including lpips before running equivalence') from exc
    if args.ssim:
        import skimage.metrics  # Fail before generation if optional dependency is absent.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    experiment = ICLightExperiment()
    salad = load_salad(args.metric_device, args.salad_repo)
    lpips_model = freeze(lpips.LPIPS(net='alex', version='0.1', verbose=False)).to(args.metric_device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    default_negative = released_global_negative_prompt()
    records = []
    with (output / 'records.jsonl').open('x') as handle:
        for row in rows:
            folder = output / re.sub(r'[^A-Za-z0-9_.-]', '_', row['sample_id'])
            folder.mkdir(exist_ok=False)
            source_path = Path(row['source_path'])
            with Image.open(source_path) as opened:
                source = opened.convert('RGB')
            source.save(folder / 'source.png')
            negative = row.get('negative_prompt') or default_negative
            strength = (experiment.adapter.RAIN_HIGHRES_DENOISE if row['condition'].lower() == 'rain'
                        else experiment.adapter.DEFAULT_HIGHRES_DENOISE)
            with observe_sampling(experiment) as released_stages:
                temporary = released_generation(experiment, source_path, row['prompt'], negative,
                                                 args.seed, strength, folder)
            released_path = folder / 'released_baseline.png'
            temporary.rename(released_path)
            with Image.open(released_path) as opened:
                released = opened.convert('RGB')
            with observe_sampling(experiment) as explicit_stages:
                explicit, trace = experiment.generate(source, row['prompt'], negative, args.seed,
                                                      strength, None, None, Guidance(scale=0))
            assert trace == [], 'Guidance must be completely disabled for this diagnostic'
            explicit_path = folder / 'explicit_baseline.png'
            explicit.save(explicit_path)
            metrics = comparison_metrics(released, explicit, salad, lpips_model,
                                          args.metric_device, args.ssim)
            configs = {
                'released': sampling_configuration(experiment, source, source_path, row['prompt'],
                                                    negative, args.seed, strength, released_stages, True),
                'explicit': sampling_configuration(experiment, source, source_path, row['prompt'],
                                                    negative, args.seed, strength, explicit_stages),
            }
            warnings = diagnostic_warnings(metrics, *thresholds)
            if configs['released'] != configs['explicit']:
                warnings.append('Observed sampling configurations differ')
            width, height = released.size
            panel = Image.new('RGB', (width * 3, height + 24), 'white')
            draw = ImageDraw.Draw(panel)
            for i, (label, image) in enumerate([('source', source), ('released baseline', released),
                                               ('explicit baseline', explicit)]):
                panel.paste(image.resize((width, height)), (i * width, 24))
                draw.text((i * width + 6, 4), label, fill='black')
            panel.save(folder / 'comparison.png')
            record = {'sample_id': row['sample_id'], 'source_id': row['source_id'],
                      'condition': row['condition'], 'source_path': str(source_path),
                      'prompt': row['prompt'], 'negative_prompt': negative, 'seed': args.seed,
                      'released_output_path': str(released_path), 'explicit_output_path': str(explicit_path),
                      'comparison_path': str(folder / 'comparison.png'), **metrics,
                      'warnings': warnings, 'configuration_comparison': configs,
                      'configuration_equal': configs['released'] == configs['explicit'],
                      'pixel_units': 'RGB uint8 levels [0,255]',
                      'lpips_model': 'alex, version 0.1, pretrained',
                      'salad_repo': args.salad_repo,
                      'versions': {'torch': torch.__version__, 'diffusers': __import__('diffusers').__version__}}
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()
            records.append(record)
            print_metrics(row['sample_id'], metrics)
            if warnings:
                print('WARNING: ' + '; '.join(warnings))
                print('Configuration comparison:\n' + json.dumps(configs, indent=2))
    means = mean_metrics(records)
    summary = {'num_samples': len(records), 'mean': means,
               'warning_thresholds': {'salad_cosine_min': args.warn_salad_cosine,
                                      'lpips_max': args.warn_lpips, 'pixel_mae_max': args.warn_pixel_mae,
                                      'psnr_min': args.warn_psnr},
               'num_samples_with_warnings': sum(bool(r['warnings']) for r in records)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print_metrics(f'Mean ({len(records)} samples)', means)
    print('Thresholds are diagnostics, not hard scientific acceptance criteria.')


if __name__ == '__main__':
    main()
