"""Read-only paired appearance diagnostics and source/v4/v5 contact sheets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, night_first_v5 as pilot, run


def compare(state, output, execute=False):
    config, jobs = pilot.load_plan(state)
    results = run.read_rows(state, 'results.jsonl')
    by_id = {row['sample_id']: row for row in results}
    if len(by_id) != len(results) or not set(by_id) <= {j['sample_id'] for j in jobs}:
        raise ValueError('Duplicate or unplanned results')
    for job in jobs:
        row = by_id.get(job['sample_id'])
        if row is None:
            continue
        if row != run.sealed(row) or any(row.get(k) != v for k, v in job.items()):
            raise ValueError('Result seal or frozen job changed')
        for field in ('source', 'output'):
            run.validate_image(row, field+'_path', field+'_sha256')
        run.validate_image(job['comparison'])
    if not execute:
        return dict(planned=len(jobs), completed=len(results), model_loaded=False)
    if not results:
        raise ValueError('No completed image pairs to compare')
    if output.exists():
        raise FileExistsError('Use a new comparison output directory')
    output.mkdir(parents=True)
    import torch
    torch.set_num_threads(4)
    verifier = pilot.quality_v3.QwenQualityVerifierV3(device='cpu', clip_device='cpu')
    verifier._ensure_weather()
    diagnostic = pilot.appearance.NightAppearanceSignalV5(verifier.evaluator)
    metrics = []
    for job in jobs:
        row = by_id.get(job['sample_id'])
        if row is None:
            continue
        with Image.open(row['source_path']) as source, Image.open(job['comparison']['output_path']) as old:
            previous = diagnostic.measure(source, old)
        current = row['night_appearance']
        if current['definition'] != previous['definition']:
            raise ValueError('Old/new appearance definitions differ')
        metrics.append(dict(sample_id=row['sample_id'], city=row['city'], place_id=row['place_id'],
            seed=row['seed'], variant=row['prompt_variant'], source_path=row['source_path'],
            old_path=job['comparison']['output_path'], new_path=row['output_path'],
            source_sha256=row['source_sha256'], old_sha256=job['comparison']['output_sha256'],
            new_sha256=row['output_sha256'], geometry_weather_passed=row['geometry_weather_passed'],
            candidate_passed=row['passed'], old_appearance=previous, new_appearance=current,
            margin_changes={k:current['night_margins'][k]-previous['night_margins'][k]
                            for k in current['night_margins']}))
        common.write_jsonl(output/'pairs.jsonl', metrics)
    sheets = []
    for start in range(0, len(metrics), 4):
        batch = metrics[start:start+4]
        sheet = Image.new('RGB', (1200, len(batch)*330), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(batch):
            y = index*330
            draw.text((8,y+3),f"{row['city']}/{row['place_id']} | {row['variant']} | source / v4 / v5",fill='black')
            for column,key in enumerate(('source_path','old_path','new_path')):
                with Image.open(row[key]) as image:
                    thumb = ImageOps.contain(image.convert('RGB'), (396,297))
                    sheet.paste(thumb,(column*400,y+25))
        name = f'comparison_{start//4+1:02d}.jpg'
        sheet.save(output/name,quality=93)
        sheets.append(name)
    summary = dict(planned=len(jobs), completed=len(metrics), pending=len(jobs)-len(metrics),
        old_regional_candidate=sum(r['old_appearance']['appearance_status']=='candidate' for r in metrics),
        new_regional_candidate=sum(r['new_appearance']['appearance_status']=='candidate' for r in metrics),
        new_geometry_weather_passed=sum(r['geometry_weather_passed'] for r in metrics),
        new_combined_candidate=sum(r['candidate_passed'] for r in metrics),
        eligible_for_training=0, plan_fingerprint=config['fingerprint'],
        result_manifest_sha256=common.file_sha256(state/'results.jsonl'),
        implementation_sha256=common.file_sha256(__file__),
        interpretation='purposefully selected pilot; uncalibrated CLIP rankings, not night realism or hallucination truth',
        comparison_policy=config['comparison_policy'], sheets=sheets)
    common.write_json(output/'summary.json',summary)
    lines=['# V5 同源、同 seed 夜间对照','',
        '只读 CLIP 区域诊断，不训练；三列分别为真实源图、旧 v4、新 v5。',
        '样本包含先前发现的问题图，不能据此估计总体成功率；CLIP 排名不是夜景真实性证明。','',
        '```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','',
        '| 地点 | 旧区域状态 | 新区域状态 | 中心 margin 变化 | 下部 margin 变化 | 新几何/天气通过 |',
        '|---|---|---|---:|---:|---|']
    for r in metrics:
        lines.append(f"| {r['city']}/{r['place_id']} | {r['old_appearance']['appearance_status']} | "
            f"{r['new_appearance']['appearance_status']} | {r['margin_changes']['center_proxy']:.3f} | "
            f"{r['margin_changes']['lower_proxy']:.3f} | {r['geometry_weather_passed']} |")
    lines += ['',*[f'![源图/v4/v5]({name})' for name in sheets], '',
              '所有图片仍需独立内容审核；Unknown 不进入训练。没有修改旧图片或旧审核结果。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,default=pilot.DEFAULT_OUTPUT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    state=args.state.resolve()
    print(json.dumps(compare(state,args.output_dir.resolve() if args.output_dir else state.parent/'comparison',
                             args.execute),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
