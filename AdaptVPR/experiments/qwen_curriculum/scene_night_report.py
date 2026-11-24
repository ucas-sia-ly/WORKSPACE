"""Read-only reports with explicit appearance/transient slots across resource forks."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from experiments.qwen_curriculum import common, run, scene_aware_night as scene


def pair_slots(jobs):
    pairs = {}
    for job in jobs:
        slot = 'transient' if job['with_transient'] else 'appearance'
        pair = pairs.setdefault(job['pair_id'], {})
        if slot in pair:
            raise ValueError('Duplicate pair slot')
        pair[slot] = job
    return list(pairs.values())


def report(directory):
    directory = Path(directory).resolve()
    cfg, _ = scene.load_plan(directory)
    states = [Path(cfg['resource_parent']['state'])] if cfg.get('resource_parent') else []
    states.append(directory)
    jobs, images, audits, origins, resource_records = {}, {}, {}, {}, []
    reserved = 0
    with ExitStack() as locks:
        for state in states:
            locks.enter_context(run.run_lock(state))
            config, plan = scene.load_plan(state)
            _, saved, attempts = scene.engine()._read_state(state, plan, config)
            quality = scene.quality_records(state, saved)
            reserved += len(attempts)
            for job in plan:
                sid = job['sample_id']
                if sid in jobs and jobs[sid] != job:
                    raise ValueError('Resource forks changed the same job')
                jobs[sid] = job
                if sid in saved:
                    if sid in images:
                        raise ValueError('A job was generated in multiple resource profiles')
                    images[sid] = quality.get(sid, saved[sid])
                    origins[sid] = config['service']['performance']['resident_dit_blocks']
                    file = state/'audit'/f'{sid}.json'
                    if file.exists():
                        entry = json.loads(file.read_text())
                        if entry != run.sealed(entry) or entry['output_sha256'] != saved[sid]['output_sha256']:
                            raise ValueError('Audit differs from exact image')
                        audits[sid] = entry
            for file in (state.parent/'resource_runs').glob('*.json'):
                resource_records.append(dict(file=str(file), **json.loads(file.read_text())))
        pairs = pair_slots(jobs.values())
        root = directory.parent/'review'
        root.mkdir(parents=True, exist_ok=True)
        sheets = []
        for offset in range(0, len(pairs), 4):
            batch = pairs[offset:offset+4]
            sheet = Image.new('RGB', (1200, 345*len(batch)), 'white')
            draw = ImageDraw.Draw(sheet)
            for index, pair in enumerate(batch):
                job = next(iter(pair.values()))
                y = index*345
                draw.text((5, y+3), f"{job['city']}/{job['place_id']} | {job['style_name']} | source / appearance / + transient", fill='black')
                for col, slot in enumerate(('source', 'appearance', 'transient')):
                    item = pair.get(slot)
                    value = images.get(item['sample_id']) if item else None
                    if slot == 'source':
                        path, caption = job['source_path'], 'real source'
                    elif value:
                        sid = item['sample_id']
                        status = audits.get(sid, {}).get('verdict', {}).get('status', 'not audited')
                        path = value['output_path']
                        caption = f"{slot}; VLM={status}; resident blocks={origins[sid]}"
                    else:
                        path = None
                        caption = 'planned, not generated (budget)' if item else 'no transient authorized'
                    x = col*400
                    draw.text((x+5, y+20), caption, fill='black')
                    if path:
                        with Image.open(path) as photo:
                            thumb = ImageOps.contain(photo.convert('RGB'), (396, 300))
                            sheet.paste(thumb, (x+(400-thumb.width)//2, y+43+(300-thumb.height)//2))
            name = f'pairs_{offset//4+1:02d}.jpg'
            sheet.save(root/name, quality=92)
            sheets.append(name)
        summary = dict(unique_planned=len(jobs), generated=len(images), reserved_requests=reserved,
            actual_styles=dict(Counter(jobs[s]['style_name'] for s in images)),
            transient_requests_saved=sum(jobs[s]['with_transient'] for s in images),
            geometry_weather_passed=sum(v.get('base_quality_passed', False) for v in images.values()),
            audit_counts=dict(Counter(audits.get(s, {}).get('verdict', {}).get('status', 'not audited') for s in images)),
            exploratory_candidates=sum(e['candidate_passed'] for e in audits.values()),
            training_eligible=0, states=[str(p) for p in states], resources=resource_records, sheets=sheets)
        common.write_json(root/'summary.json', summary)
        lines = ['# Scene-conditioned night pilot', '',
            '源图 / 外观控制 / 外观加临时物体；缺失图片明确留空，不把物体图当作外观控制。',
            '跨资源配置配对标出各自常驻block数。相同权重、Prompt、seed不保证逐像素相同。',
            'VLM审核与规划同模型家族，candidate不是独立语义真值。没有启动训练。', '',
            '```json', json.dumps({k:v for k,v in summary.items() if k!='resources'}, ensure_ascii=False, indent=2), '```', '']
        for name in sheets:
            lines.extend([f'![source / appearance / transient]({name})', ''])
        lines += ['## 资源记录', '']
        for record in resource_records:
            lines.append(f"- `{record['file']}`: VRAM={record.get('peak_device_vram_gib', 0):.2f}GiB, "
                         f"RSS={record.get('peak_owned_rss_gib', 0):.2f}GiB, error={record.get('error')}")
        (root/'report.md').write_text('\n'.join(lines)+'\n')
        return summary
