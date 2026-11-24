"""Night-only, source-filtered style coverage; old frozen scene runs stay intact."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import campaign, common, run, night_diversity_policy as policy

ROOT = common.WORKSPACE_ROOT/'outputs/scene_night_balanced_v1'
_RUNTIME = None


def validate_audit(value, job):
    for key in ('structure_preserved', 'is_night', 'landmarks_visible', 'style_matched'):
        if key not in value or (value[key] is not None and type(value[key]) is not bool):
            raise ValueError('Explicit true/false/null required: '+key)
    for key in ('permanent_changes', 'transient_changes'):
        if not isinstance(value.get(key), list) or any(not isinstance(v, str) for v in value[key]):
            raise ValueError('Missing audit evidence: '+key)
    if value.get('surface_state') not in ('dry', 'wet', 'mixed', 'uncertain'):
        raise ValueError('Invalid observed surface state')
    policy.text(value.get('evidence'), 'comparison evidence', 3)
    conditions = [value[key] for key in ('structure_preserved', 'is_night', 'landmarks_visible')]
    if False in conditions or value['permanent_changes']:
        status = 'reject'
    elif all(v is True for v in conditions):
        status = 'candidate'
    else:
        status = 'unknown'
    # Style fidelity controls diversity accounting, not whether a sound night
    # augmentation is rejected. Transient counts and wetness are not hard gates.
    return {**value, 'status': status}


def make_jobs(sources, plans, paired_transients, seed):
    if paired_transients:
        raise ValueError('Use integrated transient changes; no exact transient pairs in this policy')
    jobs = []
    for source, plan in zip(sources, plans, strict=True):
        policy.validate_plan(plan, len(plan['variants']))
        for variant in plan['variants']:
            key = common.fingerprint([policy.VERSION, source['source_sha256'], plan, variant['style_id'], seed])
            job = {k: source[k] for k in ('city', 'place_id', 'source_path', 'source_sha256', 'source_dimensions')}
            job.update(sample_id='qnight_'+key[:24], condition='night', style_id=variant['style_id'],
                prompt_policy=policy.VERSION, prompt=policy.render_prompt(variant), negative_prompt='',
                prompt_variant=variant['style_id'], style_name=variant['style_name'],
                seed=common.candidate_seed(seed, key, 0), source_sample_id=source['sample_id'],
                pair_id=key, with_transient=False, transient_policy='integrated plausible changes allowed',
                scene_plan_sha256=common.fingerprint(plan),
                appearance={**variant, 'requested_mechanism': policy.STYLES[variant['style_id']]['mechanism']},
                protected_regions=plan['protected_regions'], scene_summary=plan['scene_summary'],
                authorized_transient=None, planned_region_is_verified_mask=False)
            job['record_sha256'] = common.fingerprint(job)
            jobs.append(job)
    return jobs


def runtime():
    """Private adapter: reuse durable HTTP/resume/verification, not old policy."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    name = 'experiments.qwen_curriculum._balanced_night_runtime'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('scene_aware_night.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    module.planning, module.ROOT = policy, ROOT
    module.make_jobs, module.validate_audit = make_jobs, validate_audit
    old_choose_sources = module.choose_sources
    def choose_sources(rows, ids, maximum, excluded, real_root, seed):
        if not ids:
            # Multiple weather rows can refer to the same real place. Select
            # one deterministic source per place before applying the bound.
            places = {}
            for row in sorted(rows, key=lambda r: common.fingerprint([seed, r['sample_id']])):
                key = row['city'], int(row['place_id'])
                if key not in excluded:
                    places.setdefault(key, row)
            ids = [r['sample_id'] for r in list(places.values())[:maximum]]
        return old_choose_sources(rows, ids, maximum, excluded, real_root, seed)
    module.choose_sources = choose_sources
    old_definition = module.quality_definition
    def definition(**kwargs):
        value = old_definition(**kwargs)
        value.update(version=policy.VERSION, catalog_sha256=common.file_sha256(policy.CATALOG_PATH),
            adapter_sha256=common.file_sha256(__file__), legacy_reader_sha256=common.file_sha256(policy.legacy.__file__),
            planning_model_calls_per_source=dict(source_analysis=1, composer_drafts=2, total_maximum=3),
            source_analysis_sampling='greedy; cached independently of bounded composer drafts',
            eligibility='night + permanent structure + useful place evidence; transient and wet/dry changes allowed')
        return value
    module.quality_definition = definition
    previous_verifier = module.CandidateVerifier
    class NightVerifier(previous_verifier):
        def evaluate(self, source, output, condition):
            value = super().evaluate(source, output, condition)
            value.update(quality_version=policy.VERSION, content_status='pending_night_structure_style_audit')
            return value
    module.CandidateVerifier = NightVerifier
    load = module.load_plan
    def load_plan(directory):
        cfg, jobs = load(directory)
        planner = campaign.config_file(Path(directory)/'planner_config.json')
        snapshot = campaign.config_file(Path(directory)/'coverage_snapshot.json')
        if planner['policy']['coverage_snapshot'] != snapshot:
            raise ValueError('Frozen style-coverage snapshot changed')
        if any(job['condition'] != 'night' or job.get('style_id') not in policy.STYLES for job in jobs):
            raise ValueError('Only catalogued night mechanisms are enabled')
        return cfg, jobs
    module.load_plan = load_plan
    module.report = report
    module.engine().DOMAINS = ('night',)
    _RUNTIME = module
    return module


def report(directory):
    directory = Path(directory)
    module = runtime()
    cfg, jobs = module.load_plan(directory)
    with run.run_lock(directory):
        _, results, attempts = module.engine()._read_state(directory, jobs, cfg)
        quality = module.quality_records(directory, results)
        counts = policy.coverage_snapshot([directory])
        opportunities = Counter(); analysed = 0
        for file in (directory/'source_analysis').glob('*.json'):
            analysis = json.loads(file.read_text())
            if 'selected' not in analysis:
                continue
            if analysis != run.sealed(analysis):
                raise ValueError('Source capability receipt changed')
            opportunities.update(policy.eligible_styles(analysis['source'])[0]); analysed += 1
        summary = dict(planned=len(jobs), saved=len(results), calls_reserved=len(attempts),
            by_style=counts['counts'], night_only=True, training_eligible=0,
            analysed_sources=analysed, eligible_source_counts=dict(opportunities),
            base_candidates=sum(v['base_quality_passed'] for v in quality.values()))
        out = directory.parent/'review'; out.mkdir(parents=True, exist_ok=True)
        common.write_json(out/'summary.json', summary)
        lines = ['# Balanced night augmentation', '',
            '风格机制用于覆盖调度，最终Prompt由VLM逐图撰写。车辆/行人与干湿变化允许；结构与夜晚效果单独检查。',
            'saved为实际生成曝光；confirmed为自动审核确认机制的探索统计，不是独立语义真值。', '',
            '|style|eligible sources|planned|saved|pending|style confirmed|style mismatch|', '|---|---:|---:|---:|---:|---:|---:|']
        planned = Counter(j['style_id'] for j in jobs)
        for key, values in counts['counts'].items():
            lines.append(f"|{key}|{opportunities[key]}|{planned[key]}|{values['saved']}|{values['pending']}|{values['confirmed']}|{values['mismatch']}|")
        lines += ['', '## Source / generated night', '']
        # Bounded preview; all records still contribute to the table.
        preview = [j for j in jobs if j['sample_id'] in results][:24]
        for start in range(0, len(preview), 6):
            batch = preview[start:start+6]
            canvas = Image.new('RGB', (800, 330*len(batch)), 'white'); draw = ImageDraw.Draw(canvas)
            for row, job in enumerate(batch):
                draw.text((5, row*330+3), f"{job['city']}/{job['place_id']} | {job['style_id']}", fill='black')
                for col, path in enumerate((job['source_path'], results[job['sample_id']]['output_path'])):
                    with Image.open(path) as opened:
                        image = ImageOps.contain(opened.convert('RGB'), (396, 297))
                        canvas.paste(image, (col*400, row*330+25))
            filename = f'night_{start//6+1:02d}.jpg'; canvas.save(out/filename, quality=92)
            lines += [f'![real source / generated night]({filename})', '']
        (out/'report.md').write_text('\n'.join(lines)+'\n')
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', nargs='?', default='prepare', choices=('prepare', 'run', 'check', 'verify', 'audit', 'report'))
    parser.add_argument('--output-dir', type=Path, default=ROOT/'state')
    parser.add_argument('--image-root', type=Path, default=ROOT)
    parser.add_argument('--coverage-state', action='append', type=Path, default=[])
    parser.add_argument('--execute', action='store_true')
    args, remaining = parser.parse_known_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.image_root = args.image_root.resolve()
    module = runtime()
    if args.execute and args.command in ('prepare', 'run', 'verify', 'audit') and os.environ.get('SCENE_RESOURCE_CHILD') != '1':
        raise ValueError('Use run_scene_night_safe.py --pipeline balanced for model execution and resource limits')
    if args.command == 'prepare':
        # Immutable historical observation; growing source libraries do not
        # silently rewrite a prepared run's sampling trajectory.
        path = args.output_dir/'coverage_snapshot.json'
        if path.exists():
            snapshot = campaign.config_file(path)
            if snapshot['requested_states'] != [str(p.resolve()) for p in args.coverage_state]:
                raise ValueError('Coverage inputs changed; use a new output directory')
        else:
            snapshot = campaign.frozen(path, {**policy.coverage_snapshot(args.coverage_state),
                'requested_states': [str(p.resolve()) for p in args.coverage_state]})
        policy.SNAPSHOT, policy.STATE = snapshot, args.output_dir
        planner_file = args.output_dir/'planner_config.json'
        if planner_file.exists() and campaign.config_file(planner_file)['policy']['catalog'] != policy.CATALOG:
            raise ValueError('Style catalog changed after input preparation; use a new output directory')
        if not any(v == '--source-manifest' or v.startswith('--source-manifest=') for v in remaining):
            remaining += ['--source-manifest', str(common.WORKSPACE_ROOT/'outputs/qwen_curriculum/state/results.jsonl')]
        if not any(v == '--service-config' or v.startswith('--service-config=') for v in remaining):
            config_path = args.output_dir.parent/'service_config_24.json'
            if not config_path.exists():
                parent = campaign.config_file(common.WORKSPACE_ROOT/'outputs/qwen_curriculum/state/all_city_campaign_config.json')
                service = parent['reference_execution']['service']
                service['performance']['resident_dit_blocks'] = 24
                campaign.frozen(config_path, dict(reference_execution=dict(service=service), purpose=policy.VERSION))
            remaining += ['--service-config', str(config_path)]
    command = [args.command, '--output-dir', str(args.output_dir), '--image-root', str(args.image_root), *remaining]
    if args.execute:
        command.append('--execute')
    module.main(command)


if __name__ == '__main__':
    main()
