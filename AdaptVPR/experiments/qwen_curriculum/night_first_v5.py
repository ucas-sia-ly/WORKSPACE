"""Bounded same-source night relighting pilot; no training or automatic expansion.

Uses the established durable Qwen request engine in a private namespace. Frozen
v3/v4 campaigns, images and service settings remain independently reproducible.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import (campaign, common, night_appearance_v5 as appearance,
    prompt_bank_v5 as prompt_bank, quality_v3, run, weather_io)

ROOT = common.WORKSPACE_ROOT / 'outputs/weather_v5_pilot'
DEFAULT_OUTPUT = ROOT / 'state'
_ENGINE = None


def quality_definition_v5(**kwargs):
    value = quality_v3.quality_definition_v3(**kwargs)
    value.update(version='5-candidate', prompt_bank=prompt_bank.POLICY_VERSION,
                 night_appearance=appearance.appearance_definition(kwargs.get('clip_model_name')),
                 training_eligibility='none before independent content audit',
                 base_quality_gate='geometry/weather AND all regional night comparisons candidate')
    return value


class CandidateVerifier(quality_v3.QwenQualityVerifierV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.definition = quality_definition_v5(thresholds=self.thresholds,
            matcher_name=self.matcher_name, img_size=self.img_size, n_kpts=self.n_kpts,
            clip_device=self.clip_device, clip_model_name=self.clip_model_name)
        self.appearance = None

    def evaluate(self, source, generated, condition):
        if condition != 'night':
            raise ValueError('This version validates strict night only')
        verdict = super().evaluate(source, generated, condition)
        self._ensure_weather()
        if self.appearance is None:
            self.appearance = appearance.NightAppearanceSignalV5(self.evaluator)
        local = self.appearance.measure(source, generated)
        base = verdict['passed'] is True
        candidate = base and local['appearance_status'] == 'candidate'
        verdict.update(geometry_weather_passed=base, base_quality_passed=candidate,
            passed=candidate, status='passed' if candidate else ('unknown' if base else 'rejected'),
            eligible_for_training=False, content_status='pending_automated_audit',
            quality_version='5-candidate', night_appearance=local)
        if not candidate and base:
            verdict['rejection_reasons'] = list(verdict.get('rejection_reasons', [])) + ['regional_night_evidence_unknown']
        return verdict


def _engine():
    global _ENGINE
    if _ENGINE is None:
        name = 'experiments.qwen_curriculum._night_first_v5_durable_engine'
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('night_first_v3.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.prompt_bank = prompt_bank
        module.ROOT, module.DEFAULT_OUTPUT, module.DEFAULT_IMAGES = ROOT, DEFAULT_OUTPUT, ROOT
        module.DOMAINS = ('night',)
        module.load_plan = load_plan
        module.QwenQualityVerifierV3 = CandidateVerifier
        module._candidate_publish = module.publish
        module.publish = publish
        module._durable_append = module._append
        module._append = bounded_append
        _ENGINE = module
    return _ENGINE


def bounded_append(path, row):
    """Enforce the cumulative HTTP reservation budget across every restart."""
    path = Path(path)
    if path.name == 'attempts.jsonl':
        config = campaign.config_file(path.parent / 'plan_config.json')
        if row['call_number'] > config['request_budget']:
            raise RuntimeError('Frozen cumulative request budget exhausted; use a new explicitly budgeted pilot')
    _engine()._durable_append(path, row)


def publish(directory, config, jobs, generated, results, attempts, state, domain, error=None):
    report = _engine()._candidate_publish(directory, config, jobs, generated, results, attempts, state, domain, error)
    report.update(candidate_passed=sum(r.get('passed') is True for r in results.values()),
                  geometry_weather_passed=sum(r.get('geometry_weather_passed') is True for r in results.values()),
                  appearance_unknown=sum(r.get('night_appearance', {}).get('appearance_status') == 'unknown'
                                         for r in results.values()),
                  accepted=0, training_eligibility='independent content audit required; candidate is not semantic truth')
    for counts in report['by_domain'].values():
        counts['candidate_passed'], counts['accepted'] = counts['accepted'], 0
    common.write_json(directory / 'summary.json', report)
    return report


def safe_roots(directory, image_root):
    for path in (Path(directory).resolve(), Path(image_root).resolve()):
        for name in ('weather_v3', 'weather_v4', 'qwen_curriculum'):
            old = common.WORKSPACE_ROOT / 'outputs' / name
            if path == old or old in path.parents or path in old.parents:
                raise ValueError('Use a new independent v5 output root')


def make_plan(rows, max_places, sample_ids, excluded, seed=42):
    if type(max_places) is not int or max_places <= 0:
        raise ValueError('A finite positive max_places budget is required')
    selected = {}
    for row in rows:
        if (row.get('condition') == 'night' and row.get('prompt_variant') in prompt_bank.VARIANTS['night']
                and (row['city'], int(row['place_id'])) not in excluded):
            if row['sample_id'] in selected:
                raise ValueError('Duplicate comparison sample ID')
            selected[row['sample_id']] = row
    if sample_ids:
        if len(set(sample_ids)) != len(sample_ids) or len(sample_ids) > max_places:
            raise ValueError('Explicit IDs must be unique and within the declared budget')
        if any(sid not in selected for sid in sample_ids):
            raise ValueError('Explicit source is excluded, missing or not a comparable strict-night variant')
        picked = [selected[sid] for sid in sample_ids]
    else:
        picked = sorted(selected.values(), key=lambda r: common.fingerprint([seed, r['sample_id']]))[:max_places]
    if not picked or len({(r['city'], int(r['place_id'])) for r in picked}) != len(picked):
        raise ValueError('Need nonempty unique training places')
    jobs = []
    for old in picked:
        variant = old['prompt_variant']
        job = {key: old[key] for key in ('city', 'place_id', 'source_path', 'source_sha256', 'source_dimensions')}
        job.update(sample_id='qwen5_' + common.fingerprint([prompt_bank.POLICY_VERSION, old['sample_id'], old['seed']])[:24],
            condition='night', prompt_variant=variant, prompt=prompt_bank.prompt('night', variant),
            negative_prompt='', prompt_policy=prompt_bank.POLICY_VERSION, seed=old['seed'],
            comparison={key: old[key] for key in ('sample_id', 'output_path', 'output_sha256', 'seed', 'prompt_policy')})
        job['record_sha256'] = common.fingerprint(job)
        jobs.append(job)
    return jobs


def prepare(args):
    safe_roots(args.output_dir, args.image_root)
    with run.run_lock(args.output_dir):
        if (args.output_dir / 'plan_config.json').exists():
            config, _ = load_plan(args.output_dir)
            requested = dict(source_manifest=str(args.source_manifest), exclude_places=str(args.exclude_places),
                             max_places=args.max_places, sample_ids=args.sample_ids, seed=args.seed,
                             request_budget=args.request_budget)
            if config['selection_request'] != requested or config['image_root'] != str(args.image_root):
                raise ValueError('Existing pilot selection differs; use a new output directory')
            return config
        if any((args.output_dir / name).exists() for name in ('plan.jsonl', 'generated.jsonl', 'attempts.jsonl')):
            raise ValueError('Unsealed pilot state already exists')
        rows = weather_io.read_rows(args.source_manifest)
        excluded = {(city, int(place)) for city, place in json.loads(args.exclude_places.read_text())}
        jobs = make_plan(rows, args.max_places, args.sample_ids, excluded, args.seed)
        chosen = {job['comparison']['sample_id'] for job in jobs}
        comparisons = [row for row in rows if row['sample_id'] in chosen]
        for row in comparisons:
            if row != run.sealed(row) or not prompt_bank.prompt_bank_v4.matches_record(row):
                raise ValueError('Comparison record seal is invalid')
            for key in ('source', 'output'):
                run.validate_image(row, key + '_path', key + '_sha256')
            source = Path(row['source_path']).resolve()
            if (source.parent != args.real_data / 'Images' / row['city']
                    or not source.name.startswith(f"{row['city']}_{int(row['place_id']):07d}_")):
                raise ValueError('Source is not the exact declared real training place')
            with Image.open(source) as image:
                if list(image.size) != row['source_dimensions']:
                    raise ValueError('Source dimensions differ')
        budget = len(jobs) if args.request_budget is None else args.request_budget
        if type(budget) is not int or budget < len(jobs):
            raise ValueError('Request budget must cover every planned place')
        common.write_jsonl(args.output_dir / 'comparison_snapshot.jsonl', comparisons)
        common.write_jsonl(args.output_dir / 'plan.jsonl', jobs)
        definition = run.execution_definition(SimpleNamespace(run_dir=args.output_dir, device='cuda', cpu_threads=4))
        modules = (sys.modules[__name__], _engine(), prompt_bank, prompt_bank.legacy, prompt_bank.prompt_bank_v4, appearance,
                   quality_v3, campaign, run, common, weather_io, _engine().resume_recovery)
        files = {Path(m.__file__).resolve() for m in modules}
        files.update(common.ADAPTVPR_ROOT / name for name in definition['implementation_sha256'])
        files.update((args.exclude_places, args.output_dir / 'comparison_snapshot.jsonl',
                      common.ADAPTVPR_ROOT / 'adapters/lightx2v_qwen_image_edit.py'))
        files.update((args.real_data / 'Dataframes').glob('*.csv'))
        service = campaign.config_file(args.service_config)['reference_execution']['service']
        files.add(args.service_config)
        config = campaign.frozen(args.output_dir / 'plan_config.json', dict(schema_version=1,
            stage='strict_night_v5_paired_pilot', num_jobs=len(jobs), num_places=len(jobs), request_budget=budget,
            domain_counts={'night': len(jobs)}, image_root=str(args.image_root), real_data=str(args.real_data),
            selection_request=dict(source_manifest=str(args.source_manifest), exclude_places=str(args.exclude_places),
                                   max_places=args.max_places, sample_ids=args.sample_ids, seed=args.seed,
                                   request_budget=args.request_budget),
            source_manifest_sha256=common.file_sha256(args.source_manifest), prompt_bank=prompt_bank.definition(),
            service=service, quality=quality_definition_v5(clip_device='cpu'),
            comparison_policy='same real source, variant, seed and deployed backend; only positive prompt changes',
            plan_sha256=common.file_sha256(args.output_dir / 'plan.jsonl'),
            files_sha256={str(p): common.file_sha256(p) for p in sorted(files)}))
        _engine().rebuild_manifest(args.output_dir, config, [])
        publish(args.output_dir, config, jobs, {}, {}, [], 'prepared', 'night')
        return config


def load_plan(directory):
    directory = Path(directory)
    config = campaign.config_file(directory / 'plan_config.json')
    safe_roots(directory, config['image_root'])
    if (config.get('stage') != 'strict_night_v5_paired_pilot'
            or config['prompt_bank'] != prompt_bank.definition()
            or config['quality'] != quality_definition_v5(clip_device='cpu')):
        raise ValueError('Frozen v5 candidate policy changed')
    if common.file_sha256(directory / 'plan.jsonl') != config['plan_sha256']:
        raise ValueError('Pilot plan checksum changed')
    for name, digest in config['files_sha256'].items():
        if common.file_sha256(name) != digest:
            raise ValueError('Frozen input or implementation changed: ' + name)
    jobs = weather_io.read_rows(directory / 'plan.jsonl')
    if (len(jobs) != config['num_jobs'] or not jobs
            or len({(j['city'], j['place_id']) for j in jobs}) != len(jobs)):
        raise ValueError('Pilot place budget changed')
    for job in jobs:
        if (job['record_sha256'] != common.fingerprint({k: v for k, v in job.items() if k != 'record_sha256'})
                or not prompt_bank.matches_record(job) or not job['sample_id'].startswith('qwen5_')):
            raise ValueError('Invalid v5 job')
    return config, jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('prepare', 'run', 'generate', 'check', 'status'))
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--image-root', type=Path, default=ROOT)
    parser.add_argument('--source-manifest', type=Path, default=common.WORKSPACE_ROOT/'outputs/weather_v4/state/results.jsonl')
    parser.add_argument('--exclude-places', type=Path, default=common.WORKSPACE_ROOT/'outputs/qwen_curriculum/state/finetuning/real_baseline_v1/excluded_places.json')
    parser.add_argument('--service-config', type=Path, default=common.WORKSPACE_ROOT/'outputs/qwen_curriculum/state/all_city_campaign_config.json')
    parser.add_argument('--real-data', type=Path, default=common.WORKSPACE_ROOT/'dataset/gsv-cities')
    parser.add_argument('--max-places', type=int, default=8)
    parser.add_argument('--request-budget', type=int, help='Cumulative reservations including failures; default one per place')
    parser.add_argument('--sample-ids', nargs='*', default=[])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--domain', choices=('night',), default='night')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--qwen-url', default='http://127.0.0.1:8001/generate')
    parser.add_argument('--request-timeout', type=float, default=600.)
    parser.add_argument('--max-calls', type=int, default=0)
    args = parser.parse_args(argv)
    for key in ('output_dir', 'image_root', 'source_manifest', 'exclude_places', 'service_config', 'real_data'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    safe_roots(args.output_dir, args.image_root)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if args.command == 'prepare':
        print(json.dumps(prepare(args), ensure_ascii=False, indent=2))
    elif args.command in ('run', 'generate'):
        if args.max_calls <= 0:
            parser.error('Generation requires an explicit positive --max-calls budget')
        load_plan(args.output_dir)
        if args.command == 'run':
            campaign.ensure_stopped()
            _engine().execute(args)
        else:
            _engine().generate(args)
    elif args.command == 'check':
        with run.run_lock(args.output_dir):
            config, jobs = load_plan(args.output_dir)
            generated, results, attempts = _engine()._read_state(args.output_dir, jobs, config)
            print(json.dumps(dict(planned=len(jobs), saved=len(generated), evaluated=len(results),
                                  calls_reserved=len(attempts), eligible_for_training=0)))
    else:
        print((args.output_dir/'summary.json').read_text())


if __name__ == '__main__':
    main()
