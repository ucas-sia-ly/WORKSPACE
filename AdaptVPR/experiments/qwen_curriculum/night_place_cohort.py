"""One real source and one precomputed night prompt per eligible training place.

Prompt work is microbatched, journaled and completed before any generation.
Generation plans are exported in bounded shards; no teacher features are cached.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

from PIL import Image

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import (balanced_night, campaign, common, run,
    night_diversity_policy as policy, source_screen_v3)

ROOT = common.WORKSPACE_ROOT/'outputs/night_place_cohort_v1'
VERSION = 'night_place_cohort_v1'


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def chunks(values, count):
    chunk = []
    for value in values:
        chunk.append(value)
        if len(chunk) == count:
            yield chunk; chunk = []
    if chunk:
        yield chunk


def select_source(item, seed, minimum):
    (city, place_id), paths = item
    existing = sorted({p for p in paths if p.is_file()}, key=lambda p: common.fingerprint(
        [VERSION, seed, city, place_id, p.name]))
    if len(existing) < minimum:
        return None, 'fewer_than_minimum_existing_images'
    for path in existing:
        try:
            with Image.open(path) as image:
                dimensions = list(image.size); image.verify()
            digest = common.file_sha256(path)
        except (OSError, ValueError):
            continue
        return run.sealed(dict(city=city, place_id=place_id, source_path=str(path.resolve()),
            source_sha256=digest, source_dimensions=dimensions, num_real_images=len(existing),
            real_pool_sha256=common.fingerprint(sorted(p.name for p in existing)),
            sample_id='qnsource_'+common.fingerprint([seed, city, place_id, path.name])[:24])), None
    return None, 'no_decodable_selected_source'


def dependencies(model):
    modules = (sys.modules[__name__], balanced_night, policy, policy.legacy, common, run, campaign, source_screen_v3)
    paths = {Path(m.__file__).resolve() for m in modules}
    paths.add(policy.CATALOG_PATH)
    paths.add(Path(__file__).with_name('scene_aware_night.py'))
    paths.add(Path(__file__).with_name('night_first_v3.py'))
    paths.update(model.glob('*.json')); paths.update(model.glob('*.txt')); paths.update(model.glob('*.safetensors'))
    return {str(p): common.file_sha256(p) for p in sorted(paths)}


def index(args):
    directory = args.output_dir
    with run.run_lock(directory):
        if (directory/'cohort_config.json').exists():
            cfg = load(directory)
            expected = dict(real_data=str(args.real_data), exclude_places=str(args.exclude_places),
                min_images=args.min_images, seed=args.seed, limit_places=args.limit_places,
                batch_size=args.batch_size, shard_size=args.shard_size, model_path=str(args.planner_model))
            if any(cfg[key] != value for key, value in expected.items()):
                raise ValueError('Frozen place cohort changed; use a new output directory')
            snapshot = campaign.config_file(directory/'coverage_snapshot.json')
            if snapshot['requested_states'] != [str(p) for p in args.coverage_state]:
                raise ValueError('Frozen coverage inputs changed; use a new output directory')
            return cfg
        if args.min_images < 4 or args.batch_size < 1 or args.batch_size > 8 or args.shard_size < 1 or args.limit_places < 0:
            raise ValueError('Require >=4 images/place, microbatch 1..8, positive shard size')
        if not list(args.planner_model.glob('*.safetensors')):
            raise ValueError('Local VLM weights missing; no download or mock fallback')
        excluded = {(city, int(place)) for city, place in json.loads(args.exclude_places.read_text())}
        # Existing parser retains exact released CSV strings and deduplicates views.
        places = source_screen_v3.eligible_places(args.real_data, args.min_images)
        ordered = sorted(((key, paths) for key, paths in places.items() if key not in excluded),
            key=lambda item: common.fingerprint([args.seed, item[0]]))
        if args.limit_places:
            ordered = ordered[:args.limit_places]
        selected, reasons = [], Counter()
        with ThreadPoolExecutor(max_workers=4) as workers:
            for part in chunks(ordered, 128):
                for row, reason in workers.map(lambda item: select_source(item, args.seed, args.min_images), part):
                    if row is None:
                        reasons[reason] += 1
                    else:
                        selected.append(row)
                print(f'index: {len(selected)} eligible sources selected', flush=True)
        if not selected:
            raise ValueError('No eligible, nonexcluded places with usable source images')
        common.write_jsonl(directory/'sources.jsonl', selected)
        coverage = campaign.frozen(directory/'coverage_snapshot.json', {
            **policy.coverage_snapshot(args.coverage_state),
            'requested_states': [str(p) for p in args.coverage_state]})
        parent = campaign.config_file(common.WORKSPACE_ROOT/'outputs/qwen_curriculum/state/all_city_campaign_config.json')
        service = copy.deepcopy(parent['reference_execution']['service'])
        service['performance']['resident_dit_blocks'] = 24
        cfg = campaign.frozen(directory/'cohort_config.json', dict(version=VERSION, condition='night',
            real_data=str(args.real_data), exclude_places=str(args.exclude_places),
            exclude_sha256=common.file_sha256(args.exclude_places), min_images=args.min_images,
            seed=args.seed, limit_places=args.limit_places, batch_size=args.batch_size, shard_size=args.shard_size,
            num_places=len(selected), source_manifest_sha256=common.file_sha256(directory/'sources.jsonl'),
            model_path=str(args.planner_model), image_root=str(directory.parent/'images'), service=service,
            coverage_fingerprint=coverage['fingerprint'], excluded_places=len(excluded),
            selection_rejections=dict(reasons), images_per_place=1, prompts_per_place=1,
            analysis_tokens=1400, compose_tokens=900, compose_drafts=2,
            inference=dict(dtype='bfloat16', attention='sdpa', padding='left', temperature=.8, top_p=.9, top_k=30),
            software={name:importlib.metadata.version(name) for name in ('torch','transformers','Pillow')},
            generation_barrier='all selected sources terminal, successful prompts frozen and exported',
            files_sha256={**dependencies(args.planner_model),
                str(args.exclude_places): common.file_sha256(args.exclude_places),
                **{str(p): common.file_sha256(p) for p in (args.real_data/'Dataframes').glob('*.csv')}}))
        common.write_json(directory/'progress.json', dict(state='indexed_not_prompted', total=cfg['num_places'],
            processed=0, ready=0, failed=0, generation_started=False))
        return cfg


def load(directory):
    cfg = campaign.config_file(Path(directory)/'cohort_config.json')
    if cfg['version'] != VERSION or cfg['condition'] != 'night':
        raise ValueError('Unsupported place cohort')
    if any(importlib.metadata.version(name) != version for name,version in cfg.get('software',{}).items()):
        raise ValueError('Inference software changed; use a new cohort')
    for name, digest in cfg['files_sha256'].items():
        if common.file_sha256(name) != digest:
            raise ValueError('Frozen cohort dependency changed: '+name)
    if common.file_sha256(Path(directory)/'sources.jsonl') != cfg['source_manifest_sha256']:
        raise ValueError('Frozen source cohort changed')
    coverage = campaign.config_file(Path(directory)/'coverage_snapshot.json')
    if coverage['fingerprint'] != cfg['coverage_fingerprint']:
        raise ValueError('Frozen style exposure snapshot changed')
    return cfg


def sealed_read(path):
    value = json.loads(Path(path).read_text())
    if value != run.sealed(value):
        raise ValueError('Changed durable receipt: '+str(path))
    return value


class BatchedPlanner(policy.legacy.LocalVisionPlanner):
    def ask(self, directory, sources, requests, phase, attempt, cfg):
        key = common.fingerprint([cfg['fingerprint'], phase, attempt,
                                  [s['sample_id'] for s in sources], requests])
        path = directory/'batches'/f'{key}.json'
        if path.exists():
            cached = sealed_read(path)
            if cached['key'] != key or cached['source_ids'] != [s['sample_id'] for s in sources]:
                raise ValueError('Cached batch differs from its exact inputs')
            if not isinstance(cached['responses'], list) or len(cached['responses']) != len(sources) or any(not isinstance(v,str) for v in cached['responses']):
                raise ValueError('Cached response batch has invalid cardinality/types')
            return cached['responses']
        reserve = path.with_suffix('.request.json')
        if reserve.exists():
            raise ValueError('Unresolved model batch; no automatic duplicate call: '+key)
        seed = common.candidate_seed(cfg['seed'], key, attempt)
        common.write_json(reserve, dict(cohort=cfg['fingerprint'], source_ids=[s['sample_id'] for s in sources],
            phase=phase, attempt=attempt, seed=seed, logical_inputs=len(sources)))
        conversations = []
        for source, request in zip(sources, requests, strict=True):
            with Image.open(source['source_path']) as opened:
                image = opened.convert('RGB'); image.thumbnail((672, 672))
            system = policy.SYSTEM if phase == 'compose' else (
                'You inspect an original photograph for visual place recognition. '
                'Describe only visible source evidence. The source may be DAY or NIGHT. '
                'Do not assume nighttime because the later task is night augmentation. '
                'Do not perform or describe an edit during this source analysis.')
            conversations.append([{'role':'system','content':[{'type':'text','text':system}]},
                {'role':'user','content':[{'type':'image','image':image},{'type':'text','text':request}]}])
        self.processor.tokenizer.padding_side = 'left'
        inputs = self.processor.apply_chat_template(conversations, tokenize=True, padding=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt').to(self.device)
        options = dict(do_sample=True, temperature=.8, top_p=.9, top_k=30) if phase == 'compose' else dict(do_sample=False)
        with self.torch.inference_mode(), self.torch.random.fork_rng(devices=[self.torch.cuda.current_device()]):
            self.torch.manual_seed(seed)
            ids = self.model.generate(**inputs, max_new_tokens=cfg['compose_tokens'] if phase=='compose' else cfg['analysis_tokens'],
                                      use_cache=True, **options)
        responses = self.processor.batch_decode(ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        if len(responses) != len(sources):
            raise RuntimeError('Batched VLM lost image/response correspondence')
        common.write_json(path, run.sealed(dict(key=key, phase=phase, source_ids=[s['sample_id'] for s in sources],
                                              seed=seed, responses=responses)))
        return responses


def analysis_request():
    return ('Inspect only this original image. Return JSON: coordinate_system="0_1000", scene_summary, '
        'protected_regions (1-3 actual permanent landmarks with description and box), features: every key '+
        json.dumps(policy.FEATURES)+' maps to {"status":"present|absent|uncertain","evidence":"specific visible evidence"}. '
        'Use explicit native 0-to-1000 coordinates. An ordinary window is not a shop; a road is not proof of traffic. '
        'Missing sky does not imply indoors. Do not invent lamps, signs or free object-placement space.')


def compose_request(source, style, feedback=''):
    return ('Write exactly ONE original night variant for this source evidence: '+json.dumps(source)+
        '\nAssigned mechanism (a direction, not a fixed prompt template): '+json.dumps(policy.STYLES[style])+
        '\nReturn JSON {"variants":[{"style_id":"'+style+'","style_name":"descriptive name",'
        '"weather":"night","surface_state":"dry|wet|mixed|unspecified",'
        '"lighting":"source-grounded mechanism","exposure":"night exposure",'
        '"appearance_prompt":"complete original English editing prompt, 40-160 words"}]}. '
        'Preserve permanent structure. Plausible people/vehicle additions/removals and wet/dry changes are allowed. '
        'Night must be apparent in surface illumination even when little sky is visible. Do not invent shops, '
        'signs or fixtures to make a mechanism fit. '+feedback)


def parse_plan(raw, source, style):
    value = policy.parse_json(raw)
    variants = value.get('variants')
    if not isinstance(variants, list) or len(variants) != 1 or not isinstance(variants[0], dict):
        raise ValueError('Exactly one night variant required')
    variants[0]['style_name'] = str(variants[0].get('style_id'))+': '+str(variants[0].get('style_name',''))
    value.update(schema_version=1, coordinate_system='0_1000', scene_summary=source['scene_summary'],
        source_evidence=source, protected_regions=source['protected_regions'], requested_style_ids=[style])
    return policy.validate_plan(policy.normalize_plan(value), 1)


def prompt_places(args):
    directory = args.output_dir
    with run.run_lock(directory):
        cfg = load(directory)
        if (directory/'prompts_complete.json').exists():
            return sealed_read(directory/'prompts_complete.json')
        campaign.ensure_stopped()
        base_counts = campaign.config_file(directory/'coverage_snapshot.json')['counts']
        allocated, failed_reasons = Counter(), Counter()
        completed = ready = failed = fresh = 0; reader = None; started = time.monotonic()
        work_time = 0
        def progress(state='prompting', error=None):
            value = dict(state=state, total=cfg['num_places'], processed=completed, ready=ready, failed=failed,
                fresh_places=fresh, elapsed_seconds=time.monotonic()-started,
                estimated_remaining_seconds=(cfg['num_places']-completed)*work_time/fresh if fresh else None,
                planned_style_counts=dict(allocated), failure_reasons=dict(failed_reasons), generation_started=False)
            if error: value['error'] = error
            common.write_json(directory/'progress.json', value)
        buffered = {}
        def receipt(source, status, **fields):
            value = run.sealed(dict(source=source, source_sha256=source['source_sha256'],
                cohort_fingerprint=cfg['fingerprint'], status=status, **fields))
            buffered[source['sample_id']] = value
            return value
        def commit_group(group, path):
            records = [buffered[s['sample_id']] for s in group]
            common.write_json(path, run.sealed(dict(cohort_fingerprint=cfg['fingerprint'],
                                                   source_ids=[s['sample_id'] for s in group], records=records)))
            for value in records:
                common.write_json(directory/'prompts'/f"{value['source']['sample_id']}.json", value)
        try:
            for group in chunks(rows(directory/'sources.jsonl'), cfg['batch_size']):
                group_path = directory/'groups'/f"{common.fingerprint([s['sample_id'] for s in group])}.json"
                buffered.clear()
                if group_path.exists():
                    committed = sealed_read(group_path)
                    if committed['cohort_fingerprint'] != cfg['fingerprint'] or committed['source_ids'] != [s['sample_id'] for s in group]:
                        raise ValueError('Committed microbatch identity changed')
                    for value in committed['records']:
                        if value != run.sealed(value): raise ValueError('Corrupt microbatch record')
                        target = directory/'prompts'/f"{value['source']['sample_id']}.json"
                        if target.exists() and sealed_read(target) != value:
                            raise ValueError('Per-place receipt differs from atomic batch commit')
                        if not target.exists(): common.write_json(target, value)
                elif any((directory/'prompts'/f"{s['sample_id']}.json").exists() for s in group):
                    raise ValueError('Prompt receipts exist without their atomic batch commit')
                pending = []
                for source in group:
                    path = directory/'prompts'/f"{source['sample_id']}.json"
                    if path.exists():
                        value = sealed_read(path)
                        if value['cohort_fingerprint'] != cfg['fingerprint'] or value['source_sha256'] != source['source_sha256']:
                            raise ValueError('Cached prompt differs from source/cohort')
                        completed += 1
                        if value['status'] == 'ready':
                            ready += 1; allocated[value['job']['style_id']] += 1
                        else:
                            failed += 1; failed_reasons[value['reason']] += 1
                    else:
                        run.validate_image(source, 'source_path', 'source_sha256'); pending.append(source)
                if not pending:
                    progress(); continue
                tick = time.monotonic()
                if shutil.disk_usage(directory).free < 5_000_000_000:
                    raise RuntimeError('Less than 5GB disk headroom; pause prompt preparation')
                if reader is None: reader = BatchedPlanner(Path(cfg['model_path']))
                try:
                    analyses = reader.ask(directory, pending, [analysis_request()]*len(pending), 'analysis', 0, cfg)
                except ValueError as error:
                    for source in pending:
                        receipt(source, 'failed', reason='unresolved_analysis_batch', error=str(error))
                    failed += len(pending); completed += len(pending)
                    failed_reasons['unresolved_analysis_batch'] += len(pending)
                    commit_group(group, group_path)
                    fresh += len(pending); work_time += time.monotonic()-tick; progress(); continue
                candidates = []; reserved = Counter()
                for source, raw in zip(pending, analyses, strict=True):
                    try:
                        evidence = policy.capabilities(policy.parse_json(raw))
                        eligible, excluded = policy.eligible_styles(evidence)
                        counts = {k:{**v, 'pending':v['pending']+allocated[k]+reserved[k]} for k,v in base_counts.items()}
                        style = policy.select_styles(eligible, 1, counts, seed=cfg['seed'], source_key=source['sample_id'])[0]
                        reserved[style] += 1
                        candidates.append(dict(source=source, evidence=evidence, style=style, excluded=excluded, feedback=''))
                    except (ValueError, KeyError, TypeError) as error:
                        receipt(source, 'failed', reason='invalid_source_analysis', error=str(error))
                        failed += 1; completed += 1; failed_reasons['invalid_source_analysis'] += 1
                for attempt in range(cfg['compose_drafts']):
                    if not candidates: break
                    prompts = [compose_request(c['evidence'], c['style'], c['feedback']) for c in candidates]
                    try:
                        replies = reader.ask(directory, [c['source'] for c in candidates], prompts, 'compose', attempt, cfg)
                    except ValueError as error:
                        for c in candidates:
                            receipt(c['source'], 'failed', reason='unresolved_composer_batch', error=str(error))
                        failed += len(candidates); completed += len(candidates)
                        failed_reasons['unresolved_composer_batch'] += len(candidates); candidates = []; break
                    retry = []
                    for c, raw in zip(candidates, replies, strict=True):
                        try:
                            plan = parse_plan(raw, c['evidence'], c['style'])
                            place_seed = common.candidate_seed(cfg['seed'], c['source']['sample_id'], 0)
                            job = balanced_night.make_jobs([c['source']], [plan], False, place_seed)[0]
                            receipt(c['source'], 'ready', plan=plan, job=job, excluded_styles=c['excluded'])
                            ready += 1; completed += 1; allocated[c['style']] += 1
                        except (ValueError, KeyError, TypeError) as error:
                            c['feedback'] = 'Previous response failed schema: '+str(error)
                            retry.append(c)
                    candidates = retry
                for c in candidates:
                    receipt(c['source'], 'failed', reason='composer_drafts_exhausted', error=c['feedback'])
                    failed += 1; completed += 1; failed_reasons['composer_drafts_exhausted'] += 1
                commit_group(group, group_path)
                fresh += len(pending); work_time += time.monotonic()-tick; progress()
                print(f'prompts: {completed}/{cfg["num_places"]}, ready={ready}, failed={failed}', flush=True)
                if completed >= 16 and ready == 0:
                    raise RuntimeError('No valid prompts in first 16 processed places; stop instead of wasting full-cohort GPU work')
        except BaseException as error:
            progress('paused', f'{type(error).__name__}: {error}'); raise
        finally:
            if reader is not None: reader.close()
        result = export(directory, cfg)
        progress('prompts_complete'); return result


def export(directory, cfg):
    module = balanced_night.runtime(); coverage = campaign.config_file(directory/'coverage_snapshot.json')
    jobs, sources, shard_paths = [], [], []; ready = failed = 0
    def flush():
        if not jobs: return
        shard = directory.parent/'shards'/f's{len(shard_paths):05d}'/'state'
        common.write_jsonl(shard/'plan.jsonl', jobs); common.write_jsonl(shard/'source_snapshot.jsonl', sources)
        snapshot = campaign.frozen(shard/'coverage_snapshot.json', {k:v for k,v in coverage.items() if k!='fingerprint'})
        model_files = {name:digest for name,digest in cfg['files_sha256'].items()
                       if Path(name).parent == Path(cfg['model_path'])}
        planner = campaign.frozen(shard/'planner_config.json', dict(request=dict(model_path=cfg['model_path']),
            policy=dict(coverage_snapshot=snapshot), files_sha256=model_files, cohort_fingerprint=cfg['fingerprint']))
        files = {name:digest for name,digest in cfg['files_sha256'].items() if name not in model_files}
        files[str(directory/'cohort_config.json')] = common.file_sha256(directory/'cohort_config.json')
        files[str(shard/'source_snapshot.jsonl')] = common.file_sha256(shard/'source_snapshot.jsonl')
        frozen = campaign.frozen(shard/'plan_config.json', dict(version=1, stage=policy.VERSION,
            planner_fingerprint=planner['fingerprint'], num_jobs=len(jobs), request_budget=len(jobs),
            domain_counts=dict(night=len(jobs)), image_root=cfg['image_root'], quality=module.quality_definition(clip_device='cpu'),
            service=cfg['service'], plan_sha256=common.file_sha256(shard/'plan.jsonl'), files_sha256=files))
        module.engine().rebuild_manifest(shard, frozen, [])
        shard_paths.append(dict(state=str(shard), config_sha256=common.file_sha256(shard/'plan_config.json'), jobs=len(jobs)))
        jobs.clear(); sources.clear()
    def all_jobs():
        nonlocal ready, failed
        for source in rows(directory/'sources.jsonl'):
            path = directory/'prompts'/f"{source['sample_id']}.json"
            if not path.exists(): raise ValueError('Generation barrier: source still lacks a terminal prompt result')
            value = sealed_read(path)
            if value['status'] != 'ready': failed += 1; continue
            ready += 1; jobs.append(value['job']); sources.append(source)
            yield value['job']
            if len(jobs) >= cfg['shard_size']: flush()
        flush()
    common.write_jsonl(directory/'night_prompts.jsonl', all_jobs())
    result = run.sealed(dict(cohort_fingerprint=cfg['fingerprint'], selected=cfg['num_places'], ready=ready,
        failed=failed, shards=shard_paths, manifest_sha256=common.file_sha256(directory/'night_prompts.jsonl'),
        generation_started=False))
    common.write_json(directory/'prompts_complete.json', result)
    return result


def generation_resource_plan(directory):
    cfg = load(directory)
    marker = Path(directory)/'prompts_complete.json'
    if not marker.exists(): raise ValueError('All prompts must be completed and exported before image generation')
    done = sealed_read(marker)
    if done['cohort_fingerprint'] != cfg['fingerprint'] or common.file_sha256(Path(directory)/'night_prompts.jsonl') != done['manifest_sha256']:
        raise ValueError('Prompt completion barrier changed')
    reserved = 0
    for shard in done['shards']:
        state = Path(shard['state'])
        if common.file_sha256(state/'plan_config.json') != shard['config_sha256']:
            raise ValueError('Exported shard changed')
        reserved += sum(1 for _ in rows(state/'attempts.jsonl')) if (state/'attempts.jsonl').exists() else 0
    return cfg, done, max(0, done['ready']-reserved)


def generate(args):
    cfg, done, remaining = generation_resource_plan(args.output_dir)
    if args.max_calls <= 0: raise ValueError('Explicit positive --max-calls required')
    campaign.ensure_stopped(); module = balanced_night.runtime(); used = 0
    if shutil.disk_usage(args.output_dir).free < 5_000_000_000+3_000_000*min(remaining,args.max_calls):
        raise RuntimeError('Insufficient disk for requested images and raw evidence')
    try:
        campaign.start_qwen({'reference_execution':{'service':cfg['service']}})
        for shard in done['shards']:
            if used >= args.max_calls: break
            state = Path(shard['state'])
            before = len(run.read_rows(state,'attempts.jsonl'))
            child = SimpleNamespace(output_dir=state,domain='night',max_calls=args.max_calls-used,
                qwen_url='http://127.0.0.1:8001/generate',request_timeout=600.,device='cuda')
            module.engine().generate(child)
            used += len(run.read_rows(state,'attempts.jsonl'))-before
    finally:
        campaign.stop_qwen()
    return dict(new_calls=used, generation_phase_only=True, next='verify/audit exported shards separately')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('index','prepare','run','check','report'))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'state')
    parser.add_argument('--image-root',type=Path)  # The cohort freezes its central image library at index time.
    parser.add_argument('--real-data',type=Path,default=common.WORKSPACE_ROOT/'dataset/gsv-cities')
    parser.add_argument('--exclude-places',type=Path,default=common.WORKSPACE_ROOT/'outputs/qwen_curriculum/state/finetuning/real_baseline_v1/excluded_places.json')
    parser.add_argument('--planner-model',type=Path,default=common.WORKSPACE_ROOT/'models/Qwen3-VL-4B-Instruct')
    parser.add_argument('--coverage-state',action='append',type=Path,default=[])
    parser.add_argument('--min-images',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--limit-places',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--shard-size',type=int,default=128)
    parser.add_argument('--max-calls',type=int,default=0)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    for name in ('output_dir','real_data','exclude_places','planner_model'):
        setattr(args,name,getattr(args,name).resolve())
    args.coverage_state = [p.resolve() for p in args.coverage_state]
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    if args.command == 'index': result=index(args)
    elif args.command in ('check','report'):
        result=json.loads((args.output_dir/'progress.json').read_text())
    elif not args.execute: result=dict(model_calls=0,next='Use the safe launcher with --pipeline places --execute')
    else:
        if os.environ.get('SCENE_RESOURCE_CHILD')!='1': raise ValueError('Use run_scene_night_safe.py --pipeline places')
        result=prompt_places(args) if args.command=='prepare' else generate(args)
    if args.command == 'prepare' and args.execute:
        result={k:v for k,v in result.items() if k!='shards'}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
