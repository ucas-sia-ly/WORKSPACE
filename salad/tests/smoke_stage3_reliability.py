"""Bounded Stage 3 GPU smoke + real-view regression and held-out spatial diagnostics.

Two updates per case, six cases. JSON only: checkpoint serialization intercepted.
No full training, Qwen calls, or changes to existing Reliability V2 artifacts.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_salad import parse_args, train
from workflow.model import SALADModel, load_checkpoint_model, FrozenSALADTeacher
from workflow.training_data import MixedGSVCitiesDataset
from workflow.reliability_cache import load_teacher_image
from workflow.local_distillation import LocalDistillationSupervisor
from workflow.reliability import reliability_region_statistics, summarize_reliability_regions
from workflow.reliability_corruption import make_generated_occlusions, make_local_corruptions
from workflow.evaluation import load_manifest, extract_descriptors, exact_retrieval, build_results
from cache_reliability_targets import build_cache, parse_args as cache_args


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.no_grad()
def recall(model, evaluation):
    model.eval()
    refs = extract_descriptors(model, evaluation.references, (224,224), 'cuda', 8, 0)
    queries = extract_descriptors(model, evaluation.queries, (224,224), 'cuda', 8, 0)
    return build_results(evaluation, exact_retrieval(refs, queries, evaluation.positives,
                         device='cuda'), False)['recall']


@torch.no_grad()
def spatial(model, prepared):
    if not model.aggregator.reliability_ot:
        return None
    model.eval()
    records = {}
    for kind in ('generated', 'generated_occlusion', 'real_copy_paste'):
        statistics = {}
        for i, pair in enumerate(prepared):
            inputs = pair['source_image' if kind == 'real_copy_paste' else 'image'].cuda()
            positive, negative = pair['positive'].cuda(), pair['negative'].cuda()
            if kind == 'generated_occlusion':
                inputs, mask = make_generated_occlusions(inputs, generator=torch.Generator(device='cuda').manual_seed(803+i))
                positive, negative = positive * ~mask, mask.float()
            elif kind == 'real_copy_paste':
                inputs, mask, _ = make_local_corruptions(inputs, torch.zeros(1, device='cuda', dtype=torch.bool),
                    generator=torch.Generator(device='cuda').manual_seed(1203+i))
                # The outside region is an unaltered real-view proxy control,
                # not a claim of semantic static-landmark ground truth.
                positive, negative = (~mask).float(), mask.float()
            _, aux = model(inputs, return_aux=True)
            values = reliability_region_statistics(aux['reliability_logits'], positive, negative, aux['assignment_dustbin'])
            for key, value in values.items():
                statistics[key] = statistics.get(key, 0.) + value
        records[kind] = summarize_reliability_regions(statistics)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-checkpoint', type=Path, required=True)
    parser.add_argument('--backbone-repo', type=Path, required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--training-manifest', type=Path, required=True)
    parser.add_argument('--real-data', type=Path, default=Path('dataset/gsv-cities'))
    parser.add_argument('--output-dir', type=Path, required=True)
    options = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(2)
    out = options.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    train_rows = list(map(json.loads, options.training_manifest.read_text().splitlines()))
    train_manifest = out/'training_manifest.jsonl'
    train_manifest.write_text(''.join(json.dumps(row)+'\n' for row in train_rows))
    train_ids = {(row['city'], int(row['place_id'])) for row in train_rows}
    cities = sorted({city for city, _ in train_ids})
    if len(train_ids) != 4:
        raise ValueError('This bounded smoke requires exactly four training places')
    data = MixedGSVCitiesDataset(options.real_data, cities=cities, images_per_place=4, min_images_per_place=4)
    evaluation = dict(references=[], queries=[], name='stage3_real_place_holdout_smoke')
    for city in cities:
        candidates = [p for p in data.places if p.city == city and (p.city, p.place_id) not in train_ids]
        candidates.sort(key=lambda p: hashlib.sha256(f'3041/{city}/{p.place_id}'.encode()).hexdigest())
        for place in candidates[:16]:
            key = f'{city}_{place.place_id}'
            evaluation['references'].append(dict(id=key, path=str(place.real_paths[0])))
            evaluation['queries'].append(dict(id='q_'+key, path=str(place.real_paths[-1]), positives=[key]))
    if len(evaluation['queries']) != 64:
        raise ValueError('Expected 64 held-out real places')
    dump(out/'real_holdout.json', evaluation)
    eval_set = load_manifest(out/'real_holdout.json', options.real_data)
    initial = load_checkpoint_model(options.teacher_checkpoint, device='cuda', backbone_repo=options.backbone_repo)
    baseline = recall(initial, eval_set)
    dump(out/'initial_real_recall.json', baseline)
    del initial; gc.collect(); torch.cuda.empty_cache()
    common = ['--real-data', str(options.real_data), '--synthetic-manifest', str(train_manifest),
        '--synthetic-places-only', '--cities', *cities, '--init-checkpoint', str(options.teacher_checkpoint),
        '--backbone-repo', str(options.backbone_repo), '--image-size','224','224','--num-trainable-blocks','1',
        '--epochs','1','--max-batches-per-epoch','2','--batch-size','2','--images-per-place','4',
        '--min-images-per-place','4','--real-sampling','coverage_pk','--seed','18','--num-workers','0',
        '--learning-rate','1e-6','--precision','16','--device','cuda','--profile-training']
    aux = ['--synthetic-aux','--synthetic-exposure-prob','1','--synthetic-global-weight','.1']
    local = ['--teacher-checkpoint',str(options.teacher_checkpoint),'--teacher-batch-size','1',
        '--pair-calibration',str(options.diagnostic_dir/'calibration.json'),
        '--pair-geometry-cache',str(options.diagnostic_dir/'geometry'),'--pair-confidence-mode','verified']
    distill = ['--feature-distill-weight','.1','--ot-distill-weight','.1']
    head = ['--reliability-ot','--reliability-context','--reliability-detach-features',
        '--reliability-head-lr','1e-4','--reliability-corruption-weight','.05','--reliability-corruption-max-images','2']
    generated = ['--reliability-generated-weight','.1']
    # Held-out spatial pairs are different places from all four training pairs.
    rows = json.loads((options.diagnostic_dir/'selected_pairs.json').read_text())['weather_pairs']
    held = [next(r for r in rows if r['condition']==weather and (r['city'],int(r['place_id'])) not in train_ids)
            for weather in ('night','rain','snow','fog')]
    manifest = out/'spatial_holdout.jsonl'; manifest.write_text(''.join(json.dumps(r)+'\n' for r in held))
    held_args = parse_args(common + aux + local + distill + head + generated + ['--output-dir',str(out/'unused')])
    held_args.synthetic_manifest = manifest
    held_args.cities = sorted({r['city'] for r in held})
    dataset = MixedGSVCitiesDataset(options.real_data, manifest, cities=held_args.cities, images_per_place=4,
                                    min_images_per_place=4, synthetic_fraction=0)
    model_config = json.loads((options.diagnostic_dir/'calibration.json').read_text())[0]['provenance']['teacher']['model_config']
    teacher = FrozenSALADTeacher(options.teacher_checkpoint, model_config, device='cuda', backbone_repo=options.backbone_repo)
    supervisor = LocalDistillationSupervisor(teacher, dataset, held_args)
    prepared, _, _ = supervisor.prepare(dict(source_paths=[r['source_path'] for r in held],
        synthetic_paths=[r['output_path'] for r in held],pair_valid=torch.ones(len(held),dtype=torch.bool)))
    for pair, row in zip(prepared,held):
        pair['source_image'] = load_teacher_image(Path(row['source_path']),(224,224))[None]
    del supervisor, teacher; gc.collect(); torch.cuda.empty_cache()
    # Fresh small V2 label cache in this experiment only. Never overwrite V2.
    cache = out/'legacy_v2_targets'
    with (out/'cache.log').open('w') as handle, contextlib.redirect_stdout(handle):
        build_cache(cache_args(['--real-data',str(options.real_data),'--synthetic-manifest',str(train_manifest),
            '--checkpoint',str(options.teacher_checkpoint),'--output-dir',str(cache),'--cities',*cities,
            '--backbone-repo',str(options.backbone_repo),'--image-size','224','224','--batch-size','2','--device','cuda']))
    cases = [
        ('C0',aux+local+distill),
        ('C1',aux+local+head+generated),
        ('C2',aux+local+head+generated+distill),
        ('C3',aux+local+head+generated+distill+['--reliability-generated-occlusion-weight','.05']),
        ('C3_no_detach',aux+local+head+generated+distill+['--reliability-generated-occlusion-weight','.05','--no-reliability-detach-features']),
        ('C1_legacy',head+['--synthetic-fraction','.5','--no-augment','--reliability-target-cache',str(cache)]),
    ]
    summaries=[]
    for name, flags in cases:
        gc.collect(); torch.cuda.empty_cache()
        destination=out/name; destination.mkdir()
        command=common+flags+['--output-dir',str(destination)]
        dump(destination/'argv.json',command)
        args=parse_args(command)
        holder,events,saved={},[],{}
        original_forward, original_step = SALADModel.forward, torch.optim.AdamW.step
        def forward(model, images, *a, **kw):
            if model.training:
                holder['model']=model
            else:
                # Student controlled-copy eval forwards may require grad;
                # frozen teachers can be recognized by requires_grad flags.
                if all(not p.requires_grad for p in model.parameters()):
                    assert all(p.grad is None for p in model.parameters())
            return original_forward(model, images, *a, **kw)
        def step(optimizer,*a,**kw):
            assert all(torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group['params'] if p.grad is not None)
            events.append('step');return original_step(optimizer,*a,**kw)
        def save(payload,path):
            if path.name=='checkpoint.pt':
                saved.update(global_step=payload['global_step'],metrics=payload['metrics'],training_config=payload['training_config'])
        with (destination/'train.log').open('w') as handle,contextlib.redirect_stdout(handle), \
             patch.object(SALADModel,'forward',forward),patch.object(torch.optim.AdamW,'step',step), \
             patch('workflow.model.atomic_save_checkpoint',save):
            train(args)
        assert len(events)==2 and saved['global_step']==2
        model=holder.pop('model')
        report=recall(model,eval_set)
        regions=spatial(model,prepared)
        result=dict(name=name,precision='16',optimizer_steps=len(events),**saved,
                    real_recall=report,initial_real_recall=baseline,heldout_spatial=regions)
        result['recall_delta_pp']={key:100*(value-baseline[key]) for key,value in report.items()}
        dump(destination/'result.json',result);summaries.append(result)
        dump(out/'summary.json',summaries)
        print(name,json.dumps(dict(real_recall=report,delta_pp=result['recall_delta_pp'],
              generated=regions['generated'] if regions else None)),flush=True)
        del model, saved, result
    print('Complete:',out/'summary.json',flush=True)


if __name__=='__main__':
    main()
