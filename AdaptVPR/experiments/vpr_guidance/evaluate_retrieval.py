"""Paired retrieval: other GSV-Cities captures are positives; source is excluded."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import torch
from PIL import Image
from .gsv_pairs import image_index, parse_filename
from .vpr import load_salad, load_boq


def retrieval_metrics(query_descriptors, database_descriptors, source_ids, database_paths):
    captures = [parse_filename(path) for path in database_paths]
    ranks, positive_cosines, skipped = [], [], []
    for descriptor, source_id in zip(query_descriptors, source_ids):
        source = parse_filename(source_id)
        eligible = [i for i, c in enumerate(captures) if c.capture_id != source.capture_id]
        positives = [j for j, i in enumerate(eligible) if captures[i].place_key == source.place_key]
        if not positives:
            skipped.append(source_id)
            continue
        similarities = database_descriptors[eligible].float() @ descriptor.float()
        # Stable tie handling in the sorted database order.
        order = torch.argsort(similarities, descending=True, stable=True).tolist()
        positive_set = set(positives)
        ranks.append(next(rank for rank, i in enumerate(order, 1) if i in positive_set))
        positive_cosines.append(float(similarities[positives].mean()))
    return {'num_queries': len(ranks), 'num_skipped_no_other_capture': len(skipped),
            'skipped_source_ids': skipped,
            'recall_at_1': sum(r <= 1 for r in ranks) / len(ranks) if ranks else None,
            'recall_at_5': sum(r <= 5 for r in ranks) / len(ranks) if ranks else None,
            'median_rank': statistics.median(ranks) if ranks else None,
            'mean_positive_cosine_similarity': statistics.mean(positive_cosines) if ranks else None}


def descriptors(model, paths):
    # One image at a time keeps this experiment small and bounded in GPU memory.
    result = []
    with torch.no_grad():
        for path in paths:
            with Image.open(path) as image:
                result.append(model.from_pil(image.convert('RGB')).cpu()[0])
    return torch.stack(result)


def paired_geometry(rows):
    pairs = [r['geometry_verification_score'] for r in rows
             if all(r.get('geometry_verification_score', {}).get(k) is not None
                    for k in ('baseline', 'guided'))]
    return {'num_pairs': len(pairs),
            'baseline_mean': statistics.mean(p['baseline'] for p in pairs) if pairs else None,
            'guided_mean': statistics.mean(p['guided'] for p in pairs) if pairs else None}


def success_check(report, geometry):
    # Full success needs all three pieces of evidence; never infer missing checks.
    salad = report['salad']
    if not salad['baseline']['num_queries']:
        return {'status': 'insufficient_evidence', 'reason': 'No eligible alternate-capture positives'}
    salad_improves = salad['guided']['recall_at_1'] > salad['baseline']['recall_at_1']
    if not salad_improves:
        return {'status': 'fail', 'reason': 'SALAD Recall@1 did not strictly improve'}
    if 'boq' not in report or geometry['num_pairs'] != salad['baseline']['num_queries']:
        return {'status': 'insufficient_evidence', 'reason': 'SALAD improved; complete held-out BoQ and geometry checks required'}
    boq = report['boq']
    heldout_ok = all(boq['guided'][key] >= boq['baseline'][key]
                     for key in ('recall_at_1', 'recall_at_5', 'mean_positive_cosine_similarity'))
    heldout_ok &= boq['guided']['median_rank'] <= boq['baseline']['median_rank']
    geometry_ok = geometry['guided_mean'] >= geometry['baseline_mean']
    return {'status': 'pass' if heldout_ok and geometry_ok else 'fail',
            'salad_recall_at_1_improves': True, 'boq_does_not_degrade': heldout_ok,
            'geometry_does_not_degrade': geometry_ok}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--database-root', type=Path, required=True,
                        help='GSV-Cities image directory, including positives AND distractors')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--with-boq', action='store_true', help='Held-out DINOv2-BoQ; never used in guidance')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--salad-repo', default='serizba/salad')
    parser.add_argument('--boq-repo', default='amaralibey/Bag-of-Queries')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    if not rows or any(r.get('route') != 'global' for r in rows):
        raise ValueError('Expected nonempty Global-route experiment records')
    identities = [(r['sample_id'], r['seed'], r['guidance_scale']) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate paired queries in records')
    database = list(image_index(args.database_root).values())
    # Fix the evaluation subset BEFORE computing baseline or guided metrics.
    place_counts = {}
    capture_ids = set()
    for path in database:
        capture = parse_filename(path)
        place_counts[capture.place_key] = place_counts.get(capture.place_key, 0) + 1
        capture_ids.add(capture.capture_id)
    eligible, skipped = [], []
    for row in rows:
        source = parse_filename(row['source_id'])
        count = place_counts.get(source.place_key, 0) - int(source.capture_id in capture_ids)
        (eligible if count > 0 else skipped).append(row)
    if not eligible:
        raise ValueError('No query has an alternate capture in the database; exact source is never a positive')
    groups = {}
    for row in eligible:
        key = (row['guidance_scale'], row['guidance_every'], row['guidance_last_n'], row['seed'])
        groups.setdefault(key, []).append(row)
    evaluations = {key: {} for key in groups}
    loaders = [('salad', load_salad, args.salad_repo)]
    if args.with_boq:
        loaders.append(('boq', load_boq, args.boq_repo))
    for name, loader, repo in loaders:
        model = loader(args.device, repo)
        database_desc = descriptors(model, database)
        cache = {}
        for key, paired_rows in groups.items():
            metrics = {}
            for variant in ('baseline', 'guided'):
                paths = [row['output_paths'][variant] for row in paired_rows]
                for path in paths:
                    if path not in cache:
                        cache[path] = descriptors(model, [path])[0]
                queries = torch.stack([cache[path] for path in paths])
                metrics[variant] = retrieval_metrics(queries, database_desc,
                                                     [r['source_id'] for r in paired_rows], database)
            evaluations[key][name] = metrics
        del model, database_desc, cache
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    result = {'database_root': str(args.database_root.resolve()), 'database_size': len(database),
              'excluded_capture_policy': 'Exact source capture excluded per query; city+place ID positives',
              'skipped_queries': [{'sample_id': r['sample_id'], 'source_id': r['source_id'],
                                   'guidance_scale': r['guidance_scale']} for r in skipped],
              'models': {'salad': args.salad_repo, 'boq': args.boq_repo if args.with_boq else None},
              'groups': []}
    for key, metrics in evaluations.items():
        geometry = paired_geometry(groups[key])
        result['groups'].append({'guidance_scale': key[0], 'guidance_every': key[1],
                                 'guidance_last_n': key[2], 'seed': key[3],
                                 **metrics, 'geometry': geometry,
                                 'success_test': success_check(metrics, geometry)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
