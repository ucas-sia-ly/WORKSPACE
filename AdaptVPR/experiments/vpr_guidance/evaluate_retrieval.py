"""Source/baseline/guided retrieval against other GSV-Cities captures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import torch
from PIL import Image
from .gsv_pairs import image_index, parse_filename
from .vpr import load_salad, load_boq


def retrieval_results(query_descriptors, database_descriptors, source_ids, database_paths,
                      query_paths=None):
    """Per-query first-positive ranks, using source identity for every query variant."""
    if len(query_descriptors) != len(source_ids) or len(database_descriptors) != len(database_paths):
        raise ValueError('Descriptor counts must match query/database metadata')
    if query_paths is not None and len(query_paths) != len(source_ids):
        raise ValueError('Query path count must match query metadata')
    captures = [parse_filename(path) for path in database_paths]
    resolved_database = [Path(path).resolve() for path in database_paths]
    results = []
    for index, (descriptor, source_id) in enumerate(zip(query_descriptors, source_ids)):
        source = parse_filename(source_id)
        query_path = Path(query_paths[index]).resolve() if query_paths is not None else None
        eligible = [i for i, c in enumerate(captures)
                    if c.capture_id != source.capture_id and resolved_database[i] != query_path]
        positives = [j for j, i in enumerate(eligible) if captures[i].place_key == source.place_key]
        # Both identity exclusion and exact resolved-path exclusion are explicit.
        assert all(captures[eligible[j]].capture_id != source.capture_id for j in positives)
        assert all(resolved_database[eligible[j]] != query_path for j in positives), \
            'Exact query path must never be a valid database positive'
        result = {'source_id': source_id,
                  'positive_database_ids': [captures[eligible[j]].capture_id for j in positives],
                  'positive_rank': None, 'positive_cosine': None, 'skip_reason': None}
        if not positives:
            result['skip_reason'] = 'No alternate capture of the same place in database'
        else:
            similarities = database_descriptors[eligible].float() @ descriptor.float()
            order = torch.argsort(similarities, descending=True, stable=True).tolist()
            positive_set = set(positives)
            result['positive_rank'] = next(rank for rank, i in enumerate(order, 1) if i in positive_set)
            result['positive_cosine'] = float(similarities[positives].mean())
        results.append(result)
    return results


def summarize_retrieval(results):
    ranks = [r['positive_rank'] for r in results if r['positive_rank'] is not None]
    cosines = [r['positive_cosine'] for r in results if r['positive_rank'] is not None]
    skipped = [r['source_id'] for r in results if r['positive_rank'] is None]
    return {'num_queries': len(ranks), 'num_skipped_no_other_capture': len(skipped),
            'skipped_source_ids': skipped,
            'recall_at_1': sum(r <= 1 for r in ranks) / len(ranks) if ranks else None,
            'recall_at_5': sum(r <= 5 for r in ranks) / len(ranks) if ranks else None,
            'median_rank': statistics.median(ranks) if ranks else None,
            'mean_positive_cosine_similarity': statistics.mean(cosines) if ranks else None}


def retrieval_metrics(query_descriptors, database_descriptors, source_ids, database_paths):
    # Preserve the existing helper API used by the original experiment tests.
    return summarize_retrieval(retrieval_results(query_descriptors, database_descriptors,
                                                source_ids, database_paths))


def recovery_metrics(source, baseline, guided):
    source_r1, baseline_r1, guided_r1 = [m['recall_at_1'] for m in (source, baseline, guided)]
    if any(value is None for value in (source_r1, baseline_r1, guided_r1)):
        loss = gain = ratio = None
        reason = 'No eligible queries; recovery ratio is not meaningful'
    else:
        loss, gain = source_r1 - baseline_r1, guided_r1 - baseline_r1
        ratio = gain / loss if loss > 0 else None
        reason = (None if loss > 0 else
                  'Generation did not reduce R@1 relative to source; recovery ratio is not meaningful')
    return {'r1_generation_loss': loss, 'r1_guidance_gain': gain,
            'recovery_ratio': ratio, 'recovery_ratio_percent': ratio * 100 if ratio is not None else None,
            'recovery_ratio_reason': reason}


def paired_statistics(baseline_results, guided_results):
    if len(baseline_results) != len(guided_results):
        raise ValueError('Baseline and guided query sets must be paired')
    pairs = []
    for baseline, guided in zip(baseline_results, guided_results):
        if baseline['source_id'] != guided['source_id']:
            raise ValueError('Baseline and guided source identities must match')
        if baseline['positive_database_ids'] != guided['positive_database_ids']:
            raise ValueError('Paired queries must have the same positive database captures')
        if (baseline['positive_rank'] is None) != (guided['positive_rank'] is None):
            raise ValueError('Paired query eligibility must match')
        if baseline['positive_rank'] is not None:
            pairs.append((baseline, guided))
    better = sum(g['positive_rank'] < b['positive_rank'] for b, g in pairs)
    equal = sum(g['positive_rank'] == b['positive_rank'] for b, g in pairs)
    worse = sum(g['positive_rank'] > b['positive_rank'] for b, g in pairs)
    return {'num_pairs': len(pairs), 'guided_rank_better': better,
            'guided_rank_equal': equal, 'guided_rank_worse': worse,
            'guided_win_rate': better / len(pairs) if pairs else None,
            'guided_non_worse_rate': (better + equal) / len(pairs) if pairs else None,
            'guided_cosine_better_fraction': sum(g['positive_cosine'] > b['positive_cosine']
                                                for b, g in pairs) / len(pairs) if pairs else None}


def aggregate_model(queries, model_name):
    results = {variant: [q['models'][model_name][variant] for q in queries]
               for variant in ('source', 'baseline', 'guided')}
    metrics = {variant: summarize_retrieval(values) for variant, values in results.items()}
    return {**metrics, **recovery_metrics(metrics['source'], metrics['baseline'], metrics['guided']),
            'paired_rank': paired_statistics(results['baseline'], results['guided'])}


def query_metadata(row):
    return {'sample_id': row['sample_id'], 'source_id': row['source_id'],
            'condition': row.get('condition'), 'seed': row['seed'],
            'guidance_scale': row['guidance_scale'], 'guidance_every': row['guidance_every'],
            'guidance_last_n': row['guidance_last_n'],
            'source_path': row['output_paths']['source'], 'baseline_path': row['output_paths']['baseline'],
            'guided_path': row['output_paths']['guided'], 'positive_database_ids': [], 'models': {}}


def per_query_record(query):
    record = {key: value for key, value in query.items() if key != 'models'}
    for model_name, variants in query['models'].items():
        for variant, result in variants.items():
            record[f'{model_name}_{variant}_rank'] = result['positive_rank']
            record[f'{model_name}_{variant}_positive_cosine'] = result['positive_cosine']
    return record


def print_model_table(model_name, report, label='overall'):
    def number(value, percent=False):
        if value is None:
            return 'n/a'
        return f'{value * 100:.2f}%' if percent else f'{value:.4f}'
    print(f'\nModel: {model_name.upper()} | {label} | queries={report["source"]["num_queries"]}')
    print('Query type | R@1 | R@5 | Median Rank | Mean Positive Cosine')
    for variant in ('source', 'baseline', 'guided'):
        m = report[variant]
        print(f'{variant.title():10} | {number(m["recall_at_1"], True)} | '
              f'{number(m["recall_at_5"], True)} | {number(m["median_rank"])} | '
              f'{number(m["mean_positive_cosine_similarity"])}')
    print(f'R@1 generation loss: {number(report["r1_generation_loss"], True)}; '
          f'R@1 guidance gain: {number(report["r1_guidance_gain"], True)}')
    print(f'Recovery ratio: {number(report["recovery_ratio"])} '
          f'({number(report["recovery_ratio"], True)})')
    if report['recovery_ratio_reason']:
        print(report['recovery_ratio_reason'])
    paired = report['paired_rank']
    print(f'Paired rank better / equal / worse: {paired["guided_rank_better"]} / '
          f'{paired["guided_rank_equal"]} / {paired["guided_rank_worse"]}')
    print(f'Guided rank win rate: {number(paired["guided_win_rate"], True)}; '
          f'non-worse rate: {number(paired["guided_non_worse_rate"], True)}; '
          f'positive-cosine win rate: {number(paired["guided_cosine_better_fraction"], True)}')


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
    parser.add_argument('--per-query-output', type=Path,
                        help='Per-query JSONL; default: <output-stem>.per_query.jsonl')
    parser.add_argument('--with-boq', action='store_true', help='Held-out DINOv2-BoQ; never used in guidance')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--salad-repo', default='serizba/salad')
    parser.add_argument('--boq-repo', default='amaralibey/Bag-of-Queries')
    args = parser.parse_args()
    per_query_path = args.per_query_output or args.output.with_name(args.output.stem + '.per_query.jsonl')
    if per_query_path.resolve() in {args.output.resolve(), args.records.resolve()}:
        raise ValueError('Per-query output must differ from summary and input records')
    if args.output.resolve() == args.records.resolve():
        raise ValueError('Summary output must differ from input records')
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
    model_names = ['salad'] + (['boq'] if args.with_boq else [])
    query_groups = {key: [query_metadata(row) for row in paired_rows]
                    for key, paired_rows in groups.items()}
    loaders = [('salad', load_salad, args.salad_repo)]
    if args.with_boq:
        loaders.append(('boq', load_boq, args.boq_repo))
    for name, loader, repo in loaders:
        model = loader(args.device, repo)
        database_desc = descriptors(model, database)
        cache = {}
        for key, paired_rows in groups.items():
            for variant in ('source', 'baseline', 'guided'):
                paths = [row['output_paths'][variant] for row in paired_rows]
                for path in paths:
                    if path not in cache:
                        cache[path] = descriptors(model, [path])[0]
                queries = torch.stack([cache[path] for path in paths])
                details = retrieval_results(queries, database_desc,
                                             [r['source_id'] for r in paired_rows], database, paths)
                for query, detail in zip(query_groups[key], details):
                    # The same positives must underlie all three variants and both models.
                    if query['models']:
                        assert query['positive_database_ids'] == detail['positive_database_ids']
                    query['positive_database_ids'] = detail['positive_database_ids']
                    query['models'].setdefault(name, {})[variant] = detail
        del model, database_desc, cache
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    for row in skipped:
        key = (row['guidance_scale'], row['guidance_every'], row['guidance_last_n'], row['seed'])
        query = query_metadata(row)
        query['skip_reason'] = 'No alternate capture in database'
        query['models'] = {name: {variant: {'source_id': row['source_id'],
                                           'positive_database_ids': [], 'positive_rank': None,
                                           'positive_cosine': None, 'skip_reason': query['skip_reason']}
                                 for variant in ('source', 'baseline', 'guided')}
                           for name in model_names}
        query_groups.setdefault(key, []).append(query)
        groups.setdefault(key, [])  # Geometry/success checks still use only eligible pairs.
    result = {'database_root': str(args.database_root.resolve()), 'database_size': len(database),
              'excluded_capture_policy': 'Exact source capture and resolved query path excluded; city+place ID positives',
              'per_query_output': str(per_query_path.resolve()),
              'skipped_queries': [{'sample_id': r['sample_id'], 'source_id': r['source_id'],
                                   'guidance_scale': r['guidance_scale']} for r in skipped],
              'models': {'salad': args.salad_repo, 'boq': args.boq_repo if args.with_boq else None},
              'groups': []}
    for key, queries in query_groups.items():
        metrics = {}
        for name in model_names:
            metrics[name] = aggregate_model(queries, name)
            conditions = sorted({q['condition'] for q in queries
                                 if isinstance(q['condition'], str) and q['condition']})
            metrics[name]['per_condition'] = {
                condition: aggregate_model([q for q in queries if q['condition'] == condition], name)
                for condition in conditions}
        geometry = paired_geometry(groups[key])
        result['groups'].append({'guidance_scale': key[0], 'guidance_every': key[1],
                                 'guidance_last_n': key[2], 'seed': key[3],
                                 **metrics, 'geometry': geometry,
                                 'success_test': success_check(metrics, geometry)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    per_query_path.parent.mkdir(parents=True, exist_ok=True)
    with per_query_path.open('w') as handle:
        for queries in query_groups.values():
            for query in queries:
                handle.write(json.dumps(per_query_record(query), allow_nan=False) + '\n')
    for group in result['groups']:
        print(f'\nScale={group["guidance_scale"]:g}; seed={group["seed"]}; '
              f'every={group["guidance_every"]}; last-n={group["guidance_last_n"]}')
        for name in model_names:
            print_model_table(name, group[name])
            for condition, report in group[name]['per_condition'].items():
                print_model_table(name, report, condition)
        print('Success test: ' + group['success_test']['status'])
    print(f'\nSkipped {len(skipped)} records without alternate captures. '
          f'Summary: {args.output}; per-query: {per_query_path}')


if __name__ == '__main__':
    main()
