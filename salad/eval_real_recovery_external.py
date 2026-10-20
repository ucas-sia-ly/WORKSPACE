#!/usr/bin/env python3
"""Read-only, post-hoc evaluation of recovery controls on local external data.

Tokyo and SF-XL use a fixed, explicitly reduced gallery. Their results are
exploratory and must never be presented as official full-gallery benchmarks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from workflow.am_statistics import exact_mcnemar, holm

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
STATE = ROOT / 'outputs/qwen_curriculum/state/finetuning/real_recovery_v1'
OUTPUT = STATE / 'posthoc_external'
CASES = ('initial', 'weak', 'randaugment', 'weak_retention', 'randaugment_retention')
DATASETS = ('nordland_probe', 'tokyo247_reduced', 'sfxl_reduced')
RADIUS = 25.0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f'Frozen protocol differs: {path}. Use a new output directory.')
    else:
        path.write_text(content)


def checkpoint(case):
    if case == 'initial':
        return ROOT / 'salad/checkpoint/dino_salad.ckpt'
    path = STATE / 'runs/42' / case / 'checkpoint_epoch_005.pt'
    audit = STATE / 'runs/42' / case / 'validation/500/audit.json'
    if not path.is_file() or not audit.is_file():
        raise FileNotFoundError(f'Missing complete 500-step recovery run: {case}')
    record = json.loads(audit.read_text())
    if record.get('optimizer_updates') != 500 or record.get('checkpoint_sha256') != sha(path):
        raise ValueError(f'Incomplete or modified recovery run: {case}')
    return path


def query_grid(queries):
    grid = {}
    for index, (_, _, x, y, _) in enumerate(queries):
        if not (math.isfinite(x) and math.isfinite(y)):
            continue  # Nine Tokyo query CSVs have NaN UTM; retain as unevaluable.
        grid.setdefault((math.floor(x / RADIUS), math.floor(y / RADIUS)), []).append(index)
    return grid


def nearby(grid, queries, x, y):
    cell = math.floor(x / RADIUS), math.floor(y / RADIUS)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for index in grid.get((cell[0] + dx, cell[1] + dy), ()):
                if (queries[index][2] - x) ** 2 + (queries[index][3] - y) ** 2 <= RADIUS ** 2:
                    yield index


def keep_negative(path, divisor, cutoff):
    # Stable across machines and Python hash randomization; no score-dependent sampling.
    number = int.from_bytes(hashlib.blake2b(path.encode(), digest_size=8).digest(), 'big')
    return number % divisor < cutoff


def make_manifest(dataset, output):
    if dataset == 'nordland_probe':
        from workflow.evaluation import load_evaluation_set
        root = ROOT / 'dataset/nordland'
        original = load_evaluation_set('Nordland', root, split='test', frame_window=10)
        count = 4096
        chosen = [round(i * (len(original.queries) - 1) / (count - 1)) for i in range(count)]
        references = [{'id': image.id, 'path': image.path.relative_to(root).as_posix()}
                      for image in original.references]
        queries = [{'id': original.queries[i].id,
                    'path': original.queries[i].path.relative_to(root).as_posix(),
                    'positives': [original.references[j].id for j in original.positives[i]]}
                   for i in chosen]
        manifest_path = output / dataset / 'manifest.json'
        write_json(manifest_path, dict(dataset=dataset, references=references, queries=queries))
        meta = dict(dataset=dataset, role='posthoc_exploratory_test',
                    gallery='full_local_gallery', positive_rule='Nordland frame +/- 10',
                    query_selection='4096 evenly spaced query indices across the complete 27592-frame test sequence',
                    full_reference_count=len(references), selected_reference_count=len(references),
                    query_count=len(queries), valid_query_count=len(queries),
                    root=str(root), manifest_sha256=sha(manifest_path),
                    initial_reference='Full 27592-query initial evaluation; ranks restricted to these same 4096 IDs')
        write_json(output / dataset / 'protocol.json', meta)
        return manifest_path, meta
    if dataset == 'tokyo247_reduced':
        root = ROOT / 'dataset/tokyo247'
        queries = []
        for csv_path in sorted((root / 'queries').glob('*.csv')):
            with csv_path.open(newline='') as stream:
                row = next(csv.reader(stream))
            if row[0] != csv_path.with_suffix('.jpg').name or row[5] not in ('D', 'S', 'N'):
                raise ValueError(f'Unexpected Tokyo query metadata: {csv_path}')
            queries.append((f'{row[5]}/{row[0]}', f'queries/{row[0]}', float(row[7]), float(row[8]), row[5]))
        def references():
            for path in sorted((root / 'database_gsv_vga').glob('*/*/*.png')):
                x, y = path.parent.name.split('_')
                yield path.relative_to(root).as_posix(), float(x), float(y)
        threshold = (1000, 250)  # ~19k random negatives plus every 25 m positive.
        source = 'Tokyo247 local query CSV and database image coordinate directories'
    elif dataset == 'sfxl_reduced':
        root = ROOT / 'dataset/SF-XL/processed/test'
        queries = []
        for group in ('queries_v1', 'queries_v2', 'queries_night', 'queries_occlusion'):
            candidates = sorted((root / group).glob('*.jpg'),
                                key=lambda p: hashlib.sha256(p.name.encode()).digest())[:32]
            for path in sorted(candidates):
                fields = path.name.split('@')
                if fields[3:5] != ['10', 'S']:
                    raise ValueError(f'Unexpected SF-XL UTM zone: {path}')
                rel = path.relative_to(root).as_posix()
                queries.append((rel, rel, float(fields[1]), float(fields[2]), group))
        database_list = root / 'database_images_paths.txt'
        def references():
            with database_list.open() as stream:
                for raw in stream:
                    rel = raw.strip()
                    fields = Path(rel).name.split('@')
                    if fields[3:5] != ['10', 'S']:
                        raise ValueError(f'Unexpected SF-XL database UTM zone: {rel}')
                    yield 'database/' + rel, float(fields[1]), float(fields[2])
        threshold = (1000, 4)  # ~11k random negatives plus every 25 m positive.
        source = 'SF-XL test database_images_paths.txt; filenames contain UTM 10S coordinates'
    else:
        raise ValueError(dataset)
    if not queries:
        raise ValueError('No query images')
    grid = query_grid(queries)
    matches = [[] for _ in queries]
    gallery = []
    full_count = 0
    for rel, x, y in references():
        full_count += 1
        found = list(nearby(grid, queries, x, y))
        if not found and not keep_negative(rel, *threshold):
            continue
        gallery.append({'id': rel, 'path': rel})
        for index in found:
            matches[index].append(rel)
    if not gallery or not any(matches):
        raise ValueError('Reduced gallery or all positives empty')
    manifest = dict(dataset=dataset, references=gallery,
                    queries=[{'id': qid, 'path': path, 'positives': matches[i]}
                             for i, (qid, path, _, _, _) in enumerate(queries)])
    meta = dict(dataset=dataset, role='posthoc_exploratory_test', gallery='fixed_reduced_gallery',
                warning='Not an official full-gallery benchmark result', source=source,
                positive_rule=f'UTM Euclidean distance <= {RADIUS:g} meters',
                negative_rule=f'blake2b(path) mod {threshold[0]} < {threshold[1]}',
                full_reference_count=full_count, selected_reference_count=len(gallery),
                query_count=len(queries), valid_query_count=sum(bool(v) for v in matches),
                query_selection='32 SHA256-minimum filename hashes per query folder',
                invalid_query_coordinates=sum(not (math.isfinite(q[2]) and math.isfinite(q[3])) for q in queries),
                query_groups={group: sum(q[4] == group for q in queries) for group in sorted({q[4] for q in queries})},
                root=str(root), manifest_sha256=None)
    manifest_path = output / dataset / 'manifest.json'
    write_json(manifest_path, manifest)
    meta['manifest_sha256'] = sha(manifest_path)
    write_json(output / dataset / 'protocol.json', meta)
    return manifest_path, meta


def run(dataset, case, output):
    from workflow.resource_limits import monitored_run
    cp = checkpoint(case)
    directory = output / dataset / case
    result = directory / 'result.json'
    receipt_path = directory / 'receipt.json'
    manifest = output / dataset / 'manifest.json'
    root = ROOT / ('dataset/nordland' if dataset in ('nordland', 'nordland_probe') else
                   'dataset/tokyo247' if dataset == 'tokyo247_reduced' else 'dataset/SF-XL/processed/test')
    protocol = dict(dataset=dataset, checkpoint_sha256=sha(cp),
                    manifest_sha256=sha(manifest) if dataset != 'nordland' else None,
                    positive_rule='Nordland frame +/- 10' if dataset.startswith('nordland') else 'UTM radius <= 25 m',
                    image_size=[224, 224], batch_size=64)
    if result.exists() or receipt_path.exists():
        if result.exists() and not receipt_path.exists():
            # Permit adoption of an evaluator invocation started before this runner.
            existing = json.loads(result.read_text())
            if (Path(existing.get('checkpoint', '')).resolve() != cp.resolve()
                    or existing.get('image_size') != [224, 224]
                    or len(existing.get('query_results', [])) != existing.get('num_queries')):
                raise ValueError(f'Unverified evaluator output: {result}')
            write_json(receipt_path, dict(protocol=protocol, result_sha256=sha(result)))
        if not result.exists() or not receipt_path.exists():
            raise ValueError(f'Partial result needs manual inspection: {directory}')
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('protocol') != protocol or receipt.get('result_sha256') != sha(result):
            raise ValueError(f'Stale result: {directory}')
        return
    command = [str(PYTHON), str(ROOT / 'salad/evaluate_salad.py'), '--checkpoint', str(cp),
               '--dataset', 'Nordland' if dataset == 'nordland' else 'manifest',
               '--dataset-root', str(root), '--split', 'test', '--output', str(result),
               '--backbone-repo', '/home/admin123/.cache/torch/hub/facebookresearch_dinov2_main',
               '--image-size', '224', '224', '--batch-size', '64', '--num-workers',
               '0' if dataset.startswith('nordland') else '2',
               '--prefetch-factor', '1', '--device', 'cuda', '--save-query-results',
               '--reference-cache-mib', '2048', '--query-chunk-size', '64',
               '--reference-chunk-size', '2048']
    if dataset != 'nordland':
        command += ['--eval-manifest', str(manifest)]
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CCR_CUDA_BUDGET_GIB='38', CCR_CUDA_RESERVE_GIB='4', OMP_NUM_THREADS='2')
    limits = dict(cuda_budget_gib=38, cuda_reserve_gib=4,
                  host_rss_limit_gib=16, host_available_reserve_gib=1.2)
    with (directory / 'evaluation.log').open('w') as log:
        log.write(' '.join(command) + '\n'); log.flush()
        monitored_run(command, handle=log, env=env, limits=limits)
    write_json(receipt_path, dict(protocol=protocol, result_sha256=sha(result)))


def summarize(dataset, output):
    path = output / dataset
    present = {}
    allowed = None
    if dataset == 'nordland_probe':
        allowed = {q['id'] for q in json.loads((path / 'manifest.json').read_text())['queries']}
        full_initial = output / 'nordland/initial/result.json'
        if full_initial.is_file():
            rows = json.loads(full_initial.read_text())['query_results']
            present['initial'] = {v['query_id']:v['rank'] for v in rows if v['query_id'] in allowed}
    for case in CASES:
        result = path / case / 'result.json'
        if result.is_file():
            rows = json.loads(result.read_text())['query_results']
            present[case] = {v['query_id']: v['rank'] for v in rows
                             if allowed is None or v['query_id'] in allowed}
    if not present:
        return
    groups = ['all'] if dataset.startswith('nordland') else (
        ['D', 'S', 'N'] if dataset == 'tokyo247_reduced' else
        ['queries_v1', 'queries_v2', 'queries_night', 'queries_occlusion'])
    table, comparisons = [], []
    for group in groups:
        for case, ranks in present.items():
            subset = {qid: rank for qid, rank in ranks.items()
                      if group == 'all' or qid.startswith(group + '/')}
            valid = [rank for rank in subset.values() if rank is not None]
            if not valid:
                continue
            row = dict(group=group, case=case, queries=len(valid),
                       invalid_queries=len(subset)-len(valid),
                       recall={f'R@{k}':sum(rank <= k for rank in valid)/len(valid) for k in (1,5,10)})
            table.append(row)
            if case != 'initial' and 'initial' in present:
                baseline = present['initial']
                if baseline.keys() != ranks.keys():
                    raise ValueError('Query identities changed between checkpoints')
                for k in (1,5,10):
                    paired = [(int(rank <= k)-int(baseline[qid] <= k))
                              for qid, rank in subset.items() if rank is not None and baseline[qid] is not None]
                    improved = sum(v == 1 for v in paired)
                    regressed = sum(v == -1 for v in paired)
                    comparisons.append(dict(group=group, case=case, k=k, improved=improved,
                                            regressed=regressed, delta_pp=100*(improved-regressed)/len(paired),
                                            p_exact=exact_mcnemar(improved, regressed)))
    family = len(groups) * 4 * 3
    for item, adjusted in zip(comparisons, holm([x['p_exact'] for x in comparisons], family)):
        item['p_holm'] = adjusted
    report = dict(dataset=dataset, cases=list(present), rows=table,
                  planned_holm_family_size=family, paired=comparisons,
                  note='Post-hoc test audit; reduced galleries are not official benchmarks.')
    temporary = path / 'report.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    temporary.replace(path / 'report.json')
    index = {(row['group'], row['case']):row for row in table}
    lines = [f'# {dataset}: post-hoc evaluation', '',
             'Single seed 42, 500 optimizer updates. Read-only tests; do not tune on these scores.',
             'Nordland probe uses all 27,592 reference images and 4,096 evenly spaced queries. Tokyo247 and SF-XL use fixed reduced galleries; their scores are exploratory, not official benchmark scores.',
             '', '| Group | Valid queries | Initial | Weak | RandAugment | Weak + retention | RandAugment + retention |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        row = index.get((group, 'initial'))
        if not row: continue
        cells = []
        for case in CASES:
            current = index.get((group, case))
            cells.append('—' if not current else f"{100*current['recall']['R@1']:.3f}%")
        lines.append(f"| {group} | {row['queries']} | " + ' | '.join(cells) + ' |')
    lines += ['', 'R@5/R@10, invalid-query counts and paired exact tests are in report.json.']
    (path / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, action='append')
    parser.add_argument('--case', choices=CASES, action='append')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prepare', action='store_true', help='Build frozen reduced-gallery manifests only')
    parser.add_argument('--execute', action='store_true', help='Run read-only GPU evaluation')
    args = parser.parse_args()
    out = args.output.resolve()
    for dataset in args.dataset or DATASETS:
        if dataset != 'nordland':
            manifest = out / dataset / 'manifest.json'
            if not manifest.exists():
                make_manifest(dataset, out)
            elif not (out / dataset / 'protocol.json').exists():
                raise ValueError(f'Missing protocol for existing manifest: {manifest}')
        if args.execute:
            for case in args.case or CASES:
                run(dataset, case, out)
        summarize(dataset, out)
        print(f'{dataset}: {out / dataset / "report.md"}')


if __name__ == '__main__':
    main()
