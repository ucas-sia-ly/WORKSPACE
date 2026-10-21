#!/usr/bin/env python3
"""Freeze external retrieval manifests (Nordland probe, reduced Tokyo247 / SF-XL).

Tokyo and SF-XL use a fixed, explicitly reduced gallery chosen without any model
score. Their results are exploratory and must never be presented as official
full-gallery benchmarks. Manifests are write-once: rerunning reproduces them
byte-identically or fails. Evaluation itself goes through the experiment runners.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/qwen_curriculum/state/external_validation'
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
        meta = dict(dataset=dataset, role='external_exploratory_validation',
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
    meta = dict(dataset=dataset, role='external_exploratory_validation', gallery='fixed_reduced_gallery',
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, action='append')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    for dataset in args.dataset or DATASETS:
        manifest, meta = make_manifest(dataset, out)
        print(f"{dataset}: {manifest} queries={meta['query_count']} references={meta['selected_reference_count']}")


if __name__ == '__main__':
    main()
