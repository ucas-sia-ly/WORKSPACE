"""CPU-only inventory of the completed 61,984 exact source/night pairs.

Images are hard-linked, never re-encoded or copied; sources are symlinks into
the real dataset. Completion is published only after hashes and uniqueness
have been checked. Building and verification never start an inference service.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = WORKSPACE/'outputs/night_dataset_v1'
CAMPAIGNS = ('night_original_no_mercury_v1', 'night_mercury_replacement_v1')
TARGET = 61984


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sealed(value, key='result_sha256'):
    body = {k: v for k, v in value.items() if k != key}
    return {**body, key: fingerprint(body)}


def read_json(path, key=None):
    value = json.loads(Path(path).read_text())
    if key and value != sealed(value, key):
        raise ValueError('Changed receipt: '+str(path))
    return value


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def input_manifests(workspace):
    retained = workspace/'outputs/night_revision_v2/state/retained_manifest.jsonl'
    yield 'retained_original', retained
    for name in CAMPAIGNS:
        root = workspace/'outputs'/name
        cfg = read_json(root/'cluster.json', 'fingerprint')
        total = 0
        for node in ('local', 'gpu2'):
            directory = root/'nodes'/node
            queue = read_json(directory/'queue.json', 'fingerprint')
            complete = read_json(directory/'completed.json', 'fingerprint')
            node_cfg = read_json(directory/'node.json', 'fingerprint')
            if (complete['node_fingerprint'] != node_cfg['fingerprint']
                    or complete['entries'] != queue['entries']):
                raise ValueError('Finish and collect the complete campaign before consolidation')
            for entry in queue['entries']:
                state = Path(entry['state'])
                if sha256(state/'plan_config.json') != entry['config_sha256']:
                    raise ValueError('Changed execution configuration')
                plan = read_json(state/'plan_config.json', 'fingerprint')
                summary = read_json(state/'summary.json')
                if (summary['state'] != 'domain_complete' or summary['saved'] != plan['num_jobs']
                        or summary['evaluated'] != plan['num_jobs']):
                    raise ValueError('Incomplete primary image collection: '+str(state))
                total += plan['num_jobs']
                yield name, state/'results.jsonl'
        if total != cfg['source_total']:
            raise ValueError('Assigned budget differs from frozen campaign')


def collect_inputs(workspace):
    seen_ids = set(); seen_places = set(); source_paths = set()
    for campaign, path in input_manifests(workspace):
        for row in rows(path):
            sample = row['sample_id']; city = row['city']; place = int(row['place_id'])
            if (not re.fullmatch(r'qnight_[0-9a-f]{24}', sample)
                    or not isinstance(city, str) or city in ('.', '..') or '/' in city or '\\' in city
                    or place < 0 or row.get('condition') != 'night' or row['style_id'] == 'mercury_vapor'):
                raise ValueError('Invalid identity or excluded style')
            if sample in seen_ids or (city, place) in seen_places or row['source_path'] in source_paths:
                raise ValueError('Duplicate generated image, place or exact source')
            if row != sealed(row):
                raise ValueError('Changed generated result receipt: '+sample)
            seen_ids.add(sample); seen_places.add((city, place)); source_paths.add(row['source_path'])
            yield campaign, path, row


def exact_link(source, target, *, symlink=False):
    source = Path(source); target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_file() or not os.path.samefile(source, target):
            raise ValueError('Existing pair file has another identity: '+str(target))
        return
    if symlink:
        target.symlink_to(source)
    else:
        os.link(source, target)


def consolidate_one(value, output):
    campaign, receipt_path, row = value
    source = Path(row['source_path']); image = Path(row['output_path']); raw = Path(row['raw_output_path'])
    sidecar = raw.with_suffix('.json')
    for path, digest in ((source, row['source_sha256']), (image, row['output_sha256']),
                         (raw, row['raw_output_sha256']), (sidecar, row['metadata_sha256'])):
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError('Source/image/sidecar missing or changed: '+str(path))
    pair = output/'pairs'/row['city']/str(int(row['place_id']))
    paths = dict(source_link=pair/('source'+source.suffix.lower()), output_path=pair/'night.png',
                 raw_output_path=pair/'night_full.png', metadata_path=pair/'generation.json')
    exact_link(source, paths['source_link'], symlink=True)
    for old, new in ((image, paths['output_path']), (raw, paths['raw_output_path']), (sidecar, paths['metadata_path'])):
        exact_link(old, new)
    record = dict(row)
    record.update(original_result_sha256=row['result_sha256'],
        original_output_path=str(image), original_raw_output_path=str(raw),
        origin_receipt=str(receipt_path), generation_campaign=campaign,
        **{k: str(v) for k, v in paths.items()})
    # Quality state is preserved verbatim: integrity is not content approval.
    return sealed(record)


def verify_one(row):
    if row != sealed(row):
        raise ValueError('Dataset row checksum changed: '+row['sample_id'])
    for field, digest in [('source_path', 'source_sha256'), ('source_link', 'source_sha256'),
                           ('output_path', 'output_sha256'), ('raw_output_path', 'raw_output_sha256'),
                           ('metadata_path', 'metadata_sha256')]:
        if sha256(row[field]) != row[digest]:
            raise ValueError('Dataset pair changed: '+row[field])
    if not os.path.samefile(row['source_link'], row['source_path']):
        raise ValueError('Original source symlink points to another image')
    return row


def build(output=ROOT, workspace=WORKSPACE, expected=TARGET, workers=2):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output/'complete.json').exists():
            result = status(output)
            if result['count'] != expected:
                raise ValueError('Existing dataset has a different budget')
            return result
        counts = Counter(); cities = Counter(); styles = Counter(); manifest = output/'manifest.jsonl.building'
        tsv = output/'pairs.tsv.building'; database = output/'index.sqlite.building'
        database.unlink(missing_ok=True)
        db = sqlite3.connect(database)
        db.execute('CREATE TABLE pairs(sample_id TEXT PRIMARY KEY, city TEXT, place_id INTEGER, '
            'style_id TEXT, source_path TEXT, source_link TEXT, output_path TEXT, raw_output_path TEXT, '
            'generation_campaign TEXT, UNIQUE(city,place_id))')
        columns = ['sample_id', 'city', 'place_id', 'style_id', 'source_path', 'source_link',
                   'output_path', 'raw_output_path', 'generation_campaign']
        count = 0
        try:
            with manifest.open('w') as handle, tsv.open('w', newline='') as table, ThreadPoolExecutor(max_workers=workers) as pool:
                writer = csv.DictWriter(table, fieldnames=columns, delimiter='\t', extrasaction='ignore')
                writer.writeheader()
                iterator = iter(collect_inputs(workspace))
                while batch := list(itertools.islice(iterator, 64)):
                    for row in pool.map(lambda v: consolidate_one(v, output), batch):
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        writer.writerow(row)
                        db.execute('INSERT INTO pairs VALUES(?,?,?,?,?,?,?,?,?)', tuple(row[k] for k in columns))
                        counts[row['generation_campaign']] += 1; cities[row['city']] += 1; styles[row['style_id']] += 1
                        count += 1
                    if count % 1024 == 0:
                        db.commit()
                        atomic_json(output/'progress.json', dict(state='verifying_and_linking', count=count, target=expected))
                        print(f'Verified exact source/night pairs: {count}/{expected}', flush=True)
                handle.flush(); os.fsync(handle.fileno()); table.flush(); os.fsync(table.fileno())
            if count != expected:
                raise ValueError(f'Expected {expected} unique exact pairs, found {count}')
            db.commit(); db.close()
            for old, new in [(manifest, output/'manifest.jsonl'), (tsv, output/'pairs.tsv'), (database, output/'index.sqlite')]:
                os.replace(old, new)
            summary = dict(version='night_dataset_v1', count=count, expected=expected,
                by_campaign=dict(counts), by_city=dict(cities), by_style=dict(styles),
                manifest_sha256=sha256(output/'manifest.jsonl'), pairs_tsv_sha256=sha256(output/'pairs.tsv'),
                index_sha256=sha256(output/'index.sqlite'), image_storage='hardlinks, no re-encoding or additional image copies',
                source_storage='symlinks to exact original real images', source_integrity='SHA256 verified',
                normalized_raw_sidecar_integrity='SHA256 verified', verification='content audit still pending',
                cpu_workers=workers, gpu_used=False)
            atomic_json(output/'complete.json', sealed(summary))
            atomic_json(output/'progress.json', dict(state='complete', count=count, target=expected))
            return summary
        finally:
            db.close()


def status(output):
    summary = read_json(output/'complete.json', 'result_sha256')
    for name, field in [('manifest.jsonl', 'manifest_sha256'), ('pairs.tsv', 'pairs_tsv_sha256'), ('index.sqlite', 'index_sha256')]:
        if sha256(output/name) != summary[field]:
            raise ValueError('Dataset index changed: '+name)
    return summary


def verify(output, workers=2):
    summary = status(output); count = 0; ids = set(); places = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator = iter(rows(output/'manifest.jsonl'))
        while batch := list(itertools.islice(iterator, 64)):
            for row in pool.map(verify_one, batch):
                key = row['city'], int(row['place_id'])
                if row['sample_id'] in ids or key in places or row['style_id'] == 'mercury_vapor':
                    raise ValueError('Duplicate or excluded pair in manifest')
                ids.add(row['sample_id']); places.add(key); count += 1
    if count != summary['count']:
        raise ValueError('Dataset cardinality changed')
    return dict(verified=count, all_source_and_image_hashes_match=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('build', 'status', 'verify', 'pair'))
    p.add_argument('--output-dir', type=Path, default=ROOT)
    p.add_argument('--workers', type=int, choices=(1, 2, 4), default=2)
    p.add_argument('--city'); p.add_argument('--place-id', type=int)
    args = p.parse_args(); output = args.output_dir.resolve()
    if args.command == 'build':
        result = build(output, workers=args.workers)
    elif args.command == 'verify':
        result = verify(output, args.workers)
    elif args.command == 'pair':
        if args.city is None or args.place_id is None:
            p.error('pair requires --city and --place-id')
        with sqlite3.connect('file:'+str(output/'index.sqlite')+'?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            row = db.execute('SELECT * FROM pairs WHERE city=? AND place_id=?', (args.city, args.place_id)).fetchone()
            if row is None:
                p.error('This place is not in the completed cohort')
            result = dict(row)
    else:
        result = status(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
