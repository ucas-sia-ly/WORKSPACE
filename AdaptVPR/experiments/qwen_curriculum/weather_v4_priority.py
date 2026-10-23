"""Run the existing v4 plan with previously ungenerated places first.

The immutable base plan, prompts, quotas and saved images are unchanged. A
separate sealed snapshot fixes scheduling order across pause/resume. Historical
images are never imported into this generation campaign.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import (campaign, common, night_first_v4 as v4,
                                        resume_control, run, weather_io)

DEFAULT_STATE = v4.DEFAULT_OUTPUT


def place(row):
    return row['city'], int(row['place_id'])


def prioritize(jobs, seen):
    """Stable partition: preserve random base-plan order inside both groups."""
    return [j for j in jobs if place(j) not in seen] + [j for j in jobs if place(j) in seen]


def historical_places(legacy_root, v3_root):
    proofs, seen = [], set()
    index = legacy_root / 'images/metadata.jsonl'
    summary = json.loads((legacy_root / 'images/summary.json').read_text())
    if common.file_sha256(index) != summary['metadata_sha256']:
        raise ValueError('Historical image index checksum mismatch')
    for path, sealed in ((index, False), (v3_root / 'state/generated.jsonl', True)):
        rows = weather_io.read_rows(path)
        # Do not silently ignore an interrupted final historical append.
        for line in path.read_text().splitlines():
            if line.strip(): json.loads(line)
        for row in rows:
            if sealed and run.sealed(row) != row:
                raise ValueError('Historical generated record seal mismatch')
            if not Path(row['output_path']).is_file():
                raise ValueError(f"Historical saved image missing: {row['output_path']}")
            seen.add(place(row))
        proofs.append({'path': str(path), 'sha256': common.file_sha256(path),
                       'generated_records': len(rows)})
    return seen, proofs


def prepare_priority(directory=DEFAULT_STATE):
    directory = Path(directory).resolve()
    with run.run_lock(directory):
        config, jobs = v4.load_plan(directory)
        receipt = directory / 'priority_config.json'
        if receipt.exists():
            return load_priority(directory)[0]
        seen, proofs = historical_places(common.WORKSPACE_ROOT / 'outputs/qwen_curriculum',
                                         common.WORKSPACE_ROOT / 'outputs/weather_v3')
        snapshot = directory / 'previously_generated_places.jsonl'
        rows = [{'city': city, 'place_id': pid} for city, pid in sorted(seen)]
        # If preparation stopped between the two atomic publications, replay
        # only the identical snapshot; never overwrite a conflicting snapshot.
        if snapshot.exists() and weather_io.read_rows(snapshot) != rows:
            raise ValueError('Incomplete priority snapshot differs; inspect before retrying')
        if not snapshot.exists(): common.write_jsonl(snapshot, rows)
        counts = Counter(j['condition'] for j in jobs if place(j) not in seen)
        body = {'schema_version': 1, 'base_plan_fingerprint': config['fingerprint'],
                'policy': 'unseen places first, then previously generated places; stable within both groups',
                'snapshot': str(snapshot), 'snapshot_sha256': common.file_sha256(snapshot),
                'controller_sha256': common.file_sha256(Path(__file__)),
                'historical_inputs': proofs, 'previously_generated_places': len(seen),
                'unseen_places': sum(place(j) not in seen for j in jobs),
                'unseen_by_domain': dict(counts)}
        campaign.frozen(receipt, body)
        return config


def load_priority(directory):
    directory = Path(directory).resolve()
    config, jobs = v4.load_plan(directory)
    priority = campaign.config_file(directory / 'priority_config.json')
    if (priority['base_plan_fingerprint'] != config['fingerprint']
            or priority['controller_sha256'] != common.file_sha256(Path(__file__))):
        raise ValueError('Priority controller or base plan changed; preserve the frozen schedule')
    snapshot = directory / 'previously_generated_places.jsonl'
    if str(snapshot) != priority['snapshot'] or common.file_sha256(snapshot) != priority['snapshot_sha256']:
        raise ValueError('Priority place snapshot changed')
    rows = weather_io.read_rows(snapshot)
    seen = {place(row) for row in rows}
    if len(seen) != len(rows) or len(seen) != priority['previously_generated_places']:
        raise ValueError('Invalid historical place snapshot')
    return config, prioritize(jobs, seen)


def install(directory=DEFAULT_STATE):
    """Bind the stopped background service; never start generation here."""
    directory = Path(directory).resolve()
    load_priority(directory)
    campaign.ensure_stopped()
    override = resume_control.install_systemd_override(directory, controller=Path(__file__))
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    return override


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'install', 'run', 'check', 'status'))
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_STATE)
    args, extra = parser.parse_known_args(argv)
    directory = args.output_dir.expanduser().resolve()
    if args.command in ('prepare', 'install', 'check', 'status') and extra:
        parser.error('unexpected arguments: ' + ' '.join(extra))
    if args.command == 'prepare':
        prepare_priority(directory)
        print((directory / 'priority_config.json').read_text())
    elif args.command == 'install':
        print(install(directory))
    elif args.command == 'status':
        print(json.dumps(resume_control.status(directory), ensure_ascii=False, indent=2))
    elif args.command == 'check':
        with run.run_lock(directory):
            config, jobs = load_priority(directory)
            generated, results, attempts = v4._engine()._read_state(directory, jobs, config)
            print(json.dumps({'planned': len(jobs), 'saved': len(generated),
                              'evaluated': len(results), 'calls_reserved': len(attempts)}))
    else:
        # The private v4 engine owns the same request ledger and recovery rules.
        # All future loads in its execute/generate lifecycle use this schedule.
        v4._engine().load_plan = load_priority
        v4.main(['run', '--output-dir', str(directory), *extra])


if __name__ == '__main__':
    main()
