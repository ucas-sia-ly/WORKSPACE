"""Fork frozen prompts into a smaller resident cache without adding requests.

Recovers completed parent sidecars first. Parent receipts remain unchanged;
only new jobs use the new resource identity, in a separate output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import campaign, common, run, scene_aware_night as scene

ROOT = common.WORKSPACE_ROOT/'outputs/scene_night_diversity_48g'


def prepare(parent, directory, image_root, blocks):
    campaign.ensure_stopped()
    scene.check_roots(directory, image_root)
    with run.run_lock(parent), run.run_lock(directory):
        if (directory/'plan_config.json').exists():
            return scene.load_plan(directory)[0]
        cfg, jobs = scene.load_plan(parent)
        saved, results, attempts = scene.engine()._read_state(parent, jobs, cfg)
        # A stopped HTTP client can leave a complete image and exact sidecar.
        for job in jobs:
            sid = job['sample_id']
            if sid in results:
                continue
            recovered = saved.get(sid) or scene.engine().recover_saved(parent, cfg, job, attempts)
            if recovered is None:
                continue
            if sid not in saved:
                scene.engine()._original_append(parent/'generated.jsonl', recovered)
                saved[sid] = recovered
            result = run.sealed({**recovered, **scene.DeferredVerifier().evaluate()})
            scene.engine()._original_append(parent/'results.jsonl', result)
            results[sid] = result
        scene.engine().rebuild_manifest(parent, cfg, list(results.values()))
        scene.publish(parent, cfg, jobs, saved, results, attempts, 'resource_profile_retired', 'all')
        remaining = cfg['request_budget']-len(attempts)
        selected = [j for j in jobs if j['sample_id'] not in results][:remaining]
        if not selected:
            raise ValueError('No remaining family budget/jobs')
        service = copy.deepcopy(cfg['service'])
        previous = service['performance']['resident_dit_blocks']
        if not 0 < blocks < previous:
            raise ValueError('Resource fork must reduce resident blocks')
        service['performance']['resident_dit_blocks'] = blocks
        common.write_jsonl(directory/'plan.jsonl', selected)
        source_ids = {j['source_sha256'] for j in selected}
        common.write_jsonl(directory/'source_snapshot.jsonl', [s for s in
            common.read_jsonl(parent/'source_snapshot.jsonl') if s['source_sha256'] in source_ids])
        planner = campaign.config_file(parent/'planner_config.json')
        files = {**cfg['files_sha256'], str(Path(__file__).resolve()): common.file_sha256(__file__),
                 str(parent/'plan_config.json'): common.file_sha256(parent/'plan_config.json')}
        campaign.frozen(directory/'planner_config.json', dict(version=1, request=planner['request'],
            files_sha256=files, parent_fingerprint=planner['fingerprint']))
        value = campaign.frozen(directory/'plan_config.json', dict(
            version=1, stage=cfg['stage'], planner_fingerprint=campaign.config_file(directory/'planner_config.json')['fingerprint'],
            num_jobs=len(selected), request_budget=len(selected),
            domain_counts=dict(Counter(j['condition'] for j in selected)), image_root=str(image_root),
            quality=cfg['quality'], service=service, files_sha256=files,
            plan_sha256=common.file_sha256(directory/'plan.jsonl'),
            resource_parent=dict(state=str(parent), reserved_requests=len(attempts),
                                 family_request_budget=cfg['request_budget'], original_resident_blocks=previous),
            omitted_jobs=[j['sample_id'] for j in jobs if j not in selected and j['sample_id'] not in results]))
        common.write_json(parent/'resource_successor.json', dict(state=str(directory), fingerprint=value['fingerprint']))
        scene.engine().rebuild_manifest(directory, value, [])
        scene.publish(directory, value, selected, {}, {}, [], 'prepared', 'all')
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-state', type=Path, default=common.WORKSPACE_ROOT/'outputs/scene_night_diversity_demo/state')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'state')
    parser.add_argument('--image-root', type=Path, default=ROOT)
    parser.add_argument('--resident-blocks', type=int, default=24)
    args = parser.parse_args()
    cfg = prepare(args.parent_state.resolve(), args.output_dir.resolve(), args.image_root.resolve(), args.resident_blocks)
    print(dict(planned=cfg['num_jobs'], request_budget=cfg['request_budget'], parent=cfg['resource_parent'],
               resident_blocks=cfg['service']['performance']['resident_dit_blocks']))


if __name__ == '__main__':
    main()
