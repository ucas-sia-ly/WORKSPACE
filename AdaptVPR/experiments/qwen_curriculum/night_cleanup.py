"""Explicit, audited cleanup after exact-pair consolidation; never follows links.

The plan contains individual file identities. Applying it refuses changed files,
protected dependencies, or paths outside the narrow output/source scopes.
Datasets, downloaded weights, final checkpoints and Reliability V2 are retained.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import time

from experiments.qwen_curriculum import night_dataset as dataset

WORKSPACE = dataset.WORKSPACE
CODE = WORKSPACE/'AdaptVPR/experiments/qwen_curriculum'
LOG = WORKSPACE/'outputs/maintenance/night_cleanup_20261113'
CURRENT = {'night_dataset_v1', 'night_generation_completion_v1', 'night_mercury_replacement_v1',
           'night_original_no_mercury_v1', 'night_place_fast_v1', 'night_place_cohort_v4',
           'night_cluster_v1'}
FROZEN = CURRENT-{'night_dataset_v1', 'night_generation_completion_v1'} | {'night_revision_v2', 'night_revision_images_v1'}
CONFIGS = {'cluster.json', 'node.json', 'cohort_config.json', 'config.json', 'plan_config.json', 'revision_config.json'}


def files(root):
    if not root.exists():
        return
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(directory)/name
            if path.is_symlink():
                yield path; dirs.remove(name)
        for name in names:
            yield Path(directory)/name


def protected_dependencies():
    protected = set()
    for name in FROZEN:
        for path in files(WORKSPACE/'outputs'/name):
            if path.name not in CONFIGS or path.is_symlink():
                continue
            value = dataset.read_json(path)
            protected.update(Path(p) for p in value.get('files_sha256', {}))
    # Preserve exact split and snapshot inputs even if not in files_sha256.
    protected.add(WORKSPACE/'outputs/qwen_curriculum/state/finetuning/real_baseline_v1/excluded_places.json')
    protected.add(WORKSPACE/'outputs/coverage_run/checkpoint_epoch_050.pt')
    # References to actual checkpoint inputs in source configs and protocols.
    for directory in [WORKSPACE/'salad/configs', WORKSPACE/'outputs/qwen_curriculum/state/finetuning']:
        for path in files(directory):
            if path.suffix != '.json' or path.stat().st_size > 2*1024**2:
                continue
            try:
                value = dataset.read_json(path)
            except (ValueError, OSError):
                continue
            def visit(obj):
                if isinstance(obj, dict):
                    for key, child in obj.items():
                        if ('checkpoint' in key or key in ('exclude_places', 'manifest', 'synthetic_manifest')) and isinstance(child, str):
                            candidate = Path(child)
                            if not candidate.is_absolute():
                                candidate = (path.parent/candidate).resolve()
                            if candidate.exists():
                                protected.add(candidate)
                        if key not in ('files_sha256',):
                            visit(child)
                elif isinstance(obj, list):
                    for child in obj:
                        visit(child)
            visit(value)
    return protected


def source_closure(protected):
    keep = {p for p in protected if p.parent == CODE}
    for name in ('__init__.py', 'night_dataset.py', 'night_cleanup.py',
                 'night_generation_completion.py', 'night_original_queue_recovery.py',
                 'run_scene_night_safe.py', 'content_audit_v42.py', 'audit_content_v42.py'):
        keep.add(CODE/name)
    # Training code outside this generation directory still uses some helpers.
    # Parse its imports, without preserving obsolete generation CLIs as roots.
    external = [p for base in [WORKSPACE/'salad', WORKSPACE/'AdaptVPR'] for p in base.rglob('*.py')
                if CODE not in p.parents and 'tests' not in p.parts and p.name != 'complete_night_v4.py']
    pending = list(keep)+external; visited = set()
    while pending:
        path = pending.pop()
        if path in visited or not path.exists():
            continue
        visited.add(path)
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeError):
            continue
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if module == 'experiments.qwen_curriculum':
                    imported.update(alias.name.split('.')[0] for alias in node.names)
                elif module.startswith('experiments.qwen_curriculum.'):
                    imported.add(module.split('.')[2])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('experiments.qwen_curriculum.'):
                        imported.add(alias.name.split('.')[2])
        for name in imported:
            target = CODE/(name+'.py')
            if target.exists() and target not in keep:
                keep.add(target); pending.append(target)
    return keep


def entry(path, reason):
    stat = path.lstat()
    return dict(path=str(path), reason=reason, size=stat.st_size, inode=stat.st_ino,
                device=stat.st_dev, mtime_ns=stat.st_mtime_ns, symlink=path.is_symlink(),
                sha256=dataset.sha256(path) if path.suffix in ('.py', '.md', '.json') and not path.is_symlink() else None)


def make_plan():
    summary = dataset.status(dataset.ROOT)
    if summary['count'] != dataset.TARGET:
        raise ValueError('Complete exact-pair consolidation first')
    LOG.mkdir(parents=True, exist_ok=True)
    protected = protected_dependencies(); keep_code = source_closure(protected)
    generated = set()
    for row in dataset.rows(dataset.ROOT/'manifest.jsonl'):
        generated.update(Path(row[k]) for k in ('original_output_path', 'original_raw_output_path'))
        generated.add(Path(row['original_raw_output_path']).with_suffix('.json'))
    protected |= generated
    removed_modules = {p.stem for p in CODE.glob('*.py') if p not in keep_code}
    code_candidates = {p for p in CODE.glob('*.py') if p.stem in removed_modules}
    # Explicit old entry point; it targets retired weather_v4 experiments.
    code_candidates.add(WORKSPACE/'salad/complete_night_v4.py')
    code_candidates |= {p for p in (CODE/'tests').glob('test_*.py')
                        if any(re.search(r'\b'+re.escape(n)+r'\b', p.read_text()) for n in removed_modules)}
    tests = WORKSPACE/'salad/tests/test_complete_night_v4.py'
    if tests.exists():
        code_candidates.add(tests)
    old_docs = ('ALL_23_CITIES.md', 'CONTENT_AUDIT_V42.md', 'CONTENT_AUDIT_V4.md', 'RESUME.md', 'WEATHER_V3.md', 'WEATHER_V4.md')
    code_candidates |= {CODE/n for n in old_docs}
    code_candidates |= {WORKSPACE/'salad/docs'/n for n in ('night_generation_v4.md', 'night_generation_v5.md',
                       'night_generation_balanced.md', 'scene_aware_night.md')}
    actions = {}
    def add(path, reason):
        if path in protected or not (path.exists() or path.is_symlink()):
            return
        actions[str(path)] = entry(path, reason)
    for path in code_candidates:
        add(path, 'unused_generation_entry_test_or_document')
    for top in (WORKSPACE/'outputs').iterdir():
        if top.name in {'night_dataset_v1', 'maintenance', 'night_generation_completion_v1'}:
            continue
        if top.name in CURRENT:
            for path in files(top):
                rel = path.relative_to(top)
                if 'incoming' in rel.parts:
                    add(path, 'verified_remote_transfer_cache')
                elif path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp') and path not in generated:
                    add(path, 'unselected_or_duplicate_generation_artifact')
        elif top.name == 'qwen_curriculum':
            for path in files(top):
                rel = path.relative_to(top)
                if rel.parts[0] == 'images' or 'history' in rel.parts or 'imported' in rel.parts:
                    add(path, 'retired_weather_generation_not_in_current_cohort')
                elif path.name.startswith('checkpoint_epoch_'):
                    add(path, 'intermediate_training_checkpoint_final_retained')
                elif path.name == 'sampling_coverage.jsonl':
                    add(path, 'large_per_image_exposure_trace_metrics_retained')
        elif top.name == 'coverage_run':
            for path in files(top):
                if path.name.startswith('checkpoint_epoch_') or path.name == 'sampling_coverage.jsonl':
                    add(path, 'old_training_intermediate_final_and_referenced_initial_retained')
        elif top.name == 'night_revision_v2':
            # Keep the original retirement inventory, source mapping and all
            # frozen hash inputs; remove unsuccessful alternative V2 drafts.
            for path in files(top):
                rel = path.relative_to(top)
                if any(part in ('batches', 'groups', 'prompts', 'runtime_patches', 'repairs', 'previews') for part in rel.parts):
                    add(path, 'unused_revised_prompt_trial_not_production')
        elif top.is_dir() and not top.is_symlink():
            for path in files(top):
                add(path, 'retired_generation_experiment_not_in_current_cohort')
        else:
            add(top, 'retired_generation_profile')
    # Keep historical results/final models, remove redundant smoke tensors and
    # intermediate snapshots. Reliability V2 retains its previous protection.
    for path in files(WORKSPACE/'salad/outputs'):
        rel = path.relative_to(WORKSPACE/'salad/outputs')
        if 'reliability_v2' in rel.parts:
            continue
        if path.name.startswith('checkpoint_epoch_') or path.name == 'sampling_coverage.jsonl':
            add(path, 'intermediate_training_snapshot_final_and_reports_retained')
        elif path.suffix in ('.pt', '.npy', '.npz') and any(k in rel.parts[0] for k in ('smoke', 'probe', 'pilot')):
            add(path, 'reproducible_smoke_or_probe_tensor')
    for path in files(WORKSPACE/'AdaptVPR/tmp'):
        add(path, 'completed_inference_service_temporary_output')
    counts = Counter(a['reason'] for a in actions.values())
    plan = dict(version=1, workspace=str(WORKSPACE), dataset_manifest_sha256=summary['manifest_sha256'],
        protected=[str(p) for p in sorted(protected)], preserved_code=[str(p) for p in sorted(keep_code)],
        actions=list(sorted(actions.values(), key=lambda a:a['path'])),
        files=len(actions), logical_bytes=sum(a['size'] for a in actions.values()), by_reason=dict(counts),
        created_unix=time.time(), policy='keep datasets, models, final checkpoints, training reports, splits and Reliability V2')
    dataset.atomic_json(LOG/'plan.json', dataset.sealed(plan))
    return dict(files=plan['files'], logical_gib=plan['logical_bytes']/1024**3,
                by_reason=plan['by_reason'], removed_modules=sorted(removed_modules), plan=str(LOG/'plan.json'))


def allowed(path):
    return (path.is_relative_to(WORKSPACE/'outputs') or path.is_relative_to(WORKSPACE/'salad/outputs')
            or path.is_relative_to(WORKSPACE/'AdaptVPR/tmp') or path.is_relative_to(CODE)
            or path == WORKSPACE/'salad/complete_night_v4.py'
            or path == WORKSPACE/'salad/tests/test_complete_night_v4.py'
            or path.parent == WORKSPACE/'salad/docs')


def apply():
    plan = dataset.read_json(LOG/'plan.json', 'result_sha256')
    if plan['workspace'] != str(WORKSPACE):
        raise ValueError('Cleanup plan belongs to another workspace')
    summary = dataset.status(dataset.ROOT)
    if summary['manifest_sha256'] != plan['dataset_manifest_sha256']:
        raise ValueError('Consolidated dataset differs from cleanup plan')
    protected = set(map(Path, plan['protected']))
    # Validate the whole plan before the first unlink, never recurse into data
    # or model directories through a symlink.
    for action in plan['actions']:
        path = Path(action['path'])
        if not allowed(path) or path in protected or not path.is_absolute() or '..' in path.parts:
            raise ValueError('Deletion escapes authorized scope or protected input: '+str(path))
        for parent in path.parents:
            if parent == WORKSPACE:
                break
            if parent.is_symlink():
                raise ValueError('Deletion traverses a symbolic link: '+str(path))
        if not (path.exists() or path.is_symlink()):
            continue
        stat = path.lstat()
        if (stat.st_ino, stat.st_dev, stat.st_size, stat.st_mtime_ns) != (
                action['inode'], action['device'], action['size'], action['mtime_ns']):
            raise ValueError('File changed after cleanup plan: '+str(path))
        if action['sha256'] and dataset.sha256(path) != action['sha256']:
            raise ValueError('File bytes changed after cleanup plan')
    free_before = shutil.disk_usage(WORKSPACE).free; count = 0
    with (LOG/'deleted.jsonl').open('a') as log:
        for action in plan['actions']:
            path = Path(action['path'])
            if path.exists() or path.is_symlink():
                path.unlink(); count += 1
                log.write(json.dumps(action, ensure_ascii=False)+'\n')
        log.flush(); os.fsync(log.fileno())
    # Remove only now-empty directories, bottom up, in the same output scopes.
    for root in (WORKSPACE/'outputs', WORKSPACE/'salad/outputs', WORKSPACE/'AdaptVPR/tmp'):
        for directory, dirs, names in os.walk(root, topdown=False, followlinks=False):
            path = Path(directory)
            if path == root or path.is_symlink() or path == LOG:
                continue
            try:
                path.rmdir()
            except OSError:
                pass
    result = dict(deleted_files=count, actual_freed_gib=(shutil.disk_usage(WORKSPACE).free-free_before)/1024**3,
        dataset_pairs=summary['count'], plan_sha256=dataset.sha256(LOG/'plan.json'),
        preserved=plan['policy'])
    dataset.atomic_json(LOG/'result.json', dataset.sealed(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'apply'))
    args = parser.parse_args()
    print(json.dumps(make_plan() if args.command == 'plan' else apply(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
