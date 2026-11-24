"""Bounded, sequential scene-night inference for a 48GB GPU / 32GB host.

This launcher changes neither frozen prompts nor model settings. It includes
the separate systemd generator in host-memory accounting and releases that
service on failure. Monitoring is sampled, not an OOM-proof reservation.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common, plan_scene_transients as transients, scene_aware_night as workflow

SERVICE = 'adaptvpr-lightx2v.service'
GIB = 1024 ** 3


def service_pid():
    return int(subprocess.check_output(
        ['systemctl', '--user', 'show', SERVICE, '--property=MainPID', '--value'], text=True).strip())


def snapshot(group=-1):
    memory = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2}
    available = memory['MemAvailable']
    owner = service_pid()
    service_peak = 0
    if owner:
        try:
            relative = next(line.split('::', 1)[1] for line in Path(f'/proc/{owner}/cgroup').read_text().splitlines()
                            if line.startswith('0::'))
            service_peak = int((Path('/sys/fs/cgroup')/relative.lstrip('/')/'memory.peak').read_text())/GIB
        except (FileNotFoundError, ProcessLookupError, StopIteration):
            pass
    rss = 0
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group or int(path.name) in (os.getpid(), owner):
                rss += int((path / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    # Device-wide usage includes the generator's allocations and other users.
    gpu = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                                  '--format=csv,noheader,nounits'], text=True, timeout=10)
    return dict(available_gib=available/GIB, owned_rss_gib=rss/GIB,
                gpu_used_gib=float(gpu.strip())/1024,
                service_cgroup_peak_gib=service_peak,
                swap_used_gib=(memory['SwapTotal']-memory['SwapFree'])/GIB)


def task_cgroup(proc_file=Path('/proc/self/cgroup'), root=Path('/sys/fs/cgroup')):
    """Use the dedicated service cgroup, never a shared desktop/user cgroup."""
    relative=next((line.split('::',1)[1] for line in proc_file.read_text().splitlines()
                   if line.startswith('0::')),None)
    if relative is None:
        raise RuntimeError('Task swap monitoring requires cgroup v2')
    root=root.resolve();group=(root/relative.lstrip('/')).resolve()
    if (not group.is_relative_to(root) or not group.name.endswith('.service') or
        group.name.startswith('user@')):
        raise RuntimeError('Task swap monitoring requires a dedicated systemd service')
    maximum=(group/'memory.max').read_text().strip()
    if maximum=='max':
        raise RuntimeError('Task swap monitoring requires a finite service memory limit')
    if (group/'memory.swap.max').read_text().strip()!='0':
        raise RuntimeError('Owned service must retain MemorySwapMax=0')
    return group


def task_swap(group):
    try:
        return int((group/'memory.swap.current').read_text().strip())/GIB
    except (OSError,ValueError) as error:
        raise RuntimeError('Cannot read task cgroup swap accounting') from error


def owned_swap(group,include_generator=False):
    """Count both owned service cgroups during generation, without double count."""
    total=task_swap(group)
    owner=service_pid() if include_generator else 0
    if owner:
        try:
            generator=task_cgroup(Path(f'/proc/{owner}/cgroup'))
            if generator!=group: total+=task_swap(generator)
        except (OSError,RuntimeError):
            # Normal shutdown may remove its cgroup between the PID and file read.
            if service_pid(): raise
    return total


def violation(value, args):
    if value.get('disk_free_gib',float('inf'))<getattr(args,'reserve_disk_gib',5):
        return 'output filesystem free space below reserve'
    if value['available_gib'] < args.reserve_ram_gib:
        return 'system available RAM below reserve'
    if value['owned_rss_gib'] > args.max_rss_gib:
        return 'launcher + child process group + generator RSS above limit'
    if value['gpu_used_gib'] > args.max_gpu_gib:
        return 'device-wide VRAM above limit'
    if getattr(args,'swap_scope','system')=='task':
        if value.get('task_swap_used_gib') is None:
            return 'task swap accounting unavailable'
        if value['task_swap_used_gib']>getattr(args,'max_task_swap_gib',0):
            return 'task cgroup swap usage exceeded allowance'
    elif value.get('new_swap_gib', 0) > getattr(args, 'max_new_swap_gib', 1):
        return 'system swap usage increased beyond allowance'
    return None


def call_budget(requested, reserved, frozen_budget):
    if requested <= 0:
        raise ValueError('run --execute requires explicit positive --max-calls')
    remaining = frozen_budget-reserved
    if remaining <= 0:
        raise ValueError('Frozen cumulative generation budget exhausted; no automatic retry')
    return min(requested, remaining)


def service_memory_controls(high_gib, max_gib):
    if not (0 < high_gib <= max_gib < float('inf')):
        raise ValueError('Service memory high/max must be finite, positive and ordered')
    # cgroup accounting includes file cache, which process RSS omits. Keeping
    # that cache bounded avoids streaming checkpoints displacing desktop RAM.
    return (f'MemoryHigh={int(high_gib*GIB)}\nMemoryMax={int(max_gib*GIB)}\n'
            'MemorySwapMax=0\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'run', 'verify', 'audit', 'report'), default='check', nargs='?')
    parser.add_argument('--pipeline', choices=('scene', 'balanced', 'places'), default='scene')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-calls', type=int, default=0)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--max-sources', type=int, default=2)
    parser.add_argument('--variants', type=int, default=2)
    parser.add_argument('--max-jobs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--source-ids', nargs='*', default=[])
    parser.add_argument('--coverage-state', action='append', type=Path, default=[])
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--planner-model', type=Path)
    parser.add_argument('--reserve-ram-gib', type=float, default=4)
    parser.add_argument('--max-rss-gib', type=float, default=20)
    parser.add_argument('--max-gpu-gib', type=float, default=44)
    parser.add_argument('--max-new-swap-gib', type=float, default=1)
    parser.add_argument('--swap-scope', choices=('system','task'),default='system')
    parser.add_argument('--max-task-swap-gib',type=float,default=0)
    parser.add_argument('--reserve-disk-gib',type=float,default=5)
    parser.add_argument('--service-memory-high-gib', type=float, default=12)
    parser.add_argument('--service-memory-max-gib', type=float, default=16)
    args = parser.parse_args(argv)
    active = workflow
    script = Path(transients.__file__).resolve()
    root = common.WORKSPACE_ROOT/'outputs/scene_night_diversity_48g'
    if args.pipeline == 'balanced':
        from experiments.qwen_curriculum import balanced_night
        active, script, root = balanced_night.runtime(), Path(balanced_night.__file__).resolve(), balanced_night.ROOT
    if args.pipeline == 'places':
        from experiments.qwen_curriculum import night_place_fast as night_place_cohort
        # Explicit old directories retain their frozen reader; the default is the
        # measured successor. Never interpret an old cohort under a new protocol.
        if args.output_dir and (args.output_dir/'cohort_config.json').exists():
            version=json.loads((args.output_dir/'cohort_config.json').read_text())['version']
            readers={'night_place_fast_v1':'night_place_fast',
                     'night_place_cohort_v1':'night_place_cohort',
                     'night_place_cohort_v4':'night_place_cohort_v4'}
            if version not in readers: raise ValueError('Unknown frozen place cohort')
            night_place_cohort=importlib.import_module('experiments.qwen_curriculum.'+readers[version])
        script, root = Path(night_place_cohort.__file__).resolve(), night_place_cohort.ROOT
        if args.command=='prepare' and night_place_cohort.__name__.endswith('.night_place_fast'):
            from experiments.qwen_curriculum import resume_night_prompts
            script=Path(resume_night_prompts.__file__).resolve()
        if args.command in ('verify', 'audit'):
            raise ValueError('Verify/audit exported shards with --pipeline balanced after generation')
    if args.command == 'prepare' and args.pipeline not in ('balanced', 'places'):
        raise ValueError('This launcher prepares the balanced pipeline; use the existing scene preparation entry for old plans')
    args.output_dir = args.output_dir or root/'state'
    args.image_root = args.image_root or root
    args.output_dir = args.output_dir.resolve()
    ownership=args.output_dir/'cluster_ownership.json'
    if args.command=='run' and args.execute and ownership.exists():
        raise RuntimeError('This cohort is assigned to the two-machine generator. '
                           'Use ./ccr-night resume/status; the old all-shard launcher would duplicate remote work.')
    if args.command == 'report':
        if args.pipeline == 'places':
            print((args.output_dir/'progress.json').read_text())
            return 0
        elif args.pipeline == 'balanced':
            report = balanced_night.report
        else:
            from experiments.qwen_curriculum.scene_night_report import report
        result = report(args.output_dir)
        print(json.dumps({k: v for k, v in result.items() if k != 'resources'}, ensure_ascii=False, indent=2))
        return 0
    actual = args.execute and args.command in ('prepare', 'run', 'verify', 'audit')
    command = [sys.executable, '-B', str(script), args.command,
               '--output-dir', str(args.output_dir), '--image-root', str(args.image_root.resolve())]
    if args.command == 'prepare' and args.pipeline != 'places':
        command += ['--max-sources', str(args.max_sources), '--variants', str(args.variants),
                    '--max-jobs', str(args.max_jobs), '--seed', str(args.seed)]
        if args.source_ids:
            command += ['--source-ids', *args.source_ids]
        for state in args.coverage_state:
            command += ['--coverage-state', str(state.resolve())]
        if args.source_manifest:
            command += ['--source-manifest', str(args.source_manifest.resolve())]
    if args.planner_model:
        command += ['--planner-model', str(args.planner_model.resolve())]
    if not actual:
        return subprocess.call(command)
    if any(not 0 < x < float('inf') for x in (args.reserve_ram_gib, args.max_rss_gib,
                                            args.max_gpu_gib, args.max_new_swap_gib,args.reserve_disk_gib)):
        raise ValueError('Resource limits must be finite and positive')
    if not 0 <= args.max_task_swap_gib < float('inf'):
        raise ValueError('Task swap limit must be finite and nonnegative')
    if args.swap_scope=='task' and not (args.command in ('prepare','run') and args.pipeline=='places'):
        raise ValueError('Task swap scope supports place Prompt preparation and image generation')
    if service_pid():
        raise RuntimeError('Generator already running; inspect/reuse its existing task before launching another')
    if args.command == 'run':
        if (args.output_dir/'resource_successor.json').exists():
            raise ValueError('This resource profile was retired; use its recorded resource_successor')
        if args.pipeline == 'places':
            cfg, done, remaining = night_place_cohort.generation_resource_plan(args.output_dir)
            command += ['--max-calls', str(call_budget(args.max_calls, done['ready']-remaining, done['ready']))]
        else:
            cfg, jobs = active.load_plan(args.output_dir)
        if cfg['service']['performance']['resident_dit_blocks'] > 24:
            raise ValueError('This canvas exceeded the VRAM reserve with 32 resident blocks; reprofile to <=24 first')
        attempts = [] if args.pipeline == 'places' else active.engine()._read_state(args.output_dir, jobs, cfg)[2]
        if parent := cfg.get('resource_parent'):
            current = len(common.read_jsonl(Path(parent['state'])/'attempts.jsonl'))
            if current != parent['reserved_requests']:
                raise ValueError('Parent request count changed; family budget must be reviewed')
        if args.pipeline != 'places':
            command += ['--max-calls', str(call_budget(args.max_calls, len(attempts), cfg['request_budget']))]
    command.append('--execute')
    owned_cgroup=task_cgroup() if args.swap_scope=='task' else None
    initial = snapshot()
    initial['disk_free_gib']=shutil.disk_usage(args.output_dir).free/GIB
    if owned_cgroup is not None:
        initial['task_swap_used_gib']=owned_swap(owned_cgroup,args.command=='run')
    if initial['gpu_used_gib'] > 2:
        raise RuntimeError('GPU already occupied; leave the existing workload untouched')
    if reason := violation(initial, args):
        raise RuntimeError('Launch refused: '+reason)
    memory_controls = service_memory_controls(args.service_memory_high_gib, args.service_memory_max_gib)
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'OPENBLAS_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'SCENE_RESOURCE_CHILD': '1'}
    directory = args.output_dir.parent/'resource_runs'
    directory.mkdir(parents=True, exist_ok=True)
    name = f'{time.time_ns()}_{args.command}'
    peak_rss, peak_gpu, minimum = initial['owned_rss_gib'], initial['gpu_used_gib'], initial['available_gib']
    peak_new_swap = 0
    peak_task_swap=initial.get('task_swap_used_gib',0)
    peak_service_memory = 0
    started = time.monotonic()
    process = None
    override = None
    profile_env = None
    error = None
    code = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        common.write_json(directory/name/'guard_policy.json',dict(
            swap_scope=args.swap_scope,task_cgroup=str(owned_cgroup) if owned_cgroup else None,
            max_task_swap_gib=args.max_task_swap_gib,max_new_system_swap_gib=args.max_new_swap_gib,
            system_swap_role='informational only' if owned_cgroup else 'stop on growth',
            includes_generator_cgroup=args.command=='run',reserve_disk_gib=args.reserve_disk_gib,
            reserve_ram_gib=args.reserve_ram_gib,max_gpu_gib=args.max_gpu_gib,max_rss_gib=args.max_rss_gib))
        if args.command == 'run':
            runtime = Path(os.environ['XDG_RUNTIME_DIR'])/'systemd/user'/f'{SERVICE}.d'
            runtime.mkdir(parents=True, exist_ok=True)
            candidate = runtime/'99-scene-night-owned.conf'
            if candidate.exists():
                raise RuntimeError('A previous owned service override remains; inspect before reuse')
            profile_env = directory/f'{name}.env'
            profile_env.write_text('LIGHTX2V_RESIDENT_BLOCKS='+
                                   str(cfg['service']['performance']['resident_dit_blocks'])+'\n')
            with candidate.open('x') as handle:
                # EnvironmentFile overrides Environment, so append a later file
                # rather than changing the user's persistent .env or unit.
                handle.write('[Service]\nEnvironmentFile='+str(profile_env)+'\n'+memory_controls)
            override = candidate
            subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
        with (directory/f'{name}.log').open('x') as log:
            print('Bounded inference log:', directory/f'{name}.log', flush=True)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=env)
            while True:
                value = snapshot(process.pid)
                value['disk_free_gib']=shutil.disk_usage(args.output_dir).free/GIB
                if owned_cgroup is not None:
                    value['task_swap_used_gib']=owned_swap(owned_cgroup,args.command=='run')
                    peak_task_swap=max(peak_task_swap,value['task_swap_used_gib'])
                value['new_swap_gib'] = max(0, value['swap_used_gib']-initial['swap_used_gib'])
                peak_new_swap = max(peak_new_swap, value['new_swap_gib'])
                peak_service_memory = max(peak_service_memory, value['service_cgroup_peak_gib'])
                peak_rss = max(peak_rss, value['owned_rss_gib'])
                peak_gpu = max(peak_gpu, value['gpu_used_gib'])
                minimum = min(minimum, value['available_gib'])
                if reason := violation(value, args):
                    raise RuntimeError('Resource guard stopped inference: '+reason)
                code = process.poll()
                if code is not None:
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                    break
                time.sleep(1)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        # This command started from an inactive service; only generation owns it.
        if args.command == 'run' and process is not None:
            subprocess.run(['systemctl', '--user', 'stop', SERVICE], check=True, timeout=45)
        if override is not None and override.exists():
            override.unlink()
            subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
        if profile_env is not None:
            profile_env.unlink(missing_ok=True)
        record = dict(command=command, exit_code=code, error=error,
                      wall_seconds=time.monotonic()-started, peak_owned_rss_gib=peak_rss,
                      peak_device_vram_gib=peak_gpu, minimum_available_ram_gib=minimum,
                      initial_system_swap_gib=initial['swap_used_gib'], peak_new_system_swap_gib=peak_new_swap,
                      peak_task_swap_gib=peak_task_swap,
                      task_cgroup=str(owned_cgroup) if owned_cgroup else None,
                      peak_generator_cgroup_gib=peak_service_memory,
                      limits=dict(reserve_ram_gib=args.reserve_ram_gib, max_rss_gib=args.max_rss_gib,
                                  max_gpu_gib=args.max_gpu_gib, max_new_swap_gib=args.max_new_swap_gib,
                                  swap_scope=args.swap_scope,max_task_swap_gib=args.max_task_swap_gib,
                                  reserve_disk_gib=args.reserve_disk_gib,
                                  service_memory_high_gib=args.service_memory_high_gib,
                                  service_memory_max_gib=args.service_memory_max_gib, service_swap_max_gib=0), sampling_seconds=1,
                      accounting='RSS includes launcher, child process group and systemd generator; shared pages may repeat')
        common.write_json(directory/f'{name}.json', record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
