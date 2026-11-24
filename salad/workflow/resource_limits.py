"""Opt-in CUDA allocator budget and Linux process-tree memory monitoring.

CUDA limits cover PyTorch's allocator, not driver/third-party allocations. Host
limits are sampled guards, not cgroup hard limits; leave room for the OS/apps.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

GIB = 1024**3


def configure_cuda_budget(device):
    import torch
    device = torch.device(device)
    requested = float(os.environ.get('CCR_CUDA_BUDGET_GIB', '0'))
    if device.type != 'cuda' or requested <= 0:
        return None
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    reserve = float(os.environ.get('CCR_CUDA_RESERVE_GIB', '4')) * GIB
    free, total = torch.cuda.mem_get_info(device)
    budget = min(requested * GIB, free - reserve)
    if budget < GIB:
        raise RuntimeError('Insufficient free CUDA memory after the configured reserve; retry when GPU is free')
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    record = dict(device=str(device), allocator_budget_gib=budget/GIB,
                  free_gib=free/GIB, device_total_gib=total/GIB, reserve_gib=reserve/GIB)
    print('CUDA resource budget: '+json.dumps(record), flush=True)
    return record


def memory_snapshot(process_group, parent_pid=None):
    """Conservative RSS sum (shared pages can be counted more than once)."""
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    if available is None:
        raise RuntimeError('Linux MemAvailable is required for host memory guard')
    resident = 0
    pagesize = os.sysconf('SC_PAGE_SIZE')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == process_group or int(path.name) == parent_pid:
                resident += int((path/'statm').read_text().split()[1]) * pagesize
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return dict(rss_bytes=resident, available_bytes=available)


def memory_violation(snapshot, limits):
    if snapshot['rss_bytes'] > limits['host_rss_limit_gib'] * GIB:
        return 'experiment process-tree RSS exceeded host_rss_limit_gib'
    if snapshot['available_bytes'] < limits['host_available_reserve_gib'] * GIB:
        return 'system MemAvailable fell below host_available_reserve_gib'
    return None


def stop_process_group(process):
    # Only the new session we started, including its DataLoader workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def monitored_run(command, *, handle, env, limits):
    before = memory_snapshot(-1, os.getpid())
    if reason := memory_violation(before, limits):
        raise RuntimeError('Resource guard refused launch: '+reason)
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True)
    peak = before['rss_bytes']; minimum = before['available_bytes']
    started = time.monotonic()
    try:
        while True:
            snapshot = memory_snapshot(process.pid, os.getpid())
            peak = max(peak, snapshot['rss_bytes']); minimum = min(minimum, snapshot['available_bytes'])
            if reason := memory_violation(snapshot, limits):
                raise RuntimeError('Resource guard stopped this experiment: '+reason+
                                   '; resume from the last complete epoch when resources are available')
            try:
                code = process.wait(timeout=.5)
                if code:
                    raise subprocess.CalledProcessError(code, command)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        stop_process_group(process)
        raise
    finally:
        handle.write('\nRESOURCE '+json.dumps(dict(peak_tree_rss_gib=peak/GIB,
            minimum_system_available_gib=minimum/GIB, wall_seconds=time.monotonic()-started,
            sampling_interval_seconds=.5, limits=limits))+'\n')
        handle.flush()
