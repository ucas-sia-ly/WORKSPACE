"""Host RAM limits and an out-of-cgroup safety watchdog for 32 GB machines."""
from pathlib import Path
import os
import subprocess
import time

GIB = 1024**3
HOST_MAX = 10 * GIB
QWEN_MAX = 6 * GIB
MIN_AVAILABLE = 6 * GIB
UNIT = "adaptcities-generation.service"
CONTAINER = "adaptcities-qwen3-vl-4b"


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot determine available RAM")


def cgroup_path(pid="self"):
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            return Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    raise RuntimeError("cgroup v2 is required for the memory limit")


def group_stats(path):
    result = {"path": str(path)}
    for key in ("memory.current", "memory.peak", "memory.max", "memory.high", "memory.swap.current", "memory.swap.max", "pids.current"):
        value = (path / key).read_text().strip()
        result[key] = int(value) if value.isdigit() else value
    result["memory.events"] = dict(line.split() for line in (path / "memory.events").read_text().splitlines())
    return result


def require_limits():
    stats = group_stats(cgroup_path())
    if not isinstance(stats["memory.max"], int) or stats["memory.max"] > HOST_MAX or stats["memory.swap.max"] != 0:
        raise RuntimeError("Run via scripts/start_adaptcities.py: a <=10 GiB cgroup and zero swap are required")
    if os.getenv("TORCH_COMPILE_DISABLE") != "1" or os.getenv("TORCHINDUCTOR_COMPILE_THREADS") != "1":
        raise RuntimeError("The 32 GB resource profile requires compilation disabled and one compiler thread")
    if available_memory() < MIN_AVAILABLE:
        raise RuntimeError("Less than 6 GiB system RAM available; refusing to start inference")
    return stats


def pressure_requires_stop(available, consecutive_low):
    return available < 3 * GIB or (available < MIN_AVAILABLE and consecutive_low >= 3)


def watchdog(task):
    from adaptcities import atomic_json
    low = 0
    started = time.monotonic()
    group = None
    qwen_group = None
    while True:
        info = subprocess.run(["systemctl", "--user", "show", UNIT, "-p", "ActiveState", "-p", "ControlGroup"], capture_output=True, text=True)
        values = dict(line.split("=", 1) for line in info.stdout.splitlines() if "=" in line)
        state = values.get("ActiveState", "inactive")
        if values.get("ControlGroup"):
            group = Path("/sys/fs/cgroup") / values["ControlGroup"].lstrip("/")
        if state not in ("active", "activating", "deactivating") and time.monotonic() - started > 10:
            subprocess.run(["docker", "stop", "-t", "5", CONTAINER], capture_output=True)
            atomic_json(task / "memory_watchdog.json", {"state": "stopped", "unit_state": state, "at": time.time()})
            return
        available = available_memory()
        low = low + 1 if available < MIN_AVAILABLE else 0
        sample = {"state": "monitoring", "available_bytes": available, "minimum_available_bytes": MIN_AVAILABLE,
                  "host_limit_bytes": HOST_MAX, "qwen_limit_bytes": QWEN_MAX, "at": time.time(), "unit_state": state}
        if group and group.exists():
            try:
                sample["host"] = group_stats(group)
            except OSError:
                pass
        if qwen_group is None or not qwen_group.exists():
            query = subprocess.run(["docker", "inspect", "--format", "{{.State.Pid}}", CONTAINER], capture_output=True, text=True)
            try:
                pid = int(query.stdout.strip())
                qwen_group = cgroup_path(pid) if pid else None
            except (ValueError, OSError):
                qwen_group = None
        if qwen_group and qwen_group.exists():
            try:
                sample["qwen"] = group_stats(qwen_group)
            except OSError:
                pass
        if pressure_requires_stop(available, low):
            sample["state"] = "stopped_low_memory"
            atomic_json(task / "memory_watchdog.json", sample)
            subprocess.run(["systemctl", "--user", "stop", "--no-block", UNIT], check=False)
            subprocess.run(["docker", "stop", "-t", "5", CONTAINER], capture_output=True)
            return
        atomic_json(task / "memory_watchdog.json", sample)
        time.sleep(2)
