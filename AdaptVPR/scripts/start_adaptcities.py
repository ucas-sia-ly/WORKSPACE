#!/usr/bin/env python3
"""Start the local services and run/resume AdaptCities independently of a terminal."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--watchdog", action="store_true")
    args = parser.parse_args()
    task = ROOT.parent / "AdaptCities"
    logs = task / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from resource_limits import UNIT, available_memory, MIN_AVAILABLE, require_limits, watchdog
    if args.watchdog:
        watchdog(task)
        return
    if not args.foreground:
        if available_memory() < MIN_AVAILABLE + 10 * 1024**3:
            raise RuntimeError("Need at least 16 GiB available RAM before starting the protected services")
        subprocess.run(["systemd-run", "--user", "--unit", UNIT, "--collect",
                        "--working-directory", str(ROOT),
                        "-p", "MemoryHigh=8G", "-p", "MemoryMax=10G", "-p", "MemorySwapMax=0",
                        "-p", "OOMPolicy=stop", "-p", "KillMode=control-group", "-p", "TimeoutStopSec=20",
                        "-p", "CPUQuota=400%", "-p", "TasksMax=256", "-p", "Nice=10",
                        "-p", f"StandardOutput=append:{logs / 'pipeline.log'}",
                        "-p", f"StandardError=append:{logs / 'pipeline.log'}",
                        sys.executable, str(Path(__file__).resolve()), "--foreground"], check=True)
        try:
            subprocess.run(["systemd-run", "--user", "--unit", "adaptcities-memory-watchdog", "--collect",
                            "-p", "MemoryMax=128M", "-p", "MemorySwapMax=0", "-p", "Nice=5",
                            sys.executable, str(Path(__file__).resolve()), "--watchdog"], check=True)
        except BaseException:
            subprocess.run(["systemctl", "--user", "stop", UNIT])
            raise
        print(f"Protected service {UNIT}: 10 GiB host + 6 GiB Qwen; log: {logs / 'pipeline.log'}")
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv(task / "runtime.env", override=True)
    require_limits()
    sys.path.insert(0, str(ROOT / "scripts"))
    from adaptcities import lock, atomic_json
    with lock(task / "supervisor"):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        commands = [
            ["bash", str(ROOT.parent / "services/qwen3-vl-4b/service.sh"), "start"],
            ["bash", str(ROOT / "scripts/start_generation_services.sh"), "--wait"],
            [sys.executable, str(ROOT / "scripts/adaptcities.py"), "prepare"],
            [sys.executable, str(ROOT / "scripts/adaptcities.py"), "run"],
        ]
        try:
            for command in commands:
                atomic_json(task / "startup.json", {"state": "starting", "pid": os.getpid(), "command": command,
                                                     "complete": False, "updated_at": time.time()})
                print("Running:", " ".join(command), flush=True)
                subprocess.run(command, env=env, cwd=ROOT, check=True)
        except BaseException as exc:
            atomic_json(task / "startup.json", {"state": "stopped_incomplete", "pid": os.getpid(), "error": str(exc),
                                                 "complete": False, "updated_at": time.time()})
            raise


if __name__ == "__main__":
    main()
