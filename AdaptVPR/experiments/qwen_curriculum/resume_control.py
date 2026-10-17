"""Pause/resume the owned generation services, without suspending GPU processes.

Only the new resumable coordinator is selected by the systemd override. The
historical generation source, plans and journals are left intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qwen_curriculum import common

DEFAULT_OUTPUT = common.WORKSPACE_ROOT / "outputs/qwen_curriculum/generation_23city"
COORDINATOR_UNIT = "qwen-curriculum-23city.service"
QWEN_UNIT = "adaptvpr-lightx2v.service"
SYSTEMCTL = "/usr/bin/systemctl"


def systemctl(*args, timeout=120):
    return subprocess.run([SYSTEMCTL, "--user", *args], text=True,
                          capture_output=True, timeout=timeout, check=False)


def unit_state(unit):
    result = systemctl("show", unit, "--property=LoadState", "--property=ActiveState",
                       "--property=SubState", "--property=MainPID", timeout=15)
    if result.returncode:
        raise RuntimeError(f"Cannot inspect {unit}: {result.stderr.strip() or result.stdout.strip()}")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if fields.get("LoadState") != "loaded":
        raise RuntimeError(f"Service {unit} is not loaded: {fields.get('LoadState', 'unknown')}")
    try:
        pid = int(fields["MainPID"])
        active = fields["ActiveState"]
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Incomplete service state for {unit}: {fields}") from exc
    return {"unit": unit, "active_state": active, "sub_state": fields.get("SubState"),
            "main_pid": pid}


def stopped(state):
    return state["active_state"] in {"inactive", "failed"} and state["main_pid"] == 0


def configured_output_dir():
    override = Path.home() / ".config/systemd/user" / f"{COORDINATOR_UNIT}.d" / "resume.conf"
    if override.exists():
        marker = "# qwen-resume-output-dir: "
        for line in override.read_text(encoding="utf-8").splitlines():
            if line.startswith(marker):
                return Path(json.loads(line[len(marker):])).resolve()
    return DEFAULT_OUTPUT.resolve()


def status(directory=None):
    directory = Path(directory).resolve() if directory is not None else configured_output_dir()
    report = {"output_dir": str(directory), "units": {
        unit: unit_state(unit) for unit in (COORDINATOR_UNIT, QWEN_UNIT)}}
    report["resources_released"] = all(stopped(state) for state in report["units"].values())
    path = directory / "summary.json"
    report["summary"] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return report


def pause(directory=None):
    # The generator child belongs to the coordinator's systemd control group.
    # Qwen is a separate unit and must also be stopped to release resident weights
    # and cancel an HTTP request that is still executing after its client exits.
    errors = []
    for unit in (COORDINATOR_UNIT, QWEN_UNIT):
        try:
            result = systemctl("stop", unit)
            if result.returncode:
                errors.append(f"{unit}: {result.stderr.strip() or result.stdout.strip()}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{unit}: {exc}")
    report = status(directory)
    if errors or not report["resources_released"]:
        pending = [unit for unit, state in report["units"].items() if not stopped(state)]
        details = "; ".join(errors + ([f"Still running: {', '.join(pending)}"] if pending else []))
        raise RuntimeError(f"Pause could not confirm generation resources were released: {details}")
    report["action"] = "paused"
    report["message"] = "生成协调器和 Qwen 服务均已停止；这两个服务占用的资源已释放。"
    return report


def _systemd_argument(value):
    value = str(value)
    if any(char in value for char in ("\n", "\r", "\0")):
        raise ValueError("Invalid newline or null in systemd argument")
    return '"' + value.replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"') + '"'


def install_systemd_override(directory=DEFAULT_OUTPUT, *, user_unit_dir=None, python=None, controller=None):
    """Write an override; do not reload, enable or start any service here."""
    directory = Path(directory).resolve()
    user_unit_dir = Path(user_unit_dir) if user_unit_dir else Path.home() / ".config/systemd/user"
    interpreter = Path(python) if python else Path.home() / "miniconda3/envs/AdaptVPR/bin/python"
    if controller is None:
        entry = "v2_campaign.py" if (directory / "migration_config.json").exists() else "resume_campaign.py"
        controller = Path(__file__).with_name(entry).resolve()
    else:
        controller = Path(controller).resolve()
    command = " ".join(_systemd_argument(arg) for arg in
        (interpreter, "-u", controller, "run", "--output-dir", directory))
    # StandardOutput is a single setting, not an ExecStart argument list; quotes
    # inserted after append: would become part of the filename.
    log_path = str(directory / "coordinator.log").replace("%", "%%")
    body = (f"# qwen-resume-output-dir: {json.dumps(str(directory), ensure_ascii=False)}\n"
        "[Service]\n"
        "ExecStart=\n"
        f"ExecStart={command}\n"
        "ExecStopPost=/usr/bin/systemctl --user --no-block stop adaptvpr-lightx2v.service\n"
        "KillMode=control-group\n"
        "KillSignal=SIGINT\n"
        "TimeoutStopSec=45\n"
        "Restart=no\n"
        f"StandardOutput=append:{log_path}\n"
        f"StandardError=append:{log_path}\n")
    path = user_unit_dir / f"{COORDINATOR_UNIT}.d" / "resume.conf"
    common._atomic_write(path, [body])
    return path


def resume(directory=None):
    configured_output = configured_output_dir()
    directory = Path(directory).resolve() if directory is not None else configured_output
    if directory != configured_output:
        raise RuntimeError(f"The background service is configured for {configured_output}, not {directory}; "
                           "stop it and install the matching systemd override first")
    state = unit_state(COORDINATOR_UNIT)
    if state["active_state"] in {"active", "activating"}:
        return {"action": "already_running", "output_dir": str(directory), "unit": state,
                "message": "生成任务已经运行，无需重复启动。如果当前进程使用旧入口，新版恢复流程将在下次 pause/resume 后生效。"}
    if not stopped(state):
        raise RuntimeError(f"Coordinator has not finished stopping: {state}")
    result = systemctl("start", COORDINATOR_UNIT)
    if result.returncode:
        raise RuntimeError(f"Cannot resume generation: {result.stderr.strip() or result.stdout.strip()}")
    after = unit_state(COORDINATOR_UNIT)
    if after["active_state"] == "failed":
        raise RuntimeError(f"Generation failed during startup; inspect coordinator.log: {after}")
    return {"action": "resumed", "output_dir": str(directory), "unit": after,
            "message": "已启动恢复流程；完成的图继续复用，中断任务由恢复账本处理。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pause", "resume", "status"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to the output directory currently bound to the background service")
    args = parser.parse_args()
    try:
        action = {"pause": pause, "resume": resume, "status": status}[args.command]
        print(json.dumps(action(args.output_dir), ensure_ascii=False, indent=2))
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
