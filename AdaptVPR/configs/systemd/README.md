# Persistent local generation services

These user service files match the local workspace under
`~/github/WORKSPACE` and the Conda environment at
`~/miniconda3/envs/AdaptVPR`. Edit the files if either location changes.
The default Qwen adapter reads its model paths from `AdaptVPR/.env`; configure
and validate those weights before starting it. Qwen handles Global, Local and
Dual routes and the staged curriculum. The IC-Light unit is optional for
explicit historical diagnosis runs.

From the AdaptVPR project directory, install the definitions:

```bash
mkdir -p "$HOME/.config/systemd/user" tmp/logs tmp/service_outputs/lightx2v
install -m 644 configs/systemd/adaptvpr-lightx2v.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now adaptvpr-lightx2v
```

Unlike transient `systemd-run` units, these definitions survive reboot.
Enabling them starts the adapters when the user service manager starts
(normally at login). This does not enable login lingering or start the
planner endpoint on port 23002.

Check service state and readiness before generation:

```bash
systemctl --user status adaptvpr-lightx2v --no-pager
curl --noproxy '*' http://127.0.0.1:8001/health
```

An active process is not sufficient: the health response must report
`status=ok`, `generator_ready=true`, and no initialization error. The disk
offload Qwen pipeline loads its weights lazily; health checks do not verify
that a complete image inference will succeed.

After changing Qwen settings in `.env`, restart that adapter:

```bash
systemctl --user restart adaptvpr-lightx2v
```

Qwen logs remain under `tmp/logs/lightx2v_8001.log`. Avoid force-restarting the same adapter through
the tmux/nohup launcher while these units are running.

To stop Qwen and turn off its automatic startup:

```bash
systemctl --user disable --now adaptvpr-lightx2v
```

## Optional historical IC-Light service

Install and enable this unit only when requesting a historical IC-Light run.
Configure its checkout and weights in `.env` first. Enabling it does not change
the route agent's Qwen backend.

```bash
mkdir -p tmp/service_outputs/iclight
install -m 644 configs/systemd/adaptvpr-iclight.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now adaptvpr-iclight
systemctl --user status adaptvpr-iclight --no-pager
curl --noproxy '*' http://127.0.0.1:8002/health
```

Its log is `tmp/logs/iclight_8002.log`. Restart it after changing its own settings
with `systemctl --user restart adaptvpr-iclight`; remove its automatic startup
with `systemctl --user disable --now adaptvpr-iclight`.
The alternative launcher enables this optional service with `--with-iclight`;
use either systemd or the launcher to manage a given adapter.
