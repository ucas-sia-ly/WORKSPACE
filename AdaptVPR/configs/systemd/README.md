# Persistent local generation services

These user service files match the local workspace under
`~/github/WORKSPACE` and the Conda environment at
`~/miniconda3/envs/AdaptVPR`. Edit the files if either location changes.
Both adapters read the model paths from `AdaptVPR/.env`; configure and
validate those local weights before starting the services.

From the AdaptVPR project directory, install the definitions:

```bash
mkdir -p "$HOME/.config/systemd/user" tmp/logs tmp/service_outputs/iclight tmp/service_outputs/lightx2v
install -m 644 configs/systemd/adaptvpr-iclight.service "$HOME/.config/systemd/user/"
install -m 644 configs/systemd/adaptvpr-lightx2v.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now adaptvpr-iclight adaptvpr-lightx2v
```

Unlike transient `systemd-run` units, these definitions survive reboot.
Enabling them starts the adapters when the user service manager starts
(normally at login). This does not enable login lingering or start the
planner endpoint on port 23002.

Check service state and readiness before generation:

```bash
systemctl --user status adaptvpr-iclight adaptvpr-lightx2v --no-pager
curl --noproxy '*' http://127.0.0.1:8002/health
curl --noproxy '*' http://127.0.0.1:8001/health
```

An active process is not sufficient: the health responses must report
`status=ok`, `generator_ready=true`, and no initialization error. The disk
offload Qwen pipeline loads its weights lazily; health checks do not verify
that a complete image inference will succeed.

After changing `.env`, restart the adapters:

```bash
systemctl --user restart adaptvpr-iclight adaptvpr-lightx2v
```

Logs remain under `tmp/logs/iclight_8002.log` and
`tmp/logs/lightx2v_8001.log`. Avoid force-restarting the same adapters through
the tmux/nohup launcher while these units are running.

To stop the services and turn off automatic startup:

```bash
systemctl --user disable --now adaptvpr-iclight adaptvpr-lightx2v
```
