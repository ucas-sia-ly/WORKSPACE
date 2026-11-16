# 两台电脑协同生成 night

2026-11-10最新状态：用户已停用新修订Prompt，改用[原Prompt双机续生成，排除汞灯](night_original_no_mercury.md)。`./ccr-night`现管理这一独立续生成队列。原混合队列锁不解除；[Prompt修订](night_revision_v2.md)和[修订图实验](night_revision_images.md)的结果与日志均保留但已暂停。下文旧目录和旧服务说明作为历史记录保留。

主机统一入口为仓库根目录的 `./ccr-night`。使用 SSH 免密连接 `ccr-gpu2`，不需要通过 RustDesk 启动、暂停或查看生成任务。SSH 主机密钥已通过原 RustDesk 会话核实；私钥仅留在主机。当前连接使用同一局域网内的 `epic.local`，未开放公网生图接口，也未复制账户密钥或项目 `.env`。

## 日常操作

```bash
cd /home/admin123/github/WORKSPACE

./ccr-night status                 # 本机状态和最近同步的第二台状态
./ccr-night pause                  # 两边完成当前图片后暂停，释放模型
./ccr-night resume                 # 从已有队列、请求记录和图片恢复

./ccr-night pause --node local      # 只暂停本机，第二台继续
./ccr-night resume --node local
./ccr-night pause --node gpu2       # 只暂停第二台
./ccr-night resume --node gpu2

./ccr-night sync                   # 立即回收第二台已提交结果，不启动生成
./ccr-night manifest               # 流式导出统一 manifest，不启动生成

ssh ccr-gpu2                      # 进入第二台终端，可运行其他命令
ssh ccr-gpu2 nvidia-smi
journalctl --user -u ccr-night-controller.service -f
ssh ccr-gpu2 journalctl --user -u ccr-night-gpu2-worker.service -n 30 --no-pager
```

`pause` 提交暂停意图后立即返回，正在进行的图片完成并写入日志后，worker 才停止后端。因此看到命令返回时 GPU 仍在工作是正常的。可用 `status` 查看 `paused` 和 `backend_pid=0`。`resume` 不会重新生成已经提交的图片。首次启动、迁移权重和内存保护停止的状态也会记录，不能把服务存在等同于已经成功生成。

### 2026-11-04 主机资源策略修复

原主机在可用RAM瞬间低于4GiB时永久退出，另有“启动前显存已用超过2GiB即拒绝”的固定判断。2GiB不是生图模型的显存额度，也不是本次OOM：控制worker实际只占约151MiB。原worker的2GiB cgroup上限与生图后端16GiB上限是两个独立进程的限额。

主机现在通过 `night_cluster_resources.py` 加载独立、带SHA和节点身份的 `nodes/local/resource_policy_v2.json`。原冻结worker、node、模型、任务及采样参数不改写，第二台暂保持已验证的原策略。

- 启动按实际显存余量判断：至少42GiB空闲显存、10GiB系统可用RAM，稳定30秒再加载。桌面已使用超过2GiB显存并不直接拒绝；其他GPU任务使模型放不下时仍等待。
- 运行时可用RAM低于2GiB持续60秒，完成当前图片后释放后端，进入 `waiting_for_resources`；短暂波动恢复后继续运行。
- 可用RAM低于0.75GiB、剩余显存低于1.5GiB，或自身RSS越界时，立即停止本节点后端。原cgroup硬限不取消，不杀其他程序。
- 只有明确的资源等待才自动恢复，冷却从120秒递增到最多900秒，且要求资源重新稳定。输入SHA、所有权、监控故障和不明确的POST错误仍停止，不无限重试。
- 未确认完成的旧请求不能自动重复POST；先复用已有raw/sidecar恢复，不能恢复时明确报错。手动pause期间不自动恢复。

`waiting_for_resources` 表示后台还在等待，不是生成中，也不是永久失败。状态记录等待原因、资源读数和下次最早重试时间。模型卸载后主机内存恢复，并不能反推停止瞬间也有相同余量。当前策略是对32GB主机的可调整运行预算，不保证与任意其他重负载同时使用时无卡顿。

两台电脑处于不同网络时，`epic.local` 不可用；应另行配置两端私有网络连接，再修改 SSH 的 `HostName`，保留已验证的 `HostKeyAlias`。当前没有将另一台设备冒充成已加入本机的 Tailscale 网络。

### resume 返回成功但没有生成

`resume` 成功表示 systemd 接受启动请求，worker 仍可能在异步预检查阶段失败。先看 `./ccr-night status` 中该节点的 `state`、`error`，再查看对应 worker 日志。

当前冻结 worker 的启动保护要求整卡已用显存不超过 2GiB，超过会报 `GPU is already in use`。这个检查没有区分桌面与模型：浏览器、桌面或 RustDesk 的合计显存也可能触发，不能据此断言存在其他训练。先用 `nvidia-smi` 确认实际进程；减少不需要的 GPU 应用后再执行 `./ccr-night resume --node local`，不必重建分片或删除输出。失败后控制器不会循环强行重启该节点。

2026-11-03 本机曾因桌面等应用合计约 2.06GiB 被启动保护拦截；实际检查后重新恢复成功。启动检查及运行中的 44GiB/系统内存保护是不同的检查。没有为这次恢复修改冻结 worker、队列身份或放宽运行资源限额。

## 分工与恢复

- 使用已经封存的 61,984 条 Prompt，每地点一图，不重新加载 VLM 或修改风格策略。
- 保留原有图片和原始计划。中断过的原分片留在本机恢复；完全没有启动的分片才可分给第二台。
- 每个分片 128 个地点，每台最多一个尚未完成的分片。主机按实际完成速度继续分配，较快设备自然领取更多。
- 主机先持久保存 ownership，再传输输入，最后发布节点队列。SSH 断线不会自动转移 ownership，避免两边重复调用同一任务。
- worker 先检查原图、计划、实现 SHA 和后端身份。请求前预留日志，保存 raw/sidecar 后提交结果；恢复时优先找回已保存的 raw。
- 第二台的计划是独立的派生执行计划，明确记录原计划 SHA 和新的硬件执行配置，原冻结计划保持不变。
- 收集先进入 `incoming/`，校验 job、请求预留、PNG、尺寸、SHA 和 sidecar，再发布结果。已经提交的前缀不得回退或覆盖。

已生成图片仍为 `pending_verification`，不等于审核通过，也不自动进入真实/生成微调训练。

## 设备和资源

本机实测为 RTX 4090 48GB / 32GB RAM。第二台实测为 RTX 4090 **24GB** / **64GB RAM**，Ubuntu 20.04；两台显存不能合并使用。

本机保留已验证的 BF16、Lightning 4 steps、24 个 GPU 常驻 DiT block 和原 source-aspect 画布。显存采样上限 44GiB、可用内存预留 4GiB、后端内存硬限 16GiB、swap 禁止。

第二台提供 BF16 分块卸载和可选 CPU 缓存配置。GPU 采样上限 22GiB、RSS 上限 48GiB、系统可用 RAM 预留 8GiB。其旧系统使用 cgroup v1，后端以普通用户 `epic` 运行在独立系统 unit；root 的启动前检查实际验证 52GiB RAM/合计内存硬限及 swappiness，不仅检查 systemd 显示的属性。限定 sudo 权限只用于启动/停止这个后端，普通任务没有新增通用免密 sudo。

已做同一源图、同一 seed 的短程对照。标准 `disk24g` 的第二张耗时 32.62 秒；只缓存 CPU DiT 的 `disk24g_ditcache24` 为 28.42 秒，RSS 峰值 40.20GiB，两组共四张输出逐像素一致。缓存 encoder/VAE 的 `disk24g_cpu64` 在首图触发 RSS 保护（55.23GiB），不采用。短程耗时不能视为所有图片的固定速度。

生产采用 `disk24g_ditcache24`。另测的 `disk24g_phase24` 仅在去噪阶段使用 24 个 GPU 常驻 block，等待所有 CUDA stream 完成后释放前缀再进入 VAE 解码；它通过资源和输出一致性检查，但两张耗时 33.40/34.06 秒，每图装载/释放开销抵消了去噪加速，因此未采用。独立性能身份记录实现 SHA，原冻结实现保持不变。逐项测量见 `outputs/night_cluster_v1/deployment/report.md`。

首次权重约 54.55GiB，仅传实际需要的 encoder/VAE、原配置、LoRA 和已经合并的 BF16 分块。省略未使用的原始 DiT payload，保留原预处理 manifest 作来源证据；新的 portable manifest 由主机 SHA 锚定，第二台每次启动全量校验使用的文件，不重复合并 LoRA。

资源监控属于采样保护，不是显存预约；超限只停止当前节点拥有的后端，不终止其他应用。

## 输出和维护

控制目录为 `outputs/night_cluster_v1/`：

- `cluster.json`、`ownership.json`：冻结来源和持久任务归属。
- `nodes/local/`、`nodes/gpu2/`：队列、状态、资源峰值和恢复记录。
- 本机图片继续放在 `outputs/night_place_fast_v1/images/`。
- 第二台图片校验后回收到 `outputs/night_cluster_v1/nodes/gpu2/images/`。
- `generated_manifest.jsonl`：`manifest` 命令导出的统一清单，保留两边正确路径。
- `deployment/`：首次迁移、环境、限额、性能测量与验收记录。

主机控制器需要在线才能分配新分片和回收图片；断线期间第二台可完成已领取分片，之后等待。第二台 SSH 和用户后台管理已配置为退出桌面后仍可访问。电脑重启后，在主机运行一次 `./ccr-night resume`。

旧的 `ccr-night-place-images-v1.service` 和全量 safe launcher 已加 ownership 检查，协同任务期间请统一使用新入口，避免重新遍历已分给第二台的地点。不覆盖之前训练、评估、Reliability V2 或 Prompt 实验记录。
