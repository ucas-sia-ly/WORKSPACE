# 修订 Night Prompt 的双机生图

本方案已于2026-11-10按用户要求暂停并弃用。当前`./ccr-night`转向[旧Prompt非汞灯续生成](night_original_no_mercury.md)，以下保留为历史记录。现有修订图不删除，但不计入当前结果。

2026-11-10 用户授权：第二主机立即使用已提交的新Prompt生图；第一主机继续完成全部Prompt，完整导出且VLM进程退出后自动加入。两边沿用同一份逐图Prompt、seed、4-step BF16和source-aspect画布，不恢复已退役的旧图片队列。

`outputs/night_revision_images_v1/` 为独立生图控制目录。调度器每30秒读取修订计划中已经原子提交、逐条索引一致的ready结果，只发布这些原始Prompt。不把failed/Unknown改成ready。每台最多一个待完成分片，先持久登记ownership，再传源图和配置，最后发布队列；断线不转交已有任务。任务完成后按实际速度领取下一片。

第一主机的准入同时检查：`prompts_complete.json`内容与SHA有效、所有Prompt已完整导出、`ccr-night-revision-prompts-v2.service`的MainPID为0，再由本机资源检查确认显存与内存稳定可用。不因VLM失败、暂停或只完成部分文本而启动本机生图。

## 保存位置

最终标准尺寸图、原始大图和sidecar全部保存在第一主机：

- 第一主机生成：`outputs/night_revision_images_v1/images/night/`，原始图在同级`images/raw/night/`。
- 第二主机回传：`outputs/night_revision_images_v1/nodes/gpu2/images/night/`，原始图在该节点`images/raw/night/`。
- `incoming/gpu2/`是回传校验暂存；发布使用硬链接，同一图不会因暂存与正式路径多占一份空间。

第二主机仅临时保存正在生成或未确认回传的图。整个分片完成、主机校验PNG/尺寸/SHA/sidecar/请求预留关系并落盘后，才向第二主机发送精确文件清单确认。第二主机再次核对归属、完成记录、路径及SHA，再删除其图片副本；源图和历史请求日志保留。传输失败不会删除第二主机唯一副本。

所有新图仍为`pending_verification`，不会自动当作合格训练数据。

## 日常命令

```bash
cd /home/admin123/github/WORKSPACE
./ccr-night status
./ccr-night pause --node gpu2
./ccr-night resume --node gpu2
./ccr-night pause --node local
./ccr-night resume --node local
./ccr-night pause                 # 两边完成当前图后释放后端
./ccr-night resume                # 第二主机继续；第一主机仍受Prompt完成门槛约束
./ccr-night sync
./ccr-night manifest

journalctl --user -u ccr-night-revision-images-controller.service -f
ssh ccr-gpu2 journalctl --user -u ccr-night-revision-gpu2-worker.service -f
```

主机统一入口检测新协议后转向新队列。旧`night_cluster_v1`计划、结果和修订锁不改写。退出RustDesk不影响SSH或这两个后台服务。

## 资源与恢复

第一主机保持48GB显存/32GB系统内存配置，生成后端MemoryHigh=12G、MemoryMax=16G、不用swap；启动要求42GiB空闲显存及10GiB可用RAM。第二主机实际为24GB显存/64GB内存，使用已经验证的`disk24g_ditcache24`，不缓存encoder/VAE，不减少采样步数。

第二主机后端既有52G硬内存限制保留，新worker在RSS到46GiB时要求完成当前图后回收模型，50GiB为RSS硬保护，避免旧队列运行数天后因缓存增长到48GiB突然中止。仅资源等待可在冷却及稳定检查后自动恢复；身份、文件SHA或不明确POST错误保持停止，不盲目重复请求。

第一主机启动时磁盘仅余约50GiB。新任务沿用20GiB磁盘余量保护；空间不足时暂停分配新分片，不删除已有结果，也不偷偷改成长期保存在第二主机。继续完成全量所需空间取决于实际PNG大小，可用`df -h`查看。

运行记录：`cluster.json`为冻结执行协议，`feed.json`为追加的已通过Prompt清单，`ownership.json`为持久归属，`nodes/*/queue.json`为节点队列，`status.json`为汇总状态，`generated_manifest.jsonl`为按需导出的新图清单。CPU验收日志在`deployment/cpu_tests.log`。
