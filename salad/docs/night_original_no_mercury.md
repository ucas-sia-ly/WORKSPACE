# 恢复旧 Prompt：双机续生图，全部排除汞灯

2026-11-10 用户看过修订图后要求停止新方案，恢复之前的Prompt，使用两台机器生成除汞灯以外的图片。

## 执行范围

- 使用`night_place_fast_v1`已经冻结的原始逐图Prompt，原图、sample_id、seed、灯光类别均逐项核对，保持不变；不调用VLM，不重新写Prompt。
- 原计划61,984个地点，排除所有2,066个汞灯地点，其余共59,918个。
- 保留已完成的46,712张非汞灯图，仅续生成剩余13,206张，覆盖其余29类。续生成清单中每张旧图的请求次数均为0，不会重复提交旧的不明确请求。
- 汞灯地点本轮直接排除，不改配其他29类。
- 新修订方案的31张已生成图和Prompt日志保留在原独立目录，已停用，不混入当前清单。新的VLM与修订生图控制器均保持停止。

旧混合分片中包含已删除的汞灯记录，不能直接恢复整个旧分片。因此新建`outputs/night_original_no_mercury_v1/`，仅承接旧计划中从未请求过的非汞灯任务；原计划和旧日志不改写。

## 运行命令

```bash
cd /home/admin123/github/WORKSPACE
./ccr-night status
./ccr-night pause
./ccr-night resume
./ccr-night pause --node local
./ccr-night resume --node local
./ccr-night pause --node gpu2
./ccr-night resume --node gpu2
./ccr-night sync
./ccr-night manifest

journalctl --user -u ccr-night-original-images-controller.service -f
journalctl --user -u ccr-night-original-local-worker.service -f
ssh ccr-gpu2 journalctl --user -u ccr-night-original-gpu2-worker.service -f
```

两台现在均可生图，不再等待新Prompt全部完成；本机仍会等待停止中的VLM完全退出，以及资源稳定可用。每台最多一个待完成分片，先持久保存归属再发布队列，断线不会转交当前任务。pause完成当前图片的提交后释放该节点后端；resume复用已有提交及raw恢复记录，不重复POST。

## 保存与资源

最终标准图、原始大图和sidecar全部保存在第一主机：

- 第一主机生成：`outputs/night_original_no_mercury_v1/images/night/`及`images/raw/night/`。
- 第二主机回传：`outputs/night_original_no_mercury_v1/nodes/gpu2/images/night/`及该节点`images/raw/night/`。
- 原46,712张仍在原路径，不搬移、不重生成。`./ccr-night manifest`合并这些原结果与本轮新结果，并拒绝汞灯或重复sample_id。

第二主机仅临时保存新图；整个分片提交且第一主机校验PNG、尺寸、SHA、sidecar和请求预留关系后，再发确认清理第二主机副本。回传暂存与正式文件通过硬链接发布，避免重复占空间。

保持已经验证的BF16、4 steps、source-aspect画布与两台设备各自的模型放置方案。第一主机为48GB显存/32GB系统内存，生图后端16GiB硬内存限制及swap禁用保留；第二主机24GB显存/64GB内存使用`disk24g_ditcache24`，52GiB硬内存限制、46GiB RSS软回收及50GiB RSS硬保护保留。

部署时第一主机可用RAM约9.5GiB，新的运行策略以8GiB稳定可用RAM准入，同时保留2GiB持续低内存保护与0.75GiB紧急保护，不结束其他应用。第一主机采用8GiB磁盘余量保护；实际余量使用`df -h`查看，空间不足时等待，不删除原结果。具体峰值以`nodes/*/telemetry.json`为准。

全部新图仍是`pending_verification`，不自动成为已审核训练数据。配置/预算清单在`cluster.json`、`prepared.json`、`feed.json`；每张任务的原Prompt对照证明在对应分片的`prompt_receipts.jsonl`。CPU验收日志在`deployment/cpu_tests.log`。


## 自动接批与队列补发

2026-11-11修复了一处队列同步中断后的等待问题：主机已持久保存第54批归属和队列，第二台仍只有前53批；旧分配器把第54批算作未完成任务，既不再分配也不补发队列。

控制器现在通过`night_original_queue_recovery.py`接入补发检查，原冻结代码与计划不改写。每次给第二台分配任务前，先比较双方密封队列；第二台必须是主机队列的完整前缀。只补发已经归属于第二台的缺失条目，先校验对应计划、Prompt证明和真实源图文件的SHA，再发布并读取确认。队列分歧、远端超前或文件缺失时拒绝覆盖；网络断线或确认丢失后下一轮重新核对，不重新分配地点或重复提交图片。

该次恢复无需重启任一worker；第二台自动接到了第54批，状态由waiting_for_work转为generating。本机正在处理的图片不受影响。恢复策略及代码SHA在`queue_delivery_policy.json`，实际补发回执在`deployment/queue_deliveries/`。60项相关CPU测试通过。
