# 23 城市生成：暂停、断电恢复和中断任务重试

长时间生成使用入口 `resume_campaign.py`，保留原来的 Qwen 模型、画布、4 步 BF16 参数、天气分配、随机种子及原阶段计划和账本。后台服务 `qwen-curriculum-23city.service` 的恢复覆盖配置指向该入口，无需设置开机自启。

**审核版本与恢复限制：** 当前规则 v2 使用最少16个匹配点，保留内点率、homography 位移及天气审核；已有23城市任务冻结的是旧规则，尚未迁移。规则和源码均有冻结校验，更新 `quality.py` 后不能直接按下列命令恢复旧任务。切换到当前实现需要独立新运行或显式迁移并保留原始记录，不能覆盖旧配置、签名结果或账本。以下操作说明适用于原实现及配置通过校验的任务。

## 暂停并释放电脑

```bash
cd /home/admin123/github/WORKSPACE
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/resume_control.py pause
```

命令先停止生成协调器及其子进程，再同步停止独立的 Qwen 服务。只有两者均为 `inactive` 或 `failed` 且主进程 PID 为 0，才报告资源已释放。Qwen 常驻模型也会退出，方便把显存和内存用于其他工作。不要使用 `kill -STOP`/`SIGSTOP` 暂停进程：挂起会保留显存和内存。

停止时正在生成的那张也会被纳入下次恢复。已经完成的图，包括审核不通过但已完成审核的图，都不会再次生成。原始输出已保存但审核未完成时，先恢复图片并补审核；没有可用输出的中断请求在独立重试账本中重新请求，不删除或改写原来的失败记录。

## 继续生成

```bash
cd /home/admin123/github/WORKSPACE
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/resume_control.py resume
```

关闭终端后服务继续运行。正常暂停、终端中断、进程被杀或电脑断电后，都使用同一命令恢复；无需重新选择源图、重新准备计划或删除 `attempts.jsonl`/`results.jsonl`。任务已经运行时，此命令只报告状态，不启动第二份任务。

重复失败时控制器会停止并保留错误，避免无限消耗生成时间。修复资源或服务问题后继续恢复；查看 `summary.json`、恢复账本和 `coordinator.log` 定位原因。磁盘空间仍按剩余图数检查，空间不足时先扩容或释放空间。

## 查看进度

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/resume_control.py status

tail -f outputs/qwen_curriculum/generation_23city/coordinator.log
```

`status` 只读，不启动模型。暂停命令释放的是上述两个生成服务的资源；其他训练或应用进程不受影响。

默认输出目录是 `outputs/qwen_curriculum/generation_23city`。如果已有计划位于其他磁盘，先让服务处于停止状态，再为该路径安装对应覆盖配置，并在控制命令中使用同一个 `--output-dir`；不要搬动已冻结计划中引用的图片或日志。不要同时运行旧 `all_city_campaign.py run`、新前台入口和后台服务。

全部图片生成及审核完成后，仍按 [全部 23 城市训练说明](ALL_23_CITIES.md#3-全23城训练命令)执行原始 SALAD 的微调和六域 SVOX 测试。
