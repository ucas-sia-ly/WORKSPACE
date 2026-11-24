# 原版夜景续跑与汞灯替换

2026-11-13 的独立续跑入口：`./ccr-night`。不修改既有计划、图片、源图或历史模块的 SHA。

当前范围为原来的 61,984 个地点。保留 46,712 张历史非汞灯图片，接着完成原版非汞灯队列 13,206 张，再为原汞灯的 2,066 个源图生成替代版本。最后一项包括已生成后删除的 1,611 张以及此前未生成的 455 张。

替代任务复用已保存的原版源图分析，使用 `night_style_catalog.json` 的原版 29 类（排除 mercury_vapor）、原版 V4 composer 和 Qwen3-VL-4B，按可用风格及覆盖量分配。保留原版允许的合理人车和地面干湿变化，保持永久地点结构。没有采用试验版 V3 Prompt。

## 运行次序

第二台立即继续旧队列，第一台先完成替代 Prompt（batch 64）。Prompt 完成且 VLM 退出后第一台加入旧队列；各机器完成已分配的旧任务后，释放旧 worker/backend，再接独立的替代队列。已完成的 Prompt、图片不重新生成；中断的文本请求从已保存响应恢复，缺失响应可重算。图片 POST 仍使用原有请求保留与 raw 恢复协议，不盲目重复提交。

任务分发根据 SSH 验证的远端完成账本和持久化的任务归属推进，不等待大文件回传。回传独立运行，支持 rsync partial 恢复；全部图片和 sidecar 校验后，才确认第一台提交并清除第二台副本。第二台仍未回传的数量与第一台已收齐数量不是同一个统计。

## 常用命令

在项目根目录运行：

```bash
./ccr-night status
./ccr-night pause                     # 两台：完成当前图片/Prompt batch 后释放模型
./ccr-night resume                    # 两台：续跑，含尚未完成的 Prompt
./ccr-night pause --node local         # 仅第一台
./ccr-night resume --node local
./ccr-night pause --node gpu2          # 仅第二台
./ccr-night resume --node gpu2
```

暂停生成不会停止后台回传。恢复不重建已分配分片，不重做已提交图片。Prompt 暂停等待当前 batch 完成，图片暂停等待当前单图完成；不是冻结进程占着模型内存。

```bash
journalctl --user -fu ccr-night-completion.service
journalctl --user -fu ccr-night-mercury-prompts.service
journalctl --user -fu ccr-night-collect-original.service
ssh ccr-gpu2 'journalctl --user -fu ccr-night-original-gpu2-worker.service'
```

全部回传完成后可导出合并清单（仅校验文件完整性，不代表质量审核通过）：

```bash
./ccr-night manifest
```

## 输出和资源

- 原队列：`outputs/night_original_no_mercury_v1`。
- 汞灯替代：`outputs/night_mercury_replacement_v1`。
- 接续状态与合并清单：`outputs/night_generation_completion_v1`。
- 第一台沿用原 48GB BF16 生图配置、44GiB GPU 上限；Prompt allocator 42GiB，进程系统内存硬上限 16GiB，不与第一台生图并行。
- 第二台沿用原 24GB BF16 磁盘卸载配置，GPU 上限 22GiB、进程 RSS 上限 50GiB（该机器为 64GB RAM）。
- 两台生图仍保留已有的运行时压力检测与释放缓存后重试，不改变采样精度和步数。

第二台当前 SSH 地址为 `192.168.8.107`，沿用 `ccr-gpu2` 主机密钥严格验证。新 Wi-Fi 下该机器信号曾为 -78dBm；大文件传输重传明显。分离回传可以避免 GPU 因此闲置，但不能提高无线链路本身的速度。若条件允许，第二台改用网线或更稳定的 Wi-Fi 可以明显缩短最终收齐时间。
