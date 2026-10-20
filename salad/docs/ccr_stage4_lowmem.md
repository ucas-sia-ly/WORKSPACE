# Stage 4：32 GiB 主机内存恢复方案

旧实验 controls/A 和 global-screen/A 因 MemAvailable 低于 3 GiB 被停止；
日志中最低分别约 2.986、2.947 GiB。不是 CUDA OOM。
后续 freeze-global / confirm-b0 / local 报错是筛选尚未完成导致的依赖错误。
校准、诊断和 B4 两 batch smoke 已完成，但不能替代正式筛选。

`configs/ccr_stage4_lowmem.json` 使用独立目录 `ccr_stage4_lowmem_v2`。
训练与评估 DataLoader 使用 0 workers，避免 worker 数据集副本及多 batch
预取占用；代价是 CPU 解码可能使 GPU 等待。保持继承的 P/LR/block、种子、
预算和 3 GiB 内存保护。不能保证在其他应用持续占用内存时仍能完成训练。
运行前建议保存工作并关闭不用的浏览器、编辑器窗口等；不自动终止其他进程。

Stage 3 原始协议不修改；Stage 4 的 worker 设置写入新配置/计划/训练配置，
恢复检查使用同一设置。worker 数变化可能改变随机增强流，因此新目录的所有
A/B 组统一使用该设置，不能将旧 worker=4 的 checkpoint 续接到新实验。
旧结果保留，不改写旧计划的源码哈希；旧目录仅继续用于历史报告。
新目录重新校准及 smoke，不把旧 smoke 当作新协议的验收证据。

以下是 **fish** 命令；每块中用 `and` 保证上一步失败即停止该块。

```fish
set PY /home/admin123/miniconda3/envs/AdaptVPR/bin/python
set RUN salad/run_ccr_stage4.py
set CFG salad/configs/ccr_stage4_lowmem.json

# 只预检查和冻结，不训练
"$PY" "$RUN" --config "$CFG"
and "$PY" "$RUN" --config "$CFG" --phase freeze

# 先检查新运行配置是否可以工作；两 batch 不保证整轮内存安全
"$PY" "$RUN" --config "$CFG" --phase smoke --method A --execute

# 上述成功后，再逐块执行。不要一口气粘贴所有块。
"$PY" "$RUN" --config "$CFG" --phase controls --execute
and "$PY" "$RUN" --config "$CFG" --phase global-screen --execute
and "$PY" "$RUN" --config "$CFG" --phase freeze-global

"$PY" "$RUN" --config "$CFG" --phase calibrate --execute
and "$PY" "$RUN" --config "$CFG" --phase diagnose --execute
and "$PY" "$RUN" --config "$CFG" --phase smoke --method B4 --execute

# 用户决定启动正式训练；只有筛选成功后执行
"$PY" "$RUN" --config "$CFG" --phase confirm-b0 --execute
and "$PY" "$RUN" --config "$CFG" --phase local --method B-Feature --seed 43 --execute

"$PY" "$RUN" --config "$CFG" --phase report
```

中断后重跑**同一条命令**，从最后一个完整 epoch 恢复；若尚未保存任何完整
epoch，将重新开始该组。不要手工把下一阶段当作恢复命令。
报告的 `pilot_progress` 列出资源停止、未完成、已审计的运行；
`pilot_resource_stopped` 与 `formal_partial` 不代表实验方法已经失败。

历史报告刷新（不训练、不修改旧 checkpoint）：

```fish
"$PY" "$RUN" --phase report
```
