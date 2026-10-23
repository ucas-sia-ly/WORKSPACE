# 当前 v4 生成的暂停与恢复

后台已绑定 `weather_v4_priority.py run --output-dir outputs/weather_v4/state`，默认只生成 night，完成后停止，不自动切换其他域。图片保存到 `outputs/weather_v4/night/`；原始高分辨率图及服务元数据在 `raw/night/`，账本及固定优先队列在 `state/`。

```bash
cd /home/admin123/github/WORKSPACE
conda activate AdaptVPR

# 启动或继续；关终端、重启电脑后仍用此命令
python AdaptVPR/experiments/qwen_curriculum/resume_control.py resume

# 停止生成并释放 Qwen 占用的显存、内存
python AdaptVPR/experiments/qwen_curriculum/resume_control.py pause

# 查看进度
python AdaptVPR/experiments/qwen_curriculum/resume_control.py status
tail -f outputs/weather_v4/state/coordinator.log
```

优先生成旧 `qwen_curriculum` 和 `weather_v3` 实际未生成过的 place，再生成已覆盖的 place；旧计划中只有排期、没有保存生成图的 place 仍视为未生成。统计包含旧生成失败审核的已保存图，不仅包含 accepted。优先队列已密封，resume 不重新抽样或重排。

v4 原有 20 张保留，已保存任务不重复调用；完整 raw 中断时先恢复补审核，无完整输出的中断请求在显式 resume 后使用原参数重试。pause 同时停止协调器和 Qwen 服务。

暂停后可检查完整性，不加载模型或调用 Qwen：

```bash
python AdaptVPR/experiments/qwen_curriculum/weather_v4_priority.py check
```

新图先作为候选，几何/天气通过不等于可训练，仍须独立内容审核。详见 [WEATHER_V4.md](WEATHER_V4.md)。
