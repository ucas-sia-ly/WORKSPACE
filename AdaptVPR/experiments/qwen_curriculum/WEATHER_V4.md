# v4：优先覆盖此前未生成的地点

使用既有 `prompt_bank_v4.py`：保持已有物体和文字，仅改变光照天气；night 要求干燥道路，不因提示词新增车辆或灯光。Qwen 模型、采样参数、分辨率和六域比例沿用当前 v4 固定计划。只优先执行 night，完成后停止。

`outputs/weather_v4/` 为独立数据根，`night/` 等目录放训练尺寸图，`raw/<域>/` 放高分辨率图与侧车；`state/` 保存固定计划和恢复账本。原有 20 张 v4 及两个旧库均保留。

新入口 `weather_v4_priority.py` 通过 v4 私有生成引擎复用预约、持久化、恢复及服务释放机制。基础 `plan.jsonl`、提示词、seed、配额、样本 ID 及旧审核均未修改；只按冻结的历史 place 集合稳定分组：未生成优先，已生成随后，各组保留原随机顺序。新队列由 `state/priority_config.json` 和 `previously_generated_places.jsonl` 密封，历史目录后续变化不会悄悄重排本轮。

历史实际保存记录为旧库 21077 张和 v3 3858 张，合并去重后 17908 个 place。v4 合格计划共 62499 个 place，其中 44602 个此前未生成。night 的 27499 个任务中，18038 个此前未生成，优先执行；已有 v4 保存记录会跳过生成。旧图审核是否通过不影响“是否已经生成”的统计；单纯排过计划、没有保存图片不算已生成。

生成的几何/天气通过图仍是候选，不能直接视为已完成内容审核；使用独立内容审核后导出的训练清单。现有 `night_v4_completion_v3` 是已冻结的 20 张等输入实验，不会自动纳入新增图片，需要另开审核快照。

后台控制命令见 [RESUME.md](RESUME.md)。已完成部署，当前保持暂停。普通 resume/pause 无需再次安装；如果以后主动切换后台到其他任务，要先停服务再重新绑定：

```bash
python AdaptVPR/experiments/qwen_curriculum/weather_v4_priority.py install
```

本次未启动生成、训练或测试集评估。
