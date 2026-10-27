# Night：场景适配、风格覆盖与开放Prompt

当前入口是 `AdaptVPR/experiments/qwen_curriculum/run_scene_night_safe.py --pipeline balanced`，默认目录 `outputs/scene_night_balanced_v1/`。代码、CPU测试与输入预检查已完成；2026-10-26已按用户要求运行小批次，保存3张，联合候选1张，未启动训练。执行记录见 `outputs/scene_night_balanced_v1/review/execution.md`。本批累计请求预算已用完，继续实验需另建批次。

目标保持为：`You plan image-conditioned augmentation for visual place recognition.` 以及 `These are open-ended descriptions, not a fixed list of prompt templates.` 风格库定义照明机制与适用条件，最终完整英文Prompt仍由VLM针对原图撰写，不拼接v3固定模板冒充看图规划。

## 保留什么，允许改变什么

- 保留相机视角、道路拓扑、建筑/立面轮廓、门窗布局、固定地标和已有标识的形状/文字；不得为了灯光风格凭空造商店、招牌或灯杆。
- 允许合理增加、删除或改变车辆、行人，允许路面干湿变化。没有“必须保持原有人车数量”“只能新增一辆车”或精确配对框要求。
- 仍需保留可用的地点结构证据。车辆或行人遮住几乎全部地标不能因为是临时物体就自动通过。
- 所有生成记录的 `condition` 和 `weather` 均为 `night`。`surface_state` 单独记录dry/wet/mixed/unspecified，不把湿路面升级为独立rain域；本入口不开展其他天气类别生成。
- 少天空的图片照样可以生成夜景。不能只换黑天空而保留白天照明；夜晚判断关注建筑与地面的照明场、光照衰减和阴影。

VLM可以减少不适配Prompt引发的幻觉，但并不能保证结构正确。几何检查和同家族VLM审核也不是独立语义真值，不把通过率等同于VPR增益。

## 30个可扩展机制类别

定义位于 `AdaptVPR/experiments/qwen_curriculum/night_style_catalog.json`，可增加类别，里面没有完整生成Prompt。

|方向|类别ID|
|---|---|
|路灯光谱与组合|sodium_lamps、low_pressure_sodium、led_lamps、warm_led、neutral_led、mercury_vapor、metal_halide、mixed_sodium_led|
|原有建筑开口与标识|mixed_window_street、shopfronts、existing_sign_glow、residential_windows、porch_entrance|
|原有特殊空间|covered_fluorescent、industrial_security|
|低照度与时段|sparse_lamps、dimmed_late_night、blue_hour|
|车辆相关|headlights、tail_lights、passing_vehicle_lights、short_light_trails|
|地面与空气|wet_reflections、damp_matte、light_haze|
|环境与曝光|moonlit_ambient、overcast_skyglow、highlight_exposed、shadow_lifted、high_iso_documentary|

包含v3原有6个night类别。`blue_hour`要求已经呈现夜间地面照明；若结果仍像白天，夜晚审核不能通过。机制名称是控制目标，不是从白天源图推断出来的真实灯泡型号。

检索参考来自真实摄影页面与照明资料，未把搜索结果里的AI插图作为真实夜景证据，也未下载或混入训练数据：

- [James Maher：夜间街头与城市摄影实景](https://jamesmaherphotography.com/night-city-street-photography/)：参考局部光源、曝光、移动主体等不同夜间表现。
- [Joe Redski：夜间街拍示例](https://joeredski.com/blog/2021/6/8/5-cool-night-street-photography-tips)：参考可用光与路面反射。
- [Edison Tech Center：钠灯资料与照片](https://edisontechcenter.org/sodium-lamps/)及[汞灯资料](https://edisontechcenter.org/mercury-vapor-lamps/)：区分照明机制。
- [Omar Ramadan的月夜街景](https://unsplash.com/photos/full-moon-illuminates-a-city-street-at-night-Lsyw49spJY8)：自然光与街灯组合的参考；生成时不要求在画面里添加月亮。

这是工程上的类别设计，不是文献声称的完整夜景分类，也不保证Qwen能实现每个类别。

## 选择与复现

1. VLM先读取原图，给出带证据的present/absent/uncertain能力清单和实际地标。每个源图一次分析，保存原始响应与已封存结果。
2. 程序按条件排除机制。没有店面不选shopfronts，没有已有车流不选headlights/tail_lights；如果有明确可行的道路空位，可以独立选择passing_vehicle_lights，让Prompt明确引入合理车辆。没有可见路灯不代表不能使用画面外的合理照明，但不能画出新灯杆。
3. 从适用类别里按 `saved - mismatch + pending + 本批已分配数` 优先选曝光少者。同优先级由seed、源图SHA、style_id确定性打破。每图不重复选同一机制，不强塞不适用类别。
4. VLM只为选中的机制写开放描述，保持源图锚点；格式修正不重新换风格。每源图最多一次场景分析、两次组织Prompt的草稿，合计最多3次看图模型调用；不进行无界重试。继承入口的planner_requests_per_source=2指组织Prompt的read草稿次数；完整模型调用上限另记录在最终plan_config的quality.planning_model_calls_per_source中。
5. 生图后记录实际曝光，再独立审核结构、夜晚效果和风格实现。夜晚/结构合格而style_matched=false时可以是night候选，但不计为该风格成功覆盖。Unknown不冒充确认。

`saved`只是请求类别下实际保存的图片；`confirmed`是自动审核确认风格的探索统计。两者不混用。已预定但未完成的任务属于pending，避免不同源图同时把同一个稀缺类别分配很多次。

覆盖率与源图在prepare时冻结，恢复时不追随正在增长的图片库重新排序。新一批用新目录读取最新记录，并通过多个 `--coverage-state` 累计历史。当前输入预检查读取 `outputs/qwen_curriculum/state`，其中10,518条night记录缺少可映射到新类别的style_id/variant；明确记为unclassified，没有猜测类别，也没有额外调用VLM给上万张旧图重新分类。因此本批新类别从零统计，批内通过已分配数控制平衡。

默认源图来自不断增长的 `outputs/qwen_curriculum/state/results.jsonl` 中记录的**真实source_path**，不把已生成图当成真实源图。沿用训练地点排除清单，按地点去重，封存所选原图SHA；新增图片只影响下一批，不改变已准备批次。

## 小批次命令

以下使用Bash。当前无GPU预检查已完成；`--execute`才调用模型。默认2个源地点，每图2种机制，最终生图预算为4张。

```bash
cd /home/admin123/github/WORKSPACE
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
RUN=AdaptVPR/experiments/qwen_curriculum/run_scene_night_safe.py

# 不调用模型：输入、排除清单、历史风格计数快照。
"$PY" -B "$RUN" prepare --pipeline balanced \
  --coverage-state outputs/qwen_curriculum/state

# 本地Qwen3-VL：原图能力分析 → 稀缺风格分配 → 逐图Prompt。
"$PY" -B "$RUN" prepare --pipeline balanced --execute \
  --coverage-state outputs/qwen_curriculum/state

# 检查冻结的计划；不调用模型。
"$PY" -B "$RUN" check --pipeline balanced

# 最多4次请求，累计预算也生效；结束后释放Qwen，再做几何/CLIP。
"$PY" -B "$RUN" run --pipeline balanced --execute --max-calls 4

# 独立前向检查结构、night和风格实现，不检查人车数量是否与源图一致。
"$PY" -B "$RUN" audit --pipeline balanced --execute
"$PY" -B "$RUN" report --pipeline balanced
```

中断后用相同命令恢复。若生成已保存、几何检查中断，单独执行 `verify --pipeline balanced --execute`。失败预留仍占预算，不自动补足。准备阶段源图分析请求若未完整落盘，明确停止，不悄悄重发。

下一批对每条命令加同一组 `--output-dir outputs/scene_night_balanced_v2/state --image-root outputs/scene_night_balanced_v2`；prepare再同时提供 `--coverage-state outputs/qwen_curriculum/state --coverage-state outputs/scene_night_balanced_v1/state`，即可优先补足已有运行中较少的适用机制。不要在已有目录修改输入清单、seed或覆盖快照。

## 资源与输出

复用24常驻DiT block配置，不跳过任何block或Lightning去噪步。看图规划、生图、几何/CLIP和看图审核依次执行，只有一个生成请求在运行。默认44GiB整卡显存上限、20GiB所监控进程RSS上限、至少4GiB系统可用RAM，以及新增swap超过1GiB停止。采样保护不等于硬件预留。此次3张生图及几何检查实测139.52秒，显存峰值35.84GiB，系统可用内存最低14.50GiB，未观察到新增swap；不外推其他输入或分辨率。

小批次启动时发现此前生图曾因新增swap停止，因此安全入口另外为生成服务临时设置cgroup MemoryHigh=12GiB、MemoryMax=16GiB和MemorySwapMax=0；它们包含RSS之外的文件缓存，防止流式读权重把桌面匿名内存挤入swap。只限制本次拥有的生成服务，不修改全局swap设置，结束后删除临时配置。服务超过硬限仍可能被内核终止，不能把限制视作完成保证。资源日志另记 `peak_generator_cgroup_gib`，保留原有全系统内存与新增swap保护。

- `state/coverage_snapshot.json`：历史计数快照与文件摘要。
- `state/source_analysis/`：逐图场景能力、被排除类别与选择结果。
- `state/scene_plans/` / `state/plan.jsonl`：真实VLM原文、最终Prompt、style_id、源图与seed。
- `state/quality_results.jsonl` / `state/audit/`：几何/夜晚/结构与风格证据。
- `review/report.md` / `review/summary.json`：逐机制适用源图数、planned、saved、pending、confirmed、mismatch；两列原图/生成图图版。
- `resource_runs/`：资源测量与执行日志。

不改写旧v3/v5或已冻结的严格物体配对实验。旧入口不带 `--pipeline balanced` 时保留原行为。本入口不启动训练，探索候选不自动混入原训练manifest。

已运行31项CPU测试，包含旧策略回归，以及新策略的场景排除、稀缺调度、允许人车/干湿变化、night限制、两阶段规划恢复、模拟HTTP生图恢复、审核与报告链路。模拟链路不代表Qwen实际风格实现能力，仍需上述显式小样本执行确认。
