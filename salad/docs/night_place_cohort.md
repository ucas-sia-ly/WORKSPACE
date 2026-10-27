# 全量地点：先确定全部night Prompt，再单独生图

每个拥有至少4张**不同且实际存在**真实图的训练地点，稳定选择1张真实源图，生成1个night Prompt。不是从旧生成图记录抽地点，也不是每个地点生成多种天气或多张图。

保留30类照明机制、场景适配排除、稀缺机制优先、VLM逐图开放描述；保留永久地点结构，允许合理的人车增删与干湿变化。只启用night。源图选择不把Sobel、纹理或低亮度冒充建筑语义真值，也不使用SVOX test错例选择源图。

本次索引为62,374个地点，排除140个校准/验证地点。v1短程检查暴露了坐标字段别名问题，已停止并保留原始响应；v2检查又发现模型会照抄JSON示例占位词；v3有限检查仍频繁缺少框。沿用v4 Prompt策略：当前任务只需Prompt，不要求VLM框选地标；整图框仅作为永久结构保护约定，明确记录origin=deterministic_preservation_contract、usable_as_correspondence_or_semantic_evidence=false。这不是检测结果、语义掩码或局部蒸馏监督，仍允许人车和干湿变化。可见照明条件仍必须由VLM逐图提供；格式失败最多重新看图纠正1次，仍然失败则单列跳过。v1/v2/v3检查记录均保留，不作为最终队列。

默认目录：`outputs/night_place_fast_v1/`。沿用真实训练地点排除清单；CSV重复行不增加图片数，缺失文件不计数。按seed/city/place_id/真实文件名稳定选源图，选中图不可解码时尝试该地点下一个稳定候选，不在地点间冒充精确配对。

## 执行阶段

1. **CPU索引**：扫描完整GSV-Cities地点池、检查文件、每地点选1张，冻结源图SHA、输入清单、软件版本、模型与代码摘要。4个CPU读取线程；不调用GPU。
2. **全量Prompt**：看图模型只加载一次。每组64张，先批量读取真实场景证据，再给每图分配适用且较少的机制，批量生成完整英文Prompt。分析是greedy；组织Prompt保留temperature=0.8/top_p=0.9/top_k=30。每源图最多2次分析（第2次仅纠正第1次格式失败）、2次Prompt草稿；不同阶段的token上限分别为1400/900，单图处理尺寸仍最多672像素边长。只缓存文本记录，不缓存教师特征。
3. **导出**：所有选中源图均有终态后，汇出成功Prompt及每128条一份的冻结生图计划。失败/Unknown单列，不用模板伪造结果，也不升级成可靠性Negative。未完成全量Prompt时禁止进入生图。
4. **生图**：需要使用者另外启动；本次不自动运行。

微批次的成员、seed与响应一起封存，左padding并按统一输入长度截取生成tokens，防止图片/响应错位。每组结果先原子提交，再写每地点记录；中断后修复缺失记录而不重复调用已提交批次。本地Prompt任务的未完成调用按下述恢复规则重新处理；生图HTTP请求仍使用独立预算日志。未出现有效Prompt的前16个地点会触发停止，避免模型/格式异常浪费全量GPU时间。

分配时维护30类计数，不逐图重新读取或遍历整个历史；生成计划按128条分片，避免每保存一张都扫描数万条记录。所有Prompt准备完毕后才切换到Qwen-Image生图服务，可以省去交替加载的开销；不会降低模型本身的逐图计算成本。

## 当前后台任务

Prompt任务：`ccr-night-place-prompts-fast-v1.service`，只执行prepare。

```bash
cd /home/admin123/github/WORKSPACE
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
RUN=AdaptVPR/experiments/qwen_curriculum/run_scene_night_safe.py

# 实时进度、成功/失败数量、风格分配、粗略剩余时间。
"$PY" -B "$RUN" check --pipeline places
journalctl --user -u ccr-night-place-prompts-fast-v1.service -f

# 停止；之后重新start会读取已有缓存继续。
systemctl --user stop ccr-night-place-prompts-fast-v1.service
systemctl --user start ccr-night-place-prompts-fast-v1.service
```

资源配置：Prompt服务MemoryHigh=12GiB、MemoryMax=16GiB、MemorySwapMax=0、CPUQuota=4核；此外沿用44GiB整卡显存、20GiB所监控进程RSS、至少4GiB可用RAM、任务自身swap为0的监控（全系统swap增长仅记录）。不改全局swap策略，不启动训练，不启用开机自启动。OOM或其他进程带来的资源压力仍可能停止任务；不把限制当作完成保证。

## 输出

- `state/sources.jsonl`：每个合格地点唯一的真实源图，记录真实图数量与地点完整文件名池摘要。
- `state/cohort_config.json`、`coverage_snapshot.json`：冻结协议和历史计数。
- `state/batches/`：真实批量VLM请求预留和原始响应。
- `state/groups/`：原子批次提交。
- `state/prompts/`：每地点Prompt或明确失败原因。
- `state/progress.json`：可随时读取的状态，不是完成标记。
- `state/night_prompts.jsonl`：全量处理结束后汇出的成功Prompt。
- `state/prompts_complete.json`：全量Prompt处理完成屏障，记录成功/失败及生图分片。
- `shards/s*/state/`：可恢复的生图计划，中心图片目录为 `images/`。

全量完成后的失败地点不会被悄悄删除，完成标记明确区分selected、ready与failed。再次新增真实图片或改变规则时创建新cohort，不修改已冻结的源图选择。


## 48GB显存 / 32GB内存的吞吐配置

2026-10-26已实测4/8/16/32/64张批次，模型、BF16、672像素边长上限、1400/900输出token上限和Prompt策略均未缩减。以下是短程代表性测量，包含源图分析和Prompt组织，不是长期吞吐保证或图像质量验证。

| 每批图片数 | 每秒图片数 | PyTorch显存保留峰值GiB | 测量结束整卡显存GiB |
|---|---:|---:|---:|
| 4 | 0.229 | 9.90 | 11.18 |
| 8 | 0.325 | 11.38 | 12.65 |
| 16 | 0.403 | 14.83 | 16.14 |
| 32 | 0.480 | 21.59 | 22.89 |
| 64 | 0.517 | 36.10 | 37.40 |

选择64张一组。另用当前最大640×480输入集中进行64张压力检查（仅性能样本会重复真实输入，不导出为地点Prompt）：所有分析/Prompt均通过格式检查，PyTorch保留峰值39.09GiB、整卡约40.40GiB、进程RSS约2.06GiB、系统可用内存约16.47GiB；没有新增swap或保护中止。

新任务增加42GiB PyTorch allocator上限，保留44GiB整卡采样保护、至少4GiB系统可用内存、16GiB服务内存硬限及禁止服务swap。显存上限不保证其他程序不会抢占资源；触发保护会暂停，不反复OOM重试。

新协议`night_place_fast_v1`原样导入旧v4已提交的88条源图、Prompt和job，保留原始receipt SHA；旧任务与原始记录不覆盖。导入前缀仍按原4张分组恢复；其余地点固定64张一组。更改批次会改变未来随机采样轨迹，不声称与4张批次逐字相同。所有新批次的成员、seed和原始响应仍封存。中断时未提交的旧调用留档，可能在新协议下重新处理，不影响88条已完成Prompt。

性能证据：`outputs/night_prompt_speed_selection_v2.json`；迁移屏障：`outputs/night_place_fast_v1/state/migration_complete.json`。旧v4任务已停止，新任务不启用开机自启动。49项CPU测试涵盖迁移、分组、配置选择、恢复和已有采样行为。


## 中断后的Prompt恢复

本地VLM Prompt准备使用`resume_night_prompts.py`接入，原systemd启动命令保持不变。

- 有完整且校验通过的原始响应：直接复用，继续校验和提交，不重复前向。
- 只有request预留、没有完整响应：原request移入`state/prompt_recovery/<batch-key>/`留档，使用相同输入、批次成员和seed重新生成；不会把整批地点记为失败。
- 批次已提交、单地点记录尚未写全：从原子批次提交补齐。
- 已完成Prompt保持原样；缓存损坏或输入不一致时暂停报错，不转换为地点失败。
- 只有需要新推理时才加载VLM；保存的响应可以先在CPU上完成恢复提交。

`state/prompt_recovery_policy.json`记录恢复代码SHA和原cohort摘要，不修改原实验配置。`state/prompt_recovery/`保留每次未完成请求及重试事件。本策略只处理本地Prompt推理，不改变生图请求的预算规则，也不自动重试已经完成的质量/格式拒绝记录。

运行中临时暂停/恢复（保留RAM和显存）可用freeze/thaw；进程已退出则用start，读取已保存进度继续：

```bash
systemctl --user start ccr-night-place-prompts-fast-v1.service
```

退出时未保存的推理会重做；最终文字不承诺GPU跨进程逐位一致，所有恢复结果都会保存原始响应。当前资源保护触发的停止同样适用上述恢复规则。


## 启动报Generator already running时

2026-10-27排查确认，重启电脑后`adaptvpr-lightx2v.service`自动启动，而Prompt任务要求生图服务和生成协调器都停止。该服务的HTTP进程可以处于空闲状态；低显存占用或health返回model_loaded不能单独证明没有生成请求。

本次已结合启动后日志无生图请求、无端口连接、生成协调器停止的证据，停止空闲生图服务并取消它的开机自启（保留unit，后续生图入口仍能手动启动），再恢复Prompt任务。没有中断其他训练或启动生图。

如再次遇到启动失败，查看本次启动的日志：

```bash
systemctl --user status ccr-night-place-prompts-fast-v1.service --no-pager
journalctl --user -u ccr-night-place-prompts-fast-v1.service -n 30 --no-pager
```

不要在已有生图任务时直接强行停止生成服务。Prompt服务保持原命令和原目录，启动成功后仍从已保存的响应和提交继续。


## 全系统swap增长导致误停的修复

2026-10-27 08:45任务再次因全系统swap增长约1.19GiB而停止；该轮最低系统可用RAM仍约13.08GiB，进程RSS峰值约4.61GiB、显存峰值约43.09GiB，均未触及相应保护线。全系统swap不能归因到这个Prompt任务。

当前Prompt unit显式使用`--swap-scope task --max-task-swap-gib 0`：读取专属systemd service cgroup的`memory.swap.current`，全系统swap变化只记日志。启动前校验专属cgroup、有限`memory.max`和`memory.swap.max=0`；无法获取自身swap账目时停止，不静默关闭监控。

保留MemoryMax=16GiB、MemorySwapMax=0、PyTorch allocator上限42GiB、44GiB整卡保护和至少4GiB系统可用RAM。其他程序造成实际RAM不足或整卡显存超限时依然停止。其他旧入口默认仍使用原系统swap策略。

每次启动的实际策略保存于`resource_runs/<run-id>/guard_policy.json`。退出报告另记`peak_task_swap_gib`、task_cgroup与swap_scope，系统swap仍可用于诊断。配置备份及变更理由保存在`resource_guard_repairs/`。60项CPU测试包含全系统swap增长但任务swap为0、真实内存/显存超限、读取失败、共享cgroup拒绝与既有恢复逻辑。

## 将来生图时

必须先有`prompts_complete.json`。例如先试生32张：

```bash
"$PY" -B "$RUN" run --pipeline places --execute --max-calls 32
```

此阶段只生图，后端在分片之间保持加载，结束后释放；不在同一时间加载几何/CLIP或VLM审核模型。之后按导出分片，使用`--pipeline balanced --output-dir <分片state路径>`分别verify/audit/report。每地点一次请求的预算与总调用预算均生效，失败请求不自动追加。全量图像与raw证据可能占用上百GB磁盘，启动前检查所请求规模的空间。
