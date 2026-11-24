# 双机拒绝夜景修复

## 当前停止范围：只完成第一轮（最新指示）

用户要求第一轮彻底结束后停止，后续由Codex逐张查看仍Reject图片。本轮主控制器为 `ccr-night-repair-v6.service`，入口 `night_repair_round_one.py`；独立协议 `round_one_stop_protocol_v1.json`。这里只完成原源图第0轮所有Prompt、生图和审核，不自动领取第1/2轮同地点替代源图。下文较早的自动换源策略不再自动执行。

完成时写 `round_one_finished.json`，阶段为 `stopped_after_round_one`，不把整个修复研究标为完成。后续普通resume也不会越过边界。第一轮仍按“全部Prompt→集中双机生图→集中审核”；Pass/Unknown接受、Reject保留。

自动导出 `outputs/night_repair_v1/round_one_review/`：

- `reject_places.jsonl` / `reject_places.tsv`：全部仍Reject地点、精确原图/旧图/新图、原Prompt/修订Prompt、原拒绝证据和新审核结果。
- `planning_failures.jsonl`：未成功生图的文本/源图问题，单独保留。
- `summary.json`：真实计数与清单哈希。

不会现在启动Codex逐图复核或下一次生图。下一阶段由用户另行安排：对于太阳光影难处理的源图考虑同地点换源，其他问题按实际图像小改Prompt；审核阈值和原始标签保持不变，不保证每张图必然通过。

## 当前阶段调度（2026-11-17更新）

按用户要求，主控制器 `ccr-night-repair-v5.service` 使用 `night_repair_staged.py`：**本轮全部Prompt → 双机集中生图 → 集中审核 → 仅Reject进入下一轮**。原源图为第0轮，最多两轮同地点换源仍需等上一轮审核后才能确定，不提前生成无必要的替代图。

Prompt阶段暂停图片请求派发，两台不会一边生图一边插入新Prompt批次。现有Prompt、冻结计划、已保存图片及请求预留均复用。阶段进度保存在 `staged_state.json`，调度协议为 `staged_protocol_v1.json`，`./ccr-night-repair status` 包含当前阶段。暂停/恢复命令不变；恢复Prompt阶段不会提前重启远程生图。Prompt阶段第二台生图GPU暂时空闲是此分阶段策略的预期行为。

图片阶段不插入Prompt或审核；两台集中完成本轮所有已规划图片，再由第一台统一执行原规则审核。无需改变审核模型、阈值、Prompt规则、采样参数或每地点尝试上限。

## 当前接受规则（2026-11-17更新）

按用户最新指示，**Pass 与 Unknown 均接受，只有 Reject 触发同地点换源生成**。原始检测标签、分数、校准和阈值不变；Unknown只是按用户规则被接受，不改写成检测器Pass。原文中“所有未通过都换源”的描述是旧策略，以本节为准。

使用 `night_repair_acceptance.py` 与独立 `acceptance_protocol_v1.json`；主控制器为 `ccr-night-repair-v4.service`。已生成且判Unknown的图片从待重试队列移出，停止该地点后续生图。取消类别隔离后仍Unknown的440张原图也保留。此前原始DINO审核的15,247张Unknown无需重生成，连同上述440张导出到 `accepted_original_unknowns.jsonl`。新图接受清单继续使用 `accepted_replacements.jsonl`，附 `raw_audit_decision`、`accepted_unknown` 与接受策略指纹；不覆盖历史审核文件，不自动改动训练准入标记。

Prompt格式错误是“未生图的文本规划失败”，不能作为已通过图片；它与图片审核Unknown分开统计。每地点最多两张不同替代源图的上限不变。

只处理第一轮8,546对Reject和第二轮5,242对Reject，共13,788个互不重叠地点，不纳入原有15,247个Unknown。用户已取消shopfronts整类隔离；原冻结审核记录保留，本轮调用同一校准分类器并显式传入空隔离列表，不修改阈值、权重或Sinkhorn。

其中1,924对仅因类别被拒，先实际补查DINO内容：1,421 Pass不重生成、63 Reject进入修复、440 Unknown单列不重生成。实际需修复11,927个地点。

## 执行

Qwen3-VL-4B读取真实源图、原失败生成图、原Prompt和规则证据，检查矛盾、白天/黄昏措辞、常识和光源依据，针对性修正Prompt。原图不变先生成一次，重新执行LightGlue/CLIP和DINO联合规则；只要未通过，换同city/place_id的另一张不同真实图，Qwen看新图重新写Prompt、生成、复查。替代源图最多两张，始终排除已尝试路径及相同图片SHA256。每个地点至多3张新图；不能找到适合源图时不伪造结果。

新源图不继承旧图的语义证据、局部索引或mask。DINO异常/窗口不是语义幻觉真值。shopfronts可用，但实际源图须有相应店铺玻璃，不靠类别名绕过内容规则。

第一台48GB GPU负责Qwen Prompt、规则审核及自己的生图批次；第二台24GB GPU并行使用原有24GB磁盘卸载生图后端。两台只领取不同批次，地点计数与重试次数统一由第一台SQLite管理。第二台不用额外下载8GB VLM模型；第一台在第二台生图时继续规划、审核或生图。每台首批4张，后续批次64张。VLM与生图模型在第一台分阶段运行。

### Prompt 最终复核修订

首批实测发现，成对读取源图和失败生成图时，4B VLM有时混淆两者，还会保留“保持白天/亮蓝天空”或“金属门发光”等错误。新增 `night_repair_prompt_gate.py`，在成对诊断后，只给Qwen看真实源图，复核并修订最终Prompt。输出采用简单的三个字符串字段，减少复杂JSON失效；对明确要求白天、黄昏、实体门发光或拒绝夜景的文本再做有限重试。此文本检查不能保证图像没有语义错误，生成后仍执行原有两轮规则。

原 `protocol.json`、8张首批生成结果和已经派发的计划保留。新增 `prompt_gate_protocol_v2.json` 固定复核指令、代码SHA及原协议指纹；新生图计划同时绑定这份复核代码。`pre_gate_drafts` 保存原草稿，不覆盖原Prompt。`prompt_gate_protocol.json` / `preflight_v1` 是未通过的小样本预检记录，没有派发生图。没有调整CLIP、几何或DINO阈值，也不改变每地点最多两个替代源图的上限。

使用完整的本地Qwen3-VL-4B；8B目录只有未完成下载分片，不能当成可用模型。复核与生图模型分阶段加载。

## 进度与控制

```bash
./ccr-night-repair status
./ccr-night-repair pause
# 等当前图片保存、两个后端释放且主控制器退出后：
./ccr-night-repair resume --execute
```

暂停是完成当前图片并保存后退出、释放模型，不是freeze进程；恢复读取原冻结任务和请求日志，不重新提交已完成请求。第二主机网络失败、图片哈希不符、结果传输歧义或未知POST会停止等待处理，不擅自重复生成。

主任务：`ccr-night-repair-v2.service`（v1已保留断点后停用）。轻量状态/已知无效文本恢复：`ccr-night-repair-monitor-v1.service`。监控只对确认未构建生图任务的无效Qwen文本做分流，不重试不确定的图片请求。远程临时服务成功退出后即使已被systemd卸载，也可从完整结果清单回收；回收前核验精确配对、Prompt、seed、图片目录和哈希。

## 文件

- `outputs/night_repair_v1/inputs.jsonl`：原拒绝队列与规则证据。
- `protocol.json`：输入、代码、模型、服务、规则和两次替代源图上限。
- `state.sqlite`：地点状态、节点所有权、每轮源图、Prompt、生成与审核结果。
- `batches/*/inference`、`prompt_raw_*.json`：Qwen请求与原文、断点复用。
- `batches/*/plan.jsonl`：冻结生成任务与精确配对。
- `batches/*/results.jsonl`：生成结果；远程图经哈希核验回收到第一台相同目录。
- `accepted_replacements.jsonl`：本轮通过两个规则的候选；不自动替换正式图库或准入训练。
- `restored_original_pass.jsonl`：取消类别隔离后实际通过的原图。
- `policy_recheck_unknown.jsonl`：取消隔离后证据不足的原图。
- `unresolved_places.jsonl`：两次替代源图机会用尽或没有可用源图。
- `live_status.json`：监控记录的节点与地点进度；`run.log`为主日志。

原图、旧生成图、原Prompt和之前审核记录不覆盖。没有启动训练。整个流程可能持续数日，耗时取决于新图数量和替代轮数；运行状态必须以日志和真实输出计数为准。

### 2026-11-17 调度提速

主服务更新为 `ccr-night-repair-v3.service`，由 `night_repair_pipeline.py` 编排，`./ccr-night-repair` 的使用方式不变。原冻结的 Prompt、生图与审核模块均不改；单独的 `pipeline_protocol_v1.json` 记录调度代码 SHA 和预取策略。

远端生成完成并回传校验后，进入 audit 队列，不再阻塞其下一批。第一台优先准备远端下一批冻结 Prompt（启动时先准备16地点，以后按64地点预取），最多预取一批；轻量监控在后台接续派发，因此第一台自己生图时，第二台也能接下一批。每台仍只运行一个生图后端，等待审核最多积压4批；达到上限优先审核，避免无限积压。资源保护、精确配对、去重账本、每地点最多两次替代源图及审核阈值不变。

第一台的 VLM、审核和生图继续分阶段使用GPU。首次填充预取队列、模型加载、网络传输或资源不足仍可能短暂空闲；不承诺两台GPU永远满载。优先消除已完成批次被审核队列阻挡的等待。远端回收和派发分别有跨进程文件锁，控制器与监控不能并发复制/重复派发同一批。

调度相关新增7项CPU测试，与原有修复测试合计31项通过；未运行训练。
