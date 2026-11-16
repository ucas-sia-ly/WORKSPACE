# 双机拒绝夜景修复

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
