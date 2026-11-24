# Night Prompt 修订：移除汞灯并重新核实源图

本修订方案已按用户最新要求暂停。当前使用[原Prompt非汞灯双机续生成](night_original_no_mercury.md)，不继续运行本页的VLM任务。以下计划与记录保留供对照。

2026-11-10 起，原 `night_place_fast_v1` / `night_cluster_v1` 图片队列保持暂停。旧计划不原地修改；`ccr-night resume` 遇到 `prompt_revision_requested.json` 会明确拒绝，防止再次调用旧 Prompt。新的准备入口只写文本 Prompt，不启动生图或训练。

## 修订范围

暂停完成后原计划共61,984个地点，已完成48,323张。汞灯 `mercury_vapor` 共2,066个地点，其中已生成1,611张，全部进入重新分配。新队列为15,272个地点：2,066个原汞灯地点及13,206个其他未生成地点；保持每个地点原选定真实源图，不额外改变地点池。

其他风格已生成的46,712张保留。汞灯PNG、raw、sidecar及双机同步副本依据精确SHA与路径清单删除；原图、计划和历史日志保留。保留清单在 `outputs/night_revision_v2/state/retained_manifest.jsonl`，仍是待审核生成图，不自动成为合格训练数据。

## 新的源图证据和Prompt

新策略和29类目录独立放在 `night_diversity_policy_v2.py` / `night_style_catalog_v2.json`。原30类目录及冻结依赖不修改。

- 建筑窗、住宅建筑窗、车辆窗分别判断；住宅窗光需要建筑窗和住宅用途证据均为present。uncertain不会算作支持。
- 店铺展示橱窗、普通玻璃门、广告牌分别判断；玻璃门和广告牌不能证明店铺存在。
- 有顶通道需要实际可见顶棚及通道深度；普通入口不等于有顶通道。
- 固定灯具必须确实可见，才能作为画内灯具。否则明确画外光源位置，仅描述既有受照表面、方向、衰减，不让模型增建灯排。
- `MUST PRESERVE`单独列出视角、建筑轮廓、门窗布局、道路拓扑、标识内容、固定设施和地点结构细节。`MAY CHANGE`列出光照、曝光、色温、合理人车变化，以及声明的地面干湿状态。
- 每条Prompt仍由VLM看源图后开放式撰写；29类仅是可选照明机制，不是固定完整Prompt。先按确认的场景条件排除不合适类别，再优先选择曝光较少的合适类别，不为类别均衡编造物体。

结构化字段与正文有明显矛盾时自动退回重写，例如无灯具却要求附墙灯排、无建筑窗却要求窗光。此检查和源图VLM均不是生成图片结构保留的证明；新图生成后仍需独立自动审核，不要求人工逐图审核。

## 运行和恢复

```bash
cd /home/admin123/github/WORKSPACE
set -x PY /home/admin123/miniconda3/envs/AdaptVPR/bin/python # fish
# bash使用：PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python

"$PY" -B AdaptVPR/experiments/qwen_curriculum/night_revision.py status

# 后台Prompt任务只做源图分析和文本撰写
systemctl --user start ccr-night-revision-prompts-v2.service
journalctl --user -u ccr-night-revision-prompts-v2.service -f

# 完成当前文本批次并提交后停止；再次start继续
systemctl --user stop ccr-night-revision-prompts-v2.service
```

资源以本机48GB显存/32GB系统内存为预算；BF16/SDPA，allocator上限42GiB，后端cgroup有限RAM且禁swap。新schema增加了文本长度，批量大小以本轮真实短测为依据，不直接套用旧batch64的结果。原图片队列保持关闭；用户后来单独授权的[新双机生图队列](night_revision_images.md)可让第二主机同时消费已提交的新Prompt。

输出目录 `outputs/night_revision_v2/`：

- `state/inventory.json`、`tasks.jsonl`：稳定原始库存及15,272个精确源图任务。
- `state/retirement/`：双机SHA清理记录及删除后复查。
- `state/revision_config.json`：冻结新策略、实现、完整模型文件SHA、分组及曝光基数。
- `state/batches/`：持久化VLM请求和完整原始响应；已有响应先复用。
- `state/groups/`、`prompts/`：原子批次提交和逐地点分析/Prompt，保留旧sample映射。
- `state/progress.json`、`resources.json`：进度、失败原因与资源。
- `state/night_prompts.jsonl`、`prompts_complete.json`、`shards/`：所有源图有明确终态后导出。

中断的是纯文本推理时，缺失响应的请求可记录重算沿革后以相同seed重算；已有响应和已提交Prompt不重做。这不适用于不明确的生图POST。输入SHA、配置或所有权异常会停止，不当作无效源图偷偷跳过。

### 2026-11-10：Prompt 失败恢复

首次暂停时192个源图中185个通过、7个失败：4个源图响应的标牌引号破坏JSON、2个中文描述被英文词数要求拒绝、1个neutral_led只写ambient而缺少人工光源。它们是文本规划失败，不是图片后端失败，也没有发现token截断。

修复优先要求源图描述英文、避免直接引用标牌文字；如仍返回中文，内部证据使用中文字符长度校验，最终生图Prompt仍要求英文。JSON备用解析只对完整源图对象描述字段内成对的未转义引号插入转义，保留原字符串、字段和状态；结构歧义、缺字段或截断均拒绝修复。每次修复记录字段、行号及插入转义数。

根据该源图证据显示可用光源和必需光源组，重试逐字段反馈错误。没有放宽场景证据、夜间效果、固定设施或Unknown约束，也不通过自动补事实、忽略验证来提高通过率。

原配置、原子批次和已通过Prompt不改写。原9份依赖封存在`state/runtime_patches/original_files/`；`state/runtime_patch.json`及编号补丁冻结修复代码的SHA和前版代码归档，后续请求也绑定该执行身份。失败批次的额外重试写到`state/repairs/`下按执行身份区分的收据，只覆盖上一版仍失败的记录；逐地点Prompt是可恢复索引。已通过源图和旧批次SHA应与`state/repair_before.json`及后续恢复结果快照一致。

每个执行版本每批次最多额外恢复一轮；已提交的恢复结果重启后复用，不无限重试。纯格式问题可复用带SHA的原始分析响应，只重新撰写尚未成功的Prompt，避免重新看图浪费GPU。仍无法核实或无法合规撰写的源图保留明确失败原因。`progress.json`区分恢复前失败数、已恢复数和当前失败数。导出前仍核对每个源图、批次及修复沿革，补丁不会授权生图。

仅修复已提交批次中的失败项（与后台服务互斥，先stop）：

```bash
"$PY" -B AdaptVPR/experiments/qwen_curriculum/night_revision.py plan --repair-only --execute
```

正常`systemctl --user start ccr-night-revision-prompts-v2.service`会先复用成功结果、修复已提交失败项，然后继续尚未处理的源图。恢复和继续都不启动图片生成。

新图片生成需另行明确启动。原旧队列不能直接resume来生成新Prompt；新旧sample_id及输出目录分离，完成标记不会触发自动生图。

用户随后已明确授权第二主机立即生图、第一主机在Prompt完成后自动加入；该授权由独立的`night_revision_images_v1`控制器执行，见[新运行入口](night_revision_images.md)。本文件的Prompt命令本身仍不启动图片后端。
