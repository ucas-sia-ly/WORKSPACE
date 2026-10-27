# 逐图 VLM 夜景规划与临时物体增广

当前推荐策略已调整为仅night、允许合理人车与干湿变化、按场景筛选并均衡30个风格机制，见 [night_generation_balanced.md](night_generation_balanced.md)。本页保留旧版严格物体配对协议及其历史记录；其中“必须保持人车”“精确新增数量”等约束不再代表新策略。

本入口面向 GSV-Cities → SALAD VPR：让真实看图的模型为每张源图提出不同的夜间外观，并在可行时增加临时街道活动。风格不局限于人工列出的 LED、钠灯等少数模板；模型自行组合场景相关的光源、天气、曝光和夜间阶段。永久地点结构始终作为同地点标签的依据。

## 和 v3/v5 的区别

- `Qwen3-VL-4B-Instruct` 实际读取源图，输出场景摘要、受保护地点区域、空地上的物体提案和多条英文编辑 Prompt。生成器收到的是保留的逐图 prose，加一段简短的授权约束；不会再用固定天气模板覆盖它。
- `lighting`、`exposure`、`style_name` 和 `appearance_prompt` 是自由文本。`weather` 使用 dry/wet/rain/snow/fog 作为可统计元数据，名字不是风格模板或质量标签。
- 每个风格先生成 appearance 控制图；有可行物体时，再用**相同源图、风格描述和生成 seed**生成 appearance+transient 对照。不同 Prompt 仍会使整幅图重新采样，不能声称只改变了某些像素。
- 当前临时物体支持 vehicle/person/cyclist/scooter，可扩展。已有窗户亮灯、湿地面、雾、雪、眩光或不同曝光属于外观变化。增加新的建筑、商铺、招牌、路灯杆或交通灯会改写地点，仍不授权。
- 允许有限的临时遮挡：提案框最多占全图15%，最多覆盖任一保护框25%。这些是显式的编辑预算，不是根据 Recall 或图片通过率拟合的质量阈值。
- 新增物体必须是一个额外对象，放在可见空位；不能把移动已有行人或点亮已有车辆冒充新增。非法表面、边界、遮挡超预算的提案单独排除，不因此丢掉合法外观方案，也不强行制造物体配额。
- 模型的 overall editability 是外观编辑的描述，不当作检索 hardness，也不要求 medium 场景一律禁止物体。每个物体单独检查其表面和空间提案。

## 实现和记录

新增 `AdaptVPR/experiments/qwen_curriculum/scene_prompting.py`、`scene_aware_night.py` 与 `plan_scene_transients.py`；旧 v3/v4/v5 和公共 AdaptVPR 入口不修改。物体规划可以独立读取原图，继承已经冻结的外观描述，避免把外观和空位判断塞进同一个任务。

源图来自训练地点清单，按 city/place_id 排除现有校准/验证地点。记录源图 SHA256、真实看图模型的权重/配置 SHA256、采样参数、原始输出、坐标系统、被拒物体提案、最终完整 Prompt 与每次生图的 durable reservation。

Qwen3-VL 的0–1000坐标必须显式声明，程序确定性转换为0–1坐标；不猜测未知单位。所有框都是**规划位置**，不是已验证的分割 mask，不用于 Reliability 负标签。每张源图最多两个规划草稿，格式修正也占预算；格式失败不会回退到伪造场景或固定 Prompt。

新增物体为合格“提案”不代表已经被正确生成。后续 VLM 会再次对照真实源图与生成图，分别检查：永久结构、预期天气/光照、实际新增对象类型与数量、合法地面位置和地标可见性。审核失败或响应格式不合格保持 reject/unknown；不能把低 DINO 相似度或低匹配率直接视作生成错误。

## 运行顺序与资源

为适应4090 48GB + 32GB RAM，模型依次运行：看图规划 → 释放看图模型 → Qwen生图 → 关闭Qwen服务 → 几何/CLIP → 释放检查进程 → 看图复查。尤其在系统只剩约4GB可用内存时，不让看图模型、生成器和CLIP同时常驻。

生图沿用现有 Qwen-Image-Edit-2511 / Lightning 4 steps、guidance=1、BF16磁盘卸载配置。正向 Prompt 包含必要约束；不依赖当前后端不会编码的 negative prompt。没有训练或修改网络权重，不下载新模型。

当前外观父计划为 `outputs/scene_conditioned_night_v2/state/`，初版对照为 `outputs/scene_night_diversity_demo/`，资源调整后的有效入口默认目录为 `outputs/scene_night_diversity_48g/`。初版冻结2张源图、4种逐图风格、6次累计生图请求；已预留2次，其中1张完整输出从中断的HTTP请求恢复。剩余4次转入新目录，不追加预算。未执行的最后一种风格留待新的显式实验，不伪装为已经生成。

初版32常驻block首次推理实测峰值46.72GiB，触发44GiB监控上限。新目录通过 `reprofile_scene_night.py` 继承完整Prompt、seed、源图和模型权重，只将常驻DiT block改为24；仍运行全部block及4步去噪。两次运行的资源身份单独记录，不覆盖旧协议，不承诺图片逐位相同。启动器仅在本次运行期间追加systemd运行时EnvironmentFile，结束恢复用户原服务配置，不改持久 `.env`。

24 block实测：4张生成及随后的几何/CLIP检查共138.73秒，整卡显存采样峰值41.87GiB，监控进程RSS合计峰值5.51GiB，系统可用内存最低21.56GiB。该耗时包含服务启动、模型加载与检查，不等于稳定单张延迟；不外推吞吐或承诺其他分辨率的峰值。资源原始记录在 `outputs/scene_night_diversity_48g/resource_runs/`。

推荐通过 `run_scene_night_safe.py` 启动。它默认保留4GiB系统可用内存，监控启动器、子进程组及独立systemd生成服务的RSS合计（上限20GiB），同时监控整卡显存（上限44GiB），并在系统swap较启动时增加超过1GiB时停止。RSS不含全部内核文件缓存；本轮系统确实出现swap占用，早期记录尚无逐运行swap增量，不能声称完全未使用swap。新增swap监控已通过CPU边界测试，尚未重新生图验证。

1秒采样不构成硬件级预留或OOM保证；共享内存页可能重复计数。若已有生成服务或其他GPU任务，拒绝重复启动，不自动停止它们。日志和实测资源写入 `resource_runs/`，失败后清理本次启动的生成服务。对当前固定画布，安全入口在启动前拒绝超过24常驻block的旧配置，避免重复发生已知的首次推理峰值问题。

```bash
cd /home/admin123/github/WORKSPACE
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
RUN=AdaptVPR/experiments/qwen_curriculum/run_scene_night_safe.py

# 核查已冻结的计划与图片；不调用模型。
"$PY" -B "$RUN" check

# 继续当前已冻结的小批次；每次与累计预算都生效，不自动扩大。
"$PY" -B "$RUN" run --execute --max-calls 4

# run正常结束会在关闭生成器后执行几何/CLIP。若检查中断，用此命令恢复。
"$PY" -B "$RUN" verify --execute

# 本地Qwen3-VL对照源图/生成图，按本次授权自动复查。
"$PY" -B "$RUN" audit --execute

# 只汇总已有文件并生成图版，不调用模型。
"$PY" -B "$RUN" report
```

失败的已预留请求仍占累计预算。安全入口将本次请求数限制为冻结预算的剩余量，不会通过重试偷偷扩容；预算用完后可继续 `verify`、`audit` 和 `report`。父目录的 `resource_successor.json` 标记其已退役，安全入口拒绝继续在父计划追加生图；子计划检查父请求计数，发现变化则停止。

新实验另建目录，使用底层规划入口；默认最多2张源图、每图2种风格，开启配对物体时上限为8张：

```bash
# 默认准备输入和预算，不运行视觉模型。
RUN=AdaptVPR/experiments/qwen_curriculum/scene_aware_night.py
SERVICE=outputs/scene_night_diversity_48g/service_config_24.json
"$PY" -B "$RUN" prepare --paired-transients \
  --service-config "$SERVICE" \
  --output-dir outputs/scene_night_next/state --image-root outputs/scene_night_next

# 显式运行真实看图模型，保留其逐图规划。
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -B "$RUN" prepare --execute --paired-transients \
  --service-config "$SERVICE" \
  --output-dir outputs/scene_night_next/state --image-root outputs/scene_night_next

# 查看所有计划中的风格、物体授权与完整Prompt，再显式生成。
cat outputs/scene_night_next/state/plan.jsonl
"$PY" -B AdaptVPR/experiments/qwen_curriculum/run_scene_night_safe.py run --execute --max-calls 8 \
  --output-dir outputs/scene_night_next/state --image-root outputs/scene_night_next
```

增加 `--source-ids <已有生成记录sample_id> ...` 可固定源图；它仍读取该记录的真实 source_path。`--seed` 控制规划抽样与新生成 seed，已保存的规划 JSON 决定恢复结果。不同软件/硬件的采样未承诺逐位一致，恢复不会重新写一套 Prompt。

## 输出如何使用

- `state/scene_plans/`：实际场景、原始 VLM 响应、多个方案、筛掉的物体提案。
- `state/object_plans/`：独立空位规划原始输出、提案和拒绝理由；外观父计划的位置保存在 `planner_config.json` 中。
- `state/plan.jsonl`：完整生成指令、配对关系、源图与授权信息。
- `state/generated.jsonl`：已落盘图片。`state/results.jsonl` 在生图阶段只记录 pending_verification；它不是最终质量结果。
- `state/quality_results.jsonl` / `quality_summary.json`：生成服务退出后的真实几何/CLIP检查。
- `state/audit/` / `audit_summary.json`：逐图授权复查。看图模型的自查不是独立语义真值。
- `state/candidate_manifest.jsonl`：通过当前自动规则的探索候选；尚不直接标记训练合格。
- `review/report.md` / `pairs_*.jpg`：源图、外观、外观+临时物体三列对照。

当前5张输出的实测资源、审核漏检和适用限制见 `outputs/scene_night_diversity_48g/review/findings.md`。安全入口的report会合并资源父目录，按明确的appearance/transient字段放列，标出未执行预算和各图资源配置；不能根据数组位置推断控制图。

无需人工审核才能执行链路。生成质量、实际域多样性与检索收益仍需分别验证；候选通过率和好看的图都不能替代 A/M 多种子真实验证。正式训练需另外冻结配置与曝光预算，本入口不启动训练。

CPU测试：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=AdaptVPR \
 "$PY" -m unittest discover -s AdaptVPR/experiments/qwen_curriculum/tests -p 'test_scene_*.py'
```
