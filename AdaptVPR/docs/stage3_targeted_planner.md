# Fixed-target scene-aware editability planning

BoQ 与上一阶段 adapter 决定位置。这里直接消费已保存的 generation mask，只判断该
位置能否进行真实 local occlusion，以及适合的有限 occluder family。不重新选样、重建
mask 或修改像素；不接入原 `run.py`、scheduler、prompt agent、editor、verifier 或训练。

## 独立入口

在 AdaptVPR 目录中，使用环境变量配置已有本地 Qwen3-VL 权重：

```bash
export ADAPTVPR_TARGETED_PLANNER_MODEL_PATH=/absolute/path/to/Qwen3-VL-4B-Instruct
python scripts/audit_targeted_planner.py \
  outputs/stage3_targeted/generation_masks/generation_masks.jsonl \
  --targets ../../Bag-of-Queries/outputs/stage3/targets/targets.jsonl \
  --count 20 --seed 0 --backend local
```

所有样本沿用输入 manifest 的顺序；默认 20 条即上一阶段 seed=0 选出的同一批 20 条。
`--seed` 传至 VLM，**不控制位置、不重新抽样**。本地 backend 只读取已存在的目录，
禁止自动下载。`ADAPTVPR_TARGETED_PLANNER_DEVICE` 默认 `cuda`，可显式配置。

也支持既有 Qwen OpenAI-compatible endpoint：

```bash
export ADAPTVPR_PLANNER_API_BASE=http://YOUR_HOST:YOUR_PORT/v1
export ADAPTVPR_PLANNER_MODEL=YOUR_QWEN_MODEL_NAME
export ADAPTVPR_PLANNER_API_KEY=YOUR_KEY
python scripts/audit_targeted_planner.py \
  outputs/stage3_targeted/generation_masks/generation_masks.jsonl \
  --targets ../../Bag-of-Queries/outputs/stage3/targets/targets.jsonl \
  --backend openai --output outputs/stage3_targeted/planner_audit_http
```

HTTP 请求使用 `temperature=0`、`seed` 和 strict JSON schema response format；服务若不
支持则显式失败，不转成 mock 或自由文本。此入口没有 cloud fallback，也不自动加载
`.env`。本地 Transformers 使用 greedy decoding（`do_sample=False`），要求严格 JSON
并执行严格后验校验，**没有 token-level grammar constraint**。跨设备/库版本的逐位
一致性不作保证。测试验证 seed 传递和输入/模板确定性，不把 mock 等同于真实 VLM。

## 输入绑定与 VLM 视图

独立读取原 BoQ manifest，复用 reader 的 schema、唯一 sample_id、image_key、place_key、
尺寸、哈希和 export manifest 校验。再对齐 adapter record 的身份与 source/vulnerability
路径和 SHA256，验证 generation mask SHA256、尺寸、binary、非空、面积和 overlap。
VLM 不接收 mask 对象、metadata 或路径，只收到独立的 RGB 副本：

1. 未改变坐标系的原图；不做 EXIF 旋转。
2. 原图上的固定 generation mask，45% magenta tint。
3. 固定 mask bbox 向四周扩展的原图 context crop，默认宽高乘 2，越界裁切。

crop 坐标仅由本地代码产生，记录在 plan 的 `context_crop_xyxy` 供审计，**不是 VLM
输出或新目标位置**。crop 不改变 mask。原始 binary mask 不做任何 resize；VLM processor
可以按自己的输入需求缩放 RGB 视图。每次调用后验证原图与两种 mask 的像素哈希，
整个 audit 后再次核对源文件哈希。

## 决策 schema

必须恰好包含七个字段：

```json
{
  "editable": true,
  "region_type": "road",
  "support_surface": "paved_ground",
  "occluder_family": "parked_vehicle",
  "object_name": "parked car",
  "confidence": 0.9,
  "reason": "Visible support and adequate room within the fixed target."
}
```

region taxonomy：`road, sidewalk, parking_area, grass, vegetation, building_front,
wall_or_fence, sky, unknown`。family taxonomy：`parked_vehicle, construction_barrier,
traffic_cones, temporary_sign, vegetation, scaffolding, construction_tarp, none`。

有限 support taxonomy：`paved_ground, soil_ground, building_base, facade_attachment,
none, unknown`。`building_base` 必须是在 mask 内可见的建筑底部支撑；
`facade_attachment` 必须是在 mask 内可见的锚点/固定结构，不能仅凭“这是墙面”推断。
物体、所需支撑/接触和阴影应能容纳在固定区域内。附近但位于 mask 外的地面不算可用支撑。

`object_name` 也是有限规范名，必须与 family 一致。`reason` 为审计说明，永不拼入
diffusion prompt。拒绝结果必须使用 `family=none, object_name=none`，仍须报告实际
region/support 和数值 confidence；confidence 表示判断把握，拒绝也可高置信度。

不接受 code fence、prose、重复 key、缺字段、坐标字段、prompt 字段、任意 object name、
类型自动转换、非有限 confidence 或超出 [0,1] 的值。schema 错误允许一次同图、同 seed、
同位置的重新评估，附上字段校验错误；每次原始响应和错误均保存。**不补造字段，也不
修改解析失败的 JSON**。仍失败则 `status=schema_error`、decision=null、prompt 为空，
计入 rejected 和独立 schema_error_count。网络/模型错误直接停止 audit 并记录 ERROR。

## 硬规则与固定模板

规则只能否决，不能改选 family 或位置。最终拒绝把 family/object 设为 none，并保留
`proposed_decision`、实际 region/support 和 `policy_rejections`。

| family | region | support |
|---|---|---|
| parked_vehicle | road / parking_area | paved_ground |
| construction_barrier / traffic_cones | road / sidewalk | paved_ground |
| temporary_sign | road / sidewalk / parking_area / grass / wall_or_fence | 与 region 相符的 paved_ground / soil_ground / facade_attachment |
| vegetation | grass / vegetation | soil_ground |
| scaffolding | building_front | building_base |
| construction_tarp | building_front | building_base / facade_attachment |

sky 总是拒绝；none/unknown 支撑总是拒绝。本版采用保守策略：unknown 区域无论置信度
均拒绝，任何区域 confidence < 0.70 也拒绝（可通过 `--min-confidence` 配置）。这比
“仅 unknown 且低 confidence 拒绝”更严格。grass/vegetation 优先 vegetation，grass
可在有合理支撑时使用 temporary_sign。无明显支撑的悬空建筑区域拒绝。

每个 family 只有一个版本化固定 template 和固定 negative；没有自由 prompt 参数。
共享约束包含 grounded、scale、perspective、lighting、shadows、scene geometry 和
outside-mask preservation。共享 negative 包含用户要求的全部十二项。none 对应空正向
prompt 和固定 negative，禁止作为可编辑计划。这里只生成计划，不调用 diffusion；
模板中的“只改 mask 内”是未来 editor 的约束，不能据此声称已经证明像素隔离或 realism。

## 输出与审计边界

默认 `outputs/stage3_targeted/planner_audit/`；已有目录不覆盖。

- `targeted_edit_plans.jsonl`：schema_version=1、contract=TargetedEditPlan、route=local、
  seed、身份和 Stage2 metadata、immutable mask 身份、原始/最终决策、重试记录、固定 prompts。
- `planner_summary.json`：editable/rejected、region/family counts、low-confidence、schema
  错误、policy 拒绝统计、模型配置、输入和代码哈希。region/family/confidence 分布仅统计
  schema-valid 的最终决策，schema errors 独立计数，避免伪造成 unknown 模型判断。
- `planner_instructions.json`：实际 system/user 指令。
- 每图 `source.png`、原字节复制的 `vulnerability_mask.png` 和 `generation_mask.png`、
  `generation_overlay.png`、`context_crop.png`、`comparison.png`、`planner.json`、`raw_response.txt`。
- `contact_sheet.jpg`：便于人工浏览，RGB 展示缩放不参与 mask 或模型采样。

VLM 对 region、支撑与尺度的判断可能出错；硬规则只验证这些声明的兼容性，不构成
geometry verification 或人工 ground truth。editable 表示候选计划通过本阶段约束，
不代表生成图像已通过真实性测试。本阶段不生成图片。

## 测试

```bash
python -m unittest discover -s tests -p test_targeted_planner.py -v
python -m unittest discover -s tests -p test_targeted_mask_adapter.py -v
python -m unittest discover -s tests -p test_targeted_inputs.py -v
```

覆盖严格 schema、所有 family/region 组合、支撑/悬空/sky/unknown/低置信度、固定模板、
三张图与 seed 传递、错误 schema 的可追溯重试、mask 不变、错图/错 mask 拒绝、
crop 边界、文件输出与统计。纯合约和 CLI help 测试禁止导入模型或原 pipeline。

## 本次真实 20 图 audit 结果

固定上一阶段 20 个 target，seed=0，本地 Qwen3-VL-4B-Instruct。最终产物位于
`outputs/stage3_targeted/planner_audit/`：editable=0，rejected=20，其中 17 个合法 VLM
拒绝和 3 个 schema/跨字段一致性错误拒绝。最终状态为 `COMPLETE_WITH_SCHEMA_ERRORS`，
CLI 返回 2；不是全量 schema 通过。low-confidence=0 仅统计 17 个合法响应。

有效 region 分布：building_front=6，wall_or_fence=3，sky=3，vegetation=2，road=1，
grass=1，unknown=1，sidewalk=0，parking_area=0。有效 family 分布全部为 none=17；
另 3 条没有伪造 region/family。这里是 **VLM 声称的类别，不是人工真值**。

三个协议错误样本为索引 01、06、13，均在一次同位置重试后仍存在 editable/family/
object_name 不一致；保存了两次 raw response。调试早期两轮记录分别保留在
`planner_audit_initial_schema_errors/`、`planner_audit_initial_insertion_ambiguity/`，
不计入最终汇总。最终指令明确是新增遮挡物，不要求场景里已有遮挡物，并为三张图
分别标注用途。

这次审计暴露出 4B VLM 在多图定位和跨字段一致性上的局限，以及偏保守、置信度偏高的
判断。不能把“20 个均拒绝”解释为 20 个位置在物理上均不可编辑，也不能声称 planner
的语义准确率或生成 realism 已经通过验证。硬规则与 schema 成功阻止了不一致计划进入
可编辑输出。按阶段要求在 audit 后停止，没有调用 diffusion 或为追求接受率改动 mask。

单元测试及回归共 57 项通过：planner 23、mask adapter 14、input reader 20。
`planner_audit/validation.json` 记录测试命令、20 个 ID 顺序一致、保存 mask 原字节一致、
全部检查文件存在，以及当前实现代码与 audit provenance 的 SHA256 对齐结果。
