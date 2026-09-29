# Stage3-dev scene planner

`targeted/scene_planner.py` 是独立的附近场景/family 评估器。
`targeted/planner.py`、旧 client、fixed-mask policy 和 templates 均保留为 baseline。
新入口不使用旧 generation-mask adapter，不调用 diffusion，也不选择最终空间位置。

```bash
conda run -n AdaptVPR python scripts/stage3_audit_scene_planner.py \
  --count 20 --seed 0 \
  --model-path /home/admin123/视频/workspace/models/Qwen3-VL-4B-Instruct
```

默认读取 `Bag-of-Queries/outputs/stage3/dev/` 的 cohort 和 vulnerability export，按
冻结 cohort 的顺序取前 20 张。`--seed` 仅传给 VLM，不重选样本。默认输出独立目录
`outputs/stage3_dev/scene_planner_audit/`，已有目录拒绝覆盖。仅加载已存在的本地
Qwen3-VL 权重，不自动下载、不调用外部服务。模型配置、软件版本、代码哈希和输入
哈希记录在 summary 中。

## 输入边界

VLM 恰好收到三个独立 RGB 图像副本：

1. 原图，保持原始 decoded orientation，不做 EXIF transpose。
2. 原图上的 15% vulnerability overlay，45% magenta tint。
3. 未标记的 ROI context crop，默认扩展 ROI 外接范围的宽高各两倍，越界裁切。

ROI 来自 BoQ dev NPZ 中已有的 `attention_roi_token_mask`：16×16 bool、38 token、
connected_topk，沿用 `round(0.15*256)`。本地代码用 floor-coordinate nearest 将该
ROI 投影到原图，仅用于 overlay 和 crop；不改动任何连续 scientific map，也不把
显示插值结果用于 coverage 数值计算。不读取或生成 generation mask。

crop 坐标由本地代码固定计算，保存在 `view_metadata.context_crop_xyxy` 供输入
审计；它不是 VLM 的坐标输出，也不是最终放置区域。原图/overlay 同尺寸，crop
保留原始像素；VLM processor 可自行缩放 RGB 输入。

输入 reader 校验完整 dev-50、cohort/export 哈希、50 个唯一 place、与 Stage2-100
disjoint、顺序/身份绑定、选中 SOURCE 与 NPZ 字节哈希，以及原图尺寸、NPZ 内部身份
和 ROI 类型/预算。审计完成后再次检查所读文件哈希。

## 六字段 schema

```json
{
  "editable": true,
  "region_type": "mixed",
  "support_surface": "paved_ground",
  "feasible_families": ["construction_barrier", "traffic_cones"],
  "confidence": 0.9,
  "reason": "The nearby sidewalk and road provide visible paved support."
}
```

`editable` 表示附近场景至少存在一个语义合理的 family，不表示该物体能塞入
vulnerability ROI，更不表示最终 placement/geometry/realism 已验证。允许利用 crop
内、tint 外的真实支撑；不应借用全图中无关的遥远场景。没有已有 occluder 不构成
拒绝理由，模型应评估新物体的场景合理性。

闭集 family 保留全部七类：`parked_vehicle`, `construction_barrier`, `traffic_cones`,
`temporary_sign`, `vegetation`, `scaffolding`, `construction_tarp`。数组可含多个 family，
无重复、不含 `none`；`editable=false` 必须返回空数组，true 必须非空。此次 20 张
pilot 不据此删减 taxonomy，完整 50-dev audit 后再决定是否精简。

region：`road`, `sidewalk`, `parking_area`, `grass`, `vegetation`, `building_front`,
`wall_or_fence`, `mixed`, `sky`, `unknown`。

support：`paved_ground`, `soil_ground`, `building_base`, `facade_attachment`,
`mixed`, `none`, `unknown`。

禁止额外字段、坐标、bbox、mask、prompt 和任意 object_name。`reason` 仅为简短
英文场景/支撑观察；禁止数值坐标/尺寸和编辑命令，也不转为生成 prompt。严格解析器
拒绝重复 JSON key、缺字段、markdown、类型转换、非有限置信度、越界值、任意 family
和跨字段不一致。文本检查拒绝 reason 中的数字、显式几何/prompt 字样和常见编辑命令；
这是协议检查，不是对自然语言语义的完整证明。

不沿用 fixed-mask 的区域内 fit/support 硬规则，也不自动按置信度阈值改写决策。
观察性 family 声明仍可能出错，需要人工 audit 和后续独立的空间可行性验证。

## 推理与审计

使用已有 `LocalQwenClient` 的本地权重 loader，但新 wrapper 完全独立构造三图
标签和 `decide()` 消息，避免旧 client 的 “ONLY editable region” 标签泄漏进新任务。
greedy decoding，最多 512 new tokens；没有 token-level grammar constraint，最终
输出必须通过严格后验 schema 校验才能进入 `decision`。

schema 错误允许一次同三图、同 seed 的重试；保留全部原始回答和错误。不补造字段，
不修改原始输出；仍失败则 `decision=null, status=schema_error`，独立计数，CLI 返回 2。
模型/加载错误记录 ERROR，不回退到 mock。真实模型调用与测试 double 明确分开。

输出包括：

- `scene_assessments.jsonl`：六字段 decision、identity、视图哈希、本地 crop metadata、
  原始响应/重试与配置；没有 generation prompt 或最终坐标。
- `summary.json`：模型 editable/rejected/schema-error、region/support/family 统计、
  置信度、输入/代码 provenance。family 按集合出现次数计数，可能总和超过样本数。
- `planner_instructions.json`：本次实际 VLM 输入指令、三图标签、schema。
- 每图 `source.png`、`vulnerability_overlay.png`、`roi_context_crop.png`、
  `comparison.png`、`scene_assessment.json` 和各次 raw response。
- `contact_sheet.jpg`、`AUDIT.md`、`human_review.csv`：审计入口；人工标签列留空，
  不把模型评估伪造成 human ground truth。

```bash
python -m unittest discover -s tests -p test_scene_planner.py -v
python -m unittest discover -s tests -p test_targeted_planner.py -v
```

## 本次 dev-20 pilot

本地 Qwen3-VL-4B-Instruct、seed=0；固定前 20 个 dev。结果为 19 个候选 family
assessment、1 个拒绝、0 个 schema error、0 次重试。七类保持不变：vegetation=19、
parked_vehicle=15、temporary_sign=11、construction_tarp=9、traffic_cones=7、
construction_barrier=2、scaffolding=1。18 条使用 mixed region/support。

这些不是人工准确率或 placement 成功率。部分 tarp 理由依赖假设性支撑，拒绝样本
Lisbon:6940 的 support label 与 reason 也有矛盾；详见输出目录中的
`MODEL_AUDIT_NOTES.md`。人工标签保持空白。没有根据这 20 张调整 taxonomy 或重跑
模型以改变接受率，也没有运行剩余 30 张。

新模块 11 项测试、旧 fixed-mask planner 23 项回归测试通过。实际 20 条输出的顺序、
identity、三图像素/文件哈希、六字段 schema 和输入/代码 provenance 均已核验。
旧 planner/client/policy/templates/audit script 的字节哈希保持不变。
