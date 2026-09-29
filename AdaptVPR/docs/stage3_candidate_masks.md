# Stage3 family-specific Core / Render candidates

新增 `targeted/family_constraints.py`、`targeted/candidate_masks.py`、
`targeted/render_mask.py`。旧 `targeted/mask_adapter.py` 保留并标记为 **ellipse
baseline**；只有模块说明变更，旧函数计算行为和统一 6% 默认值保留供 baseline 比较。
新路径不调用 baseline adapter、VLM、diffusion、训练或编辑器。

```bash
conda run -n AdaptVPR python scripts/stage3_build_candidate_masks.py
```

默认消费 `outputs/stage3_dev/scene_planner_audit/` 中已有的 20 条 scene assessment，
按原 dev 顺序生成，输出 `outputs/stage3_dev/candidate_masks/`。只有 schema-valid、
editable=true 的 family 集合产生候选；拒绝/schema-error 样本也保留明确的零候选
记录。不补跑剩余 dev，不重调 VLM，不根据候选通过率修改 family 判断。

## 可修改的 family 先验

配置位于 `configs/targeted_family_constraints.json`，可通过 `--config` 指定另一份。
每次输出保存完整配置、原文件哈希及 canonical 配置哈希。以下为首版几何搜索范围，
都是**相对原图总像素面积**，不是相对 ROI 面积；它们是待 dev 审计的先验，尚未
经过真实物体大小或透视校准。

| Family | Core 形状 | 面积范围 | 宽 / 高 |
| --- | --- | --- | --- |
| parked_vehicle | 横向 compact superellipse，指数四 | 3%–12% | 2.0、3.0 |
| construction_barrier | 横向 compact superellipse，指数四 | 1.5%–8% | 2.5、4.0 |
| traffic_cones | 竖向实心三角 footprint | 0.4%–2.5% | 0.55、0.8 |
| temporary_sign | 竖向矩形 footprint | 0.6%–3.5% | 0.4、0.65 |
| vegetation | 紧凑无孔 irregular radial blob | 2%–10% | 0.8、1.2 |
| scaffolding | 局部 facade 方向对齐矩形 | 4%–16% | 0.6、1.4 |
| construction_tarp | 局部 facade 方向对齐矩形 | 3.5%–14% | 0.8、1.6 |

每个范围默认均匀采样三个面积值；所有配置中的 aspect/angle/center 组合都保留。
实际 rasterized 面积必须落在配置范围内才通过该 gate，不因像素取整偷偷扩宽范围。
边界裁剪、面积不合法、不连通或 coverage 不合格的候选仍保存，附拒绝原因。

Core 在原图尺寸以像素中心直接 rasterize，不先画小图再 resize。vehicle/barrier
为横向圆角紧凑体；cone 是单个竖向三角支撑的几何假设，并不生成自由的多物体分布。
vegetation 的径向边界由固定三阶/五阶谐波产生，SHA-256(image_key/family/seed)
确定相位，无全局随机状态。VLM 只提供已有的闭集 family，不提供 shape、坐标或 mask。

facade frame 来自 ROI 邻近图像的强梯度正交方向投票，矩形枚举该方向 ±5° 及中心角。
证据不足时明确记录 `image_axis_fallback`，不是谎称识别了 facade。所有候选均标记
`facade_plane_verified=false` / `support_contact_verified=false`；这不是单应性、平面
重建或物体真实落地证明，后续仍需独立几何与语义验证。

中心集合由 binary ROI centroid 周围配置的 3×3 offset lattice、ROI bbox 中心、
最多四个按固定 token 权重排序且有最小间隔的 ROI anchor 构成。列表去重、固定排序；
不存在按最终评分保留 top-k 的步骤。超出图像的中心不枚举；落在图内但 footprint
越界的候选保留并标记 clipping。`seed` 只改变 irregular blob 的确定性相位，不重选
source，也不选择最优 candidate。

## 指标和明确的阈值口径

令 `C` 为原尺寸 Core，`R` 为以 floor-coordinate nearest 投影到原尺寸的 binary
vulnerability ROI，`w[t]` 为从 NPZ 原样读取的 token 权重。

- **target precision** = `|C ∩ R| / |C|`。
- **binary ROI coverage** = `|C ∩ R| / |R|`。
- **vulnerability-weighted coverage** = `Σ w[t] a[t] / Σ w[t]`，分母覆盖**全图 token**。
- `a[t]` = 原图中属于 token t 的像素里被 Core 覆盖的比例。
- area 同时保存像素数和原图面积占比。
- centroid distance 从 Core centroid 到 binary ROI centroid，保存像素距离及除以
  原图对角线的归一化距离；像素中心坐标为 `(x+0.5,y+0.5)`。
- connectivity 保存 Core 的八邻域连通分量数，通过要求为一。

默认 `weight_map=raw_attention_map`，沿用先前导出的主表示。配置可显式改为
`attention_map`、`intervention_map` 或 `fused_map`，但必须作为不同的记录完整配置
运行；程序不尝试多个 map 后择优。原始 float 权重从不插值、重采样、二值化或变成
PNG。`a[t]` 是 native Core 与原 token cell 的精确面积比例，即使原图宽高不能被 16
整除，各 token 权重质量也不因对应 cell 像素数不同而改变。

另存 `roi_conditioned_weighted_coverage = Σ w[t] R[t] a[t] / Σ w[t] R[t]` 和
`binary_roi_token_coverage`，仅供解释与审计。**它们不替代主 weighted coverage 的
分母，也不参与阈值 gate。** 全图总权重为零时 weighted coverage 为 null，所有该项
gate 失败并注明原因，不用 epsilon 或全一权重伪造结果。

`tau_target_precision=0.7`。每个候选同时记录 weighted coverage 阈值 **0.3、0.5、0.7**
下的通过/失败和原因；三个实验档位全部保留。没有默认最佳档位，也没有“若严格档失败
就回退宽松档”的逻辑。候选 `selected=false`，summary 的 chosen threshold 和 selected
candidate 均为空。

注意：若原始 attention 权重分布较均匀，小面积 footprint 的全图 weighted coverage
可能远低于 0.3。零通过也是有效 dev 结果，不能通过改用 ROI 分母、Render Mask 或
挑选另一种 weight map 隐藏这个结果。

## Core 与 Render 分离

Core 是候选真实 occluder footprint 的几何假设，全部科学指标只基于 Core。
Render 是 Core 的有界 Euclidean dilation，允许有限边缘、接触阴影及融合空间。
默认 requested radius 为原图短边的 0.8%，四舍五入后不超过 6 px；新增面积不超过
Core 的 30%，Render 总面积不超过原图 20%。这些上限均在配置中。

从 requested radius 向零递减，取首个满足面积上限的完整整数半径 dilation；不会
对边缘像素按 vulnerability 分数重排。Render 必须包含全部 Core，不能改善 Core 的
precision/coverage。没有可行扩张可退至零半径，并记录 effective radius；Core 自身
超过 render 总面积上限时记录 render infeasible，保留 Core 而不伪造扩张。

## 全量保存与复核

- `constraints.json` / `summary.json`：完整配置、指标公式、输入/代码哈希、所有档位统计。
- `candidates.jsonl`：每个候选的参数、六类指标、阈值诊断、Core/Render 路径和哈希。
- `samples.json`：每张 SOURCE 的全部 family、候选数和每档通过数；零候选明确记原因。
- 每图目录中的 `masks/*_core.png` / `*_render.png`：**全部**候选（含失败），原尺寸，
  uint8 `{0,255}`，无软 mask。
- `core_token_occupancy.npz`：候选 ID 与 float64 token cell coverage，shape `[N,16,16]`。
- `vulnerability_token_inputs.npz`：所用 ROI 和未经改变的 floating weights，含 image_key。
- 每图 `candidates.jsonl`、`summary.json`、`source.png`、`vulnerability_overlay.png`。

不覆盖已有输出。mask PNG 保存后逐一回读验证，结束时再次核验只读输入和代码哈希。
失败状态明确记录 ERROR，不把部分产物标为完成。

```bash
python -m unittest discover -s tests -p test_candidate_masks.py -v
python -m unittest discover -s tests -p test_targeted_mask_adapter.py -v
```

新增依赖为 SciPy（连通分量和 Euclidean distance transform），已写入 requirements。

## 当前 dev-20 导出

首轮以默认配置生成 5,706 个候选、11,412 张 Core/Render PNG。0.3 档共四个
parked_vehicle 候选通过全部 Core gate，来自 PRS:3993（三个）和 Osaka:3358（一个）；
0.5、0.7 档均为零。没有选定 winner，没有更换 weight map 或降低阈值。

原始 attention 的全图 weighted coverage 最大约 0.367802。facade 类型中 450 个候选
使用局部梯度方向，1,584 个明确记录为 image-axis fallback；这些都不是 facade 平面
验证。全部 5,706 个落盘候选均通过独立文件/数值一致性检查；12 项新测试和 14 项
ellipse baseline 回归测试通过。详细统计见输出目录 `AUDIT.md` 和 `validation.json`。
