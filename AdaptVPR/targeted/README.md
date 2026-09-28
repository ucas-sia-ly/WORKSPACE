# Vulnerability → compact generation mask

独立的后续 scene-aware planner 已新增，见
[Stage3 targeted planner](../docs/stage3_targeted_planner.md)。下面仍描述纯几何 adapter；
planner 复用其已有 mask，仅判断固定区域可编辑性，不改变这里的适配行为。

这是独立的几何适配层。Vulnerability mask 是 retrieval sensitivity 的目标，
不能直接等同于适合真实物体插入的形状。本模块只生成新的 binary generation mask；
不修改原 target、生成模型或现有 inpainting 流程，不调用 diffusion/VLM，不训练。

```python
from targeted.mask_adapter import adapt_generation_mask

generation_mask, diagnostics = adapt_generation_mask(
    vulnerability_mask,       # PIL 1/L，或 bool/uint8 二维数组
    target_ratio=0.06,         # 占整幅图像面积，配置范围 0.04–0.08
    min_overlap=0.70,
    image_height=H,
    image_width=W,
)
if generation_mask is None:
    print(diagnostics["failure_reason"])
```

成功返回一张新的 PIL `L` 图像，像素严格为 0/1，尺寸与输入一致。失败返回
`(None, diagnostics)`，不会输出勉强生成的区域。空 mask 是明确失败；非法参数、
非 binary 输入或 H/W 不匹配抛出 `ValueError`。输入支持 0/1、0/255 或 1-bit；
soft mask、RGB mask、混合 1/255 前景会被拒绝。原始输入不被修改。

## 构建规则

1. 在原始分辨率直接 rasterize 实心椭圆：默认宽高比候选
   `1, 0.75, 4/3, 0.5, 2`，最大长短轴比不超过 2。使用像素中心的解析边界，
   无 resize、bbox 填充、thresholding、dilation 或 morphological smoothing。
   平滑轮廓由椭圆形状直接提供。
2. 面积尽量接近 `target_ratio * H * W`。`area_tolerance=0.05` 表示相对目标面积
   ±5%：目标 6% 时允许 5.7%–6.3%；另强制实际面积仍在整图 4%–8% 内。
   像素取整后无合法面积时明确失败。
3. 在全图搜索，不限制为原 vulnerability bbox。利用逐行前缀和精确计算
   `overlap_ratio = |M_edit ∩ M_vuln| / |M_edit|`，即候选区域内的 vulnerability
   密度。优先最大 overlap；同分时优先靠近 vulnerability centroid，再比较面积
   偏差、compactness 和固定坐标顺序。
4. 初始位置包括规则网格、图像边界、原 mask 的 centroid 和最多 32 个最大连通块
   的 centroid。默认步长 `max(1, min(H,W)//32)`；每个形状最好的 8 个位置在邻域内
   做逐像素 refinement。没有随机生成形状，结果与全局 RNG 状态无关。
5. 成功结果必须是一个 8-connected component，满足面积、overlap 和 compactness。
   原始连通块数也按 8 邻域定义。最终 mask 不经任何 bilinear/nearest resize；
   可视化中的缩略图只缩放 RGB 展示图片，不参与输出 binary mask。

## Compactness 的明确定义

使用平移/尺度归一化的二阶矩 compactness：

```text
Q = A² / (2πJ)
J = 对 mask 内所有像素，累加距 centroid 的平方距离，再加每个单位像素的内在矩 1/6
```

理想连续圆盘为 1，长条、分散区域和不规则轮廓得分更低。计算包含像素的内在矩，
避免离散像素中心估计造成圆盘得分虚高。默认要求新 mask 的 Q 不低于旧 mask；
可用 `min_compactness_gain` 要求明确的最小提升。原图如果已近乎最优紧凑，不强行
声称一定有严格提升；不满足该约束时返回失败。本次 20 个真实样本均有严格提升。

这里的 compactness 不代表场景语义、物体落地位置或视觉真实性。椭圆可以覆盖墙面、
树木等敏感区域；本阶段不能据此宣称物体插入 realism 已被验证。

## Diagnostics 与失败语义

包含用户要求的所有字段：`vulnerability_area`、`generation_area`、`overlap_pixels`、
`overlap_ratio`、`centroid_vulnerability`、`centroid_generation`、
`centroid_distance_normalized`、`num_components_before`、`num_components_after`。

面积单位为像素；额外记录占整图比例。Centroid 为 `[x,y]`，像素中心坐标从
`(0.5,0.5)` 起；距离除以图像对角线 `sqrt(W²+H²)`。另记录搜索参数、实际面积误差、
候选数、前后 compactness、椭圆范围及失败原因。

主要失败原因：

- `empty_vulnerability_mask`：输入无前景。
- `insufficient_vulnerability_area_for_required_overlap`：即使全部原前景都落入候选，
  也不足以满足最小合法面积的 overlap，属于确定的面积上界判断。
- `no_integer_area_in_requested_tolerance`：面积窗口内没有合法整数像素数。
- `no_template_meets_area_size_and_compactness_constraints`：给定形状族内没有满足
  面积、尺寸及 compactness 的模板。
- `no_compact_candidate_meets_overlap_in_configured_search`：当前搜索没有满足 overlap
  的候选；**不等同于数学上证明所有形状或位置均不可行**。可显式调整允许的配置后重试，
  本函数不会静默降低 overlap、缩小目标面积或扩大 bbox。

## 随机 20 图可视化

在 AdaptVPR 目录运行：

```bash
python scripts/visualize_generation_masks.py \
  ../../Bag-of-Queries/outputs/stage3/targets/targets.jsonl \
  --count 20 --seed 0 --target-ratio 0.06 --min-overlap 0.70
```

输入通过既有 model-free reader 的 schema、身份、尺寸与哈希校验。先按 sample_id
排序，再使用独立 `random.Random(seed).sample` 抽样；seed 只影响选样，不影响
单个 mask 的适配。脚本复用 reader 校验整个 manifest，再为抽中的记录保存可视化。

默认输出 `outputs/stage3_targeted/generation_masks/`。每个样本保存 `source.png`、
`vulnerability_mask.png`、`vulnerability_overlay.png`、`side_by_side.png` 和
`diagnostics.json`；成功时另存 binary `generation_mask.png` 与
`generation_overlay.png`。失败时 comparison 明确标注 FAIL，不伪造 generation mask。

目录内还有 `contact_sheet.png`、`generation_masks.jsonl` 和 `summary.json`。
分布只统计成功样本，失败另计数并记录原因；百分位使用线性插值。所有 source/原 mask
只读，导出路径独立于 BoQ artifact；已有输出目录不会被覆盖。

```bash
python -m unittest discover -s tests -p test_targeted_mask_adapter.py -v
python -m unittest discover -s tests -p test_targeted_inputs.py -v
```

本阶段不运行 `test_targeted_editor.py`：其中有上一阶段的真实 Diffusers 采样测试，
不属于本次纯 mask adapter 验证范围。可视化集成测试在独立进程禁止导入 torch、
diffusers、transformers、TargetedEditor、planner 和 verifier。
