# CCR-VPR Stage 2B：同视角 Patch 对应与监督可信度

本阶段只增加对应接口、离线诊断和几何证据缓存，不修改主训练 Loss，
不创建教师特征缓存。运行入口为 `salad/diagnose_pair_correspondence.py`。

## 接口与坐标

```python
from workflow.pair_correspondence import build_pair_correspondence

result = build_pair_correspondence(
    source_image, generated_image,       # PIL RGB，精确 manifest 源图配对
    source_teacher_aux, generated_teacher_aux,
    calibration=calibration,             # 独立校准数据拟合，无固定相似度默认值
    search_radius=1, alignment_tolerance=1,
    geometry=geometry_evidence,          # 可选的 LightGlue/RANSAC 局部内点
    quality=manifest_row,
    edit_mask=None,                      # 已知编辑区域：只排除正监督
    occlusion_mask=None, destruction_mask=None,
)
```

一次调用处理一对图像。局部特征接受 `[1,C,H,W]` 或 `[C,H,W]`；
Score/OT 接受 Stage 2A 的 `score_logits`、`log_assignment` 和 `assignment_keep`。
输出在**生成图的 Patch 网格**上，形状为 `[Hg,Wg]`：

| 输出 | 含义 |
|---|---|
| `match_index` | 对应的真实源图展平索引，`source_y * Ws + source_x`；没有正监督时为 `-1` |
| `positive_conf` | 正向证据强度；A/B 的计算方式不同，不是已校准的正确概率 |
| `negative_conf` | 明确遮挡/破坏 mask 在该 Patch 的面积比例；没有证据时为零 |
| `unknown_mask` | 正负强度均为零，不参与监督 |
| `match_diagnostics` | 候选索引、双向一致性、余弦、双向最小最佳/次佳差距、结构误差、位移、几何投票、A/B 分类、原始审核记录等 |

索引方向固定为 generated → source，避免后续蒸馏反向索引错误。
所有目标均 detached；计算不消耗全局 RNG。相似度、局部统计采用 FP32。

坐标假设为全帧 resize；不隐式猜测 crop、padding 或翻转。
图像原始尺寸和两侧网格可以不同：用归一化 Patch 中心寻找对应侧最近坐标，
再在该位置的局部邻域搜索。`search_radius=1` 是 3×3，包含一个 Patch 的偏移。
`alignment_tolerance` 范围为闭区间 `[0, search_radius]`。
旧 `reliability.build_local_targets()` 已同步修正此边界；默认 radius=3/tolerance=1 行为不变。
radius=tolerance 时没有“超出容差”的候选，不会产生旧接口的远距离位移负标签。

LightGlue 坐标必须附带其**实际输入 tensor 的 H/W**，不是原图大小或猜测的 512。
点坐标为零起始 XY 像素中心，映射为 `floor((x + 0.5) * Wpatch / Wcanvas)`；
越界坐标不夹入边缘 Patch。生成图 mask 使用原图 H/W，按完整像素单元与 Patch
单元的面积重叠投影，可处理不能整除的尺寸，任何已知遮挡/编辑重叠都阻止正监督。
重复纹理检查按 `patch_stride=14` 重建教师的全帧输入网格；其他步长须显式传入。

## 证据规则

**A：高置信几何/结构对应。** 必须是双向一致、位移在容差内、且存在足够
最佳/次佳区分的教师对应；另外满足校准后的相对结构一致性，或得到局部几何
同一索引的支持。教师与有效局部几何明显冲突时保留 Unknown。
相对结构使用局部自相似，复用 `reliability.py` 的空间移位实现，未复用其固定经验阈值。

**B：困难 Positive 候选。** 选中对应的教师余弦低于校准界线，但独立 LightGlue
内点支持局部对应。要求至少两个不同内点、生成→源与源→生成的严格多数一致，
重投影满足原 matcher 的 RANSAC 容差，位移在允许范围内。
这仍是候选，不是建筑、静态地标或语义正确性真值。B 的几何一致性强度可能比 A
的特征强度大；报告分别保存两类分布，不能把两类值直接解释为概率并比较。

重复像素 Patch 的歧义检查优先于 A/B。它用于防止位置编码给完全重复纹理制造
唯一性；近似重复仍依赖特征差距与几何证据，不声称解决所有重复立面问题。

**Negative：** 只接收明确提供的 `occlusion_mask` 或 `destruction_mask`。
普通 `edit_mask` 仅排除正监督，不等于内容已破坏。不能从天气名称、提示词、低余弦、
Sobel、高 Keep 或 LightGlue 失配推导负标签。没有证据时允许整张图全部 Unknown。
输入 mask 与正监督不冲突；重叠区域优先保留明确的负证据。

manifest 的显式质量拒绝会阻止正标签，但不会生成负标签。
整图审核通过只表示这对图像允许进入诊断，不会把全图升级为可靠监督。
Teacher Score/OT/Keep 仅做诊断，不作为标签真值。

## 校准与诊断设计

生产接口没有默认余弦/差距/结构阈值。不给 `calibration` 时关闭教师单独建立的
正监督；仍可输出独立几何候选。

本次实验按 manifest 路径的 SHA256 和固定 seed 抽取四种天气各 6 对，
共 24 个不同地点。另取四种天气各 3 个地点，总计 12 个，保证与诊断地点不重叠：

1. 把校准真实图和另一个地点的生成图组成错配对，获得 3072 个局部候选统计。
2. 余弦上界和相对结构误差下界从错配分布取分位数。
3. 最佳/次佳差距下界从这些校准源图的同图复制对照取分位数。
   实测发现仅用错配分布的差距上界会过度过滤平滑但正确的 Patch，因此未采用。
4. 显式扫描 `--null-tail-probabilities .01 .025 .05`；预览固定使用第一个操作点，
   不自动选择覆盖率最高的一组。
5. 校准地点之外，对四个源图分别做复制、一 Patch 平移、已知遮挡、降低亮度、
   完全重复纹理、无关图像等 24 项构造对照。

分位数操作点不是已验证的误匹配概率，Patch 统计并非独立样本；
本次小规模构造对照也不能替代真实天气的人工逐 Patch 标注。
校准记录包含教师路径/SHA256、manifest SHA256、具体错配路径、采样 seed 和方法。
更换教师、预处理、Patch 网格或搜索半径后应重新校准。

## 本次实测结果

最终输出：`salad/outputs/ccr_stage2b_diagnostics/index.html` 与 `report.json`。
教师沿用 Stage 2A 的完整已训练 checkpoint，SHA256：
`f06772e6b22c5335efccdf0afd23a1afe18442703299281d23b0641286589513`。
输入 224×224，16×16 Patch 网格；每个天气 6 对、1536 个 Patch。
Teacher 在线 GPU 前向，LightGlue 在 CPU 执行；没有 optimizer 或训练步骤。

| 天气 | A 覆盖 | B 候选覆盖 | 合计 Positive | Negative | Unknown | A 强度中位数 | B 强度中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Night | 18.10% | 16.73% | 34.83% | 0% | 65.17% | 0.167 | 0.800 |
| Rain | 12.70% | 15.95% | 28.65% | 0% | 71.35% | 0.176 | 0.750 |
| Snow | 14.71% | 13.02% | 27.73% | 0% | 72.27% | 0.202 | 0.800 |
| Fog | 16.60% | 15.10% | 31.71% | 0% | 68.29% | 0.214 | 0.750 |

没有真实破坏/遮挡 mask 的天气样本保持零 Negative，这是证据缺失的正确处理。
以上为候选覆盖率，**不是天气对应准确率**。未对这 24 对进行人工逐 Patch 真值标注。
预览可用于检查建筑、道路、车辆、植被等位置；有几何对应仍不代表该处是静态地标。
两类强度只表示各自证据，不可根据 B 的数值更大推断 B 更可靠。
完整分位数和逐对数据保存在报告中。

第一操作点 0.01 的实测校准值：余弦下界 0.814825、差距下界 0.002914、
相对结构误差上界 0.112121。这些值只保存在此次校准结果中，没有硬编码进模块。
0.025 / 0.05 操作点也已运行；因 A/B 分类与歧义下界同时变化，总覆盖率不要求单调。
代码没有自动根据天气覆盖率选择操作点。

独立构造对照，每类四个源图、合计 1024 个 Patch：

| 对照 | 正标签数 | 负标签数 | 有正标签位置的坐标准确率 |
|---|---:|---:|---:|
| 同图复制 | 1018（99.41%） | 0 | 100% |
| 向右平移一个 Patch | 720（70.31%） | 0 | 99.44%（4 个错误） |
| 已知遮挡 | 876 | 64 | 100%；64 个遮挡 Patch 均无正标签 |
| 降低亮度、结构保留 | 684（66.80%） | 0 | 99.85%（1 个错误） |
| 完全重复墙面纹理 | 0 | 0 | 全部 Unknown |
| 不同地点无关图像 | 3 | 0 | 3 个误标，其余 1021 个 Unknown |

无关对照的 3/1024 误标和位移对照的少量错误说明仍需真实天气人工核验；
特别是 B 只能作为困难候选交给后续阶段评估，不能宣称已经获得无误差监督。
本次通过 44 项针对性 CPU 单元/回归测试，包含 Stage 2A 教师与旧可靠性接口。
几何缓存约 4.6 MiB，预览和 JSON 约 21 MiB；没有保存教师特征。

## 几何缓存与输出

复用 `AdaptVPR/experiments/qwen_curriculum/quality.py` 的严格 matcher 初始化和
homography 有效性检查，以及生产 JPEG95 → vismatch 的处理路径。
只运行 SuperPoint-LightGlue 和 RANSAC，不启动 CLIP 或新的图像生成。
没有隐式 SIFT fallback；几何失败会明确记录，后续以未知证据处理。

缓存键包含两侧 RGB 内容哈希、原始尺寸和 matcher 参数。缓存仅存内点坐标、H、
实际匹配画布大小和相关统计，不保存 DINO/Score/OT 特征。
诊断 manifest 的源图和生成图会核对已有 SHA256。
可选 mask 字段为 `edit_mask_path`、`occlusion_mask_path`、`destruction_mask_path`，
有对应 `*_sha256` 时也会校验。没有字段就不虚构 mask。

独立实验目录包含：

- `index.html`：可人工逐对检查的预览入口；包含真实图、生成图、对应线、Positive/Negative/Unknown 热力图。
- `report.json`：每种天气覆盖率、A/B 各自置信度分位数、所有操作点、构造对照结果。
- `selected_pairs.json`、`calibration.json`：完整样本和校准来源。
- 每对图像的 PNG 和 JSON：像素/特征对应诊断，便于追查单个 Patch。
- `geometry/` 或指定的共享几何缓存目录：轻量几何证据。

## 运行命令

从工作区根目录运行。下面只进行有限数量的教师前向和局部匹配，没有训练步骤：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=salad \
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/diagnose_pair_correspondence.py \
  --manifest outputs/qwen_curriculum/generation_23city_v2/training_manifest.jsonl \
  --teacher-checkpoint outputs/qwen_curriculum/ratio_700_vpr_scratch_8to1_4to1/generated_8to1/checkpoint.pt \
  --backbone-repo /home/admin123/.cache/torch/hub/facebookresearch_dinov2_main \
  --output-dir salad/outputs/ccr_stage2b_manual \
  --pairs-per-weather 6 --calibration-pairs 12 \
  --null-tail-probabilities .01 .025 .05 \
  --device cuda --matcher-device cpu --seed 42
```

输出目录必须是新目录。可用 `--geometry-cache 路径` 复用已校验内容身份的几何结果。
CPU 单元与回归测试：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=salad:salad/tests /home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  -m unittest test_pair_correspondence test_reliability_supervision test_frozen_teacher -v
```
