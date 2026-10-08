# 生成端与验证器诊断

本目录延续 `outputs/gen_diagnosis/strategy_grid.py` 的诊断。实验代码独立于生产生成器、prompt 规则和验证门槛；保留原有工作区修改。

比较对象是固定的 8 张 Bangkok 源图 × 5 种天气，每个源图/天气只有一个 seed。源图取自原候选清单的 `sorted(unique(source_path))[::5][:8]`；seed 与旧实验相同。结果只能解释这个小样本诊断，不能代表跨城市效果或 VPR Recall 改善。

## 运行

在工作区根目录、AdaptVPR 环境中执行：

```bash
python AdaptVPR/experiments/generation_diagnosis/runner.py --mode iclight --output-dir outputs/gen_diagnosis/prompt_ablation
python AdaptVPR/experiments/generation_diagnosis/runner.py --mode qwen --num-sources 2 --output-dir outputs/gen_diagnosis/qwen_comparison_2src
python AdaptVPR/experiments/generation_diagnosis/verifier_controls.py
python AdaptVPR/experiments/generation_diagnosis/inspect_generated.py --run-dir outputs/gen_diagnosis/prompt_ablation --out outputs/gen_diagnosis/generated_geometry_iclight
python AdaptVPR/experiments/generation_diagnosis/inspect_generated.py --run-dir outputs/gen_diagnosis/qwen_comparison_2src --out outputs/gen_diagnosis/generated_geometry_qwen
python AdaptVPR/experiments/generation_diagnosis/rectify_qwen.py --run-dir outputs/gen_diagnosis/qwen_comparison_2src --out outputs/gen_diagnosis/qwen_rectification --evaluate-rectified
python AdaptVPR/experiments/generation_diagnosis/report.py --run-dir outputs/gen_diagnosis/prompt_ablation --run-dir outputs/gen_diagnosis/qwen_comparison_2src
```

`runner.py` 每张图完成后写入并刷新 `results.jsonl`；相同命令可继续未完成的实验。`generation_config.json` 固定输入哈希、代码哈希、prompt 和参数；改变配置或实现需要新输出目录。`--plan-only` 只记录实验计划。运行前要求已下载的模型和配置，Qwen 模式要求真实本地 HTTP 服务；不使用 mock。

### Qwen 画布修复后的复跑

2026-10-08 的 adapter 已使用 `source_aspect_v1`，为每个源图显式传入 `[height,width]` 目标画布；400×300 源图对应 1472×1104。HTTP response 和 `sampling` 记录策略、源尺寸、目标画布和实际原始尺寸。上游模型内部的 VAE/VL 图像缩放仍存在，这个修复不保证局部结构完全不变。

此前 `qwen_comparison_2src/` 的 20 张输出和报告描述的是旧 1664×928 画布。实现哈希已经改变，应保留旧记录，在新目录执行对照：

```bash
python AdaptVPR/experiments/generation_diagnosis/runner.py --mode qwen --num-sources 2 --seed 42 --output-dir outputs/gen_diagnosis/qwen_source_aspect_2src
python AdaptVPR/experiments/generation_diagnosis/inspect_generated.py --run-dir outputs/gen_diagnosis/qwen_source_aspect_2src --out outputs/gen_diagnosis/geometry_source_aspect_2src
python AdaptVPR/experiments/generation_diagnosis/report.py --run-dir outputs/gen_diagnosis/prompt_ablation --run-dir outputs/gen_diagnosis/qwen_source_aspect_2src --output-dir outputs/gen_diagnosis/report_source_aspect
```

新图库单独使用修复后的 Qwen run；同时输入旧/新 Qwen run 会因相同 method/source/condition/seed 键而由后者替换前者，不能这样构造画布 A/B 汇总。请分别保留两个图库与逐图分数。

## 对照设计与解释

- IC-Light：原版噪声初始化与源图初始化 SDEdit 0.85，各比较原 prompt、删除 `Avoid`/`Do not` 完整句子的 prompt、精简且天气置前的正向 prompt，共 240 张。删除句子的实验仍保留阴天的 `no strong shadows`；阴天/雾原 prompt 没有这些句子，作为不变对照。精简正向组还改变了措辞与天气强度，不能单独归因为去掉否定句。
- Qwen：用户选择本轮先用前 2 张源图 × 5 天气 × 2 prompt，共 20 张。脚本默认 8 源图，可在新的输出目录扩样。跨生成栈比较只使用同一 2 源图的 10 对；不能把 Qwen 的 10 张与 IC-Light 的全部 40 张直接作通过率比较。步数 4、guidance 1；该模型和采样器与 IC-Light 不同，因此只能比较这两个配置后的生成栈。原始输出与按生产包装器方式缩回源尺寸的输出都保存；当前默认输出宽高比可能不等于源图，需同时检查原始图。
- 验证器：8 源图 × 6 种构造明确的变换。恒等、极暗 gamma、极浓均匀雾不移动像素坐标；已知投影变换改变几何；替换四分之一画面破坏局部源图对应。极端光度变换不代表真实天气质量。匹配数、内点数、空间覆盖和估计 H 位移只用于诊断。
- 补充几何诊断默认仅检查精简正向组，IC-Light 80 张、Qwen 10 张；复用既有图，不增加生成调用。Qwen 对齐脚本要求完整 2 源图 20 张数据，将原始生成像素按拟合 H 变换到源图网格；缺失区域透明且不评分，没有源图混合或补边。H 拟合与重评使用同一图对，bilinear 与原 LANCZOS 还存在重采样差异，因此分数不构成独立结构真值。下一步优先修复上游源图宽高比。

所有 Global 判定仍为 `s_geo >= 0.78` 且 `s_div >= 0.15`。`s_geo` 是任意拟合 H 下的匹配内点比例，未检查 H 是否接近恒等、匹配是否足够或是否覆盖整幅图；`s_div` 是 CLIP 图像余弦距离，未检查天气语义。应分别看几何失败、差异不足、目视结构和目视天气，不把联合通过直接等同于合格增强样本。

新图保存 PNG，匹配器仍使用生产 evaluator 的 JPEG95 临时输入。旧 grid 的最终图为 JPEG95，因此以新实验内部配对作为 prompt 因果比较；旧采样对照与新对照分开列出。重跑的 80 张原 prompt 控制重新编码为 JPEG95 后，与旧 grid 的对应文件逐字节相同；分数小幅变化应结合最终保存格式解释。

## 产物

接续的天气信号校准脚本为 `calibrate_weather_signal.py`。它复用本地 CLIP，以审核文件中的源图/输出哈希和冻结生成记录核验配对，并按源图留一选门槛：

```bash
python AdaptVPR/experiments/generation_diagnosis/calibrate_weather_signal.py \
  --reviews outputs/gen_diagnosis/manual_reviews_iclight.jsonl \
  --out outputs/gen_diagnosis/weather_calibration_new
```

本次已重算的结果位于 `outputs/gen_diagnosis/weather_calibration_recomputed/`；历史 IC-Light 诊断的固定 `shift > 6` 保留 yes 79.1%、weak 55.3%、no 4.3%，按源图留一 balanced accuracy 为 0.8224。标签仍是原非盲 agent 目视诊断，不是独立真值。`--scores` 可复用脚本产生的哈希绑定缓存；`--legacy-probe` 仅显式导入旧无图像身份的分数，并保留来源不可验证的限制。IC-Light 的阈值不用于新的 Qwen 质量门限。原闭环生成与生成器 LoRA 训练实验已由 [分阶段 Qwen 数据构建](../qwen_curriculum/README.md) 替代；本目录保留历史诊断工具。

完整解释见 `outputs/gen_diagnosis/REPORT.md`；逐图证据、汇总 CSV 和本地 HTML 图库位于 `outputs/gen_diagnosis/report/`。原始旧 grid 与 photometric probe 保留。

CPU 协议检查：

```bash
PYTHONPATH=AdaptVPR python -m unittest experiments.generation_diagnosis.test_runner experiments.generation_diagnosis.test_rectify_qwen -v
```
