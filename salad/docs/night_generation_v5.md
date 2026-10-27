# VPR 夜间生成 v5：整幅场景重新照明与区域检查

v5 是独立的小样本入口，使用真实源图重新生成，不对旧 v4 图二次编辑。旧 v3/v4 的 Prompt、计划、图像、审核结果均保持原样。默认计划最多 8 个地点；生成必须显式指定正的 `--max-calls`，不会自动扩展到全城市。

## 已完成的小样本结果（2026-10-26）

8 次调用、8 张完整落盘、8 张完成审核，没有重复请求。原几何/天气检查 8/8 通过；用同一套新增区域规则评估，旧 v4 为 2/8 candidate，新 v5 为 3/8 candidate，其余为 Unknown。没有把任何图片直接写入训练清单。

- Miami 3866 的白墙/停车场和 Miami 4855 的施工场景，夜间局部照明更明显。
- Lisbon 5443 从黑天空配白天道路，变为带局部灯照和暗区的道路；区域语义证据仍不足，保持 Unknown。
- Bangkok 2981 相比旧图更好地保留了门窗，但灯具是否完全来自原图仍需独立内容证据。
- Madrid 1724 主要变暗，仍不能据此证明已经得到合理夜间照明。
- 8 组下部裁剪的夜间相对竞争类别 margin 都提高，中心裁剪为 6/8 提高。这是未校准的 CLIP 描述性结果，不是统计显著性、真实照片质量或 VPR Recall 收益。

完整对照：[报告](../../outputs/weather_v5_pilot/comparison/report.md)、[前四组](../../outputs/weather_v5_pilot/comparison/comparison_01.jpg)、[后四组](../../outputs/weather_v5_pilot/comparison/comparison_02.jpg)。每组从左到右是源图、v4、v5。

生成进程总耗时约 276.6 秒（包括服务管理与审核），资源监控期间系统最低可用内存 3.44 GiB；推理中一次 GPU 快照占用约 41,001 MiB，该快照不是精确峰值。监控日志的进程组 RSS 不包含 systemd 启动的 Qwen 服务，不能当成整个生成系统的内存占用。结束后已确认 Qwen 服务 inactive、PID=0。29 项新增 CPU 测试通过；没有启动训练，也没有扩展批量生成。

## 针对当前问题的变化

- 夜间指令作用于墙面、道路、人行道、植被，即使天空完全不可见也必须改变日间光照。
- 区分“太阳投影”和物体边缘：移除日光直射、日光高亮与太阳投影，保留建筑、标线、文字和物体几何。
- 只允许已有光源产生局部照明；无光源时使用弱夜间环境光，允许暗处细节自然不可见，不要求每个立面都清晰明亮。
- 保留 LED、钠灯、稀疏照明、已有车灯四种变体；strict night 不再包含 blue_hour。
- 复用已有 CLIP，对完整图、中心裁剪、下部裁剪分别比较夜间、晴天、阴天和“保留日间光照的假夜景”文本集合。裁剪只是空间代理，不声称分割出了天空。
- 任一区域夜间解释没有胜过其他解释，标记 Unknown。所有区域通过也只是 candidate，不是语义真实或夜间相机真实性证明。没有新增平均亮度淘汰线，也没有拟合任意 margin 阈值。
- `base_quality_passed` 包含几何/天气与区域检查；这样后续内容审核不能误把区域 Unknown 升级为训练合格。所有生成记录初始 `eligible_for_training=false`。

实现位于 `AdaptVPR/experiments/qwen_curriculum/` 下的 `prompt_bank_v5.py`、`night_appearance_v5.py`、`night_first_v5.py` 和 `compare_night_v5.py`。

## 后端和资源

当前部署是 Qwen-Image-Edit-2511 + 已融合的 Lightning 4-step BF16 权重，继续使用 4 steps、guidance=1 和源图宽高比画布。不能只提高步数/CFG，就声称它变成了未蒸馏模型的高质量模式。

guidance=1 的当前 LightX2V 路径不编码 negative prompt，所以所有重要结构/光照约束都写在正向 Prompt 中。没有通过全图压暗、增加噪声或伪造新灯具来后处理生成结果。

复用已有磁盘卸载与 32 个常驻 block 配置；Qwen 串行生成，CLIP 在 CPU 上复用一个模型，匹配器沿用原有设置。没有加载额外大型视觉语言模型。本轮没有训练、微调或修改网络权重。

## 当前固定对照

`outputs/weather_v5_pilot/state/plan_config.json` 冻结了 8 个地点：此前诊断发现的 6 个问题图，以及 2 个普通夜景对照。每组沿用旧 v4 的真实源图、seed、变体、画幅和部署后端，仅修改正向 Prompt。选择是有意覆盖问题场景，不能据此估计整个数据池的成功率。

默认累计 HTTP 请求预算等于计划地点数。失败/中断请求也占预算；只有已经落盘的完整图能够不耗新请求地恢复。每次启动的 `--max-calls` 只是本次上限，不能绕过冻结的累计上限。需要额外重试预算时，在新的计划准备阶段显式传入 `--request-budget`，不改写已有计划。

```bash
cd /home/admin123/github/WORKSPACE
PY=/home/admin123/miniconda3/envs/AdaptVPR/bin/python
CLI=AdaptVPR/experiments/qwen_curriculum/night_first_v5.py

# 只核验现有计划、记录和图片，不调用模型。
"$PY" -B "$CLI" check
"$PY" -B "$CLI" status

# 仅在确实需要继续未完成的已冻结小试验时执行。
# 完整落盘图会恢复，已完成任务不会重复生成。
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -B "$CLI" run --max-calls 8

# 比较入口默认只检查已有配对，不加载 CLIP。
"$PY" -B AdaptVPR/experiments/qwen_curriculum/compare_night_v5.py

# 显式计算旧/新图的同一套区域指标，输出独立报告与三列对照图。
# 输出目录若已存在会拒绝覆盖；重新分析请给一个全新目录。
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -B \
  AdaptVPR/experiments/qwen_curriculum/compare_night_v5.py \
  --output-dir outputs/weather_v5_pilot/comparison --execute
```

需要新的一批时使用新的输出位置；准备阶段不调用 Qwen：

```bash
"$PY" -B "$CLI" prepare \
  --output-dir outputs/weather_v5_pilot_next/state \
  --image-root outputs/weather_v5_pilot_next --max-places 8

# 显式启动新批次，最多8次，结束后释放服务。
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -B "$CLI" run \
  --output-dir outputs/weather_v5_pilot_next/state \
  --image-root outputs/weather_v5_pilot_next --max-calls 8
```

默认新批次从已有 v4 真实源图清单确定性选择符合排除规则的 strict-night 配对。使用不同 `--seed` 可以改变抽样；Qwen 生成 seed 仍沿用所选旧图的 seed，以便同源比较。若要选择明确的场景，在 prepare 时提供 `--sample-ids <v4_id> ...`。已有 plan 不允许临时更换源图、seed、Prompt 或实现。

## 判断效果

查看 `comparison/report.md`、`pairs.jsonl` 与 `comparison_*.jpg`，同时核对：中心/下部日夜 margin 变化、几何是否仍通过、建筑与文字是否保留，以及是否有残留太阳投影或新增灯具。所有样本和失败结果均保留，不按“看起来最好”挑选报告。

区域 CLIP 检查尚未在独立真实夜景上校准。局部裁剪可能丢掉夜间上下文，所以 Unknown 不等于生成错误；同样，candidate 也不保证灯具、文字和局部结构完全没有编造。不得为了增加这 8 张的通过率调整规则或直接启动全量生成。相机退化增强仍是独立训练消融，不能修复错误的照明关系。

CPU 测试：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=AdaptVPR \
  "$PY" -m unittest discover -s AdaptVPR/experiments/qwen_curriculum/tests -p '*v5.py'
```
