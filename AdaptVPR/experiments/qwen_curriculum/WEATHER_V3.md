# 新一轮：覆盖 place，先完成 night

后台入口为 `night_first_v3.py`。本轮数据独立存放于 `outputs/weather_v3/`，与旧计划的 `outputs/qwen_curriculum/` 并列；状态位于 `outputs/weather_v3/state/`。旧两万多张及其账本保留，本轮使用独立计划。

本次已将原 `outputs/qwen_curriculum/images/weather_v3/` 图片和原 `outputs/qwen_curriculum/state/weather_v3/` 状态迁至上述独立目录，旧根目录已移除；4 份一次性迁移回执已清理，历史归档保留在 `outputs/qwen_curriculum/state/history/`。

当前因人工发现局部内容问题而保持暂停。已有 `accepted` 表示通过当时的自动审核，仍可能包含新增车辆、源图拼接畸变等问题，不能视为人工确认无缺陷。目录整理不会自动重启生成或修改既有审核结果。

## 源图和比例

23 城市共有 62514 个满足至少 4 个视角的 place。CPU 检查解码、曝光、模糊和结构纹理，按每个 place 的固定随机视角顺序挑选，14 个地点没有合格源图。随后复用经过完整校验的原始 SALAD 可靠度记录，要求有效同地点正样本相似度至少 0.15，不使用旧困难度分位：替换 124 个低可靠视角，再排除 1 个没有同时通过画质与可靠度要求的地点。

最终 62499 个 place，每个 place 只安排 1 张。全局无放回随机排列地点，再独立打乱域配额，不按城市轮流生成。seed 固定为 42，暂停和恢复不会重排。

| 域 | 比例 | 计划数 |
|---|---:|---:|
| night | 44% | 27499 |
| rainy_night | 14% | 8750 |
| rain | 15% | 9375 |
| snow | 14% | 8750 |
| sun | 8% | 5000 |
| fog | 5% | 3125 |

整数配额按最大余数法分配，overcast 不安排任务。目前只执行 night，完成后停止并释放 Qwen，不自动切到其他域。源图筛选不保证生图一定通过；生成后仍执行当前 `quality_v3.py` 的结构和天气审核，拒绝图保留记录，不反复生成。

## 图片与复用

```text
outputs/weather_v3/
  night/ rainy_night/ rain/ snow/ sun/ fog/  训练尺寸图，按域独立
  raw/<域>/                               高分辨率图与服务元数据
  accepted/<域>/                          审核通过入口，链接不重复占空间
  state/
    source_screen/                        可恢复的 CPU 筛选
    source_reliability/                   可靠度证明、替换及排除记录
    plan.jsonl                            固定六域计划
    attempts.jsonl                        POST 前持久记账
    generated.jsonl                       完整保存的图片
    results.jsonl                         审核结果
    training_manifest.jsonl               本轮通过审核清单
    summary.json                          进度和每域计数
    coordinator.log                       后台日志
```

Pilot 有 71 个完整 prompt、模型、采样与审核设置兼容的合格候选。遵守一地一图、随机分域和源图可靠度后，实际复制复用 15 张，其中 night 10 张。保留原 seed 和来源，原 pilot 文件不移动。旧 21077 张与当前 prompt 没有精确匹配，未混入本轮。

因此 night 初始需新增 27489 张，剩余数量以 `state/summary.json` 为准。其他域目前只有可复用图，没有新增生成调用。

## 常用命令

```bash
cd /home/admin123/github/WORKSPACE
conda activate AdaptVPR

python AdaptVPR/experiments/qwen_curriculum/resume_control.py pause
python AdaptVPR/experiments/qwen_curriculum/resume_control.py resume
python AdaptVPR/experiments/qwen_curriculum/resume_control.py status
```

关闭终端后继续运行。pause 同时停止协调器与 Qwen 服务，释放其显存和内存。重启后仍用同一 resume 命令。

请求在 POST 前记账。完整 raw 和 sidecar 已保存时先恢复补审核；已保存图审核中断时只补审核；没有完整可确认输出时，下次显式 resume 使用原参数独立重试一次，再失败就停止。未完成请求的输入保留到图片提交后。

查看日志：

```bash
tail -f outputs/weather_v3/state/coordinator.log
```

暂停后做完整性检查，不加载模型、不新增调用：

```bash
python AdaptVPR/experiments/qwen_curriculum/night_first_v3.py check
```

再次核对已有准备结果（CPU，不启动生成）：

```bash
python AdaptVPR/experiments/qwen_curriculum/night_first_v3.py prepare
```

Qwen 仍为 2511、Lightning 4steps BF16、guidance 1、原始比例高分辨率画布和 32 个常驻 DiT blocks，采用当前 prompt bank。不要同时启动旧生成入口。

当前不自动微调；旧 SALAD 训练说明属于旧计划，本轮训练数据应使用这里独立的审核通过清单。
