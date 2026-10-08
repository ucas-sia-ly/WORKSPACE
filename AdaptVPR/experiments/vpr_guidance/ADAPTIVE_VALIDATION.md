# 自适应生成与天气信号验证

日期：2026-10-08。接续 Claude 停在天气 contrast 门槛扫描处的实现；本记录区分机制检查、资源计数和待完成的检索效果实验。

## 已完成实现

- `adaptive_candidates.py` 将实验 IC-Light 两阶段生成、原 Global 验证、CLIP 天气变化和当前 SALAD 学生评分放在一个进程。默认 `SDEdit 0.85 + positive prompt`；Global 阈值仍是 0.78/0.15。
- CLIP 模型、源图/生成图特征、源图 VAE 条件，以及学生的真实正图、负池与真实视图校准均可复用。最终 JPEG 的 verifier 与学生预处理保持原协议。
- 质量与 plausibility 合格的候选达到 utility/mining 搜索目标后停止；预算耗尽则选已有合格候选，允许无样本退出。保留相同实验采样器的 `--sampling-policy fixed`。
- `run_loop.py --generation-mode adaptive` 使用在线 selected 继续学生训练与 full 的 LoRA；原固定 K 流程保留。自适应候选属于各分支当前学生，不跨学生共用。
- 固定全数据集的正图随机抽样上下文，消除候选组成/评分顺序造成的 Monte Carlo 差异。独立评分与选择保留 `weather_ok=False` 的拒绝状态。
- 配置绑定 prompt、源图、checkpoint、LoRA、模型目录清单、数据元信息与代码；候选保存图像/行哈希。模型目录清单是 size/mtime 检查，不是全部模型字节哈希；checkpoint、LoRA 和选中源图有内容哈希。忽略 `.git`/`__pycache__` 等运行缓存。
- 逐行 durable checkpoint、截断尾行恢复、损坏图片后的同组后缀重生成，以及完整结果的无模型复用。

## 天气门槛复核

新脚本：`experiments/generation_diagnosis/calibrate_weather_signal.py`。

已在 CPU 上以本地 CLIP 重新计算全部审核图对，并核对 frozen generation config、源图和输出图像 SHA-256。产物：`outputs/gen_diagnosis/weather_calibration_recomputed/`。旧 probe 仅用于数值核对，不作为新图像身份的证据；最大 shift 差为 `5.53e-05`。

160 条审核中排除 8 条 uncertain，剩余 152 条，来自 8 张源图；yes/weak/no 分别为 91/38/23。固定尺度为 `100 × [cos(image, condition_text) − cos(image, clear_day_text)]`，shift 为生成图减源图。它不是概率或经过校准的 log odds。

| 检查 | 结果 |
| --- | --- |
| shift 区分 yes/weak 与 no 的 AUC | 0.9097 |
| shift 区分 yes 与 weak/no 的 AUC | 0.7836 |
| 固定 `shift > 6` 的 yes 保留 | 72/91，79.1% |
| 同门槛 weak 保留 | 21/38，55.3% |
| 同门槛 no 保留 | 1/23，4.3% |
| 48 张原 metric passes 中保留 | 25/48 |
| 按源图留一、仅用训练源图选门槛的 balanced accuracy | 0.8224 |
| 同留一检验的 precision / recall / specificity（yes/weak vs no） | 0.9709 / 0.7752 / 0.8696 |

留一检验选出的门槛范围为 4.4773–5.6172；这支持保留可调门槛，而不是证明 6 是普适最优值。所有天气和方法按源图整体分 fold，但仍是同一城市、同一诊断和同一非盲 agent 目视标签，不能当独立人工真值或跨城市效果。

## 回归与真实 GPU 检查

执行：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -m unittest discover \
  -s AdaptVPR/experiments/vpr_guidance/tests -p 'test_*.py'
```

**133 项通过**。新增检查包括逐张/不同大小批次评分一致、真实校准 leave-one-out、最终 JPEG 预处理、质量门槛、固定/自适应预算、实际 main 的中断恢复/图像哈希修复、无模型复用、天气拒绝无法被独立 scorer 重新接纳，以及闭环命令和缓存签名。

真实 GPU 使用现有 `outputs/vpr_guidance_smoke/loop_codex/full/round_1/student/checkpoint.pt`，它是旧的小型机制检查 checkpoint，**不是完整 Bangkok real-only benchmark**。负池 64 places、真实校准 16 places、batch 32 places、每 place 4 images、16 negative draws；默认天气/停止门槛。现有 Qwen 服务和诊断任务未被中断。

| 小样本 | 实际生成 / 固定 K 上限 | Global 通过 | 最终 eligible / selected groups | 提前停止组 | 避免调用 |
| --- | --- | --- | --- | --- | --- |
| 原发布 prompt 分布过滤后抽取 6 组（全为 overcast） | 24 / 24 | 9 | 2 / 2 | 0 | 0 |
| 2 张诊断源图 × snow/fog/night，6 组 | 20 / 24 | 13 | 2 / 2 | 2 | 4，16.7% |

第一组位于 `outputs/vpr_guidance_adaptive_smoke/adaptive/`，保留的是最后两项缓存/边界修正前的代码指纹。最终代码的均衡检查位于 `outputs/vpr_guidance_adaptive_smoke/balanced/`，输入为同目录上级的 `balanced_prompts.jsonl`。均衡组的 fog 在第 1 张、night 在第 3 张达到目标；另外 4 组耗尽预算。两个达标候选的 mining probability 分别为 0.4375、0.375，utility 为 0.2672、0.2510。

均衡组单次运行约 62.5 秒，含模型加载/真实描述符/生成/验证；并行 Qwen 活动存在，**未运行独立固定 K 的时间或质量对照**。16.7% 指相对预定生成调用上限的减少，不是端到端加速比。两组完整实验重跑均输出 `validated complete ... no models loaded`；最终代码均衡组在无法访问 CUDA 的沙箱内也成功跳过模型。

`balanced/selected_contact_sheet.jpg` 保留源图和 selected 的并排检查。街道布局近似保留，但地面/立面细节仍被重绘，夜晚候选也可能接近暮色。通过这些启发式门槛不能认证局部结构或天气真实性。

## 尚未得到的结论

这轮完成了自适应机制和天气校准，没有重新进行完整 SALAD 训练、三分支 LoRA 多轮、多 seed 或外部 Recall 比较。发布模型近乎饱和且本次学生来自微型 fixture，不能由这些样本推断正式收益。

后续效果实验应使用充分训练的同起点 Bangkok 学生；在共享固定候选池上比较 random/select，再独立评估 adaptive 预算策略；保持 final 数据曝光一致，加入 matched real-only 对照，并完成完整 SVOX 各天气 Recall。自适应和固定 K 的搜索分布不同，匹配 place/prompt 组成并不能消除该差异。
