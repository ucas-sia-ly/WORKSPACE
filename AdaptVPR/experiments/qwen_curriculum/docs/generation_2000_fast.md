# 4090 48 GiB 的 Qwen 2000 张续跑

本轮目标是包括原图产物在内累计2000张，不是再生成2000张。沿用困难源图计划，night=1000、snow=400、fog=400、rain=200，四城市各500个源图，最多每地点2个不同源图。

## 性能配置与实测

保持 Qwen-Image-Edit-2511、已合并 Lightning 4-step BF16 权重、guidance=1、torch_sdpa 和 `source_aspect_v1` 画布。文本编码器、VAE、DiT 模块跨请求复用，前32/60个 DiT block 常驻 GPU，剩余 block 沿用双缓冲磁盘加载。只改变模型寿命和权重放置，不跳去噪步骤，不量化，不改变图像尺寸。LightX2V 的固定源码保持不变；适配器的 `/health.performance` 记录独立性能实现与内容哈希。

同一夜间源图、提示词和 seed=1541609810 的实测：原方案63.53秒，首次缓存加载66.39秒，缓存就绪后26.99秒。新旧1472×1104原始 PNG 的 SHA-256 均为 `a325dc2297abeb4f97de83bde6905c7f677f3729d74dd40370ac769fa0278eb8`，该样本逐像素相同。其余图片耗时随画幅和输入变化；单样本一致性不替代全体结构/天气验收。报告位于 `outputs/qwen_curriculum/performance_48g/parity_result.json`。性能诊断的重复源图不进入2000张训练清单。

`AdaptVPR/.env` 配置 `LIGHTX2V_MEMORY_PROFILE=resident_bf16_48g`、`LIGHTX2V_RESIDENT_BLOCKS=32`；磁盘 BF16 block 权重继续启用。用户服务的 `48g-performance.conf` 将 OMP/MKL 线程限制为4。缓存加载保留显存余量，设备显存不是48 GiB时应使用默认磁盘配置。原CPU offload全模型模式不能在32GB主存中直接启用。

## 恢复与进度

原 `generation_1000` 有998张成功输出、2条未知结果记录。两条记录均找到了源输入路径、seed和画布匹配的唯一原始图片，放入 `generation_2000/recovered_saved` 零调用恢复、重新验收。原结果与请求账本不修改；恢复证据明确记录原错误结果哈希和侧车不能独立证明的提示词/服务身份。

新1000张使用相同源图、域和种子计划，在 `generation_2000/additional_1000_fast` 生成，独立冻结新服务身份和运行配置。累计 `results.jsonl` 保留每条记录的原执行指纹和签名，按原2000条计划顺序汇总，引用原图片路径。验收通过清单另存 `training_manifest.jsonl`，完整训练实验使用全部成功输出，包含自动验收未通过图，延续此前700张实验口径。

```bash
cd /home/admin123/github/WORKSPACE
systemctl --user status qwen-curriculum-2000-fast.service
cat outputs/qwen_curriculum/generation_2000/summary.json
tail -f outputs/qwen_curriculum/generation_2000/fast_generation.log
```

关闭聊天不会停止 systemd 用户任务。中断或重启后使用同一目录恢复（不要与已在运行的后台任务同时启动）：

```bash
systemctl --user start adaptvpr-lightx2v.service
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/fast_campaign.py run \
  --output-dir outputs/qwen_curriculum/generation_2000
```

再次启动会核对历史账本、图片和代码指纹，尝试恢复已预留调用对应的已落盘图片。不会重复未知结果的POST。累计2000张均生成、验收和完整核对后才发布 `state=complete`，并停止Qwen服务释放显存。

## 2000张固定池训练

确认累计 `summary.json` 为 `state=complete`、`generated_images=2000`、`quality_evaluated=2000` 后执行。训练读取各阶段的原签名与原图片，不把旧执行指纹重新签成新生成结果。训练快照保存在新实验目录，原700张实验保留。

```bash
cd /home/admin123/github/WORKSPACE
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py run \
  --generation-run-dir outputs/qwen_curriculum/generation_2000 \
  --output-dir outputs/qwen_curriculum/ratio_2000_vpr_scratch_8to1_4to1 \
  --num-images 2000 \
  --epochs 50 --learning-rate 6e-5 --trainable-blocks 4 --batch-size 8 \
  --stop-qwen-service
```

文件名仍为 `test_700_ratio_experiment.py`，实际数量由 `--num-images 2000` 控制。命令串行训练四组：生成8:1/配对全真、生成4:1/配对全真，然后评估完整SVOX六子集。只加载DINOv2预训练，SALAD聚合器随机初始化，FP32、workers=0，每组50轮。8:1生成组每轮16000张真实图+2000张生成图；4:1生成组每轮8000张真实图+2000张生成图。各对照组使用相同源槽位、相同训练预算。

若选择比较 `reliability_ot` 新模块，先完成上述2000张原版基线，再用同一个图片池运行以下命令；700张基线无法用于2000张同池模块对照。

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/test_700_ratio_experiment.py run \
  --generation-run-dir outputs/qwen_curriculum/generation_2000 \
  --output-dir outputs/qwen_curriculum/ratio_2000_reliability_ot_8to1_4to1 \
  --num-images 2000 \
  --epochs 50 --learning-rate 6e-5 --trainable-blocks 4 --batch-size 8 \
  --reliability-ot \
  --baseline-dir outputs/qwen_curriculum/ratio_2000_vpr_scratch_8to1_4to1 \
  --stop-qwen-service
```
