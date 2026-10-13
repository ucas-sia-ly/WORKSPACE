# 累计 7277 张困难源图 Qwen 生成与训练

保留已完成的2000张，新增5277张。用户选择保留困难度范围并让每张源图只生成一次，因此最终目标是7277张。源图仍来自四城真实训练池，可靠性条件、各城市70%–97.5%困难度范围、每地点最多2张源图均保持一致。所有源图路径、内容和样本ID全局唯一；不使用评估图片。

最终天气数量：night=3639、snow=1455、fog=1455、rain=728。按原50%/20%/20%/10%权重进行整数取整，无overcast。城市数量由该困难范围的实际容量决定：Bangkok=3658、BuenosAires=1498、LosAngeles=1033、Medellin=1088。

## 生成配置与恢复

沿用2000张快速阶段的完整服务身份和质量配置：Qwen-Image-Edit-2511，Lightning 4-step V1 BF16权重，4步，guidance=1，torch_sdpa，原发布提示词，负提示词为空，source_aspect_v1高分辨率画布。仍保留原始PNG和缩回源图尺寸的训练PNG。前32个DiT block、文本编码器、VAE和模型模块常驻复用。没有降低分辨率、量化或减少步骤。

新增任务为5个1000张批次和1个277张批次。每批沿用原生成器的独立调用预算、请求前落盘预留、每任务一次POST、未知请求不自动重试和中断图片恢复。聚合结果引用旧图片，保留原记录签名及执行指纹。所有批次完成、图片哈希和域数量验证通过后才发布`state=complete`，并停止Qwen服务释放显存。

```bash
cd /home/admin123/github/WORKSPACE
systemctl --user status qwen-curriculum-7277.service
cat outputs/qwen_curriculum/generation_7277/summary.json
tail -f outputs/qwen_curriculum/generation_7277/generation.log
```

后台任务由systemd用户服务管理，关闭聊天不影响运行。重启电脑后恢复同一任务：

```bash
systemctl --user start qwen-curriculum-7277.service
```

发生未知调用时任务停止并写入日志，恢复命令不会重复发出该POST。需检查已保存原始图片和预留记录后处理。当前服务不自动循环重启。

## 训练命令

原固定8:1/4:1脚本不能直接扩到7277张：它要求生成数量能被4整除，并要求全局不重复的真实上下文图片；8:1还需要65493个源槽位，超过四城45661张真实图片。此次训练采用现有完整四城池的精确源图替换采样。

以下命令在生成完成后执行；生成未完成时会拒绝开始训练。默认延续之前实验的全部成功输出口径，包括自动质量检查未通过的图片。训练选择清单单独记录原质量结果和原签名；原生成/验收记录不修改。

```bash
cd /home/admin123/github/WORKSPACE
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/train_scaled.py \
  --generation-run-dir outputs/qwen_curriculum/generation_7277 \
  --output-dir outputs/qwen_curriculum/train_7277_replace \
  --epochs 50 --replacement-probability 1.0
```

默认DINOv2预训练骨干、随机SALAD聚合器、后4层可训练、FP32、lr=6e-5、weight_decay=0、batch=8个地点、每地点4张、workers=0、不做图像增强、seed=42。完整真实池为5710个地点、45661张真实图。每次先采样不同真实源图，拥有生成版本的源图被替换，其余仍使用原图；源图上限为每地点2张，能保留真实上下文。1.0是条件替换概率，不表示全部训练图都是生成图。每轮从候选库采样，并非强制全部7277张都出现一次。此实验不沿用固定8:1/4:1曝光协议，不能把两种协议的结果直接视作同条件对照。可用`--replacement-probability 0.5`降低替换频率。

可在上述命令最后增加`--check-data`只检查数据；增加`--selection accepted-only`仅使用质量验收通过的图片，并选择独立输出目录。纯真实全池基线：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/train_scaled.py \
  --generation-run-dir outputs/qwen_curriculum/generation_7277 \
  --output-dir outputs/qwen_curriculum/train_7277_real_baseline \
  --selection real-only --epochs 50
```

两组具有相同真实源池、训练初始化、地点采样规则和训练预算。训练完成的`checkpoint.pt`可用同一入口的`--resume 路径 --epochs 新总轮数`续训。SVOX评估需要另行使用统一评估协议；本入口只负责训练，不自动宣称评估结果。
