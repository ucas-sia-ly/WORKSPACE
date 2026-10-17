# 全部23城市：不限总张数的困难源图生成和原始SALAD训练

本文对应已有冻结任务，尚未迁移到当前最少16个匹配点的审核规则 v2。恢复限制与迁移要求见 [RESUME.md](RESUME.md)；下列恢复命令以原实现及配置通过冻结校验为前提。

本地已逐城市核对CSV对应图片文件、地点标签及至少4张真实视角的要求：23城市共有524701张有效真实图片、62514个地点。城市名自动从`dataset/gsv-cities/Dataframes/*.csv`发现，保留OSL、PRG、PRS、TRT等原始CSV名称。

本轮全量源图筛选已完成：共79426张符合源图规则的候选，复用原7277张，新增72149张。新增生成预计需要约216.4GB存放原始PNG及训练PNG，另预留5GB；当前剩余空间和已生成张数以`summary.json`为准。按约27秒/张估算，新增生成约22.5天；这是候选数量，审核通过数量要在生成后确定。

保留原来四城的7277张生成图片和原签名结果。新增19城不设固定总张数，也不按每城250/500张截断：使用原始SALAD教师，可靠正样本相似度≥0.15，困难度位于各城70%–97.5%分位，每地点最多2张源图，保留全部符合条件的候选。源图筛选与生成后审核是两步，候选数量不等于最终审核通过数量。

为控制32GB主存和磁盘读取，新增城市逐城使用磁盘memmap特征缓存，难度负样本来自该城市内其他地点，并排除25米近邻。历史四城的原评分和生成图继续引用，不改写。源图路径、内容及样本ID全局去重。全量源图筛选已通过后台`qwen-source-mining-23city.service`运行，未自动启动Qwen生成或模型训练。

## 1. 查看全量源图筛选

```bash
cd /home/admin123/github/WORKSPACE
cat outputs/qwen_curriculum/mining_23city/progress.json
systemctl --user status qwen-source-mining-23city.service
```

等待`state=complete`。最终`all_city_selection.json`提供每城候选数、总数、新增数；`additional_sources.jsonl`是19个新增城市的全部候选。中断或重启后恢复该源图任务：

```bash
systemctl --user start qwen-source-mining-23city.service
```

也可在服务已停止时直接运行：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/mine_all_cities.py run
```

## 2. 生成全部候选并审核

源图筛选完成后先准备计划，核对最终数量和磁盘估算；该命令不启动Qwen：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/all_city_campaign.py prepare \
  --output-dir outputs/qwen_curriculum/generation_23city

cat outputs/qwen_curriculum/generation_23city/summary.json
```

累计天气比例保持night50%、snow20%、fog20%、rain10%，按累计目标精确整数分配。Qwen仍为同一2511模型、Lightning 4steps BF16、guidance1、source_aspect_v1高分辨率画布和32个常驻DiT blocks；本冻结任务记录原结构与天气验收门限。训练PNG仍LANCZOS缩回源尺寸，原始PNG保留。

不限张数可能需要比当前磁盘剩余空间更多的容量。`summary.json`中`required_free_bytes_estimate`为剩余图片预留估算（约3MB/图加5GB余量），`available_free_bytes`为实际可用空间；不足时生成入口在任何POST之前停止。若需要较大磁盘，在第一次生成前用该磁盘上的新`--output-dir`重新prepare/run；原四城图片继续引用原路径。单张稳定约27秒，可用最终新增数量估算总耗时。描述子缓存另约20GB，数字不包含已占用的旧数据。

确认计划和空间后执行生成：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
ADAPTVPR_CLIP_LOCAL_FILES_ONLY=1 \
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -u \
  AdaptVPR/experiments/qwen_curriculum/resume_campaign.py run \
  --output-dir outputs/qwen_curriculum/generation_23city
```

所有新增任务被自动拆成不超过1000次调用的阶段。全局域和源图计划不因分批而重排。已完成图片不重复生成；已保存但未审核的图先补审核；中断且没有完整输出的任务在下次显式恢复时独立重试，原失败记录保留。每次恢复时同一任务最多新调用一次，再失败就停，避免无限重试。全部任务验证后发布`state=complete`并停止Qwen服务，`training_manifest.jsonl`只含通过审核的图。

使用默认输出目录时可执行以下后台命令，关闭终端后继续运行；不要同时执行前台run。当前任务继续运行，新恢复入口在下一次暂停、恢复时接管：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/resume_control.py resume
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/resume_control.py status
```

需要电脑空闲时，使用`resume_control.py pause`同时停止协调器和独立Qwen服务，释放显存、内存；稍后使用`resume`。断电后也执行同一恢复命令。完整说明见[RESUME.md](RESUME.md)。如果使用其他磁盘上的输出目录，使用对应前台命令，或先停止服务再安装该目录的覆盖配置；不要同时运行旧入口。

## 3. 全23城训练命令

生成全部完成后先检查：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/home/admin123/miniconda3/envs/AdaptVPR/bin/python \
  AdaptVPR/experiments/qwen_curriculum/finetune_all_cities.py check-data \
  --generation-dir outputs/qwen_curriculum/generation_23city \
  --output-dir outputs/qwen_curriculum/finetune_salad_pretrained_23city \
  --epochs 10
```

正式训练及随后六域SVOX评估：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/home/admin123/miniconda3/envs/AdaptVPR/bin/python -u \
  AdaptVPR/experiments/qwen_curriculum/finetune_all_cities.py run \
  --generation-dir outputs/qwen_curriculum/generation_23city \
  --output-dir outputs/qwen_curriculum/finetune_salad_pretrained_23city \
  --epochs 10
```

加载用户指定的原始`salad/checkpoint/dino_salad.ckpt`完整骨干和原聚合器，不加载上轮微调模型；保持原始SALAD，不启用Reliability。训练参数保持与上一轮相同以便考察城市扩展：10轮、lr1e-5 cosine、AdamW weight_decay1e-4、末4个骨干block+聚合器可训、32地点×4图、224、FP32、workers0、无额外图像增强。完整23城真实池参加训练，仅合格源图按对应关系替换为生成图。实际生成图曝光比例由候选覆盖率及真实视角采样决定。

训练数量动态读取，不硬编码5004/7277。生成不完整、任一城市没有合格生成图、来源标签或图片校验失败时，拒绝进入部分城市训练。既定最后一轮之后自动评估原模型与微调模型的SVOX day/night/rain/snow/sun/overcast六个完整子集，均322×322、FP32、完整图库、25米正例；不使用SVOX挑选轮次。

查看训练进度：

```bash
cat outputs/qwen_curriculum/finetune_salad_pretrained_23city/progress.json
tail -f outputs/qwen_curriculum/finetune_salad_pretrained_23city/training.log
```

默认目录还可使用已安装的后台训练服务（与前台run二选一）：

```bash
systemctl --user start salad-finetune-23city.service
systemctl --user status salad-finetune-23city.service
```

输出`accepted_manifest.jsonl`、`accepted_summary.json`、`finetuned/checkpoint.pt`及六域`comparison.json`。同一run命令恢复最近完整epoch及已通过协议校验的评估子集。10轮训练时间随524701张真实图规模增加，不沿用旧四城任务的运行时间估计。

上一轮四城微调的六域平均R@1下降约0.25个百分点。扩展数据覆盖是新的实验设置，当前没有证据保证召回提高，仍以同协议完整测试结果为准。
