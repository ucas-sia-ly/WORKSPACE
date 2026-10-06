# 面向最终实验的 VPR-aware Global generator

唯一推荐流程：`train_full` → TensorBoard → `evaluate_all`。
最终 VPR 结论来自真实 SVOX、RobotCar-Seasons 和 Nordland。生成图的 verifier 分数、SALAD cosine、loss 仅用于 generator monitoring。

## 环境与数据

在 WORKSPACE 根目录、现有 AdaptVPR Python 环境运行：

```bash
python -m pip install -r AdaptVPR/experiments/vpr_guidance/requirements.txt
export ICLIGHT_ROOT="$PWD/IC-Light"
export ICLIGHT_BASE_MODEL_PATH="$PWD/models/stable-diffusion-v1-5"
export ICLIGHT_MODEL_PATH="$PWD/models/iclight-ckpt/iclight_sd15_fc.safetensors"
export VISMATCH_ROOT="$PWD/vismatch"
```

使用本地 `salad/`，不需要重新 clone。SALAD teacher 读取官方预训练权重；fresh SALAD 使用预训练 DINOv2 backbone + 随机初始化 SALAD aggregator，不复用 teacher 的 aggregator 权重。Torch Hub 的权重应已缓存，或运行环境应允许下载。

保留当前目录即可：

```text
dataset/
  AdaptCities/prompts/adaptcities_160k_prompts.jsonl
  gsv-cities/
    Dataframes/<city>.csv
    Images/<city>/*.jpg
  svox/images/test/
    gallery/
    queries/
    queries_night/ queries_rain/ queries_snow/ ...
  nordland/
    README.txt
    images/test/database/
    images/test/queries/
  RobotCar-Seasons/
    images/<condition>/<left|rear|right>/*.jpg
    metadata/robotcar_v2_train.txt
    metadata/robotcar_v2_test.txt
    3D-models/individual/colmap_reconstructions/001_aligned.zip ... 049_aligned.zip
```

RobotCar metadata 文件名末尾现有的 `?utm_source=chatgpt.com` 也能自动识别，可自行去掉后缀以便管理；不要同时保留两份同名 train metadata。官方 COLMAP archives 可直接读取，无需解压 `points3D.txt`。也支持原生 `images.txt` / `images.bin`。

## 1. 完整训练

```bash
python -m AdaptVPR.experiments.vpr_guidance.train_full \
  --prompts dataset/AdaptCities/prompts/adaptcities_160k_prompts.jsonl \
  --gsv-root dataset/gsv-cities --salad-root salad \
  --conditions snow night rain fog --output-dir outputs/full_run \
  --generation-passes 2 --chunk-size 128 --generator-steps-per-chunk 256 \
  --replay-rounds 2 --lora-rank 8 --lora-alpha 8 --generator-lr 1e-4 \
  --lambda-diff 1.0 --lambda-vpr 0.1 --lambda-keep 0.05 --timestep-window 10 \
  --real-ratio 8 --synthetic-ratio 1 --salad-steps 4000 --salad-workers 8 \
  --seed 42 --tensorboard-dir outputs/full_run/tensorboard
```

`--cities`、`--salad-batch-size`、`--salad-image-size` 可控制三套 SALAD 的共同设置。SALAD training 默认 image size 224，真实 evaluation 默认 322×322。这些设置对 A/B/C 完全一致。

编排按顺序执行 source preparation、online generator、B pool、C pool、fresh SALAD A/B/C，随后停止，不自动 benchmark。

- `prepare_data` 只验证 Global source、GSV dataframe label、目标 condition、源图可读性，并缓存 frozen SALAD source descriptor。输出 `source_manifest.jsonl`，不生成 baseline。
- 每 pass 在 condition 内 deterministic shuffle，然后轮流取样形成 chunk；每条 prompt 在该 pass 恰好出现一次。某 condition 耗尽后重新分配剩余 slots，并记录实际分布。
- 当前 `G_r` 对整个 chunk 完整执行 released IC-Light stage 1 + refinement。训练时的一次 predicted-x0 只用于 loss 估计，不能替代 candidate refresh。
- 每张 candidate 进入真实 `DualTraitEvaluator` 的 Global route。阈值保持 `geometry >= 0.78`、`diversity >= 0.15`。两者通过才允许进入训练，rejected 只保存审计。
- accepted 当前图片 `y_r` 编码成 latent；low-noise timestep 下计算 diffusion MSE、SALAD source cosine loss、相对于 `y_r` 的 L1 keep loss。只更新 fp32 LoRA 参数；teacher、VAE、text encoder、base UNet 固定。
- two-pass first-order gradient 保留：Pass A 对同一 predicted-x0 计算 VPR/keep gradient，Pass B 用 VJP proxy 传给 LoRA。
- 每 step 以 50% 概率选 current accepted pool，50% 选最近两个 round 的 accepted replay。replay 为空时只用 current。current 没有 accepted 时该 round 不更新；不会用 rejected 或仅凭 replay 强行补步骤。
- 每 round 训练结束再生成下个 chunk。active replay 有界，旧图片保留审计但不参与 sampling。
- online seed 为 SHA256(`base_seed, sample_id, generation_pass, round_id`) 的稳定整数。B/C final-generation seed 只依赖 `base_seed, sample_id`，不含 variant。
- B/C 同 source、prompt、negative、seed、scheduler、resolution、refinement 和 Global verifier，唯一生成器差异是 LoRA。默认不加载 SALAD，不做 inference-time guidance。
- `final_lora.pt` 发布后禁止在线 trainer 继续修改；pipeline 验证 downstream 前后的内容 fingerprint。

### A/B/C 的可比性和数据覆盖

A 只用 real GSV。B 用 original synthetic，C 用 final-LoRA synthetic。相同 architecture、初始权重 fingerprint、seed、optimizer、steps、batch size、augmentation、城市、loss/miner。

`shared_mix_plan.json` 用 **B/C 都有 accepted synthetic 的地点**，每地点 synthetic capacity 取两池的较小值。两套 dataloader 使用同一配额，因此即使最后一个 epoch 不完整，实际 real/synthetic slots 也一致。各池的其他 accepted 图片仍留在完整 manifest 供审计；训练使用共同地点覆盖以控制曝光比例这一混杂因素。

当前四条件有 20,614 条 prompts，而 62,514 个有效 real places 在每地点 4 张、8:1 时需要约 27,784 个 synthetic slots/epoch。现有数据即使全部通过，也不足以达到 8:1。系统采用两池都能达到的共同实际比例，明确输出 `coverage_limited`、目标比例及实际 slots，不把目标 8:1 写成实际比例。若需要完整 8:1，应补充 Global prompts，尤其是更多有效 GSV 地点覆盖，直到共同 accepted capacity 足够。

### 恢复

完整流程用原命令加 `--resume`。已完成阶段核验内容 hash 后跳过。source descriptor 可按 source hash、teacher weights fingerprint、preprocessing version 安全复用。

Online generator 每个完整 round 边界保存 LoRA、AdamW、global step、pass/round、完整 sample ordering、active replay、Python/sample RNG、torch/CUDA RNG、config 和 teacher/verifier fingerprint。初始未训练边界也保存，支持首个 round 中断后重做。mid-round checkpoint 明确拒绝。

中断 round 的图片目录与未提交日志尾部改名为 `.interrupted-*` 保留，再从最近完成边界重做。TensorBoard 恢复 committed journal，清除旧事件显示。单独调用 online trainer：`train_online_generator --resume <round_checkpoint>`，参数须与原 run 一致。

Fresh SALAD 支持完整 epoch checkpoint 恢复；若最后一次 checkpoint 是未完成 epoch，full pipeline 保留原目录为 `.interrupted-*` 后从同一初始化重新训练该 SALAD。最终相同步数预算可结束于半个 epoch，仍保存完整 `checkpoints/last.ckpt`。

## 2. TensorBoard

```bash
tensorboard --logdir outputs/full_run/tensorboard --host 0.0.0.0 --port 6006
```

目录为 `generator/`、`salad_A/`、`salad_B/`、`salad_C/`。Generator 默认启用，仅单独 trainer 的 `--disable-tensorboard` 能关闭。

Step curves：`generator/loss_diff`、`loss_vpr`、`loss_keep`、`loss_total`、`salad_cosine`、`lora_grad_norm`、`x0_guidance_grad_norm`、`lr`、`timestep`。

Round curves：`round/pass_rate`、`accepted_count`、`rejected_count`、`mean_s_geo`、`mean_s_div`、`mean_salad_preservation_cosine`，以及每个实际 condition 的 pass rate、geo/div、cosine、generated count。每 round 前四个样本保存 `source | generated` panel 与 condition / geo / div / cosine / acceptance caption；小于四条的末尾 chunk 则全部记录。

SALAD curves：`train/loss`、`train/b_acc`、`train/lr`；每 epoch 记录实际 `data/synthetic_fraction`、`real_to_synthetic`、`synthetic_slots`、`real_slots`。A 的 synthetic fraction 为 0，real-to-synthetic scalar 使用 0 表示无 synthetic；JSON 的该比值为 null，避免伪造有限比值。

## 3. 真实数据集评估

```bash
python -m AdaptVPR.experiments.vpr_guidance.evaluate_all \
  --a-checkpoint outputs/full_run/salad_A/checkpoints/last.ckpt \
  --b-checkpoint outputs/full_run/salad_B/checkpoints/last.ckpt \
  --c-checkpoint outputs/full_run/salad_C/checkpoints/last.ckpt \
  --salad-root salad --svox-root dataset/svox \
  --robotcar-root dataset/RobotCar-Seasons --nordland-root dataset/nordland \
  --image-size 322 --batch-size 32 --output-dir outputs/full_run/eval
```

直接加载 fresh Lightning `.ckpt` 的 `state_dict` 和 architecture hyperparameters，也支持相同 architecture 的 raw state dict。无需手动转换。A/B/C 统一 RGB、tensor bilinear antialias resize、ImageNet normalization、L2 descriptor normalization 和 CPU FAISS `IndexFlatIP` exact retrieval。完整 reference 排序分批进行，median/mean rank 不会截断在 top 10。

输出每个 A/B/C × dataset 的 JSON、per-query `.ranks.json`，以及 `evaluation_summary.json` / `.csv`。CSV 保留 Night/Rain/Snow、RobotCar Night/Other、Nordland 的 R@1，并包含实际 condition 的 R@5/R@10。不存在的 condition 留空；不填 0。RobotCar Other 是排除 `night` 与 `night-rain` 后的 condition macro-average。

SVOX 按实际 `queries*` 目录发现 conditions，`queries` 标为 normal；GT 使用本地 README 明确定义的文件名 UTM 字段，默认 25m retrieval radius。10m 是数据集构建时保证 query 有邻居的筛选阈值，与 25m retrieval 评估阈值不同。`--positive-radius` 可明确指定；协议写入 JSON。参考 [公开 retrieval 实现](https://github.com/gmberton/deep-visual-geo-localization-benchmark/blob/master/datasets_ws.py)。不会通过图像序号或相似 basename 猜 positives。

Nordland 当前 prepared layout 使用 README 定义的真实帧序号 ±10。所有 27,592 条 query 都评估；该 protocol 与 SALAD vendored subsampled GT 区分。若采用 `ref/query` 且文件匹配，优先加载 `salad/datasets/Nordland` 的原始 db/query/GT metadata。可以通过显式 metadata 使用其他已定义 split。

RobotCar-Seasons 按用户选择，默认使用官方 `robotcar_v2_train.txt` 中公开位姿的 adverse-condition 图片作为本地 evaluation queries，`overcast-reference` COLMAP 图片为 reference。读取 metadata 中实际 condition，公开 camera-to-world 4×4 转为 COLMAP world-to-camera quaternion + world camera center；默认 25m positives。输出明确标为 **`robotcar_v2_public_pose_evaluation`，不是官方 hidden-test benchmark**。该子集不用于本项目训练。

同时按官方联合阈值 (0.25m,2°)/(0.5m,5°)/(5m,10°) 计算 top-1 reference pose transfer 的 localization accuracy。JSON 将 `recall_metrics` 和 `official_metrics` 分开，并声明 `estimator=top1_reference_pose_transfer`。这是一种纯 retrieval pose estimator，未加入 local matching/PnP。见 [v2 pose convention](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/README_RobotCar_v2.md)、[官方阈值](https://www.visuallocalization.net/benchmark/)。官方 `robotcar_v2_test.txt` 没有 query GT，不能声称算出了其本地官方 test 结果。

单模型入口参数：

```text
evaluate_real --checkpoint ... --salad-root salad --dataset svox|robotcar-seasons|nordland
              --dataset-root ... --image-size 322 --batch-size 32 --output ...
```

可加 `--metadata`、`--reference-dir`、`--query-dirs`、`--positive-radius`、`--frame-tolerance`。`evaluate_all` 接受 `--svox-metadata`、`--robotcar-metadata`、`--nordland-metadata`。所有 GT 和 split 在提取 descriptors 前检查；不存在/无 positive 的 query 明确报错，不偷偷删除。

### 可选显式 metadata

默认查找 `<dataset-root>/metadata/evaluation.json`。路径相对 dataset root；positive index 相对 `references` 的原始列表。例：

```json
{
  "protocol": {"name": "official_retrieval_mapping", "source": "metadata publication or path"},
  "references": [{"path": "images/reference/1.jpg"}],
  "queries": [{"path": "images/night/2.jpg", "condition": "night", "positives": [0]}]
}
```

`positives` 也可使用 reference 完整相对路径。若提供双方 `pose={"center_m":[x,y,z],"quaternion_wxyz":[qw,qx,qy,qz]}` 与 `protocol.positive_radius_m`，可自动按 metric radius 生成 positives。Quaternion 统一 world→camera，所有 poses 必须在同一世界坐标系。缺少 poses 时 localization metrics 明确 unavailable。

## 输出与研究判断

```text
outputs/full_run/
  config.json pipeline_state.json complete.json
  generator/
    source_manifest.jsonl source_salad/ config.json
    rounds/pass_00_round_000/{accepted,rejected,records.jsonl}
    checkpoints/ final_lora.pt train.jsonl round_metrics.jsonl
  synthetic_B/{accepted,rejected,records.jsonl,synthetic_manifest.jsonl}
  synthetic_C/{accepted,rejected,records.jsonl,synthetic_manifest.jsonl}
  shared_mix_plan.json
  salad_A/ salad_B/ salad_C/  # config, mix_stats, checkpoints/last.ckpt
  tensorboard/{generator,salad_A,salad_B,salad_C}/
  eval/{evaluation_summary.json,evaluation_summary.csv,...}
```

核心问题是 **C > B 是否成立**；B > A 表示普通 domain augmentation 是否有效。summary 保存 B−A、C−B 的 macro R@1/5/10 差值；condition 指标用于判断 night/snow/rain 的变化，SVOX normal 等用于观察一般条件退化。单 seed 的差值是实验结果，不能自动等同统计显著性。

`train_generator.py` 的旧 offline CLI 已弃用，其 loss/gradient helpers 保留供 online trainer 和现有 unit tests 使用。旧 baseline cache 不兼容新 source manifest，不迁移为 online targets。旧 smoke utilities / 审计文档保留历史用途，不是推荐流程。
