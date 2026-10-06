# 最终实验重构报告

分支：`test/vpr-loss-generator`。本次完成代码、回归验证和现有数据/metadata 检查，未执行耗时完整模型训练或 benchmark，没有经验性能结论。

## 1. 旧主流程

`prepare_data` 不再加载 IC-Light 或生成 `baseline/*.png`；输出改为 `source_manifest.jsonl` 和 source SALAD descriptors。`train_generator` 的 offline CLI 明确报弃用，保留 loss / VJP / checkpoint 等 helpers 供 online trainer 与既有 unit tests 使用。旧 baseline cache 不导入新流程。历史 smoke utilities 与 AUDIT_REPORT 保留，但 README 不再推荐这些流程。没有新增 synthetic retrieval、inference-time guidance、BoQ 或 pilot 实验。

## 2. Online round 数据流

`train_online_generator` 是正式 generator 入口；`train_full` 调用它。每 pass 在 condition 内稳定 shuffle，再均衡交错构造 chunks；没有 replacement 或无限重复较小 condition。每 round：

1. 当前 LoRA `G_r` 完整执行 released Global stage 1 + refinement。
2. 收集整个 chunk 的真实 verifier 结果和 generated-image 审计。
3. 仅 accepted 图进入 current pool。
4. 用当前 `y_r` latent/noise 分布计算 diffusion MSE、SALAD source cosine loss、对 `y_r` 的 L1 keep loss；两次 forward 的一阶 VJP 更新 LoRA。
5. 将 accepted pool 加入 bounded replay，保存完整边界 checkpoint，再 refresh 下个 round。

没有 accepted 的 round 不更新，且记录 updates=0。若整个训练没有有效更新，不发布伪装成已训练的 final LoRA。

## 3. Verifier 的位置

在 candidate refresh 和任何 LoRA loss 之间，使用原 `DualTraitEvaluator.evaluate(route="global")`；geometry≥0.78、diversity≥0.15。拒绝 skipped、不一致的 passed flag 和非有限 score。Rejected 保留图片和监控 cosine，不能成为 diffusion/VPR/keep target。B/C final pools 同样筛选；只有 passed 且 eligible 的记录进入 synthetic manifest。matcher fallback 的实际策略被记录并检查一致性。

## 4. Replay 与恢复

当前池与最近 `replay_rounds=2` 个完成 round 的 accepted 池按 50/50 采样；replay 为空则只采 current。deque 有界，旧磁盘图片不代表 active replay。

每 round checkpoint 保存 LoRA、optimizer、global step、pass/round、完整 ordering、active replay 记录、sample/Python/torch/CUDA RNG、config、teacher 和 verifier fingerprint。只接受 `completed_round` 边界；首个 round 前也保存 initial boundary。中断尾部留作审计再重做；恢复检查目标图片 checksum。单元测试验证了 refresh 后确实使用更新的 generator，并验证中断恢复最终 LoRA tensors 与连续执行逐元素一致（模型 orchestration doubles，非实际模型短步数试训）。

## 5. Seeds

SHA256 对 JSON 编码的 `(base_seed, sample_id, pass, round)` 生成稳定 63-bit seed；跨进程不会受 Python hash randomization 影响。Final B/C 只编码 `(base_seed, sample_id)`，不含 variant。因此 B/C 每条 seed 相同，sample 之间不同。

## 6. Source descriptors

永久缓存的是 fixed source + frozen teacher 的 descriptor。Manifest 与 `.pt` payload 记录 source path/SHA256、teacher 完整权重 fingerprint、preprocessing version、descriptor dimension。复用前验证 metadata、finite values 和 L2 normalization，teacher 或 source 变化时报错。Prepared manifest 保留 prompt、negative、condition、city/place_id 与 released sampling policy。

## 7. TensorBoard

统一 `tensorboard/{generator,salad_A,salad_B,salad_C}`。

Generator step：diff/VPR/keep/total loss、SALAD cosine、LoRA grad norm、x0 guidance grad norm、LR、timestep。Round：pass rate、accepted/rejected counts、geo/div、preservation cosine；每 condition 同类分数与 count。每 round 最多四个 source/generated panels 和分数 caption，末尾不足四张则全部记录。默认启用。

Fresh SALAD：train/loss、train/b_acc、train/lr，以及实际 consumed batches 的 synthetic fraction、real:synthetic、real/synthetic slots，每 epoch 记录一次。A 的 synthetic fraction 为 0。

## 8. Fresh SALAD A/B/C

同 DINOv2+SALAD architecture、预训练 DINOv2 / 随机 SALAD 初始化、seed、AdamW、batch size、steps、augmentation、城市、官方 MultiSimilarityLoss/Miner。固定 seed 后记录初始完整模型 fingerprint，并检查 A/B/C 一致。teacher aggregator 权重和 teacher descriptors 不进入 fresh training。

A real-only；B original synthetic；C final-LoRA synthetic。B/C 的 `shared_mix_plan` 在共同 accepted 地点取相同容量和逐地点 quota，保证完整/partial epoch 的 synthetic exposure 一致；最后核对实际 journal。此策略控制地点覆盖和曝光混杂因素，完整原始 manifests 保留审计。

当前 snow/night/rain/fog 合计 **20,614 prompts**，有效 real places **62,514**，每地点4张、8:1 需要约 **27,784 synthetic slots/epoch**。现有 prompt 数即使100%通过也不足。输出使用共同可达到的实际比例，标记 coverage_limited，不声称实现了实际8:1。要完整8:1，需补充更多有效地点的 Global prompts，且保证 B/C 共同 accepted capacity 足够。

Final LoRA 训练结束固定。生成 B/C pool 默认没有 SALAD、latent optimization 或 inference-time feedback。Generator、generation 和三套 SALAD 使用独立进程。全流程训练完成后停止，不自动 evaluation。

## 9. 真实数据集加载与验证

现有目录无需搬动。实际检查结果：

| 数据集 | References | Queries | GT / Protocol |
|---|---:|---:|---|
| SVOX test | 17,166 | 18,634 | 已文档化的 UTM 字段，默认25m positives；实际 queries* conditions |
| Nordland prepared test | 27,592 | 27,592 | 本地 README 定义的帧序号±10；全 query traversal |
| RobotCar v2 public poses | 20,862 registered references | 1,906 | 官方 query camera-to-world poses + 49 aligned COLMAP reference models；默认25m positives |

RobotCar query 条件：dawn230、dusk187、night197、night-rain224、overcast-summer220、overcast-winter202、rain198、snow239、sun209。所有 image paths 存在；每个 query 有 positives（最少146）。从 `*_aligned.zip` 直接读取 image poses，未解压 unused 3D points。当前带 URL query suffix 的 metadata 文件名也兼容。

按用户明确选择，RobotCar 输出标为 `robotcar_v2_public_pose_evaluation`；这是公开位姿子集，不是官方 hidden test。`robotcar_v2_test.txt` 没有 GT，不能本地声称完成其官方评测。参考 [官方 v2 定义](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/README_RobotCar_v2.md)。

RobotCar 另用 top-1 reference pose transfer 计算 [官方联合阈值](https://www.visuallocalization.net/benchmark/) (0.25m,2°)/(0.5m,5°)/(5m,10°)。JSON 将 recall_metrics 与 official_metrics 分开，明确 pose estimator；未引入 local matching/PnP。

Nordland 的当前 layout 与 SALAD vendored subsampled metadata 不同。两者分别标明 protocol；如果 root 为匹配的 ref/query 布局，则优先使用 vendored GT。

所有 A/B/C evaluation 使用同 RGB、默认322×322、bilinear antialias resize、ImageNet normalization、L2 descriptors、CPU FAISS FlatIP exact retrieval。直接加载 Lightning `.ckpt` 或 raw state dict；自动用 checkpoint architecture hyperparameters。完整 rank 在全部 references 上计算，不将 mean/median rank 截断在 top10。

每个 condition 输出 R@1/5/10、median/mean rank，另有 overall 与 condition macro-average。统一 summary JSON/CSV 包含所有实际条件，未知/不存在 condition 留空；RobotCar Other 为排除 night/night-rain 的 condition macro-average。核心比较 B−A 和 C−B，以真实 adverse datasets 为依据。

## 10. 正式命令

环境变量与依赖配置见 [README](README.md)。在 WORKSPACE 根目录执行：

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

```bash
tensorboard --logdir outputs/full_run/tensorboard --host 0.0.0.0 --port 6006
```

```bash
python -m AdaptVPR.experiments.vpr_guidance.evaluate_all \
  --a-checkpoint outputs/full_run/salad_A/checkpoints/last.ckpt \
  --b-checkpoint outputs/full_run/salad_B/checkpoints/last.ckpt \
  --c-checkpoint outputs/full_run/salad_C/checkpoints/last.ckpt \
  --salad-root salad --svox-root dataset/svox --robotcar-root dataset/RobotCar-Seasons \
  --nordland-root dataset/nordland --image-size 322 --batch-size 32 \
  --output-dir outputs/full_run/eval
```

## 验证范围和当前需要调整的内容

- 现有和新增 **63 项 unit tests 全部通过**：data/label validation、released generation policy、LoRA state/fp32、VPR gradients、source cache、verifier gate、online refresh/replay/boundary resume、matched exposure、Lightning training/resume、真实 protocol、full retrieval ranks、pose metrics、Lightning/raw checkpoint loading。
- 7 个入口 `--help`、Python compile 和 `git diff --check` 通过。
- 20,614 条目标 Global prompts 的 source 与 GSV dataframe labels 全部实际核对通过。
- 真实 SVOX/Nordland/RobotCar split 和 GT 完整加载检查通过；没有提取模型 descriptors 或运行 benchmark。
- 未自动运行 4-image smoke、20-step overfit、pilot、synthetic retrieval 或完整训练。
- 数据目录与 RobotCar metadata 现在足够；若要求实际8:1，需要补充 prompts 和共同 accepted 地点覆盖。
- 当前输出文件系统剩余约53GB。以保留两个 generation passes、B/C pools、每round optimizer checkpoint、三套 SALAD checkpoints 的要求，建议正式实验准备至少100GB可用空间并将 output-dir / tensorboard-dir 指向该盘。现有数据无需搬迁，可使用绝对路径。
