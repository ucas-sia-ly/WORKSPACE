# 61,984 对真实—夜间生成图

2026-11-13 已完成生成、回传、统一汇总和文件完整性检查。

| 来源 | 图片数 |
|---|---:|
| 历史保留的原版非汞灯图 | 46,712 |
| 原 Prompt/seed 续生成 | 13,206 |
| 原汞灯地点按其他风格补生成 | 2,066 |
| 合计 | **61,984** |

覆盖 **23 个城市、61,984 个唯一地点、29 类 night 风格**。每个地点只有一个精确真实源图和一个生成结果，未混入独立 V2/V3 Prompt 试验图或旧 rain/snow/fog 图库。

## 统一目录

所有路径均相对于项目根目录：

```text
outputs/night_dataset_v1/
  pairs/<city>/<place_id>/
    source.jpg        精确真实源图（符号链接，原文件扩展名保留）
    night.png         训练尺寸生成图（硬链接）
    night_full.png    高清生成图（硬链接）
    generation.json   原始生成 sidecar（硬链接）
  manifest.jsonl      完整逐图记录
  pairs.tsv           全量原图—生成图对应表
  index.sqlite        唯一图片 ID / 城市地点索引
  complete.json       数量、风格、城市分布及清单 SHA256
```

例如 `outputs/night_dataset_v1/pairs/Chicago/5496/`，四个文件都对应同一张原图及同一次生成。

硬链接共用原有文件数据，不复制图片、不重编码。不要原地覆盖某个链接中的图片：这会同时改变相同 inode 的其他路径。需要编辑时先复制到独立位置。原图符号链接依赖 `dataset/gsv-cities` 的现有路径；迁移数据时须连同真实数据一起迁移或更新链接。

## 查询与复核

```bash
./ccr-night status
./ccr-night pair --city Chicago --place-id 5496
./ccr-night manifest
./ccr-night verify
```

`verify` 只使用 CPU，默认两个线程，检查原图、原图链接、训练尺寸图、高清图与 sidecar 的 SHA256，以及地点和图片 ID 唯一性。不会启动生图或训练。

`pairs.tsv` 可用表格软件打开，也可直接查阅。`manifest.jsonl` 保留 Prompt、seed、风格、真实源图路径、所有文件 SHA256、原结果校验值和运行来源；同时提供统一目录下的输出路径。SQL 例子：

```sql
SELECT * FROM pairs WHERE city='Chicago' AND place_id=5496;
SELECT style_id, COUNT(*) FROM pairs GROUP BY style_id;
```

## 校验范围

汇总前逐条检查冻结结果签名、预算、主机完成账本、精确源图和文件哈希。重复地点、重复生成 ID、重复源图、汞灯、缺失文件或校验不一致会阻止发布 `complete.json`。

**文件完整性通过不代表内容质量通过。** 原有 `pending_verification`、`eligible_for_training=false` 等状态保持不变，没有把未审核图自动提升为训练数据。

## 清理后的旧目录

本批生成路径中的硬链接、Prompt、任务账本和冻结依赖仍保留，便于审计。其余旧生成试验、回传缓存、临时服务输出及多余训练快照已清理。最终训练 checkpoint、正式报告、真实数据和模型未删除。旧实验日志中已删除的图库/中间快照路径仅用于历史追溯。

本批相关生成服务已停止；`./ccr-night resume` 现在会提示本批已完成，不会重新启动模型。后续新增生成应使用新的独立输出目录，不追加改写这个已封存数据集。

清理明细见 [2026-11-13 清理记录](../../docs/cleanup_20261113.md)。

## DINO＋局部结构内容审核

新增独立入口 `./ccr-night content-audit`。审核顺序为：精确图片 SHA256 → 已有几何与天气证据 → 冻结 DINO 特征与对齐局部结构联合检查。沿用 `content_audit_v42.py`，不修改原有 LightGlue/Sinkhorn/生成器或历史规则。

- 使用多尺度局部窗口内 `min(DINO差异, 结构差异)` 的均值，要求两种证据同时存在。无效投影区域不作为破坏证据。
- 阈值、窗口尺寸和有效区域下限来自独立训练地点的程序化对照；留出对照验证失败就不能使用该校准。不允许根据待审生成图降低阈值。
- `decision_source=local_content` 表示内容分数判定；`prompt_policy` 表示已有的 `shopfronts` 整类隔离，不能说成识别出局部幻觉。证据不足或不匹配时为 Unknown。
- `peak_region` 是生成图坐标系中最高联合分数窗口，不是语义错误框。正常光照变化、对齐误差也可能产生高分。
- 通过只表示通过此安全筛查。输出 `safety_pass_manifest.jsonl` **不是最终训练清单**，保留 `eligible_for_training=false`。原始 61,984 对图片、Prompt 和 manifest 均不修改。

复用本次 87 对真实特征缓存验证入口（不加载模型、不占 GPU）：

```bash
./ccr-night content-audit \
  --manifest outputs/maintenance/night_rules_87_20261113/dino_content_trial_v1/targets.jsonl \
  --base-measurements outputs/maintenance/night_rules_87_20261113/measurements.jsonl \
  --signals outputs/maintenance/night_rules_87_20261113/dino_content_trial_v1/remote_results/targets_signals.jsonl \
  --calibration outputs/maintenance/night_rules_87_20261113/dino_content_trial_v1/remote_results/calibration.json \
  --output-dir outputs/night_content_review_v1
```

默认仅检查输入文件及校准签名、打印计划。加 `--execute` 才执行；中断后原命令加 `--execute --resume`，输入、代码、校准和缓存必须一致。逐图结果写入 `audited_manifest.jsonl`，统计写入 `report.json`。断点恢复会验证图片哈希，清除自身日志末尾未写完的半行，不重复已完成推理。

对其他图片在线提取特征时，去掉 `--signals`，加入：

```text
--dino-repo /本机/DINOv2源码目录
--dino-checkpoint /本机/dinov2_vitb14_pretrain.pth
--device cuda --threads 4
```

并提供对应的 `--manifest` 和完整的 `--base-measurements`。几何/天气证据支持原有扁平 JSONL，以及此次测量的嵌套 `current_rule` 格式；每条必须有相同 sample_id、source_sha256、output_sha256。原始生成清单的 pending 状态不是几何/天气通过证据，缺失不得自动放行。本入口不重新运行 LightGlue/CLIP。

校准绑定权重、DINO 源码、PyTorch 版本、预处理、分辨率与搜索半径。只有源码目录位置可迁移；其他字段变化须独立重新校准，不能忽略检查。已验证的校准来自第二主机 PyTorch 2.8.0 环境；本机不同环境应使用缓存回放或重新校准。在线逐对推理，不保存全量特征缓存；日志保留分数和最高窗口，减少内存和磁盘开销。

2026-11-15 接入回归：新入口复现 87 对 **51 Pass / 27 Unknown / 9 Reject**，其中 6 对为内容分数拒绝，3 对为类别隔离。N062 内容分数 0.6437，高于对照校准边界 0.5966；N015、N028 仍漏检。因此此规则只是额外安全筛查，不能替代完整的结构和夜间真实性审核。未启动全量审核或训练。

### 当前全量审核范围（2026-11-15确认）

用户已取消逐图目视、Prompt修改及换源图阶段；当前只执行两轮规则审核，不生图、不训练、不改正式图库。

1. 当前LightGlue几何＋CLIP夜间筛选已完成：61,984对中53,438 Pass、8,546 Reject。
2. 仅对这53,438对执行当前DINO＋对齐局部结构联合筛查，沿用冻结校准、权重、源码、PyTorch版本及预处理，不另设阈值。Pass、Reject、Unknown分别输出；shopfronts类别隔离单独标明原因，不能说成DINO识别到局部幻觉。

本轮独立任务为`ccr-night-dino-audit-v1.service`，输入快照位于`outputs/night_dino_audit_v1_inputs`，结果位于`outputs/night_dino_audit_v1`。本机权重、源码和PyTorch 2.8.0+cu128已核对匹配现有校准。在线逐对提取，不保存全量特征缓存。

只读进度见`outputs/night_dino_audit_v1/progress.json`；完整记录见`audited_manifest.jsonl`。完成后输出`safety_pass_manifest.jsonl`、`reject_manifest.jsonl`、`unknown_manifest.jsonl`和报告。通过仍仅表示安全筛查通过，不自动准入训练。

不要用旧`ccr-night-audit resume`恢复本轮任务：旧编排包含已取消的后续阶段。本轮停止/恢复命令见`outputs/night_dino_audit_v1_inputs/README.md`。

### 拒绝样本修复的新授权（2026-11-15，后续指示）

用户随后授权仅对两轮Reject合并队列进行Qwen Prompt修正、重生成、再次审核，并允许最多两张同地点替代源图；原有Unknown不纳入。用户同时取消shopfronts整类隔离，并要求使用两台设备。本阶段以[双机拒绝夜景修复](night_repair_campaign.md)为准，不改写已完成的历史审核结果。

新增入口`./ccr-night-repair`，独立目录`outputs/night_repair_v1`。取消隔离后的1,924对补查为1,421 Pass、63 Reject、440 Unknown，需重生成修复的实际队列为11,927个地点。新内容复检显式传入空类别隔离列表，沿用同一DINO校准。
