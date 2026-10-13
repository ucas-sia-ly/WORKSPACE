# 审核通过的 Qwen 图片微调原始预训练 SALAD

本实验读取累计7277张生成记录，校验各阶段计划、执行指纹、原记录签名、原始PNG与训练PNG哈希，再提取`passed=true AND eligible_for_training=true`的5004张图片。未通过的2273张全部排除。通过图片的原签名和质量标记不改写；清单快照位于`outputs/qwen_curriculum/finetune_salad_pretrained_accepted_5004/accepted_manifest.jsonl`。

通过图片分布：night=2753、snow=917、fog=874、rain=460；覆盖3562个地点。训练保留完整四城5710个地点、45661张真实源图，采样到审核通过的困难源图时使用其生成版本，其他视角保留真实图片。这是精确源图替换，不是单独只用生成图训练。

## 完整预训练模型初始化

初始化文件为用户指定的`salad/checkpoint/dino_salad.ckpt`，SHA256：

```text
6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada
```

使用`train_salad.py --init-checkpoint`严格载入完整原始DINOv2 ViT-B/14和SALAD聚合器权重。实际188个权重张量初始化时逐值完全相等，记录在`initialization_audit.json`。原始聚合器64个cluster、128维cluster feature、256维token，描述子8448维。没有Reliability模块及其辅助损失，原checkpoint文件保持不变。

固定方案：10轮、AdamW、lr=1e-5、weight_decay=1e-4、cosine调度、训练最后4个DINO块和原聚合器、batch32个地点×4张图、224×224、FP32、workers=0、seed=42、关闭图像增强。保留前8个DINO块及图像嵌入权重；完成后检查实际更新只发生在后4个块、最后norm和聚合器。最终第10轮作为预先声明的测试模型，不使用SVOX结果挑选轮次或调参数。

## 完整 SVOX 微调前后对照

原始模型和微调模型均使用322×322、FP32、关闭TF32、PIL BILINEAR/ImageNet归一化、25米UTM正例、完整17166张图库、精确欧氏检索。每个模型仅提取一次共享图库，分别评估六个子集并报告R@1、R@5、R@10：

| 子集 | Query数 |
|---|---:|
| day | 14278 |
| night | 823 |
| rain | 937 |
| snow | 870 |
| sun | 854 |
| overcast | 872 |

不复用旧224分辨率、缺少完整元数据的历史基线。六域结果逐个原子保存，可通过checkpoint SHA、数据顺序及文件大小/修改时间、代码指纹、评估协议校验后恢复。测试集不参与困难挖掘或训练。

## 运行与文件

任务已由systemd用户服务运行，训练后自动完成原始基线和微调模型两套六域评估。查看进度：

```bash
cd /home/admin123/github/WORKSPACE
systemctl --user status salad-pretrained-accepted-finetune.service
cat outputs/qwen_curriculum/finetune_salad_pretrained_accepted_5004/progress.json
tail -f outputs/qwen_curriculum/finetune_salad_pretrained_accepted_5004/training.log
```

训练完成后的评估日志位于同目录`coordinator.log`。最终产物：

- `accepted_manifest.jsonl`：5004张已核对审核通过的原签名记录。
- `finetuned/checkpoint.pt`：微调模型、优化器、调度器和随机状态。
- `finetuned/dino_salad_finetuned_accepted5004.ckpt`：188项纯模型权重，可供原始SALAD使用；逐张量核对与完整微调checkpoint相等。
- `finetuned/checkpoint_epoch_010.pt`：固定最后一轮模型。
- `final_checkpoint_audit.json`：原始架构、初始化来源、实际可训练范围核对。
- `pretrained/evaluation/SVOX_*.json`：原始预训练模型六域结果。
- `finetuned/evaluation/SVOX_*.json`：微调模型六域结果。
- `comparison.json`：微调前后指标与百分点变化。

关闭聊天不影响运行。意外中断或电脑重启后恢复：

```bash
systemctl --user start salad-pretrained-accepted-finetune.service
```

恢复保留原10轮cosine计划，从最近完整轮次的checkpoint恢复优化器和随机状态；已经完成且协议匹配的评估子集不会重复计算。源图、生成图、初始模型、训练代码和配置发生变化时会拒绝静默继续。服务不自动循环重启失败任务。

前置CPU测试覆盖完整原checkpoint初始化、可训练/冻结权重、审核拒绝图片过滤、全真实池保留、六域完整评估及安全恢复。正式训练和最终评估使用全部训练批次与全部测试图片。
