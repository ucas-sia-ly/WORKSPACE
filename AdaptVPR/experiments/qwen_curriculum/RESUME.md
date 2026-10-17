# 23 城市生成：按最新代码暂停与恢复

当前后台入口是 `v2_campaign.py`，输出目录为 `outputs/qwen_curriculum/generation_23city_v2`。`resume_control.py` 自动读取后台服务绑定的目录；暂停、继续、查看进度不需要额外指定路径。

迁移采用当前审核 v2：最少16个匹配点，移除空间覆盖率门限，保留内点率、homography 位移和天气审核。原有16892张图片直接复用，利用已保存且经过校验的诊断数据重判，通过数从11049变为11056，未新增生成调用。Qwen 模型、4步 BF16、分辨率、提示词、种子、天气分配及79426张总目标保持不变。

原 `generation_23city` 及更早目录保留原始图片、配置、审核结果和失败账本。新结果单独签名并记录旧结果来源；新目录引用原图片，不复制整批图片。旧源码通过 Git 历史中的精确字节校验来证明历史来源，新执行使用当前代码；不覆盖原配置或伪装成旧审核结果。

## 终端准备

每次重新打开终端时执行：

```bash
cd /home/admin123/github/WORKSPACE
conda activate AdaptVPR
```

## 暂停并释放显存、内存

```bash
python AdaptVPR/experiments/qwen_curriculum/resume_control.py pause
```

协调器及其子进程先退出，独立 Qwen 服务随后停止。两者均已停止且主进程 PID 为0才报告资源已释放。其他训练或应用进程不受影响。不要用 `SIGSTOP` 挂起进程，它仍占用显存和内存。

## 继续生成，包括断电或重启后恢复

```bash
python AdaptVPR/experiments/qwen_curriculum/resume_control.py resume
```

关闭终端后仍在后台运行。已完成图片（包括审核不通过的图片）不重新生成；已有完整图片但审核未完成时先补审核；没有完整输出的中断任务使用同样参数在独立执行目录重试，原失败账本保留。每次显式恢复，同一未完成任务至多新增一次尝试，再失败就停止，修复资源问题后可再次恢复。

原已审核16892张；迁移另恢复了上次中断时已保存的1张原图，等待补审核。因此仍需新生成62533张，合计剩余待完成审核62534张。后续实时数量以新目录的 `summary.json` 为准。

## 查看进度和日志

```bash
python AdaptVPR/experiments/qwen_curriculum/resume_control.py status

tail -f outputs/qwen_curriculum/generation_23city_v2/generation.log
```

启动或错误日志位于同目录的 `coordinator.log`。恢复首先校验历史及已完成图片，此阶段尚未加载Qwen，可能需要几分钟。空间检查仍保留原始PNG与训练PNG，按剩余图数估算，不降低生成配置。

只做完整性预检（不启动模型、不新增生成调用）：

```bash
python AdaptVPR/experiments/qwen_curriculum/v2_campaign.py check
```

## 迁移重现与后续训练

本机已准备迁移结果。以下命令可再次校验已有迁移，或在独立新目录重现；执行前保持生成服务停止：

```bash
python AdaptVPR/experiments/qwen_curriculum/v2_campaign.py migrate \
  --source-dir outputs/qwen_curriculum/generation_23city \
  --output-dir outputs/qwen_curriculum/generation_23city_v2
```

`migration_config.json` 记录原始文件快照、新代码SHA和重判数量；`historical_code_proofs.json` 保存旧源码证据；`imported/` 保存新审核结果及原调用来源。中断恢复继续使用已冻结的新配置，不能删除账本、移动原图片或同时启动旧入口。

全部生成和审核完成后，[23城市训练说明](ALL_23_CITIES.md#3-全23城训练命令)中的训练入口会读取新目录，只用当前规则通过的图片微调原始SALAD，并测试六域SVOX。
