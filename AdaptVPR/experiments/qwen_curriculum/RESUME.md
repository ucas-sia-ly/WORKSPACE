# 23 城市生成：暂停、继续与当前目录

`outputs/qwen_curriculum/` 顶层只保留两个目录：

```text
images/                 统一图片库及图片索引
  train/                原尺寸训练 PNG
  raw/                  高分辨率原始 PNG
  accepted/             当前审核通过图片的入口
  metadata/             生成元数据
state/                  当前计划、账本、进度、日志与训练状态
  history/legacy.tar.gz 原始历史配置、账本和其他证据的归档
  training/             后续原始 SALAD 微调与 SVOX 测试输出
```

后台入口为 `campaign.py run --output-dir outputs/qwen_curriculum/state`。`resume_control.py` 自动读取服务绑定的目录，常用命令无需额外指定路径。

迁移快照共保存 **21010 张**，其中 **21009 张已审核、13710 张通过**，另有 **1 张中断时保存的图片等待补审核**。还需新生成 **58416 张**，全计划仍为 **79426 张**。本次整理没有新增生成调用；实时数量见 `state/summary.json`。

## 终端准备

```bash
cd /home/admin123/github/WORKSPACE
conda activate AdaptVPR
```

## 暂停并释放电脑

```bash
python AdaptVPR/experiments/qwen_curriculum/resume_control.py pause
```

协调器及其子进程先退出，独立 Qwen 服务随后停止。只有两者均已停止且主进程 PID 为 0，才报告生成资源已释放。其他训练或应用进程不受影响。不要用 `SIGSTOP` 挂起生成进程，它仍会占用显存和内存。

## 继续生成

```bash
python AdaptVPR/experiments/qwen_curriculum/resume_control.py resume
```

关闭终端后仍在后台运行。正常暂停、进程中断、电脑重启后都使用同一命令。已经完成的图片，包括审核不通过的图片，不重新生成。完整图片已保存时先补审核；没有完整输出的中断任务使用同样参数在独立执行目录重试，并保留原请求来源。每次显式恢复，同一个未完成任务最多新增一次尝试，再失败则停止，修复问题后可再次恢复。

## 查看进度与预检

```bash
python AdaptVPR/experiments/qwen_curriculum/resume_control.py status

tail -f outputs/qwen_curriculum/state/generation.log
```

启动或错误日志为 `state/coordinator.log`。恢复会先校验当前快照、账本及已完成图片，再加载 Qwen；校验阶段可能需要几分钟。状态文件中的磁盘估算继续为剩余原始 PNG 与训练 PNG 预留空间。

暂停后执行完整性预检，不启动模型或新增生成调用：

```bash
python AdaptVPR/experiments/qwen_curriculum/campaign.py check
```

当前图片和账本包含内容校验及来源记录。不要手工删账本、修改签名、移动图片或同时运行多个生成入口。

Qwen 模型、4 步 BF16、画布、提示词、种子和天气分配不变。审核继续使用当前 v2：最少 16 个匹配点、内点率至少 0.78、512 尺度的 homography 网格中位位移最多 16 像素，并通过天气审核；不设空间覆盖率门限。

全部生成和审核完成后，按 [23 城市操作说明](ALL_23_CITIES.md#原始-salad-微调与-svox-测试)执行训练，输出位于 `state/training/`。
