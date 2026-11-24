# 4090 48GB 显存 / 32GB 内存运行配置

使用新配置 `salad/configs/ccr_feasibility_4090_48g.json`，输出到独立的
`salad/outputs/ccr_feasibility_4090_48g/`。保持 A/M/B0/B4/B5、P=8、K=4、
各 500 次更新、相同损失权重与划分策略。旧配置和已有实验记录不覆盖。

```bash
cd /home/admin123/github/WORKSPACE

# CPU 预检查，不训练。
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_feasibility_4090_48g.json

# GPU 空闲后执行。中断后使用同一命令恢复。
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/run_ccr_feasibility.py \
  --config salad/configs/ccr_feasibility_4090_48g.json --execute
```

流程、结果判读和独立验证要求见 [可行性实验说明](ccr_feasibility.md)。
这里优化运行效率，不改变代理天气验证的含义，也不启动 Reliability。

## 配置与实现

| 项目 | 专用配置 | 目的 |
|---|---|---|
| 学生训练精度 | BF16 | 使用硬件混合精度，避免 FP16 scaler 溢出重试 |
| 教师、评估、KL/OT 归约 | 保持 FP32 | 保留教师和检索评估的数值协议 |
| 真实主批次 | 8 地点 × 4 图 | 不通过改变 P/K 影响对照问题 |
| 数据加载 | 4 workers，每 worker 预取 1 批 | 与 GPU 重叠加载，同时限制队列内存 |
| CPU 数值计算线程 | 4 | 控制主进程 CPU 线程竞争 |
| 评估批量 | 64 图 | 根据本机小规模真实图片读取测量选择 |
| 教师分块 | 2 图 | 同一次处理精确源图及生成图，保留原 OOM 减半机制 |
| 精确检索查询分块 | 128 | 减少分块启动开销 |
| 参考描述子 GPU 缓存 | 最多 2048MiB | 避免每个查询块重复传输参考集、重复计算范数 |
| PyTorch CUDA allocator 上限 | 最多 40GiB | 给驱动、其他分配及系统留余量 |
| CUDA 当前空闲余量 | 至少 4GiB | 实际预算取 min(40GiB, 启动时空闲显存 − 4GiB) |
| 实验进程树 RSS 停止阈值 | 20GiB | 包括父入口和本次子进程、DataLoader workers |
| 系统 MemAvailable 停止阈值 | 3GiB | 避免实验继续挤占系统剩余内存 |

参考缓存不够容纳全集或分配 OOM 时回退到原分块方式；不会改为近似检索。
仍按全参考集计算精确正例排名并保留相同的距离相等处理规则。范数计算也分块，
不额外创建整个参考集大小的平方临时矩阵。缓存预算是常驻描述子和范数预算，
距离分块及范数临时张量还会占少量额外显存，受总 allocator 预算约束。

训练加载并恢复后及时释放 CPU checkpoint 副本，再创建数据 workers。
局部配对复用已经解码的 PIL 图像，避免同一对图片再次读盘、解码；预处理逐元素一致。
保留单图学生局部微批次和原跨辅助批次加权归约，避免引入新的损失归约差异。

不启用 persistent_workers，保留当前 epoch 传递和恢复机制。
BF16 和 4 workers 是新实验配置：与旧 FP16/0-worker 的随机增强及浮点结果不承诺
逐位一致，必须各组统一使用此配置，不能中途改变精度来续跑旧实验。
不额外打开 TF32、降低评估精度或近似检索。

## 资源保护的范围

校准、训练、评估串行运行，子进程进入独立进程组。每 0.5 秒检查一次内存；
超过阈值只停止本次实验进程组，保留上一完整 checkpoint，记录异常和资源统计。
启动前已低于系统剩余内存阈值时拒绝启动。对应日志尾部的 `RESOURCE` 行包含
进程树峰值 RSS、最小系统可用内存、运行时间和实际保护配置。

RSS 将共享页面重复计数，属于保守估计。CPU 保护是采样监控，不是内核 cgroup
硬限制；PyTorch 显存限制也不包含所有驱动/第三方分配，不能约束其他程序随后抢占。
这些阈值提供余量和提前停止机制，不承诺整机所有程序的总占用永远不超过限制。
正式运行时应停止其他占用大量内存/显存的任务。若因为资源保护停止，在资源释放后
重复同一运行命令；不要用删除 plan.json 的方式绕过实验配置检查。

## 本机短程测量（2026-10-19）

结果保存在 `salad/outputs/ccr_resource_probe_4090_48g_final/measurements.json`，
资源监控在同目录 `probe.log`。环境为 RTX 4090、PyTorch 2.8.0+cu128。

| 项目 | 测量值 |
|---|---:|
| FP32 纯前向，batch 8 | 约 507 图/秒 |
| FP32 纯前向，batch 64 | 约 648 图/秒 |
| 512 张 SVOX 真图读取+提取，batch 64 / 128 / 256 | 约 467 / 426 / 363 图/秒 |
| 4096 参考 × 512 query 精确检索，原 q32 无缓存 | 0.636 秒 |
| 同一检索，q128 + GPU 参考缓存 | 0.084 秒，排名及首选参考一致 |
| 教师每对两次单图 / 一次双图前向 | 10.7 / 6.6 毫秒 |
| 两步 BF16 B4 形状计算探针的峰值 allocated | 约 2.23GiB |
| 整个批量扫描（包括 256 图）峰值 allocated / reserved | 约 6.84 / 11.93GiB |
| 包含 4 个数据 workers 的保守 RSS 峰值 | 约 7.78GiB |

较大的评估 batch 虽然纯 GPU 吞吐略高，但在这次 512 张真实图读取样本中更慢，
所以默认保留 batch 64；这不是完整数据规模上的最优性证明。
该样本有操作系统文件缓存和 worker 启动开销，完整评估速度可能不同。
GPU 计算探针使用随机图像/描述子，只有读取测量使用真实图片；不构成检索效果验证。
两步计算探针不包含真实对应构建、完整输入流水线或 checkpoint 保存开销。
reserved 包含 allocator 留下的缓存，不能把它理解为每步活跃张量需求。

复测命令如下，输出目录必须为新的目录：

```bash
/home/admin123/miniconda3/envs/AdaptVPR/bin/python salad/tests/profile_ccr_resources.py \
  --config salad/configs/ccr_feasibility_4090_48g.json \
  --output-dir salad/outputs/ccr_resource_probe_new
```

此命令只做小规模性能探针和两次临时 optimizer 更新，不保存训练权重。
没有自动启动五组完整可行性实验。
