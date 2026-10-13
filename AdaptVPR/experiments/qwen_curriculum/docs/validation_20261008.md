# 验证记录（2026-10-08）

使用 `/home/admin123/miniconda3/envs/AdaptVPR/bin/python`。新流水线 85 项、公共 Qwen 入口 57 项、SALAD 55 项、Qwen 磁盘权重 7 项 CPU 测试通过，共 204 项。修复了两个旧测试把假的 NumPy 注册到全局模块表而干扰真实 Torch 导入的问题。`git diff --check` 通过。

## 实际运行证据

- 训练图挖掘：45,661 张、5,710 个地点；选出 1000 张、806 个地点，每城市 250 张。使用发布的 dino_salad checkpoint 和训练集内 leave-one-out/远地点负例。所有选择图的质心正例 rank=1，难度表示较小的识别相似度间隔，不应称为 1000 张已经检索失败的图片。
- 冻结计划：night 500、snow 200、fog 200、rain 100，零 overcast；每源图一项，完整图片字节/尺寸与行校验通过。
- 历史 Qwen pilot：80 张四域 released 图零调用复用，58 张通过；旧门接受 41 张。新门保留了 25 张旧门因低 s_div 而拒绝的图，也以新的结构/天气检查拒绝部分旧门通过图。此比较反映筛选规则变化，不能证明模型变好或 Recall 提升。
- 新任务前 357 张审计：248 自动通过，109 拒绝；配置、代码、计划、1000 源图、生成图/原始大图和调用账本全部验证。审核捕获时有一个在途调用，未发现重复预算预留。
- 32 张分层目测发现 2 个自动接受但不适合训练的局部重绘，已写入独立哈希绑定排除文件；原始生成结果保持冻结。记录、样例和观察在 `outputs/qwen_curriculum/review_357`。
- SALAD 实际 `--check-data` 检查通过：四城市完整真实池、58 历史变体、exact-source 替换模式。正式新图训练等待生成完毕，不与 Qwen 重叠加载。
- 本地原生 SVOX 数据元信息与 25m 正例核验通过：17,166 gallery、14,278 日间与 823 夜间查询。正式性能结果将在两组四轮训练和评估完成后产生。

## 清理与保存

旧 IC-Light 自适应候选/在线 SALAD/生成器 LoRA 闭环包移除 29 个跟踪文件；历史诊断共享依赖迁入本包，历史 LoRA 推理 helper 保存在 `adapters/iclight_lora.py`，旧 checkpoint 读取测试通过。IC 服务已停用并取消自动启动；默认 Global/Local/Dual 与启动器均使用 Qwen。

完成的 Qwen 服务临时图片经字节或解码像素相同检查后，保留主 raw 图片并删除副本；另移除可重建旧 gallery assets。共 890 个文件、545.34 MiB，完整清单在 `outputs/qwen_curriculum/cleanup_manifest.json`。GSV 数据、权重、Qwen 主 raw/归一化图、source pools、SALAD checkpoint 保留。

工作区原有 `salad` 为失效 gitlink，缺少自身 `.git`；其本地训练代码修改不能显示在父仓库普通 diff 中。未改变 Git 索引；四份实际修改文件在 `outputs/qwen_curriculum/salad_integration_snapshot.tar.gz` 与相邻 SHA 清单中归档，源文件仍在原路径。当前后台运行脚本与 SALAD 源码保持冻结，完成任务前不要改动。
