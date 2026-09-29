# Stage3 人工标注小工具

在 AdaptVPR 目录、现有 AdaptVPR Python 环境中启动：

```bash
python scripts/annotate_targeted.py --open
```

浏览器访问 `http://127.0.0.1:8765`。服务器只监听本机；终端 Ctrl+C 停止。端口占用时
使用 `--port 8766`。不加载模型、不调用 diffusion、不自动填写标签。

默认读取 `outputs/stage3_targeted/planner_v2/targeted_edit_plans.jsonl` 和
`outputs/stage3_targeted/planner_human_audit.csv`。可用 `--plans` / `--csv` 指定其他同合约文件。

操作：

1. 对照原图与带轮廓的 target context crop；点击图片可查看原尺寸。界面不显示模型预测。
2. 选择区域、支撑面、是否可编辑、MVP family，填写备注。中文选项对应 CSV 的英文枚举。
3. 点击“保存并下一张”，或按 Ctrl/⌘+Enter；Alt+左右箭头切换。切换不自动保存，未保存
   修改会提示。顶部可跳转样本、跳到未完成样本，或重新读取外部修改后的 CSV。

可保存部分标签；四个选项全部填写后才记为完成。空白表示未标注，不代表 false/none。
为保持当前 v2 人工审计口径，支撑面看轮廓内可见支撑，editable 判断固定 mask 不动的
局部新增遮挡物可行性。这是旧 generation mask 审计，不是尚未实现的 Core/Canvas 审计。

保存使用既有 human audit 校验，拒绝矛盾标签，例如 editable=true 配 family=none。
只改当前 sample，保留其他行；先保存原 CSV 到同级 `human_audit_backups/`，再原子替换。
若另一个标注窗口或编辑器已更新 CSV，旧版本保存会被拒绝，需重新载入并复核，避免
覆盖新标注。建议不要同时在多个程序中编辑同一 CSV。

完成后按 [planner 文档](stage3_targeted_planner.md) 的比较命令计算人工 agreement、
precision/recall。工具不会把尚未填写的标签解释为人工拒绝。

测试（临时文件，不修改真实标签）：

```bash
python -m unittest discover -s tests -p test_targeted_annotation.py -v
```
