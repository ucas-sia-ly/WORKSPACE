# VisMatch for AdaptVPR

本目录是 [gmberton/vismatch](https://github.com/gmberton/vismatch) 的本地源码副本，
为 AdaptVPR 的生成图几何验收提供 `get_matcher("superpoint-lightglue")`。
匹配器实现、LightGlue 依赖、命令行入口、测试输入和许可证保留；
上游文档站、演示 notebook、预生成匹配展示图已清理。

该副本没有独立的上游 Git 提交记录，源码版本以工作区 Git 提交为准；
安装版本由 `pyproject.toml` 中的 `setuptools_scm` 处理。
项目安装和模型准备以 [AdaptVPR 外部组件说明](../AdaptVPR/docs/EXTERNAL_COMPONENTS.md) 为准。

```bash
python -m pip install -e ./vismatch
```

`MANIFEST.in` 使用的 `vismatch/assets/example_pairs/` 和 `example_test/`
保留用于 CLI 和上游测试。新增匹配器应遵循原有延迟导入机制。
完整 API 教程见 [上游文档](https://vismatch.readthedocs.io/en/latest/)。
许可证见 [LICENSE](LICENSE)；LightGlue 的许可证独立保留在其源码目录。
