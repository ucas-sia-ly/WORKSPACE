# LightX2V for AdaptVPR

本目录提供 AdaptVPR 的 Qwen-Image-Edit-2511 推理依赖。
上游为 [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V)，基础版本固定在
[`522609ecc121b49c20d201b3f00c3dc052821bce`](https://github.com/ModelTC/LightX2V/tree/522609ecc121b49c20d201b3f00c3dc052821bce)，
由 `.adaptvpr-source-revision` 记录。工作区包含适配本机的源码补丁，不能视为完整的上游原样副本。

保留推理包、设备后端、内核、配置、准备工具及原有运行脚本。`lightx2v/pipeline.py`
在导入时注册多个模型系列，因此这些运行源码也保留。清理范围是上游编辑器技能、
CI 配置、文档站、独立应用、演示代码、测试案例、Docker 示例和展示图片。
`assets/inputs/` 是保留脚本引用的示例输入。

项目的安装和服务命令以 [AdaptVPR 外部组件说明](../AdaptVPR/docs/EXTERNAL_COMPONENTS.md)
和 [服务启动说明](../AdaptVPR/configs/systemd/README.md) 为准。
安装本目录中的 Python 包可使用：

```bash
python -m pip install -e ./LightX2V
```

许可证见 [LICENSE](LICENSE)。完整上游教程及示例见上游仓库。
