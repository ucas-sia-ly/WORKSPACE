# Qwen3-VL-4B-Instruct 独立服务

本目录独立于 AdaptVPR；不修改其代码、`.env` 或 Conda 环境。使用 Docker 内的 vLLM 0.11.0（CUDA 12.8.1），固定镜像摘要和模型 revision，提供 OpenAI 兼容 Chat Completions API。

## 启动

前提：Docker daemon 可访问，NVIDIA Container Toolkit 已配置，GPU 0 空闲显存至少约 17 GB，可访问 Docker Hub 和 Hugging Face。首次拉取约 12.2 GB 压缩镜像和约 9 GB 模型权重；解压后建议预留至少 50 GB 磁盘空间。已检查当前机器约有 327 GB 可用磁盘空间。

```bash
cd /home/admin123/workspace/services/qwen3-vl-4b
bash service.sh start
```

下载速度决定首次部署耗时，可能需要数十分钟。脚本等待就绪最多 1 小时，随后自动验证模型列表、文本生成、两张图片识别和 JSON 输出。成功打印 `READY`。日志保存在 `logs/deploy.log`，失败或成功验收时保存最近的容器日志到 `logs/server.log`。终端中断不会删除已经启动的容器。

支持沿用当前 shell 的 HTTP(S)/ALL_PROXY、HF_ENDPOINT、HF_TOKEN；host 网络允许容器访问本机代理。Docker 镜像拉取使用 Docker daemon 的网络配置，shell 代理不一定适用于镜像拉取。

## 项目接入

- Base URL：`http://127.0.0.1:23002/v1`
- Model：`qwen3-vl-4b-instruct-remote`（也接受 `Qwen/Qwen3-VL-4B-Instruct`）
- API key：`local-placeholder`，与现有项目默认值相同，仅用于本地兼容鉴权。
- 支持 `/v1/models`、`/v1/chat/completions`，文本、图片 data URL 和 `response_format={"type":"json_object"}`。
- 本机监听 `127.0.0.1`；远程调用可用 SSH 端口转发。

AdaptVPR 默认地址、别名和 key 已匹配本服务。建议在启动项目的同一个终端加载以下文件，以禁用 mock、明确使用本地模型，并将冷启动请求超时从默认 20 秒调为 180 秒：

fish（本机用户使用的 shell）：

```fish
source /home/admin123/workspace/services/qwen3-vl-4b/client.fish
```

Bash/Zsh：

```bash
source /home/admin123/workspace/services/qwen3-vl-4b/client.env
```

然后正常运行原来的项目命令。这些配置只修改当前 shell 环境变量。fish 不支持 `client.env` 中 Bash 的 `${VAR:-default}` 语法；出现 `${ is not a valid variable in fish` 时改为加载 `client.fish`，无需重启模型服务。

## 资源与维护

默认 BF16、16K 上下文、2 个并发序列、每请求最多 4 张图片、每图最多约 100 万像素，不启用视频。采用 eager 模式减少 CUDA graph 额外显存和首次启动开销。显存预算为总显存的 35%（本机约 16.8 GiB）；多个图像生成模型同时使用时仍须核对总显存。

```bash
bash service.sh status
bash service.sh logs
bash service.sh check
bash service.sh stop
```

容器配置为 `unless-stopped`，Docker 启动后自动恢复，手动 stop 后不会自动恢复。再次 start 会复用现有容器及配置。更改 GPU 预算、key 或其他启动参数时，先 stop，再显式执行 `docker rm qwen3-vl-4b-instruct`，然后重新 start；模型缓存保留在本目录 `cache/`。例如重建时可以设置 `QWEN_GPU_MEMORY_UTILIZATION=0.4 bash service.sh start`。自定义 `QWEN_API_KEY` 时，运行验收和加载 client.env 的 shell 也须设置同一变量。

部署已完成，已通过模型列表、文本生成、双图片识别及 JSON 输出验收，并使用 AdaptVPR 原有客户端完成文本和双 JPEG 图片的 JSON 请求验证。

参考：
- [Qwen3-VL 官方部署说明](https://github.com/QwenLM/Qwen3-VL#deployment)
- [模型权重](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [OpenAI official documentation：Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
