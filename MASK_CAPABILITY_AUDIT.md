# Stage3 Mask Capability Audit

```text
NATIVE_MASK_SUPPORT = false
```

此结论专指当前 AdaptVPR 使用的 **LightX2V `qwen-image-edit-2511` / `task="i2i"`**：没有把空间 binary mask 作为独立条件送入去噪采样的实现。不能把通用 pipeline 的 `src_mask` 参数、其他模型的 mask 能力、或 HTTP 接口收下一个字段解释成 Qwen masked inpainting。

审计日期：2026-09-28。仅直接检查本地源码并运行无模型的参数传递探针；没有依据 README 推断，没有加载权重、启动服务、调用 GPU 生成、修改业务代码或现有实验产物。下文 realism 与论文适用性属于工程评估，尚无本次生成实验验证。

## 1. 六项结论

| 问题 | 本地检查结论 |
|---|---|
| 当前 pipeline 是否原生支持 mask？ | **当前 Qwen i2i 后端不支持空间 mask 条件。** 通用 `generate` 有 `src_mask`，实际对应其他任务；当前 Qwen 路径不会消费它。 |
| 是否支持 `mask_path`？ | **不支持。** `LightX2VPipeline.generate` 没有 `mask_path`、`mask` 或 `image_mask_path` 形参，也没有 `**kwargs`；直接传入会报 `TypeError`。AdaptVPR 的 request schema 也没有这些字段。 |
| 是否支持 masked inpainting？ | **不支持。** 没有目标 mask 编码、masked-image 条件或按空间 mask 在每个采样步保留原图区域的逻辑。普通 image editing 不能据此称为 masked inpainting。 |
| 是否支持 multiple conditioning images？ | **底层 Qwen 2511 实现支持；当前 AdaptVPR HTTP 链路不支持。** 底层通过逗号分隔的 `image_path` 字符串逐图读取、编码并拼接 conditioning latents，不是 Python 路径列表参数。 |
| 是否只能 image + text prompt？ | 当前 AdaptVPR 接口的语义条件是 **单张 image + text prompt**，另有 negative prompt、seed 和采样设置。底层 Qwen i2i 可以 **多张 images + text prompt**，所以不能把整个底层能力说成“只能单图”。 |
| mask 传进去是否真的参与采样？ | 通过当前 `generate_local(mask=...)`：**不参与**，未进入 HTTP payload；直接传 `src_mask`：**不参与 Qwen i2i**，转 input info 时被排除。只有将 mask 制成视觉条件图并经多图路径送入时，其像素才可作为普通图像条件影响采样；这仍不是原生空间 mask 约束。 |

## 2. 被审计版本与范围

- WORKSPACE 根目录：`/home/admin123/视频/workspace`。
- WORKSPACE Git HEAD：`4d729e1fe72b96af7ffa81ceaf069a0b97b79158`。
- `LightX2V/` **没有独立 `.git`**。本地 [`.adaptvpr-source-revision`](LightX2V/.adaptvpr-source-revision) 声明来源版本 `522609ecc121b49c20d201b3f00c3dc052821bce`；AdaptVPR adapter 的固定版本相同。
- 因此，版本标记是来源声明，不是本次独立验证的上游 Git HEAD；本报告以实际本地文件及末尾 SHA256 为准。不能用父仓库的 `git -C LightX2V rev-parse HEAD` 冒充 LightX2V 上游版本。
- Adapter 选用 `Qwen/Qwen-Image-Edit-2511`，固定 model revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`，配套 Lightning 4-step LoRA；`load_pipeline()` 明确设置 `model_cls="qwen-image-edit-2511"`、`task="i2i"`。证据：[adapter](AdaptVPR/adapters/lightx2v_qwen_image_edit.py)，第 19–29、96–145 行。

## 3. 直接查看的 generate 签名与 mask 丢失位置

本地 [pipeline.py](LightX2V/lightx2v/pipeline.py) 第 420–441 行的完整签名如下，已从该函数 AST 提取并用 `inspect.signature` 核对，未导入或初始化真实 pipeline：

```python
def generate(
    self,
    seed=42,
    prompt="",
    negative_prompt="",
    save_result_path="lightx2v_gen_result.png",
    task=None,
    image_path=None,
    action_path=None,
    video_path=None,
    image_strength=None,
    image_frame_idx=None,
    last_frame_path=None,
    audio_path=None,
    src_ref_images=None,
    src_video=None,
    src_mask=None,
    return_result_tensor=False,
    target_shape=[],
    sr_ratio=2.0,
):
    ...
```

函数第 455 行执行 `self.src_mask = src_mask`；第 467–471 行创建与 task 对应的 input info，然后调用 `update_input_info_from_dict(input_info, self)`，最后送入 runner。关键不是字段是否保存在 pipeline 对象上，而是是否进入当前 runner 的输入。

[input_info.py](LightX2V/lightx2v/utils/input_info.py) 提供了完整证据：

- 第 207–222 行：`I2IInputInfo` 有 `image_path`、prompt 和形状字段，**没有任何空间 mask 字段**。
- 第 392–422 行：`"i2i"` 映射到 `I2IInputInfo`；当前 adapter 未设置混合 `support_tasks`。
- 第 544–547 行：更新函数只遍历 `input_info.__dataclass_fields__`，其他字段不会复制进去。因此 `self.src_mask` 在此路径失去下游消费者。
- 第 86–102 行：`src_mask` 出现在 `VaceInputInfo`。另见 [default_runner.py](LightX2V/lightx2v/models/runners/default_runner.py) 第 317–330 行，VACE 的专用路径读取并向其编码函数传递该字段。这是其他任务的能力，不是 Qwen i2i 的能力。

只读探针：执行从本地 AST 提取的原函数主体及 input-info 函数，移除 `no_grad` 装饰器；用仅返回 input-info 字典的 runner 和空 logger/seed hook 替代推理依赖。结果：

```text
src_mask stored on pipeline: /mask.png ; forwarded to i2i: False
mask: generate() got an unexpected keyword argument 'mask'
mask_path: generate() got an unexpected keyword argument 'mask_path'
image_mask_path: generate() got an unexpected keyword argument 'image_mask_path'
```

该探针验证的是实际字段传递，不是声称跑过真实扩散采样。

## 4. AdaptVPR 当前调用链

| 层级 | 当前实现 | mask 状态 |
|---|---|---|
| `scripts/run_targeted.py` → `generation/targeted_inputs.py` | validate、load source/mask、写 normalized task；`route="local"`、`generated=false` | 已验证 mask，但**尚未接生成器**，属于预期 dry-run 行为。 |
| `Lightx2vGenerator.generate_local` | [lightx2v.py](AdaptVPR/generation/lightx2v.py) 第 178–193 行具有 `mask: Image.Image = None` | 调用 `_call_api` 时不传 `mask`。这个现有参数不是可用的 native mask 能力。 |
| HTTP request | 同文件第 123–132 行构建 payload | 只有 `image_path`、`prompt`、`negative_prompt`、`seed`、`infer_steps`、`guidance_scale`。即便把 mask 放进 `**kwargs`，payload 白名单也不会转发。 |
| FastAPI `GenerateRequest` | [adapter](AdaptVPR/adapters/lightx2v_qwen_image_edit.py) 第 32–38 行 | 无 mask 字段，无 conditioning-image 列表字段。 |
| FastAPI `/generate` | 第 188–190 行把单个 `image_path` 当作一个文件校验；第 206–212 行调用 pipeline | 仅传 seed、单图路径、prompt、negative prompt、结果路径。不能把 `"a.png,b.png"` 直接塞进当前 HTTP 参数绕过单文件校验。 |
| `LightX2VPipeline.generate` → Qwen runner | 按上一节映射到 `I2IInputInfo` | `mask_path` 不合法，`src_mask` 不会传到 Qwen i2i 输入。 |

补充两个已确认的问题：

1. 当前 `_mock_local(ref_image, prompt, mask)` 也没有读取 mask；它在固定中央区域画矩形（同文件第 210–223 行）。Mock 输出不能作为 mask 生效证据。
2. 从本地 adapter AST 提取 `GenerateRequest`，在当前 AdaptVPR Python 环境的 **Pydantic 2.13.5** 中构造附带 `mask`、`mask_path`、`src_mask` 的请求，`model_dump()` 仍只有上述六个字段。未知字段被忽略，并不会因此得到 422 或生效。本次只执行了 request model，未启动 FastAPI lifespan。

## 5. Qwen image-edit 实现及采样证据

### 多图条件确实进入模型

- [pipeline.py](LightX2V/lightx2v/pipeline.py) 第 127–134 行：`qwen-image-edit-2511` 转到 `model_cls="qwen_image"`，启用 `USE_IMAGE_ID_IN_PROMPT=True`。
- [qwen_image_runner.py](LightX2V/lightx2v/models/runners/qwen_image/qwen_image_runner.py) 第 237–263 行：`image_path.split(",")`，逐图调用 `read_image_input`，把 `images_list` 交给 text encoder；再逐项 VAE 编码 `vae_image_list`。
- [qwen25_vlforconditionalgeneration.py](LightX2V/lightx2v/models/input_encoders/hf/qwen25/qwen25_vlforconditionalgeneration.py) 第 172–220 行：为每张图建立 `Picture 1`、`Picture 2` 等标记，传递 `images=condition_image_list`，提取视觉与文本 hidden states；同时保留各图供 VAE 编码。
- [qwen_image/model.py](LightX2V/lightx2v/models/networks/qwen_image/model.py) 第 100–103 行：拼接全部 `image_latents`，再与待生成的 noisy latents 拼接，并送入 transformer；第 142–145 行给出当前无 CFG 分支的 noise prediction。

因此底层多图是可追踪的实际模型 conditioning，不只是 metadata。**没有在本次验证多图生成质量、最大可靠图数或显存占用。**

### 没有空间 mask 的采样操作

- [qwen_image_runner.py](LightX2V/lightx2v/models/runners/qwen_image/qwen_image_runner.py) 第 316–334 行：每步仅执行 scheduler `step_pre` → model `infer` → scheduler `step_post`。
- [qwen_image/scheduler.py](LightX2V/lightx2v/models/schedulers/qwen_image/scheduler.py) 第 488–523、574–580 行：按 seed 建立随机 latent，准备 timesteps；第 637–641 行直接执行 `scheduler.step(noise_pred, t, latents)`。
- 上述路径没有读取 target mask、构造 mask latent、拼接 masked source，或使用 mask 对非目标区域做逐步 latent 恢复/融合。
- Runner 第 466–475 行在采样后直接 VAE decode、保存，也没有按空间 mask 合成回原图。
- Text encoder 中的 `attention_mask`、`prompt_embeds_mask` 及 `_extract_masked_hidden` 是有效 token/padding 的掩码，不是图像空间的编辑区域 mask。不能以变量名含 `mask` 判定支持 inpainting。

直接传 `src_mask` 在当前数据流上不可能提供 Qwen 空间控制信号；把 mask 作为普通 conditioning image 则可能改变 noise prediction，但只具备视觉提示的语义。

## 6. 仓库里另一个“mask 接口”不构成反例

LightX2V 自带 server 与 AdaptVPR 的 FastAPI adapter 是两套入口。其 [openai_images.py](LightX2V/lightx2v/server/api/openai_images.py) 第 209–252 行确实接收上传的 `mask`，转为 `image_mask_path`；[server/schema.py](LightX2V/lightx2v/server/schema.py) 第 36 行也声明该字段。

继续跟踪后的行为是：

1. [generation/image.py](LightX2V/lightx2v/server/services/generation/image.py) 第 38–42 行调用 `_pack_image_and_mask_as_dir`。
2. [generation/base.py](LightX2V/lightx2v/server/services/generation/base.py) 第 57–89 行把 source 与 mask 转为 RGB PNG，必要时 nearest resize，然后保存成一个目录中的两张图；把 `image_path` 改成目录路径，并清空 `image_mask_path`。
3. [inference/worker.py](LightX2V/lightx2v/server/services/inference/worker.py) 第 97–100 行更新 input info 后直接进入 runner。
4. 当前 Qwen runner 第 238–241 行按逗号分隔文件路径；其 `read_image_input` 第 221–228 行对每项执行 `Image.open`，未展开该目录。按所检查的本地路径，会尝试打开目录，不能认定这套打包入口在当前 Qwen runner 下已经可用；本次未启动该服务验证报错。

即使修通目录到多图路径的转换，这也只是将 mask 图作为普通图像条件；没有补出空间 mask 采样算子。另外，对 BoQ 的 **0/1 PNG** 直接 `convert("RGB")` 不会自动把 1 变成 255；直接当视觉提示时前景几乎是黑色，需要显式生成可视化副本。这不能改变原始 binary artifact。

## 7. 三种替代方案的工程评估

以下是后续方案，**本次均未实现**。不存在通过在当前 Qwen wrapper 中再加一个 `mask` 参数即可完成的原生透传改法。

| 方案 | Spatial controllability | Realism（预期，需实测） | Implementation complexity | 是否适合论文正式实验 |
|---|---|---|---|---|
| **A. 新增真正 mask-aware inpainting backend** | 三者中最有条件实现明确的区域约束。须核实 mask 进入条件编码或逐步采样逻辑；原生 mask 条件也不自动保证边界外像素完全不变。 | 有机会在目标区域内生成连贯物体与边界；小而不规则的 mask、遮挡物几何及投影仍可能不自然。 | **高**：后端/权重/API、图像-mask 同步几何变换、采样语义、显存与复现协议均需接通；可能需要换权重或训练，不能假定当前 Lightning LoRA 兼容。 | **优先候选**。在真实 mask 生效、区域控制、自然度和 VPR 标签保持通过验证后，适合作为正式 targeted generation 实验。 |
| **B. Qwen edit + mask visualization / multiple-image conditioning** | **软空间引导**。Overlay 或第二张 mask 可视化经图像编码影响模型，但可能忽略边界、改动全图或留下标记。不能保证 exact support。 | 可能利用现有 Qwen 的全图编辑生成较自然的结果；也可能复制标记、产生错误对象或改变背景。 | **中**：复用当前模型；增加可视化、稳定图像顺序、多图 HTTP schema、逐文件校验和对齐验证。单图 overlay 原型更简单。 | 可作为明确标注的 **visual-mask conditioning baseline / supplementary experiment**；若作为独立方法，必须报告定位服从率与泄漏。不能宣称原生 inpainting 或未经验证的精确位置控制。 |
| **C. generate-then-mask-composite** | **最终像素替换范围可精确控制**，但生成对象的位置和形状不受 mask 控制。可能只粘回背景或截断物体。 | 边缘接缝、光照/阴影断裂和物体残片风险高。只看 mask 外零变化无法证明生成自然。 | **低**：全图编辑后按同尺寸 binary mask 合成，处理分辨率与色彩一致性即可。 | **仅 plumbing/debug baseline**。不能作为论文主要 mask-aware targeted generation 证据，不能把合成后区域一致归因于采样时的控制。 |

### A. 真正 mask-aware 后端需要做什么

后续链路应明确落到实际消费 mask 的新后端：

```text
run_targeted.py 的独立生成分支（仍保留 check-only/dry-run）
  → 专用 inpainting generator
  → HTTP 请求：source + binary mask + prompt + seed + backend identity
  → 专用 FastAPI schema：严格校验 mask/尺寸/哈希
  → 经源码和采样验证的 inpainting pipeline
  → mask 条件编码 / 空间约束的采样路径
```

不能把最后一层写成当前 `LightX2VPipeline.generate(mask_path=...)`。当前接口与实现并不存在该能力。若考虑使用此仓库的其他 mask-aware 模型分支，必须另做任务类型、权重、静态图输入及采样路径审计，不能直接借用 VACE 字段便宣称 Qwen 已支持。

需要固定 mask 的 1 表示可编辑区域、nearest 几何映射、实际像素面积、模型/采样版本；比较 attention 与 matched random 时固定提示语、后端和预算，防止后端更换或掩码扩张成为混杂因素。验收应覆盖移动/置空/替换 mask 的响应、mask 外差异与边界质量，不只检查参数是否出现在日志。

### B. 使用现有多图能力的具体后续路径

可选择 source + overlay/reference 图；无需新增一个决定位置的 Qwen planner，目标位置来自 BoQ mask。若后续选择此方案，准确的拟修改点是：

1. `run_targeted.py` 的独立生成分支读取 source 与原 mask，生成与 source 对齐的可视化副本，使用预先固定的编辑指令；不能更改既有 plan/prompt 行为。
2. `Lightx2vGenerator` 增加专用 targeted/multi-image 方法；保留 source，显式传递有序 `conditioning_image_paths`。不能复用当前被忽略的 `mask` 参数冒充已完成。
3. `_call_api` 的独立 targeted payload 和 FastAPI `GenerateRequest`/专用请求 schema 传递该列表；服务端逐文件校验，不再把整个列表或逗号串交给 `Path(...).is_file()`。
4. Adapter 将验证后的路径按顺序连接为 `image_path="source.png,visualization.png"`，再调用现有 `LightX2VPipeline.generate`。本地 runner 实际支持的是这个字符串格式；需要拒绝/规避文件名中的逗号歧义。
5. 下游已有 runner → Qwen25 encoder → VAE → transformer 的多图路径可以复用。标记能力为 `visual_mask_conditioning`，而不是 `native_mask`。

必须注意：runner 在未显式指定形状时使用 **最后一张 conditioning image** 的尺寸比例决定输出形状（`qwen_image_runner.py` 第 383–391 行），所以 source 与可视化应同尺寸、顺序固定。Encoder 第 133–139 行分别缩放视觉条件与 VAE 输入；视觉图经过模型预处理不会保有 binary 边界语义。原始 0/1 mask 不变，可视化副本需显式提升亮度或制作 overlay，并记录其生成规则。多个条件图会增加编码与 transformer token 工作量，具体资源消耗须实测。

Qwen edit 本身已有视觉-文本条件编码器；复用它不等于新增一个选位置的 VLM/planner，也不改变本次“只审计、不实现”的范围。

### C. 合成基线的正确边界

设原图为 `I`，无空间 mask 约束的编辑输出为 `G`，binary target 为 `M`：

```text
C = M * G + (1 - M) * I
```

`M` 只用于最终合成，未参与生成 `G` 的采样。若用 feather/dilation，会改变实际作用区域，必须另外记录；不能继续把原 mask 的 token budget 当作最终作用面积。

当前 `Lightx2vGenerator` 会把 source 重新保存为 JPEG quality=95，且把不同尺寸的生成结果用 LANCZOS 缩回 source 尺寸（`lightx2v.py` 第 103–110、165–174 行）。将来做该基线时，mask 外应来自原始 source 解码像素，不能拿 JPEG 临时参考图冒充原始保留区域；最终合成后的变化也不能证明 inpainting 能力。

## 8. 本次审计决策与文件指纹

**当前不能把 Stage3 tasks 直接接到 `generate_local(mask=...)` 并宣称 mask-aware。** 正式区域约束实验优先评估 A；B 可独立研究视觉 mask 引导并明确能力边界；C 保持 plumbing/debug 定位。没有原生支持，因此本报告不提供虚构的“mask 原生透传到现有 Qwen pipeline”的修改方案。

只新增本审计文档。已有 dry-run reader、`run.py`、plan/prompt、scheduler、verifier、AdaptVPR adapter 与 LightX2V 实现均未修改；未生成图片、未创建 commit。

用于复核的本地文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `AdaptVPR/generation/lightx2v.py` | `b77c64d7548a222f0deef7e6e83f069630a03ded730090c8a729e91282b44185` |
| `AdaptVPR/adapters/lightx2v_qwen_image_edit.py` | `d0dedddd58770ab2049af1015ce7a668bd34e704a3c0dbaf215863af87868cf2` |
| `LightX2V/lightx2v/pipeline.py` | `49fc1520bb092acf8ecb05b639ed593050ae0a0ca02bf0b5ef2cc728b3ae1ddf` |
| `LightX2V/lightx2v/utils/input_info.py` | `1e73a9d71440adb7a3b84f5ae14605cd54e7c3cd814697dfd9ad8231fc168b96` |
| `LightX2V/lightx2v/models/runners/qwen_image/qwen_image_runner.py` | `52ba8adb4e3fb55e3aed663d72caafb59d6f8091f904ae572f65740ed70688eb` |
| `LightX2V/lightx2v/models/networks/qwen_image/model.py` | `c37c8e43b1303832ac2196f765488d72839b12b5ef4cacfc1566196861939c9f` |
| `LightX2V/lightx2v/models/schedulers/qwen_image/scheduler.py` | `aaf1dc75b50f4d0fafe24bf36f760189b8e21f84b8518a06539d1f4193f3d8a7` |
| `LightX2V/lightx2v/models/input_encoders/hf/qwen25/qwen25_vlforconditionalgeneration.py` | `3cde011721bf436297c992ae9ce042bd5d1070f320a55e0bbb559a67044cb3c3` |
