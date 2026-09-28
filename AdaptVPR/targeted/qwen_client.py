"""Explicit Qwen3-VL backends. No mock fallback or imports from generation.agent."""

import base64
import io
import os

from .editability import DECISION_SCHEMA

IMAGE_LABELS = (
    "Image 1: original street photograph.",
    "Image 2: same photograph; the ONLY editable region is tinted magenta.",
    "Image 3: unmarked expanded context crop around that SAME fixed mask; not a new location.",
)

class OpenAIQwenClient:
    """Use the existing Qwen OpenAI-compatible service, with seed and strict JSON schema."""
    def __init__(self, *, endpoint=None, model=None):
        from openai import OpenAI

        endpoint = endpoint or os.environ.get("ADAPTVPR_PLANNER_API_BASE")
        model = model or os.environ.get("ADAPTVPR_PLANNER_MODEL")
        if not endpoint or not model:
            raise ValueError("Configure ADAPTVPR_PLANNER_API_BASE and ADAPTVPR_PLANNER_MODEL")
        self.model = model
        self.client = OpenAI(base_url=endpoint, api_key=os.environ.get("ADAPTVPR_PLANNER_API_KEY", "local-placeholder"),
                             timeout=180, max_retries=0)
        self.metadata = dict(backend="openai_compatible", endpoint=endpoint, model=model, temperature=0,
                             schema_constrained_decoding=True)

    def decide(self, *, system, user, images, seed):
        content = [{"type": "text", "text": user}]
        for label, image in zip(IMAGE_LABELS, images):
            content.append({"type": "text", "text": label})
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": url}})
        result = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            temperature=0, seed=seed, max_tokens=512,
            response_format={"type": "json_schema", "json_schema": {
                "name": "editability", "strict": True, "schema": DECISION_SCHEMA}},
        )
        if result.choices[0].finish_reason != "stop":
            raise RuntimeError(f"Incomplete planner response: {result.choices[0].finish_reason}")
        return result.choices[0].message.content


class LocalQwenClient:
    """Local, greedy Qwen3-VL inference. Post-validation rejects any non-schema output."""
    def __init__(self, *, model_path=None, device=None, max_image_pixels=524288):
        from pathlib import Path
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = model_path or os.environ.get("ADAPTVPR_TARGETED_PLANNER_MODEL_PATH")
        if not model_path or not Path(model_path).is_dir():
            raise ValueError("Configure an existing ADAPTVPR_TARGETED_PLANNER_MODEL_PATH (no automatic downloads)")
        self.device = device or os.environ.get("ADAPTVPR_TARGETED_PLANNER_DEVICE", "cuda")
        self.torch = torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa",
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, min_pixels=65536, max_pixels=max_image_pixels,
        )
        self.metadata = dict(backend="local_transformers", model_path=str(Path(model_path).resolve()),
                             device=self.device, transformers_version=transformers.__version__,
                             torch_version=torch.__version__, do_sample=False, max_new_tokens=512,
                             max_image_pixels=max_image_pixels, schema_constrained_decoding=False,
                             schema_validation="strict post-validation; malformed responses rejected",
                             determinism="greedy decoding; bitwise cross-device equality not guaranteed")

    def decide(self, *, system, user, images, seed):
        torch = self.torch
        content = []
        for label, image in zip(IMAGE_LABELS, images):
            content.extend([{"type": "text", "text": label}, {"type": "image", "image": image}])
        content.append({"type": "text", "text": user})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        # Isolate RNG state from the rest of the application. Greedy decoding uses no sampling.
        devices = [torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=512)
        continuation = output[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
