"""RunPod Serverless worker: Qwen-Image-Edit (instruction-based image editing).

- 日本語・英語・中国語の「指示文」で画像を編集するモデルです（タグ式プロンプトではありません）。
- フィルタリングは行いません。成人向けの利用は、18歳以上・被写体の同意がある素材のみを
  対象としてください。未成年を想起させる内容は絶対に扱わないでください。

Accepts the same job payload as the previous SDXL worker:
  input: {
    image: base64 string (source image),
    prompt: edit instruction (Japanese / English / Chinese OK),
    negative_prompt: optional,
    steps: optional int (clamped 10-50, default 40),
    seed: optional int (>=0 for reproducible results),
    ... (mode / nsfw / strength / guidance_scale are accepted and ignored)
  }
Returns {"image": base64 PNG} or {"error": "..."}.
"""

import base64
import io
import os
import traceback

import torch
from PIL import Image

import runpod
from diffusers import QwenImageEditPlusPipeline

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
DEFAULT_STEPS = 40
TRUE_CFG_SCALE = 4.0

_pipe = None


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def load_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe
    print(f"[load] loading {MODEL_ID} ...", flush=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )
    vram = _vram_gb()
    print(f"[load] GPU VRAM: {vram:.1f} GB", flush=True)
    if vram >= 55:
        pipe.to("cuda")
        print("[load] mode: full GPU (bf16)", flush=True)
    else:
        # Smaller GPUs: offload components to CPU RAM (slower, but works).
        pipe.enable_model_cpu_offload()
        print("[load] mode: model CPU offload", flush=True)
    pipe.set_progress_bar_config(disable=True)
    _pipe = pipe
    print("[load] pipeline ready", flush=True)
    return _pipe


def _decode_image(data: str) -> Image.Image:
    if isinstance(data, str) and data.startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    try:
        inp = job.get("input") or {}
        image_b64 = inp.get("image")
        if not image_b64:
            return {"error": "input.image is required"}
        prompt = (inp.get("prompt") or "").strip()
        if not prompt:
            return {"error": "input.prompt is required"}
        negative_prompt = inp.get("negative_prompt") or " "

        try:
            steps = int(inp.get("steps") or DEFAULT_STEPS)
        except (TypeError, ValueError):
            steps = DEFAULT_STEPS
        steps = max(10, min(steps, 50))

        generator = None
        seed = inp.get("seed")
        if seed is not None:
            try:
                seed_val = int(seed)
                if seed_val >= 0:
                    generator = torch.Generator(device="cuda").manual_seed(seed_val)
            except (TypeError, ValueError):
                generator = None

        pipe = load_pipe()
        image = _decode_image(image_b64)
        with torch.inference_mode():
            output = pipe(
                image=[image],
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                true_cfg_scale=TRUE_CFG_SCALE,
                guidance_scale=1.0,
                generator=generator,
            )
        return {"image": _encode_image(output.images[0])}
    except Exception as exc:  # never crash the worker; report via job result
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


# Warmup: preload the model, but never crash the worker if it fails.
# The first real job will retry the load and surface the error instead.
try:
    load_pipe()
except Exception:
    traceback.print_exc()
    print("[warmup] model preload failed (will retry on first job)", flush=True)

runpod.serverless.start({"handler": handler})
