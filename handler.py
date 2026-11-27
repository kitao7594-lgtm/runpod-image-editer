"""RunPod Serverless worker: Qwen-Image-Edit (instruction-based image editing).

- 日本語・英語・中国語の「指示文」で画像を編集するモデルです（タグ式プロンプトではありません）。
- フィルタリングは行いません。成人向けの利用は、18歳以上・被写体の同意がある素材のみを
  対象としてください。未成年を想起させる内容は絶対に扱わないでください。

Accepts the app job payload:
  input: {
    image: base64 string (source image),
    reference_images: optional list of base64 strings (max 2 extra images
                      passed to the model together with the source),
    prompt: edit instruction (Japanese / English / Chinese OK),
    negative_prompt: optional,
    steps: optional int (clamped 1-50, default 40),
    true_cfg_scale: optional float (default 4.0; use 1.0 with Lightning LoRA),
    seed: optional int (>=0 for reproducible results),
    lora_id: optional HF repo id of a LoRA (e.g. "user/my-lora"),
    lora_file: optional weight file name inside the LoRA repo,
    lora_scale: optional float (default 1.0, clamped 0.0-2.0),
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

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
DEFAULT_STEPS = 40
DEFAULT_TRUE_CFG_SCALE = 4.0

_pipe = None
_lora_key = None    # (lora_id, lora_file) currently loaded, or None
_lora_scale = None  # scale currently applied for _lora_key


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
    # Import here (not at module top level) so that an import failure is
    # reported as a job error instead of crash-looping the whole worker.
    from diffusers import QwenImageEditPlusPipeline

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


def ensure_lora(pipe, lora_id, lora_file, scale):
    """Load / unload / rescale the LoRA adapter to match the job request."""
    global _lora_key, _lora_scale
    wanted = (lora_id, lora_file) if lora_id else None
    if wanted != _lora_key:
        if _lora_key is not None:
            pipe.unload_lora_weights()
            _lora_key = None
            _lora_scale = None
            print("[lora] unloaded", flush=True)
        if wanted is not None:
            kwargs = {"adapter_name": "lora"}
            if lora_file:
                kwargs["weight_name"] = lora_file
            pipe.load_lora_weights(lora_id, **kwargs)
            _lora_key = wanted
            _lora_scale = None
            print(f"[lora] loaded {lora_id} ({lora_file or 'default file'})",
                  flush=True)
    if _lora_key is not None and _lora_scale != scale:
        pipe.set_adapters(["lora"], adapter_weights=[scale])
        _lora_scale = scale
        print(f"[lora] scale set to {scale}", flush=True)


def _decode_image(data: str) -> Image.Image:
    if isinstance(data, str) and data.startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _float_or(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        steps = max(1, min(steps, 50))

        true_cfg_scale = _float_or(inp.get("true_cfg_scale"),
                                   DEFAULT_TRUE_CFG_SCALE)
        true_cfg_scale = max(1.0, min(true_cfg_scale, 10.0))

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

        # LoRA: match the adapter state to this job's request.
        lora_id = (inp.get("lora_id") or "").strip() or None
        lora_file = (inp.get("lora_file") or "").strip() or None
        lora_scale = _float_or(inp.get("lora_scale"), 1.0)
        lora_scale = max(0.0, min(lora_scale, 2.0))
        try:
            ensure_lora(pipe, lora_id, lora_file, lora_scale)
        except Exception:
            # Never leave a half-loaded adapter behind.
            global _lora_key, _lora_scale
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
            _lora_key = None
            _lora_scale = None
            raise

        # Images: source first, then up to 2 reference images.
        images = [_decode_image(image_b64)]
        refs = inp.get("reference_images") or []
        if isinstance(refs, list):
            for ref_b64 in refs[:2]:
                if ref_b64:
                    images.append(_decode_image(ref_b64))
        if len(images) > 1:
            print(f"[job] {len(images)} input images (incl. references)",
                  flush=True)

        with torch.inference_mode():
            output = pipe(
                image=images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                true_cfg_scale=true_cfg_scale,
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
