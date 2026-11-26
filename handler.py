"""
RunPod Serverless - AI Image Editor (NSFW-capable, uncensored)

- safety_checker は無効化済み。プロンプトのフィルタリングは一切行いません。
- 成人向けコンテンツは利用者自身の責任で、18歳以上・被写体の同意がある素材のみを対象としてください。
  未成年を想起させる内容は絶対に扱わないでください。

Modes:
  img2img (default): AutoPipelineForImage2Image。任意のSDXL/SD1.5チェックポイントが使えます。
    デフォルト MODEL_ID=AstraliteHeart/pony-diffusion-v6 (NSFW対応で知られるSDXL系)
  ※作者のリポジトリは単一ファイル v6.safetensors のため from_single_file で読み込む。
    リポジトリ名に -xl を付けたものは存在しないため使わないこと。
    実写系なら例: SG161222/RealVisXL_V4.0 などに MODEL_ID で差し替え可能。
  pix2pix: timbrooks/instruct-pix2pix による指示編集 (MODE=pix2pix)

Input:
{
  "input": {
    "image": "<base64>",
    "prompt": "...",
    "negative_prompt": "",
    "mode": "img2img",          # 省略時は環境変数 MODE
    "nsfw": true,               # 情報用フラグ。フィルタには使用しません
    "strength": 0.75,           # img2img: 元画像をどれだけ変えるか 0.1-1.0
    "steps": 25,
    "guidance_scale": 7.0,
    "image_guidance_scale": 1.5,
    "seed": 42
  }
}
Output: { "image": "<base64 png>", "seed": int, "model": str, "mode": str }
"""
import base64, io, os, random
import torch
from PIL import Image
import runpod

MODE_DEFAULT = os.environ.get("MODE", "img2img")
MODEL_ID = os.environ.get(
    "MODEL_ID",
    "timbrooks/instruct-pix2pix" if MODE_DEFAULT == "pix2pix"
    else "AstraliteHeart/pony-diffusion-v6",
)
PONY_SINGLE_FILE_URL = os.environ.get(
    "MODEL_SINGLE_FILE_URL",
    "https://huggingface.co/AstraliteHeart/pony-diffusion-v6/resolve/main/v6.safetensors",
)
_pipes = {}

def load_pipe(mode: str):
    if mode in _pipes:
        return _pipes[mode]
    if mode == "pix2pix":
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, safety_checker=None
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        # 作者のリポジトリは diffusers形式ではなく単一の v6.safetensors のため
        # from_pretrained ではなく from_single_file で読み込む。
        # diffusers形式の別チェックポイントに差し替えた場合は from_pretrained を使う。
        if MODEL_ID == "AstraliteHeart/pony-diffusion-v6":
            from diffusers import StableDiffusionXLImg2ImgPipeline
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                PONY_SINGLE_FILE_URL, torch_dtype=torch.float16, use_safetensors=True
            )
        else:
            from diffusers import AutoPipelineForImage2Image
            pipe = AutoPipelineForImage2Image.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            )
        # SD1.5系で safety_checker が付く場合も明示的に無効化
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
    pipe.to("cuda")
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    _pipes[mode] = pipe
    return pipe

def b64_to_image(s: str, max_side: int) -> Image.Image:
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    # SD系は8の倍数に揃えると安定
    w, h = img.size
    return img.resize((max(64, w // 8 * 8), max(64, h // 8 * 8)), Image.LANCZOS)

def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def handler(job):
    try:
        inp = job.get("input") or {}
        if not inp.get("image"):
            return {"error": "input.image (base64) is required"}
        if not inp.get("prompt"):
            return {"error": "input.prompt is required"}
        mode = (inp.get("mode") or MODE_DEFAULT).lower()
        seed = int(inp.get("seed", random.randint(0, 2**31 - 1)))
        generator = torch.Generator("cuda").manual_seed(seed)
        pipe = load_pipe(mode)
        if mode == "pix2pix":
            out = pipe(
                prompt=inp["prompt"],
                image=b64_to_image(inp["image"], 512),
                negative_prompt=inp.get("negative_prompt", ""),
                num_inference_steps=int(inp.get("steps", 20)),
                guidance_scale=float(inp.get("guidance_scale", 7.5)),
                image_guidance_scale=float(inp.get("image_guidance_scale", 1.5)),
                generator=generator,
            ).images[0]
        else:
            out = pipe(
                prompt=inp["prompt"],
                image=b64_to_image(inp["image"], 1024),
                negative_prompt=inp.get("negative_prompt", ""),
                strength=float(inp.get("strength", 0.75)),
                num_inference_steps=int(inp.get("steps", 25)),
                guidance_scale=float(inp.get("guidance_scale", 7.0)),
                generator=generator,
            ).images[0]
        return {"image": image_to_b64(out), "seed": seed, "model": MODEL_ID, "mode": mode}
    except Exception as e:
        return {"error": str(e)}

load_pipe(MODE_DEFAULT)  # ウォームアップ
runpod.serverless.start({"handler": handler})
