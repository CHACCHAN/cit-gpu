"""
FLUX.1-dev OpenAI Images API 互換サーバ
Open WebUI の Image Generation (OpenAI engine) から呼び出される。

対応エンドポイント:
  POST /v1/images/generations
    body: { "prompt": str, "n": int, "size": "WxH", "response_format": "b64_json"|"url" }
  GET  /health
"""
import os
import io
import base64
import time
import threading

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# ---- 環境変数から設定 ----
MODEL_ID      = os.environ.get("MODEL", "black-forest-labs/FLUX.1-dev")
QUANTIZATION  = os.environ.get("QUANTIZATION", "none").lower()
DEFAULT_STEPS = int(os.environ.get("STEPS", "50"))
DEFAULT_GUID  = float(os.environ.get("GUIDANCE", "3.5"))
API_PORT      = int(os.environ.get("FLUX_API_PORT", "9090"))

# 生成は1枚ずつ逐次実行（VRAM保護のためロック）
_gen_lock = threading.Lock()
_pipe = None


def load_pipeline():
    """FLUXパイプラインをロード（マルチGPU分散）"""
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import FluxPipeline

    num_gpus = torch.cuda.device_count()
    print(f"[flux-api] Loading {MODEL_ID} (quant={QUANTIZATION}) on {num_gpus} GPU(s)...", flush=True)

    max_memory = {i: "18GB" for i in range(num_gpus)}
    max_memory["cpu"] = "32GB"

    if QUANTIZATION == "none":
        _pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            max_memory=max_memory,
        )
    elif QUANTIZATION == "fp8":
        from optimum.quanto import freeze, qfloat8, quantize
        from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
        )
        quantize(transformer, weights=qfloat8)
        freeze(transformer)
        _pipe = FluxPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16,
            device_map="balanced", max_memory=max_memory,
        )
    else:
        raise ValueError(f"Unsupported QUANTIZATION: {QUANTIZATION}")

    print("[flux-api] Pipeline ready.", flush=True)
    return _pipe


def parse_size(size: Optional[str]):
    """'1024x1024' → (1024, 1024)。未指定や不正値は1024x1024"""
    if not size or "x" not in size:
        return 1024, 1024
    try:
        w, h = size.lower().split("x")
        w, h = int(w), int(h)
        # FLUXは16の倍数推奨、極端な値はクランプ
        w = max(256, min(2048, (w // 16) * 16))
        h = max(256, min(2048, (h // 16) * 16))
        return w, h
    except Exception:
        return 1024, 1024


app = FastAPI(title="FLUX.1 Images API")


class ImageRequest(BaseModel):
    prompt: str
    n: int = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    model: Optional[str] = None
    # 拡張パラメータ（任意）
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": _pipe is not None}


@app.get("/v1/models")
def list_models():
    # Open WebUIがモデル一覧を引く場合に備える
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/images/generations")
def generate(req: ImageRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    pipe = load_pipeline()
    width, height = parse_size(req.size)
    steps = req.steps or DEFAULT_STEPS
    guidance = req.guidance or DEFAULT_GUID
    n = max(1, min(req.n or 1, 8))  # 上限8枚

    print(f"[flux-api] generate: '{req.prompt[:50]}...' size={width}x{height} n={n} steps={steps}", flush=True)

    results = []
    with _gen_lock:
        for i in range(n):
            seed = (req.seed if req.seed is not None else int(time.time())) + i
            gen = torch.Generator("cpu").manual_seed(seed)
            image = pipe(
                prompt=req.prompt,
                height=height,
                width=width,
                guidance_scale=guidance,
                num_inference_steps=steps,
                max_sequence_length=512,
                generator=gen,
            ).images[0]

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            results.append({"b64_json": b64})

    return JSONResponse({"created": int(time.time()), "data": results})


if __name__ == "__main__":
    import uvicorn
    # 起動時にモデルを事前ロード（初回リクエストのタイムアウト回避）
    load_pipeline()
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
