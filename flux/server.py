"""
FLUX.1-dev OpenAI Images API 互換サーバ
"""
import os
import io
import base64
import time
import threading
import secrets
import unicodedata
import queue
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_ID      = os.environ.get("MODEL", "black-forest-labs/FLUX.1-dev")
DEFAULT_STEPS = int(os.environ.get("STEPS", "50"))
DEFAULT_GUID  = float(os.environ.get("GUIDANCE", "3.5"))
API_PORT      = int(os.environ.get("FLUX_API_PORT", "9090"))
_API_KEY      = os.environ.get("FLUX_API_KEY", "").strip()

NUM_GPUS_PER_PIPE = 2

if not _API_KEY:
    print("[flux-api] WARNING: FLUX_API_KEY is not set.", flush=True)
else:
    print("[flux-api] API key authentication enabled.", flush=True)

# API認証
_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def verify_api_key(authorization: str = Security(_api_key_header)):
    if not _API_KEY:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, _API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")

# 翻訳
def _is_japanese(text: str) -> bool:
    for ch in text:
        name = unicodedata.name(ch, "")
        if "HIRAGANA" in name or "KATAKANA" in name or "CJK" in name:
            return True
    return False

def _translate_to_english(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="ja", target="en").translate(text)
        print(f"[flux-api] Translated: '{text[:40]}' -> '{translated[:80]}'", flush=True)
        return translated
    except Exception as e:
        print(f"[flux-api] Translation failed: {e}", flush=True)
        return text

# ワーカのプール
_worker_pool: queue.Queue = queue.Queue()
_num_pipes = 0
_pool_ready = threading.Event()


def _load_pipeline(pipe_id: int, gpu_ids: list):
    """指定GPU群にパイプラインをロード"""
    from diffusers import FluxPipeline

    print(f"[flux-api] Pipeline {pipe_id}: Loading on GPU{gpu_ids} (full bf16)...", flush=True)

    # 各GPUに均等分散（2GPUで~34GB → 各17GB）
    max_memory = {i: "17GB" for i in gpu_ids}
    max_memory["cpu"] = "32GB"

    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=max_memory,
    )

    used = sum(torch.cuda.memory_allocated(i) for i in gpu_ids) / 1e9
    print(f"[flux-api] Pipeline {pipe_id}: Ready. Total VRAM used: {used:.1f}GB across GPU{gpu_ids}", flush=True)
    return pipe


def initialize_workers():
    global _num_pipes
    total_gpus = torch.cuda.device_count()

    # GPU割り当て [0,1], [2,3]..
    gpu_groups = []
    for i in range(0, total_gpus - 1, NUM_GPUS_PER_PIPE):
        group = list(range(i, min(i + NUM_GPUS_PER_PIPE, total_gpus)))
        if len(group) == NUM_GPUS_PER_PIPE:
            gpu_groups.append(group)

    _num_pipes = len(gpu_groups)
    print(f"[flux-api] Initializing {_num_pipes} pipeline(s) on {total_gpus} GPUs...", flush=True)
    print(f"[flux-api] GPU groups: {gpu_groups}", flush=True)

    for pipe_id, gpu_ids in enumerate(gpu_groups):
        pipe = _load_pipeline(pipe_id, gpu_ids)
        _worker_pool.put((pipe_id, pipe))

    print(f"[flux-api] All {_num_pipes} pipelines ready. Max concurrent requests: {_num_pipes}", flush=True)
    _pool_ready.set()


def _parse_size(size: Optional[str]):
    if not size or "x" not in size:
        return 1024, 1024
    try:
        w, h = size.lower().split("x")
        w = max(256, min(2048, (int(w) // 16) * 16))
        h = max(256, min(2048, (int(h) // 16) * 16))
        return w, h
    except Exception:
        return 1024, 1024


# FastAPI
app = FastAPI(title="FLUX.1 Images API")


class ImageRequest(BaseModel):
    prompt: str
    n: int = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "b64_json"
    model: Optional[str] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None


@app.get("/health")
def health(auth=Security(verify_api_key)):
    available = _worker_pool.qsize()
    return {
        "status": "ok",
        "model": MODEL_ID,
        "quantization": "none (full bf16)",
        "total_pipelines": _num_pipes,
        "available_pipelines": available,
        "busy_pipelines": _num_pipes - available,
    }


@app.get("/v1/models")
def list_models(auth=Security(verify_api_key)):
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/images/generations")
def generate(req: ImageRequest, auth=Security(verify_api_key)):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if not _pool_ready.is_set():
        raise HTTPException(status_code=503, detail="Pipelines not ready yet")

    prompt = req.prompt
    if _is_japanese(prompt):
        print("[flux-api] Japanese prompt detected, translating...", flush=True)
        prompt = _translate_to_english(prompt)

    width, height = _parse_size(req.size)
    steps    = req.steps    or DEFAULT_STEPS
    guidance = req.guidance or DEFAULT_GUID
    n        = max(1, min(req.n or 1, 8))

    print(f"[flux-api] generate: '{prompt[:80]}' "
          f"size={width}x{height} n={n} steps={steps} "
          f"(pipelines available: {_worker_pool.qsize()}/{_num_pipes})", flush=True)

    # 空きパイプラインを取得（なければ待機）
    pipe_id, pipe = _worker_pool.get()
    try:
        results = []
        for i in range(n):
            s = (req.seed if req.seed is not None else int(time.time())) + i
            gen = torch.Generator("cpu").manual_seed(s)
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance,
                num_inference_steps=steps,
                max_sequence_length=512,
                generator=gen,
            ).images[0]

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            results.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})

        return JSONResponse({"created": int(time.time()), "data": results})
    finally:
        _worker_pool.put((pipe_id, pipe))


if __name__ == "__main__":
    import uvicorn
    init_thread = threading.Thread(target=initialize_workers, daemon=True)
    init_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)