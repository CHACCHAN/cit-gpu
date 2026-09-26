#!/usr/bin/env python3
"""Check the Heretic adapter and compare LoRA-enabled and -disabled requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from verify import call

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/MiMo-V2.6-Flash-RL-Uncensored-Heretic-lora.gguf"


def vram_mib() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return []  # The login node can verify the public API without GPU telemetry.
    return [int(line.strip().split()[0]) for line in result.stdout.splitlines()]


def request(base: str, key: str, model: str, prompt: str, adapter_id: int,
            enabled: bool, *, thinking: bool = False, **extra: object) -> tuple[dict, float]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0,
        "cache_prompt": False,
        "lora": [{"id": adapter_id, "scale": 1.0}] if enabled else [],
        "chat_template_kwargs": {"enable_thinking": thinking},
        **extra,
    }
    result, wall = call(base, key, "/v1/chat/completions", body)
    if not result.get("choices"):
        raise RuntimeError("Chat completion returned no choices")
    return result, wall


def visible_answer(result: dict) -> str:
    message = result["choices"][0]["message"]
    content = message.get("content") or ""
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Empty answer: {str(message)[:300]}")
    if re.search(r"(.{12,}?)\1{4,}", content):
        raise RuntimeError(f"Repeated output: {content[:300]}")
    return content


def quality_round(base: str, key: str, model: str, adapter_id: int,
                  enabled: bool) -> None:
    label = "ON" if enabled else "OFF"
    cases = [
        ("general", "What is the capital of France? Answer in one sentence.", "paris"),
        ("Japanese", "日本の首都はどこですか。都市名を答えてください。", "東京"),
        ("reasoning", "Compute 17 + 25. Give the numerical answer.", "42"),
        ("coding", "Write a Python function square(x) that returns x*x.", "def "),
    ]
    for name, prompt, expected in cases:
        result, wall = request(base, key, model, prompt, adapter_id, enabled)
        answer = visible_answer(result)
        if expected not in answer.lower():
            raise RuntimeError(f"LoRA {label} {name} failed: {answer[:250]}")
        print(f"LoRA {label} {name}: OK {wall:.2f}s; {answer[:120]!r}", flush=True)

    for thinking in (False, True):
        result, wall = request(
            base, key, model,
            "Solve 23 * 17 and explain briefly how you checked it.",
            adapter_id, enabled, thinking=thinking, max_tokens=512,
        )
        answer = visible_answer(result)
        if "391" not in answer:
            raise RuntimeError(f"LoRA {label} thinking={thinking} answer failed: {answer[:250]}")
        reasoning = result["choices"][0]["message"].get("reasoning_content") or ""
        if not thinking and reasoning.strip():
            raise RuntimeError(f"LoRA {label} thinking OFF exposed reasoning")
        if thinking and not reasoning.strip():
            raise RuntimeError(f"LoRA {label} thinking ON produced no reasoning_content")
        print(f"LoRA {label} thinking={thinking}: OK {wall:.2f}s; "
              f"reasoning_chars={len(reasoning)}, answer={answer[:100]!r}", flush=True)

    tools = [{"type": "function", "function": {
        "name": "lookup_project_code", "description": "Look up a project code by name.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]},
    }}]
    result, _ = request(
        base, key, model,
        "Use lookup_project_code to find the code for MiMo. Do not guess it.",
        adapter_id, enabled, tools=tools, tool_choice="required", max_tokens=512,
    )
    message = result["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if not tool_calls or tool_calls[0]["function"]["name"] != "lookup_project_code":
        raise RuntimeError(f"LoRA {label} tool call missing: {str(message)[:300]}")
    arguments = json.loads(tool_calls[0]["function"]["arguments"])
    if arguments.get("name", "").lower() != "mimo":
        raise RuntimeError(f"LoRA {label} invalid tool arguments: {arguments}")
    follow_up, _ = call(base, key, "/v1/chat/completions", {
        "model": model,
        "messages": [
            {"role": "user", "content": "Use lookup_project_code to find the code for MiMo. Do not guess it."},
            message,
            {"role": "tool", "tool_call_id": tool_calls[0]["id"],
             "content": '{"code":"MIMO-26"}'},
        ],
        "max_tokens": 256,
        "lora": [{"id": adapter_id, "scale": 1.0}] if enabled else [],
        "chat_template_kwargs": {"enable_thinking": False},
    })
    if "MIMO-26" not in visible_answer(follow_up):
        raise RuntimeError(f"LoRA {label} tool result was not used")
    print(f"LoRA {label} tool call, JSON and continuation: OK", flush=True)


def stream_round(base: str, key: str, model: str, adapter_id: int,
                 enabled: bool) -> tuple[float, float]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from one to ten."}],
        "max_tokens": 128, "stream": True,
        "lora": [{"id": adapter_id, "scale": 1.0}] if enabled else [],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "curl/8.0"},
    )
    start = time.monotonic()
    first = None
    done = False
    with urllib.request.urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            if json.loads(payload).get("choices") and first is None:
                first = time.monotonic() - start
    if not done or first is None:
        raise RuntimeError("Streaming did not finish")
    return first, time.monotonic() - start


def performance_round(base: str, key: str, model: str, adapter_id: int,
                      enabled: bool) -> None:
    label = "ON" if enabled else "OFF"
    prompt = "In exactly four sentences, describe how binary search works."
    before = vram_mib()
    result, wall = request(base, key, model, prompt, adapter_id, enabled,
                           max_tokens=160)
    visible_answer(result)
    after = vram_mib()
    timings = result.get("timings", {})
    usage = result.get("usage", {})
    print(f"LoRA {label} 1 concurrent: wall={wall:.2f}s, "
          f"prompt_tok/s={timings.get('prompt_per_second', 'n/a')}, "
          f"generation_tok/s={timings.get('predicted_per_second', 'n/a')}, "
          f"output_tokens={usage.get('completion_tokens', 'n/a')}, "
          f"VRAM_MiB_before={before}, after={after}", flush=True)
    ttft, stream_wall = stream_round(base, key, model, adapter_id, enabled)
    print(f"LoRA {label} streaming: TTFT={ttft:.2f}s, wall={stream_wall:.2f}s", flush=True)
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(request, base, key, model,
                               f"In four sentences, explain why {n} is a prime number.",
                               adapter_id, enabled, max_tokens=160)
                   for n in (11, 13, 17, 19)]
        peak = vram_mib()
        while not all(f.done() for f in futures):
            peak = [max(a, b) for a, b in zip(peak, vram_mib())]
            time.sleep(1)
        results = [f.result() for f in futures]
    wall = time.monotonic() - start
    for result, _ in results:
        visible_answer(result)
    tokens = sum(result.get("usage", {}).get("completion_tokens", 0)
                 for result, _ in results)
    print(f"LoRA {label} 4 concurrent: {tokens} output tokens / {wall:.2f}s "
          f"= {tokens / wall:.2f} aggregate tok/s, peak_VRAM_MiB={peak}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--api-key-file", type=Path, default=ROOT / ".secrets/api-keys")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    key = args.api_key_file.read_text().strip()
    adapters, _ = call(base, key, "/lora-adapters")
    if not isinstance(adapters, list) or len(adapters) != 1:
        raise RuntimeError(f"Expected exactly one loaded adapter: {adapters}")
    adapter = adapters[0]
    if Path(adapter["path"]).resolve() != ADAPTER or adapter["scale"] != 1.0:
        raise RuntimeError(f"Unexpected adapter path or scale: {adapter}")
    print("/lora-adapters:", json.dumps(adapters, ensure_ascii=False), flush=True)
    models, _ = call(base, key, "/v1/models")
    model = models["data"][0]["id"]
    for enabled in (False, True):
        quality_round(base, key, model, adapter["id"], enabled)
        performance_round(base, key, model, adapter["id"], enabled)
    final, _ = call(base, key, "/lora-adapters")
    if final != adapters:
        raise RuntimeError(f"Global adapter scale changed: {final}")


if __name__ == "__main__":
    main()
