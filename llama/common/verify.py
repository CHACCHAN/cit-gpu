#!/usr/bin/env python3
"""Exercise llama-server's authenticated OpenAI API, tools, streaming and all slots."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from env import load_env

# Merged into every chat request, e.g. {"reasoning_effort": "none"} for Bonsai
EXTRA_BODY: dict = {}


def call(base: str, key: str, path: str, body: dict | None = None,
         timeout: int = 1800) -> tuple[Any, float]:
    headers = {"Authorization": f"Bearer {key}", "User-Agent": "curl/8.0"}
    if body is not None:
        headers["Content-Type"] = "application/json"
        if path == "/v1/chat/completions":
            body = {**EXTRA_BODY, **body}
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.load(response)
    return value, time.monotonic() - start


def stream(base: str, key: str, model: str) -> tuple[float, float]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from one to ten."}],
        "max_tokens": 128,
        "stream": True,
        **EXTRA_BODY,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json", "User-Agent": "curl/8.0"},
    )
    start = time.monotonic()
    first = None
    done = False
    with urllib.request.urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            value = line[6:].strip()
            if value == b"[DONE]":
                done = True
                break
            item = json.loads(value)
            if item.get("choices") and first is None:
                first = time.monotonic() - start
    if not done or first is None:
        raise RuntimeError("Streaming response did not finish")
    return first, time.monotonic() - start


def chat(base: str, key: str, model: str, prompt: str,
         max_tokens: int = 64, timeout: int = 1800) -> tuple[dict, float]:
    return call(base, key, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }, timeout=timeout)


def tool_round_trip(base: str, key: str, model: str) -> None:
    messages = [{"role": "user", "content":
                 "Only the lookup_project_code tool knows the project code. "
                 "Call it with name MiMo. Do not guess the code. After the tool returns, tell me its code."}]
    tools = [{"type": "function", "function": {
        "name": "lookup_project_code",
        "description": "Look up a project's code by name.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]},
    }}]
    result, _ = call(base, key, "/v1/chat/completions", {
        "model": model, "messages": messages, "tools": tools,
        "tool_choice": "auto", "max_tokens": 1024,
    })
    message = result["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        raise RuntimeError(f"Tool call missing: {str(message)[:400]}")
    tool = tool_calls[0]
    if tool["function"]["name"] != "lookup_project_code":
        raise RuntimeError("Unexpected tool name")
    arguments = json.loads(tool["function"]["arguments"])
    if arguments.get("name", "").lower() != "mimo":
        raise RuntimeError(f"Unexpected tool arguments: {arguments}")
    messages.extend([message, {"role": "tool", "tool_call_id": tool["id"],
                               "content": '{"code":"MIMO-26"}'}])
    follow_up, _ = call(base, key, "/v1/chat/completions", {
        "model": model, "messages": messages, "max_tokens": 1024,
    })
    answer = follow_up["choices"][0]["message"].get("content") or ""
    if "MIMO-26" not in answer:
        raise RuntimeError(f"Tool result was not used: {answer[:400]}")
    print("Automatic tool call, JSON arguments and result continuation: OK", flush=True)


def parallel_round(base: str, key: str, model: str, count: int) -> None:
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(chat, base, key, model,
                               f"In three sentences, describe why {i + 2} is a prime number.",
                               128) for i in range(count)]
        results = [f.result() for f in futures]
    wall = time.monotonic() - start
    tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r, _ in results)
    if any(not r.get("choices") for r, _ in results):
        raise RuntimeError(f"{count} parallel requests did not all complete")
    print(f"{count} concurrent: {tokens} output tokens / {wall:.2f}s = "
          f"{tokens / wall:.2f} aggregate tok/s", flush=True)


def parallel_stream_round(base: str, key: str, model: str, count: int) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(lambda _: stream(base, key, model), range(count)))
    print(f"{count} concurrent SSE streams: OK, TTFT=" +
          ", ".join(f"{first:.2f}s" for first, _ in results), flush=True)


def long_context_round(base: str, key: str, model: str, target: int) -> None:
    word = f"validation{target} "
    sample, _ = call(base, key, "/tokenize", {"content": word * 1000})
    per_word = len(sample["tokens"]) / 1000
    repeat = int((target - 100) / per_word)
    result, wall = chat(base, key, model,
                        "Remember the marker ORANGE. " + word * repeat +
                        " What was the marker?", 32, timeout=7200)
    prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
    if prompt_tokens < target * 0.9 or not result.get("choices"):
        raise RuntimeError(f"Context test at {target} tokens failed: got {prompt_tokens}")
    timings = result.get("timings", {})
    print(f"Context target={target}, actual={prompt_tokens}, wall={wall:.1f}s, "
          f"new_prefill={timings.get('prompt_n', '?')} tokens, "
          f"prompt_tok/s={timings.get('prompt_per_second', '?')}", flush=True)


def all_long_round(base: str, key: str, model: str, count: int, slot_ctx: int,
                   per_server: int) -> None:
    target = slot_ctx * 250000 // 262144
    prompts = []
    for i in range(count):
        word = f"distinct{i} "
        sample, _ = call(base, key, "/tokenize", {"content": word * 1000})
        repeat = int((target - 100) / (len(sample["tokens"]) / 1000))
        prompts.append(f"Remember marker {i}. " + word * repeat +
                       " What was the marker?")
    start = time.monotonic()
    peak_busy = 0
    peak_vram: dict[str, int] = {}
    peak_util: dict[str, int] = {}
    visible = set(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(chat, base, key, model, prompt, 32, 7200)
                   for prompt in prompts]
        while not all(f.done() for f in futures):
            slots, _ = call(base, key, "/slots", timeout=15)
            peak_busy = max(peak_busy, sum(bool(s.get("is_processing")) for s in slots))
            sample = subprocess.run([
                "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, check=False)
            if sample.returncode == 0:
                for line in sample.stdout.splitlines():
                    parts = [value.strip() for value in line.split(",")]
                    if len(parts) != 4:
                        continue
                    index, uuid, memory, utilization = parts
                    if visible == {""} or index in visible or uuid in visible:
                        try:
                            peak_vram[index] = max(peak_vram.get(index, 0), int(memory))
                            peak_util[index] = max(peak_util.get(index, 0), int(utilization))
                        except ValueError:
                            pass
            time.sleep(2)
        results = [f.result() for f in futures]
    wall = time.monotonic() - start
    usages = [r.get("usage", {}) for r, _ in results]
    prompt_total = sum(usage.get("prompt_tokens", 0) for usage in usages)
    # Behind a load balancer /slots shows one replica's slots.
    if peak_busy < per_server:
        raise RuntimeError(f"Only {peak_busy} of {per_server} slots were busy together")
    if any(usage.get("prompt_tokens", 0) < target * 0.9 for usage in usages):
        raise RuntimeError(f"{count} long contexts were truncated: {usages}")
    output_total = sum(usage.get("completion_tokens", 0) for usage in usages)
    print(f"{count} x ~{target // 1000}K contexts: peak_busy={peak_busy}, {prompt_total} prompt tokens, "
          f"{output_total} output tokens, {wall:.1f}s, "
          f"VRAM peak MiB={peak_vram}, GPU util peak %={peak_util}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--slot-ctx", type=int, default=262144)
    parser.add_argument("--replicas", type=int, default=1,
                        help="llama-server replicas behind the load balancer")
    parser.add_argument("--body", type=json.loads, default={})
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--all-long", action="store_true")
    args = parser.parse_args()
    EXTRA_BODY.update(args.body)
    base = args.url.rstrip("/")
    key = load_env()["LLAMA_API_KEY"]
    unauthenticated = None
    try:
        with urllib.request.urlopen(
            urllib.request.Request(base + "/v1/models", headers={"User-Agent": "curl/8.0"}),
            timeout=15,
        ) as response:
            try:
                unauthenticated = json.load(response)
            except (json.JSONDecodeError, UnicodeDecodeError):
                unauthenticated = None  # Cloudflare Access can return a login page.
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 403):
            raise RuntimeError(f"Unauthenticated request returned HTTP {exc.code}") from exc
    else:
        if not isinstance(unauthenticated, dict) or "data" not in unauthenticated:
            unauthenticated = None
    if isinstance(unauthenticated, dict) and "data" in unauthenticated:
        raise RuntimeError("Unauthenticated /v1/models request was accepted")
    print("Bearer authentication: OK", flush=True)
    models, _ = call(base, key, "/v1/models")
    model = models["data"][0]["id"]
    print("Model:", model, flush=True)
    per_server = args.slots // args.replicas
    props, _ = call(base, key, "/props")
    if props.get("total_slots") != per_server:
        raise RuntimeError(f"Expected {per_server} slots, got {props.get('total_slots')}")
    slots, _ = call(base, key, "/slots")
    contexts = [slot.get("n_ctx") for slot in slots]
    if len(slots) != per_server or any(n != args.slot_ctx for n in contexts):
        raise RuntimeError(f"Expected {per_server} x {args.slot_ctx}-token slots, got {contexts}")
    print(f"{args.replicas} x {per_server} x {args.slot_ctx}-token slots: OK", flush=True)
    result, wall = chat(base, key, model, "Hello. Respond briefly.")
    if not result.get("choices"):
        raise RuntimeError("Chat completion failed")
    timings = result.get("timings", {})
    print(f"Chat: OK, {result.get('usage', {}).get('completion_tokens', '?')} "
          f"tokens in {wall:.2f}s; prompt={timings.get('prompt_per_second', '?')} "
          f"tok/s, generation={timings.get('predicted_per_second', '?')} tok/s", flush=True)
    ttft, stream_wall = stream(base, key, model)
    print(f"Streaming: OK, TTFT={ttft:.2f}s, wall={stream_wall:.2f}s", flush=True)
    tool_round_trip(base, key, model)
    for count in sorted({1, 2, args.slots}):
        parallel_round(base, key, model, count)
    parallel_stream_round(base, key, model, args.slots)
    if args.long_context:
        for target in (32768, 65536, 131072, 204800, 250000):
            if target < args.slot_ctx:
                long_context_round(base, key, model, target)
    if args.all_long:
        all_long_round(base, key, model, args.slots, args.slot_ctx, per_server)


if __name__ == "__main__":
    main()
