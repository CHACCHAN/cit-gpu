#!/usr/bin/env python3
"""Send a measured long prompt and report whether FreeToken kept its tokens."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def env_values():
    values = {}
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            values[key.strip()] = value.strip()
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:5051")
    p.add_argument("--target-tokens", type=int, default=260000)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    cfg = env_values()
    tokenizer = AutoTokenizer.from_pretrained(cfg["FT_MODEL_PATH"], local_files_only=True)

    def prompt_for(repeats):
        # Vary the prefix so a previous shorter sweep cannot warm this request.
        content = f"Context test {args.target_tokens}. " + " hello" * repeats
        messages = [{"role": "user", "content": content}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return messages, len(ids["input_ids"])

    lo, hi = 1, args.target_tokens
    while lo < hi:
        mid = (lo + hi) // 2
        _, count = prompt_for(mid)
        if count < args.target_tokens:
            lo = mid + 1
        else:
            hi = mid
    messages, count = prompt_for(lo)
    print(f"Prepared prompt: {count} tokens, {len(messages[0]['content'])} characters", flush=True)
    payload = {
        "model": cfg["FT_MODEL_ID"], "messages": messages,
        "max_tokens": 8, "temperature": 0, "reasoning_effort": "low",
    }
    headers = {"Content-Type": "application/json"}
    if args.url.rstrip("/").endswith(":5050"):
        headers["Authorization"] = "Bearer " + cfg["FT_API_KEY"]
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers=headers,
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read(1000).decode(errors='replace')}", flush=True)
        raise
    elapsed = time.monotonic() - start
    usage = result.get("usage", {})
    print(f"elapsed_s={elapsed:.1f} prompt_tokens={usage.get('prompt_tokens')} "
          f"completion_tokens={usage.get('completion_tokens')} "
          f"finish_reason={result['choices'][0].get('finish_reason')}", flush=True)
    if usage.get("prompt_tokens", 0) < args.target_tokens - 500:
        raise RuntimeError("Long prompt appears truncated")


if __name__ == "__main__":
    main()
