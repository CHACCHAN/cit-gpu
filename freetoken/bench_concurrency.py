#!/usr/bin/env python3
"""Fire N simultaneous streaming requests at one worker and report per-request and aggregate speed."""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


def env_values():
    values = {}
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            values[key.strip()] = value.strip()
    return values


def one(index, args, cfg):
    # A unique prefix keeps the prefix cache from sharing work between requests.
    filler = " hello" * args.prompt_words
    content = (f"Request {index} of {args.n}, tag {time.time_ns()}.{filler}\n\n"
               "Now write a long, detailed story about a lighthouse keeper. Do not stop early.")
    payload = {
        "model": cfg["FT_MODEL_ID"], "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens, "temperature": 0.7, "reasoning_effort": "low",
    }
    headers = {"Content-Type": "application/json"}
    if args.url.rstrip("/").endswith(":5050"):
        headers["Authorization"] = "Bearer " + cfg["FT_API_KEY"]
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers=headers)
    start = time.monotonic()
    first = None
    usage = {}
    chunks = 0
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            event = json.loads(line[5:])
            if event.get("usage"):
                usage = event["usage"]
            delta = event["choices"][0].get("delta", {}) if event.get("choices") else {}
            # 役割だけの先頭断片 (本文が空) は最初のトークンに数えない
            if delta.get("content") or delta.get("reasoning_content"):
                chunks += 1
                if first is None:
                    first = time.monotonic()
    end = time.monotonic()
    return {
        "i": index, "prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens", chunks),
        "ttft_s": round(first - start, 1) if first else None,
        "total_s": round(end - start, 1),
        "decode_tps": round((usage.get("completion_tokens", chunks) or 0) / max(end - (first or start), 1e-6), 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:5051")
    p.add_argument("-n", type=int, default=8, help="simultaneous requests")
    p.add_argument("--prompt-words", type=int, default=1000, help="' hello' repeats per prompt (~1 token each)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    cfg = env_values()
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as pool:
        results = list(pool.map(lambda i: one(i, args, cfg), range(args.n)))
    wall = time.monotonic() - start
    for r in results:
        print(r, flush=True)
    done = sum(r["completion"] or 0 for r in results)
    print(f"n={args.n} wall_s={wall:.1f} total_completion_tokens={done} "
          f"mean_per_request_decode_tps={sum(r['decode_tps'] for r in results) / len(results):.2f}", flush=True)


if __name__ == "__main__":
    main()
