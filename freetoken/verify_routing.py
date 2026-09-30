#!/usr/bin/env python3
"""Confirm the gateway spreads simultaneous, unrelated requests over every FreeToken worker."""

import concurrent.futures
import json
import random
import time
import urllib.request
from pathlib import Path


def configuration():
    values = {}
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            values[key.strip()] = value.strip()
    return values


def completed(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/stats", timeout=10) as response:
        return json.load(response)["requests"]["completed"]


def request(index, config):
    # 互いに無関係な長めの入力を同時に投げる。cache_aware でも負荷偏りで全 worker に分かれる。
    rng = random.Random(index)
    filler = " ".join(f"{rng.getrandbits(40):x}" for _ in range(600))
    payload = {
        "model": config["FT_MODEL_ID"],
        "messages": [{"role": "user", "content": f"Test {index}: {filler}\n\nReply with a short sentence."}],
        "max_tokens": 64,
        "temperature": 0,
        "reasoning_effort": "low",
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{config.get('FT_PORT', '5050')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + config["FT_API_KEY"]},
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    if not result.get("choices"):
        raise RuntimeError(f"Request {index} returned no choices")
    return index, round(time.monotonic() - start, 1)


def main():
    config = configuration()
    ports = [int(config.get("FT_WORKER_PORT", "5051")) + 2 * i for i in range(int(config["FT_REPLICAS"]))]
    before = [completed(port) for port in ports]
    count = len(ports) * 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(lambda i: request(i, config), range(count)))
    after = [completed(port) for port in ports]
    deltas = [end - start for start, end in zip(before, after)]
    print(f"requests={results}", flush=True)
    print(f"worker_ports={ports} completed_deltas={deltas}", flush=True)
    if any(delta < 1 for delta in deltas):
        raise RuntimeError("The gateway did not route requests to every worker")


if __name__ == "__main__":
    main()
