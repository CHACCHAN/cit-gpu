#!/usr/bin/env python3
"""Run SGLang Gateway using the existing llama API key without exposing it in argv."""

import os

from sglang_router.launch_router import launch_router
from sglang_router.router_args import RouterArgs


def main():
    key = os.environ["LLAMA_API_KEY"]
    upstreams = os.environ["LLAMA_UPSTREAMS"].split()
    if not upstreams:
        raise RuntimeError("No llama-server workers configured")
    args = RouterArgs(
        host="127.0.0.1",
        port=int(os.environ.get("LLAMA_GATEWAY_PORT", "5050")),
        worker_urls=[f"http://{address}" for address in upstreams],
        policy="cache_aware",
        balance_abs_threshold=1,
        balance_rel_threshold=1.5,
        cache_threshold=0.3,
        max_concurrent_requests=64,
        queue_size=100,
        queue_timeout_secs=600,
        health_check_interval_secs=10,
        health_check_timeout_secs=10,
        health_failure_threshold=3,
        request_timeout_secs=2400,
        prometheus_host="127.0.0.1",
        prometheus_port=int(os.environ.get("LLAMA_GATEWAY_PROMETHEUS_PORT", "29000")),
        control_plane_api_keys=[("client", "existing-api-key", key, "user")],
    )
    launch_router(args)


if __name__ == "__main__":
    main()
