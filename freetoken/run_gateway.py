"""One authenticated SGLang Model Gateway for local FreeToken workers."""

import os

from sglang_router.launch_router import launch_router
from sglang_router.router_args import RouterArgs


def main():
    key = os.environ["FT_API_KEY"]
    urls = os.environ["FT_WORKER_URLS"].split()
    if not 1 <= len(urls) <= 5:
        raise RuntimeError(f"Expected 1-5 FreeToken workers, got {len(urls)}")
    launch_router(RouterArgs(
        host="127.0.0.1",
        port=int(os.environ.get("FT_PORT", "5050")),
        worker_urls=urls,
        policy="round_robin",
        max_concurrent_requests=64,
        queue_size=100,
        queue_timeout_secs=600,
        health_check_interval_secs=10,
        health_check_timeout_secs=10,
        health_failure_threshold=3,
        request_timeout_secs=2400,
        prometheus_host="127.0.0.1",
        prometheus_port=29000,
        control_plane_api_keys=[("client", "freetoken-api-key", key, "user")],
    ))


if __name__ == "__main__":
    main()
