"""SGLang Model Gateway for local FreeToken workers. Internal only: auth_proxy.py checks the API key."""

import os

from sglang_router.launch_router import launch_router
from sglang_router.router_args import RouterArgs


def main():
    key = os.environ["FT_API_KEY"]
    urls = os.environ["FT_WORKER_URLS"].split()
    if not 1 <= len(urls) <= 8:
        raise RuntimeError(f"Expected 1-8 FreeToken workers, got {len(urls)}")
    policy = os.environ.get("FT_ROUTING_POLICY", "round_robin")
    launch_router(RouterArgs(
        host="127.0.0.1",
        port=int(os.environ.get("FT_GATEWAY_PORT", "5049")),  # 公開は auth_proxy.py の FT_PORT
        worker_urls=urls,
        policy=policy,
        # cache_aware: 同じ長い会話履歴を同じ worker へ送り、FreeToken の prefix cache を効かせる。
        # 偏りが閾値を超えたら空いている worker へ回す (llama/common/run_gateway.py と同じ値)。
        balance_abs_threshold=1,
        balance_rel_threshold=1.5,
        cache_threshold=0.3,
        max_concurrent_requests=int(os.environ.get("FT_GATEWAY_CONCURRENCY", "64")),
        queue_size=100,
        queue_timeout_secs=600,
        health_check_interval_secs=10,
        health_check_timeout_secs=10,
        health_failure_threshold=3,
        request_timeout_secs=2400,
        prometheus_host="127.0.0.1",
        prometheus_port=int(os.environ.get("FT_PROMETHEUS_PORT", "29000")),
        control_plane_api_keys=[("client", "freetoken-api-key", key, "user")],
    ))


if __name__ == "__main__":
    main()
