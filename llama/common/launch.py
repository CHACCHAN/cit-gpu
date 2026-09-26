#!/usr/bin/env python3
"""Run llama-server behind Caddy (and a Cloudflare Tunnel if .env has TUNNEL_TOKEN) in one Slurm job.

    python3 -u common/launch.py --server BIN --model GGUF [--lora GGUF] --alias NAME
        [--replicas N] [--dry-run] -- LLAMA_SERVER_ARGS...

The job's GPUs are split into N groups (default 1), one llama-server each on 127.0.0.1:5051+,
and Caddy (bin/caddy, common/Caddyfile) serves them on port 5050.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from env import load_env

ROOT = Path(__file__).resolve().parents[1]
CLOUDFLARED = ROOT.parent / "SGLang/bin/cloudflared"
CADDY = ROOT / "bin/caddy"
PORT = 5050
STOP = False


def on_signal(_signum, _frame):
    global STOP
    STOP = True


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def healthy(port, key):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/health",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=lambda p: ROOT / p, required=True)
    parser.add_argument("--model", type=lambda p: ROOT / p, required=True)
    parser.add_argument("--lora", type=lambda p: ROOT / p)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("server_args", nargs="*")
    args = parser.parse_args()

    env = load_env()
    key = env["LLAMA_API_KEY"]
    token = env.get("TUNNEL_TOKEN")
    required = [args.server, args.model, CADDY] + [p for p in (args.lora, token and CLOUDFLARED) if p]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Required file is missing: {path}")

    command = [str(args.server), "-m", str(args.model)]
    if args.lora:
        command += ["--lora", str(args.lora)]
    command += [*args.server_args, "--alias", args.alias]
    ports = [PORT + 1 + i for i in range(args.replicas)]
    visible = env.get("CUDA_VISIBLE_DEVICES")  # Unset on the login node (--dry-run)
    gpus = visible.split(",") if visible else [""] * args.replicas
    if len(gpus) % args.replicas:
        raise RuntimeError(f"{len(gpus)} GPUs cannot be split into {args.replicas} replicas")
    per = len(gpus) // args.replicas
    groups = [",".join(gpus[i * per:(i + 1) * per]) for i in range(args.replicas)]
    commands = [command + ["--host", "127.0.0.1", "--port", str(port)] for port in ports]
    for gpu, cmd in zip(groups, commands):
        print(f"command (GPU {gpu or 'all'}):", " ".join(cmd), flush=True)
    print(f"Tunnel: {'on' if token else 'off'}", flush=True)
    if args.dry_run:
        return 0

    # Resolve the libraries next to llama-server, whatever RUNPATH the build has.
    env["LD_LIBRARY_PATH"] = ":".join(
        filter(None, [str(args.server.parent), env.get("LD_LIBRARY_PATH")]))
    env["QUIC_GO_DISABLE_RECEIVE_BUFFER_WARNING"] = "1"

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    servers = []
    tunnel = balancer = None
    try:
        for gpu, cmd in zip(groups, commands):
            servers.append(subprocess.Popen(
                cmd, env={**env, "CUDA_VISIBLE_DEVICES": gpu} if gpu else env,
                start_new_session=True))
        deadline = time.monotonic() + 3600
        while not STOP and time.monotonic() < deadline:
            for server in servers:
                if server.poll() is not None:
                    raise RuntimeError(f"llama-server exited: {server.returncode}")
            if all(healthy(port, key) for port in ports):
                break
            time.sleep(5)
        else:
            if STOP:
                return 0
            raise RuntimeError("llama-server did not become healthy within 1 hour")
        env["LLAMA_UPSTREAMS"] = " ".join(f"127.0.0.1:{port}" for port in ports)
        extra = args.server_args
        env["LLAMA_SLOTS"] = extra[extra.index("--parallel") + 1] if "--parallel" in extra else "1"
        balancer = subprocess.Popen(
            [str(CADDY), "run", "--config", str(ROOT / "common/Caddyfile"),
             "--adapter", "caddyfile"], env=env, start_new_session=True)
        print("llama-server ready", flush=True)

        if token:
            tunnel = subprocess.Popen(
                [str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--loglevel", "info", "run"],
                env=env, start_new_session=True,
            )
            print("Cloudflare Tunnel started", flush=True)
        while not STOP:
            for server in servers:
                if server.poll() is not None:
                    raise RuntimeError(f"llama-server exited: {server.returncode}")
            for name, process in (("cloudflared", tunnel), ("caddy", balancer)):
                if process is not None and process.poll() is not None:
                    raise RuntimeError(f"{name} exited: {process.returncode}")
            time.sleep(5)
        return 0
    finally:
        stop_process(tunnel)
        stop_process(balancer)
        for server in servers:
            stop_process(server)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
