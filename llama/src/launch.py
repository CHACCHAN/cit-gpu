#!/usr/bin/env python3
"""Run llama-server (and a Cloudflare Tunnel if a token is given) inside one Slurm job.

    python3 -u src/launch.py --server BIN --model GGUF [--lora GGUF] --alias NAME
        [--tunnel-token FILE] [--dry-run] -- LLAMA_SERVER_ARGS...
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

ROOT = Path(__file__).resolve().parents[1]
API_KEYS = ROOT / ".secrets/api-keys"
CLOUDFLARED = ROOT.parent / "SGLang/bin/cloudflared"
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


def healthy(key):
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/health",
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
    parser.add_argument("--tunnel-token", type=lambda p: ROOT / p)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("server_args", nargs="*")
    args = parser.parse_args()

    secrets = [API_KEYS] + ([args.tunnel_token] if args.tunnel_token else [])
    required = [args.server, args.model, *secrets]
    required += [p for p in (args.lora, args.tunnel_token and CLOUDFLARED) if p]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Required file is missing: {path}")
    for path in secrets:
        if path.stat().st_mode & 0o077:
            raise RuntimeError(f"Secret file permissions are too broad: {path}")
    key = API_KEYS.read_text().strip()
    token = args.tunnel_token.read_text().strip() if args.tunnel_token else None
    if not key or "\n" in key or token == "":
        raise RuntimeError("Invalid API key or Tunnel token")

    command = [str(args.server), "-m", str(args.model)]
    if args.lora:
        command += ["--lora", str(args.lora)]
    command += [
        *args.server_args, "--alias", args.alias, "--api-key-file", str(API_KEYS),
        "--host", "0.0.0.0", "--port", str(PORT),
    ]
    print("command:", " ".join(command), flush=True)
    if args.dry_run:
        return 0

    # Resolve the libraries next to llama-server, whatever RUNPATH the build has.
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        filter(None, [str(args.server.parent), env.get("LD_LIBRARY_PATH")]))

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    server = tunnel = None
    try:
        server = subprocess.Popen(command, env=env, start_new_session=True)
        deadline = time.monotonic() + 3600
        while not STOP and time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"llama-server exited: {server.returncode}")
            if healthy(key):
                break
            time.sleep(5)
        else:
            if STOP:
                return 0
            raise RuntimeError("llama-server did not become healthy within 1 hour")
        print("llama-server ready", flush=True)

        if token:
            env = os.environ.copy()
            env["TUNNEL_TOKEN"] = token
            env["QUIC_GO_DISABLE_RECEIVE_BUFFER_WARNING"] = "1"
            tunnel = subprocess.Popen(
                [str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--loglevel", "info", "run"],
                env=env, start_new_session=True,
            )
            print("Cloudflare Tunnel started", flush=True)
        while not STOP:
            if server.poll() is not None:
                raise RuntimeError(f"llama-server exited: {server.returncode}")
            if tunnel is not None and tunnel.poll() is not None:
                raise RuntimeError(f"cloudflared exited: {tunnel.returncode}")
            time.sleep(5)
        return 0
    finally:
        stop_process(tunnel)
        stop_process(server)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
