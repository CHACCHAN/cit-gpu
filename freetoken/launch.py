#!/usr/bin/env python3
"""Start FreeToken workers, SGLang Gateway, then the existing tunnel."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
STOP = False


def load_env(path):
    if path.stat().st_mode & 0o077:
        raise RuntimeError(f"{path} must have mode 600")
    for line in path.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() and not name.lstrip().startswith("#"):
            os.environ[name.strip()] = value.strip()
    # Keep the public endpoint and key while migrating from llama.
    if not os.environ.get("FT_API_KEY"):
        os.environ["FT_API_KEY"] = os.environ.get("LLAMA_API_KEY", "")
    if not os.environ.get("FT_API_KEY"):
        raise RuntimeError("FT_API_KEY is required")
    return os.environ.copy()


def signal_stop(_sig, _frame):
    global STOP
    STOP = True


def stop(proc):
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def probe(url, key=None):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5) as response:
            if response.status != 200:
                return False
            if url.endswith("/health") and not key:
                return json.load(response).get("status") == "ok"
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=Path, default=ROOT / ".env")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    env = load_env(args.env)
    model = Path(env["FT_MODEL_PATH"])
    ft = ROOT / ".venv/bin/ft"
    router = ROOT / ".router-venv/bin/python"
    cloudflared = REPO / "SGLang/bin/cloudflared"
    if not args.dry_run and (not ft.is_file() or not router.is_file()):
        raise RuntimeError("Install FreeToken and sglang-router first (see README)")
    if not args.dry_run and env.get("TUNNEL_TOKEN") and not cloudflared.is_file():
        raise RuntimeError(f"cloudflared missing: {cloudflared}")
    if not args.dry_run and not (model / "model.safetensors.index.json").is_file():
        if env.get("FT_STAGE_MODEL", "1") != "1":
            raise RuntimeError(f"Staged checkpoint missing: {model}")
        subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "stage_model.py")],
                       env=env, check=True)
    if not args.dry_run and not (model / "config.json").is_file():
        raise RuntimeError(f"Checkpoint config missing: {model}")
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    gpus = visible.split(",") if visible else []
    if not args.dry_run and not 1 <= len(gpus) <= 5:
        raise RuntimeError(f"Slurm must assign 1-5 GPUs; CUDA_VISIBLE_DEVICES={visible!r}")
    if args.dry_run and not gpus:
        gpus = [str(i) for i in range(5)]
    replicas = int(env.get("FT_REPLICAS", str(len(gpus))))
    if not 1 <= replicas <= len(gpus):
        raise RuntimeError(f"FT_REPLICAS={replicas} does not fit {len(gpus)} visible GPUs")
    gpus = gpus[:replicas]
    first = int(env.get("FT_WORKER_PORT", "5051"))
    port = int(env.get("FT_PORT", "5050"))
    worker_ports = [first + 2 * i for i in range(replicas)]
    if port in worker_ports or port in [p + 1 for p in worker_ports]:
        raise RuntimeError("Gateway and worker ports overlap")
    name = env.get("FT_MODEL_ID", model.name)
    common = [str(ft), "serve", "--model", str(model), "--served-model-name", name,
              "--host", "127.0.0.1", "--max-seq-len-override", "262144",
              "--num-tokens", "262144", "--kv-reserve-tokens", "262144",
              "--max-running-requests",
              env.get("FT_MAX_RUNNING_REQUESTS", "1"), "--memory-ratio",
              env.get("FT_MEMORY_RATIO", "0.95"), "--moe-strategy",
              env.get("FT_MOE_STRATEGY", "offload"), "--moe-cache-size",
              env.get("FT_MOE_CACHE_SIZE", "512")]
    if env.get("FT_TEXT_MODEL_ONLY", "0") == "1":
        common.append("--text-model-only")
    if env.get("FT_DISABLE_MOE_PREFILL_OVERLAP", "0") == "1":
        common.append("--disable-moe-prefill-overlap")
    if env.get("FT_MAX_PREFILL_LENGTH"):
        common += ["--max-prefill-length", env["FT_MAX_PREFILL_LENGTH"]]
    cmds = [common + ["--port", str(worker_port)] for worker_port in worker_ports]
    for i, cmd in enumerate(cmds):
        print(f"worker {i} GPU {gpus[i]}: {' '.join(cmd)}", flush=True)
    print(f"gateway 127.0.0.1:{port}; tunnel {'enabled' if env.get('TUNNEL_TOKEN') else 'disabled'}", flush=True)
    if args.dry_run:
        return 0
    # Login/compute hosts expose CUDA 12.2 by default. Reuse the CUDA 13 toolkit
    # already installed for SGLang so FreeToken's JIT matches torch cu130.
    cuda = REPO / "SGLang/.venv/lib/python3.12/site-packages/nvidia/cu13"
    if not (cuda / "bin/nvcc").is_file():
        raise RuntimeError(f"CUDA 13 toolkit missing: {cuda}")
    env["CUDA_HOME"] = str(cuda)
    env["PATH"] = f"{cuda / 'bin'}:{ROOT / '.venv/bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    env["QUIC_GO_DISABLE_RECEIVE_BUFFER_WARNING"] = "1"
    workers = []
    gateway = tunnel = None
    try:
        # Parallel reads of the same 63 GiB expert bank overwhelm node-local disk.
        for index, (gpu, cmd, worker_port) in enumerate(zip(gpus, cmds, worker_ports)):
            workers.append(subprocess.Popen(cmd, env={**env, "CUDA_VISIBLE_DEVICES": gpu}, start_new_session=True))
            deadline = time.monotonic() + int(env.get("FT_STARTUP_TIMEOUT", "3600"))
            while not STOP and time.monotonic() < deadline:
                for worker in workers:
                    if worker.poll() is not None:
                        raise RuntimeError(f"FreeToken worker exited: {worker.returncode}")
                if probe(f"http://127.0.0.1:{worker_port}/health"):
                    print(f"worker {index} ready on {worker_port}", flush=True)
                    break
                time.sleep(5)
            else:
                if STOP:
                    return 0
                raise RuntimeError(f"FreeToken worker {index} did not become healthy")
        env["FT_WORKER_URLS"] = " ".join(f"http://127.0.0.1:{worker_port}" for worker_port in worker_ports)
        gateway = subprocess.Popen([str(router), "-u", str(ROOT / "run_gateway.py")], env=env, start_new_session=True)
        for _ in range(120):
            if gateway.poll() is not None:
                raise RuntimeError(f"SGLang Gateway exited: {gateway.returncode}")
            if probe(f"http://127.0.0.1:{port}/health", env["FT_API_KEY"]):
                break
            time.sleep(1)
        else:
            raise RuntimeError("SGLang Gateway did not become healthy")
        print(f"{replicas} workers and Gateway ready", flush=True)
        if env.get("FT_VERIFY_ROUTING", "1") == "1":
            subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "verify_routing.py")],
                           env=env, check=True, timeout=180)
        if env.get("TUNNEL_TOKEN"):
            tunnel = subprocess.Popen([str(cloudflared), "tunnel", "--no-autoupdate", "run"],
                                      env=env, start_new_session=True)
        if env.get("FT_AUTO_RENEW", "1") == "1":
            try:
                queued = subprocess.run(
                    ["squeue", "-h", "-u", env["USER"], "-n", "freetoken", "-t", "PD", "-o", "%i"],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()
                if not queued:
                    successor = subprocess.run(
                        ["sbatch", "--parsable", "--dependency=singleton", str(ROOT / "shinnarashino")],
                        cwd=ROOT, check=True, capture_output=True, text=True,
                    ).stdout.strip()
                    print(f"Queued successor job {successor}", flush=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                print(f"WARNING: could not queue successor job: {exc}", flush=True)
        while not STOP:
            for label, proc in [("worker", w) for w in workers] + [("gateway", gateway), ("tunnel", tunnel)]:
                if proc and proc.poll() is not None:
                    raise RuntimeError(f"{label} exited: {proc.returncode}")
            time.sleep(5)
        return 0
    finally:
        stop(tunnel)
        stop(gateway)
        for worker in workers:
            stop(worker)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
