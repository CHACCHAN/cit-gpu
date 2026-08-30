#!/usr/bin/env python3
"""SGLang 配信ジョブの汎用パイプライン。サイト固有の差分は持たない。

ジョブファイル (sglang.shinnarashino / sglang.tsudanuma) が .env を source し、
サイト固有の環境変数と serve 追加引数を組み立ててから exec する:

    python3 -u src/launch.py [--marker FILE]... [--dry-run] -- [serve 追加引数...]

  --marker FILE : config.json に加えて存在を確認するモデル内ファイル (複数可)
  --dry-run     : tunnel と serve を起動せず、組み上がった引数を表示して終わる
  -- 以降       : serve.base_args() に足すサイト固有引数 (そのまま渡す)

流れ: 検証 → ノード情報 → Cloudflare Tunnel → venv 構築/補正
      → モデル取得 → sglang serve (起動ウォッチドッグ付き)。
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modelfetch
import serve
import tunnel
import venvsetup

# 両サイト共通の必須変数。サイト固有の必須分はジョブファイル側で
# ${VAR:?} 展開により担保する
COMMON_REQUIRED = [
    "SG_MODEL_REPO", "SG_MODEL", "SG_SERVED_MODEL_NAME", "SG_HOST", "SG_PORT",
    "SG_API_KEY", "SG_TP_SIZE", "SG_CONTEXT_LENGTH", "SG_MEM_FRACTION_STATIC",
    "SG_MAX_RUNNING_REQUESTS", "SG_CHUNKED_PREFILL_SIZE", "SG_CUDA_GRAPH_MAX_BS",
    "SG_WEIGHT_LOAD_THREADS", "SG_REASONING_PARSER", "SG_TOOL_CALL_PARSER",
]


def require_vars(names: list) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        sys.exit(f"ERROR: required env vars missing (source .env in the job file): "
                 f"{' '.join(missing)}")


def resolve_path(workdir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else workdir / p


def print_node_info() -> None:
    e = os.environ.get
    print(f"job {e('SLURM_JOB_ID', 'local')} on {os.uname().nodename} "
          f"/ tp={e('SG_TP_SIZE')} ctx={e('SG_CONTEXT_LENGTH')}", flush=True)
    print(f"running={e('SG_MAX_RUNNING_REQUESTS')} "
          f"queued={e('SG_MAX_QUEUED_REQUESTS', 'default')} "
          f"hicache={e('SG_ENABLE_HICACHE', '0')}", flush=True)
    if e("SG_ENABLE_HICACHE") == "1" and e("SG_HICACHE_SIZE"):
        total = int(e("SG_HICACHE_SIZE")) * int(e("SG_TP_SIZE"))
        print(f"hicache {e('SG_HICACHE_SIZE')}GB/rank = {total}GB total", flush=True)

    subprocess.run(["nvidia-smi",
                    "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                    "--format=csv"], check=False)
    out = subprocess.run(["free", "-g"], capture_output=True, text=True,
                         check=False).stdout.splitlines()
    print("\n".join(out[:2]), flush=True)
    out = subprocess.run(["quota", "-s"], capture_output=True, text=True,
                         check=False).stdout.splitlines()
    if out:
        print(out[-1], flush=True)
    subprocess.run(["df", "-h", "/dev/shm"], check=False)


def main() -> None:
    argv = sys.argv[1:]
    extra: list = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)

    # ジョブファイルが SLURM_SUBMIT_DIR (= SGLang/) に cd し .env を source 済み
    workdir = Path.cwd()
    require_vars(COMMON_REQUIRED)

    model_path = resolve_path(workdir, os.environ["SG_MODEL"])
    venv = workdir / ".venv"
    cloudflared = workdir / "bin/cloudflared"

    os.environ["HF_HOME"] = str(workdir / ".hf-cache")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PIP_NO_CACHE_DIR"] = "1"
    # CUDA graph の捕獲に失敗する場合はこの行を外す
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    print_node_info()

    # scancel (SIGTERM) / Ctrl-C で子を確実に畳む
    def _terminate(signum: int, _frame) -> None:
        serve.shutdown(signal.SIGTERM)
        tunnel.shutdown()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    atexit.register(tunnel.shutdown)

    if not ns.dry_run:
        tunnel.start(cloudflared)

    venvsetup.setup(venv)
    site_packages = venvsetup.env(venv)
    venvsetup.tidy(venv, site_packages)

    modelfetch.fetch(venv, model_path, ns.marker)

    args = serve.base_args(model_path) + extra

    if ns.dry_run:
        print("--- dry run: sglang serve args ---", flush=True)
        masked, hide_next = [], False
        for a in args:
            masked.append("***" if hide_next else a)
            hide_next = a == "--api-key"
        print("\n".join(masked), flush=True)
        return

    serve.run_with_watchdog(venv, args)


if __name__ == "__main__":
    main()
