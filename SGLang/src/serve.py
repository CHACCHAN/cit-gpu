"""sglang serve の共通引数と起動ウォッチドッグ。

  base_args()         : 両サイト共通の serve 引数。サイト固有分は
                        ジョブファイルが launch.py の -- 以降で渡す
  run_with_watchdog() : sglang serve を起動し、READY (/health 200) まで
                        SG_STARTUP_TIMEOUT 秒待つ。駄目なら殺して張り直す
                        (最大 SG_STARTUP_RETRIES 回)。分散初期化がハングする
                        環境向け (README「分散初期化のハング」)。
                        この関数は戻らない (サーバの終了コードで exit する)。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

_proc: Optional[subprocess.Popen] = None


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def base_args(model_path: Path) -> List[str]:
    return [
        "--model-path", str(model_path),
        "--served-model-name", _env("SG_SERVED_MODEL_NAME"),
        "--host", _env("SG_HOST"),
        "--port", _env("SG_PORT"),
        "--api-key", _env("SG_API_KEY"),
        "--tp-size", _env("SG_TP_SIZE"),
        "--context-length", _env("SG_CONTEXT_LENGTH"),
        "--mem-fraction-static", _env("SG_MEM_FRACTION_STATIC"),
        "--max-running-requests", _env("SG_MAX_RUNNING_REQUESTS"),
        "--chunked-prefill-size", _env("SG_CHUNKED_PREFILL_SIZE"),
        "--cuda-graph-max-bs-decode", _env("SG_CUDA_GRAPH_MAX_BS"),
        "--model-loader-extra-config",
        '{"num_threads": %s}' % _env("SG_WEIGHT_LOAD_THREADS"),
        "--weight-loader-drop-cache-after-load",
        "--reasoning-parser", _env("SG_REASONING_PARSER"),
        "--tool-call-parser", _env("SG_TOOL_CALL_PARSER"),
        "--allow-auto-truncate",
    ]


def _health_ok() -> bool:
    req = urllib.request.Request(
        f"http://127.0.0.1:{_env('SG_PORT')}/health",
        headers={"Authorization": f"Bearer {_env('SG_API_KEY')}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _kill_tree(proc: subprocess.Popen) -> None:
    # start_new_session で独立した process group にしてあるので、
    # scheduler 群まで process group ごと殺せる。ノードは他ジョブと
    # 共有なので、名前ベースの pkill は使わない (同一ユーザの別ジョブの
    # sglang を巻き込むため)。
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def shutdown(sig: int = signal.SIGTERM) -> None:
    """SIGTERM (scancel) 時の後始末。launch.py のハンドラから呼ばれる。"""
    if _proc is not None and _proc.poll() is None:
        try:
            os.killpg(_proc.pid, sig)
        except ProcessLookupError:
            pass


def run_with_watchdog(venv: Path, args: List[str]) -> None:
    global _proc

    # READY までは triton JIT キャッシュ次第で大きく振れる
    # (新習志野実測: コールド 835s / ウォーム 185s)。コールド基準で取る。
    timeout = int(_env("SG_STARTUP_TIMEOUT", "1800"))
    retries = int(_env("SG_STARTUP_RETRIES", "5"))

    for attempt in range(1, retries + 1):
        _proc = subprocess.Popen(
            [str(venv / "bin/sglang"), "serve", *args],
            start_new_session=True,
        )

        ready = False
        for _ in range(timeout // 5):
            if _proc.poll() is not None:
                break
            if _health_ok():
                ready = True
                break
            time.sleep(5)

        if ready:
            print(f"server ready (attempt {attempt})", flush=True)
            sys.exit(_proc.wait())

        print(f"startup did not reach ready within {timeout}s "
              f"(attempt {attempt}); restarting", flush=True)
        _kill_tree(_proc)
        _proc = None
        time.sleep(20)

    sys.exit(f"ERROR: server failed to start after {retries} attempts")
