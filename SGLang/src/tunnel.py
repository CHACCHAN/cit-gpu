"""Cloudflare Tunnel の起動。TUNNEL_TOKEN が空なら何もしない (学内限定運用)。

token は cloudflared が環境変数 TUNNEL_TOKEN から直接読む。
停止は launch.py 側の後始末 (atexit / SIGTERM ハンドラ) で行う。
"""

from __future__ import annotations

import os
import stat
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

DOWNLOAD_URL = ("https://github.com/cloudflare/cloudflared/releases/latest/"
                "download/cloudflared-linux-amd64")

_proc: Optional[subprocess.Popen] = None


def _download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading cloudflared -> {dest}", flush=True)
    req = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "SGLang-Slurm"})
    with urllib.request.urlopen(req) as response, open(dest, "wb") as f:
        while chunk := response.read(1024 * 1024):
            f.write(chunk)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def start(cloudflared: Path) -> None:
    global _proc

    if not os.environ.get("TUNNEL_TOKEN"):
        print("TUNNEL_TOKEN empty -> cloudflared not started (cluster-internal only)",
              flush=True)
        return

    if not (cloudflared.is_file() and os.access(cloudflared, os.X_OK)):
        _download(cloudflared)

    # cloudflared (quic-go) は UDP 受信バッファを 7MiB に広げようとするが、
    # net.core.rmem_max は root でないと上げられない。実害は QUIC の
    # スループット上限だけなので、毎回出る WRN を止める。
    os.environ["QUIC_GO_DISABLE_RECEIVE_BUFFER_WARNING"] = "1"

    _proc = subprocess.Popen(
        [str(cloudflared), "tunnel", "--no-autoupdate", "--loglevel", "info", "run"]
    )
    print(f"cloudflared started: PID={_proc.pid}", flush=True)


def shutdown() -> None:
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
