#!/usr/bin/env python3
"""4GPU の NCCL all_reduce が通る環境変数の組み合わせを総当たりする。
分散初期化ハングの切り分け専用。

  srun --partition=research --nodes=1 --ntasks=1 --cpus-per-task=8 \\
       --gpus-per-task=4 --mem=64G --time=00:25:00 src/dist_smoke_sweep.py
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"

CASES = [
    ("baseline", {}),
    ("cumem-off", {"NCCL_CUMEM_ENABLE": "0"}),
    ("p2p-off", {"NCCL_P2P_DISABLE": "1"}),
    ("shm-off", {"NCCL_SHM_DISABLE": "1"}),
    ("cumem-off+p2p-off", {"NCCL_CUMEM_ENABLE": "0", "NCCL_P2P_DISABLE": "1"}),
    ("p2p-level-sys", {"NCCL_P2P_LEVEL": "SYS"}),
]


def run_case(name: str, extra: dict) -> None:
    print(f"======== {name} ========", flush=True)
    env = os.environ.copy()
    cu = next((VENV / "lib").glob("python3.*")) / "site-packages/nvidia/cu13"
    env.update({
        "CUDA_HOME": str(cu),
        "LD_LIBRARY_PATH": f"{cu / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(39600 + random.randrange(200)),
        "OMP_NUM_THREADS": "1",
        **extra,
    })
    try:
        proc = subprocess.run(
            [str(VENV / "bin/torchrun"), "--nproc_per_node=4",
             str(ROOT / "src/dist_smoke.py")],
            env=env, capture_output=True, text=True, timeout=90, check=False,
        )
        rc = proc.returncode
        lines = (proc.stdout + proc.stderr).splitlines()
    except subprocess.TimeoutExpired as exc:
        rc = 124
        def _text(x):
            return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
        lines = (_text(exc.stdout) + _text(exc.stderr)).splitlines()

    hits = [l for l in lines if "all_reduce ok" in l or "init_process_group ok" in l]
    print("\n".join(hits[-5:]), flush=True)
    print(f"exit={rc}\n", flush=True)


def main() -> None:
    for name, extra in CASES:
        run_case(name, extra)
    print("======== sweep done ========", flush=True)


if __name__ == "__main__":
    main()
