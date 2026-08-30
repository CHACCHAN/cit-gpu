#!/usr/bin/env python3
"""GPU 個体の P2P 故障を切り分ける。情報システム担当への報告にはこの出力を添える。

  srun --partition=research --nodes=1 --ntasks=1 --cpus-per-task=8 \\
       --gpus-per-task=4 --mem=64G --time=00:10:00 src/p2pdiag.py

ノードは GPU を10枚持っていてジョブは4枚しか掴まないので、
故障個体を引かないと ALL OK になる。数回試すこと。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"


def main() -> int:
    cu = next((VENV / "lib").glob("python3.*")) / "site-packages/nvidia/cu13"
    os.environ["CUDA_HOME"] = str(cu)
    os.environ["LD_LIBRARY_PATH"] = (
        f"{cu / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )

    print(f"=== node: {os.uname().nodename} ===", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
          flush=True)
    subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,serial,name",
         "--format=csv"], check=False)
    print("\n=== nvidia-smi topo -m ===", flush=True)
    out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                         text=True, check=False).stdout.splitlines()
    print("\n".join(out[:12]), flush=True)

    print("\n=== p2p_check ===", flush=True)
    try:
        rc = subprocess.run(
            [str(VENV / "bin/python"), str(ROOT / "src/p2p_check.py")],
            timeout=240, check=False,
        ).returncode
    except subprocess.TimeoutExpired:
        print("p2p_check timed out (>240s)", flush=True)
        rc = 124
    print(f"exit={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
