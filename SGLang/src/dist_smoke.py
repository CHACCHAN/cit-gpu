#!/usr/bin/env python3
"""4GPU の NCCL all_reduce だけを行う最小ジョブ。分散初期化ハングの切り分け用。

  srun --jobid=<jobid> --overlap bash -c '
    export NCCL_DEBUG=INFO MASTER_ADDR=127.0.0.1 MASTER_PORT=39501
    .venv/bin/torchrun --nproc_per_node=4 src/dist_smoke.py'

どの段階で止まったかを rank ごとに時刻付きで出す。
NCCL 自身のログは stdout に出るので、sbatch 経由なら logs/<jobid>.log 側を見る
(sglang のログは stderr = logs/<jobid>.err なので別ファイルになる)。
"""

import os
import time

import torch
import torch.distributed as dist


def stamp(rank: str, msg: str) -> None:
    print(f"[{rank}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def main() -> None:
    rank = os.environ.get("RANK", "0")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    stamp(rank, "start")
    torch.cuda.set_device(local_rank)
    stamp(rank, f"set_device({local_rank}) ok")

    dist.init_process_group("nccl")
    stamp(rank, "init_process_group ok")

    t = torch.ones(1024, device=f"cuda:{local_rank}")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    stamp(rank, f"all_reduce ok sum={t[0].item()}")

    dist.destroy_process_group()
    stamp(rank, "done")


if __name__ == "__main__":
    main()
