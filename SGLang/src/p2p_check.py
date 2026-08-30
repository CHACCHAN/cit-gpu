#!/usr/bin/env python3
"""GPU 間 P2P (CUDA IPC / PCIe peer copy) が実際に通るかを単一プロセスで確認する。

NCCL を介さないので、ハングが NCCL 側か CUDA ドライバ側かを切り分けられる。
情報システム担当への報告用の最小再現でもある。

  srun --partition=research --nodes=1 --ntasks=1 --cpus-per-task=8 \
       --gpus-per-task=4 --mem=64G --time=00:10:00 \
       .venv/bin/python src/p2p_check.py
"""

import sys
import threading

import torch


def peer_copy(src: int, dst: int, result: dict) -> None:
    a = torch.ones(1 << 20, device=f"cuda:{src}")
    b = a.to(f"cuda:{dst}", non_blocking=False)
    torch.cuda.synchronize(dst)
    result["sum"] = b.sum().item()


def main() -> int:
    n = torch.cuda.device_count()
    print(f"visible GPUs: {n}", flush=True)
    for i in range(n):
        # 故障個体を名指しできるよう UUID / PCI を必ず出す (報告用)
        props = torch.cuda.get_device_properties(i)
        uuid = getattr(props, "uuid", "?")
        bus = f"{getattr(props, 'pci_domain_id', 0):04x}:{getattr(props, 'pci_bus_id', 0):02x}:{getattr(props, 'pci_device_id', 0):02x}.0"
        print(f"  cuda:{i} {props.name} uuid=GPU-{uuid} pci={bus}", flush=True)

    print("\ncanAccessPeer matrix (src -> dst):", flush=True)
    for i in range(n):
        row = [
            "-" if i == j else ("Y" if torch.cuda.can_device_access_peer(i, j) else "n")
            for j in range(n)
        ]
        print(f"  {i}: {' '.join(row)}", flush=True)

    print("\nactual peer copy (10s timeout each):", flush=True)
    bad = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            result: dict = {}
            t = threading.Thread(target=peer_copy, args=(i, j, result), daemon=True)
            t.start()
            t.join(10.0)
            if t.is_alive():
                print(f"  {i} -> {j}: HANG (>10s)", flush=True)
                print("\n  ハングしたスレッドは回収できないのでここで打ち切る。", flush=True)
                return 1
            ok = result.get("sum") == float(1 << 20)
            print(f"  {i} -> {j}: {'ok' if ok else 'WRONG DATA ' + str(result)}", flush=True)
            bad += 0 if ok else 1

    print(f"\n{'ALL OK' if bad == 0 else str(bad) + ' FAILURES'}", flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
