#!/usr/bin/env python3
"""Build llama-server from a pinned llama.cpp commit into <dest>/build/bin.

The CUDA runtime and cuBLAS are copied next to it, so the result runs on nodes
with only the NVIDIA driver and can be rsynced to the other site.

    python3 src/build.py --repo URL --ref COMMIT --arch 86 --dest DIR
"""

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--dest", type=lambda p: ROOT / p, required=True)
    args = parser.parse_args()

    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise SystemExit("nvcc is not on PATH")
    source = args.dest / "src"
    bin_dir = args.dest / "build/bin"
    source.mkdir(parents=True, exist_ok=True)
    run(["git", "init", "-q"], source)
    run(["git", "fetch", "-q", "--depth", "1", args.repo, args.ref], source)
    run(["git", "checkout", "-q", "--force", "FETCH_HEAD"], source)
    # NATIVE=OFF: built on a login node, run on compute nodes.
    # FA_ALL_QUANTS: flash attention with a q8_0 KV cache needs it.
    run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON",
         f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}", "-DGGML_NATIVE=OFF",
         "-DGGML_CUDA_FA_ALL_QUANTS=ON", "-DLLAMA_BUILD_TESTS=OFF"], source)
    run(["cmake", "--build", "build", "-j", "16", "--target", "llama-server"], source)

    shutil.rmtree(bin_dir, ignore_errors=True)
    shutil.copytree(source / "build/bin", bin_dir, symlinks=True)
    cuda_lib = Path(nvcc).resolve().parents[1] / "lib64"
    for name in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
        shutil.copy2(cuda_lib / name, bin_dir / name)
    shutil.rmtree(source)
    print(f"Built {args.ref} into {bin_dir}", flush=True)


if __name__ == "__main__":
    main()
