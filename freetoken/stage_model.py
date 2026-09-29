#!/usr/bin/env python3
"""Stage the pinned HF checkpoint in node-local scratch, outside the home quota."""

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4"
REVISION = "be794b990578ef3031eccf9f28e675a289a09ee9"
DEST = Path(os.environ.get("FT_MODEL_PATH", "/tmp/cit-gpu-qwen3.8-flash-next"))


def main():
    if shutil.disk_usage(DEST.parent).free < 145 * 1024**3:
        raise RuntimeError(f"Need at least 145 GiB free at {DEST.parent}")
    os.environ.setdefault("HF_HOME", str(DEST.parent / "cit-gpu-hf-cache"))
    print(f"Staging {REPO}@{REVISION} to {DEST}", flush=True)
    snapshot_download(repo_id=REPO, revision=REVISION, local_dir=str(DEST), max_workers=16)
    if not (DEST / "model.safetensors.index.json").is_file():
        raise RuntimeError("Checkpoint index missing after download")
    print(f"Staged checkpoint at {DEST}", flush=True)


if __name__ == "__main__":
    main()
