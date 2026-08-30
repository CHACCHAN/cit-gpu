"""モデル取得。

config.json と追加 marker ファイルが揃っていなければ Hugging Face から取得する。
snapshot_download(local_dir=...) は blob を HF_HOME 側に置かない。
models/ と .hf-cache/ の二重保存を避けるため (quota 80GB)。
huggingface_hub は venv 側にある (土台イメージには入っていない) ので
venv の python を subprocess で使う。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

_DOWNLOAD_SNIPPET = """\
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
"""


def fetch(venv: Path, model_path: Path, markers: List[str]) -> None:
    repo = os.environ["SG_MODEL_REPO"]

    if not all((model_path / m).is_file() for m in ["config.json", *markers]):
        print(f"Downloading model: {repo} -> {model_path}", flush=True)
        model_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(venv / "bin/python"), "-c", _DOWNLOAD_SNIPPET, repo, str(model_path)],
            check=True,
        )

    if not (model_path / "config.json").is_file():
        sys.exit(f"ERROR: {model_path}/config.json not found")

    subprocess.run(["du", "-sh", str(model_path), str(venv)], check=False)
