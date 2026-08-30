"""SGLang の venv 構築と環境ごとの補正。

  setup() : venv 作成 + sglang インストール (作成済みなら何もしない)
  env()   : PATH / CUDA_HOME 等の設定と lib64 symlink farm (冪等)
  tidy()  : 起動ログを汚す既知問題の後始末 (毎回実行、冪等)

SG_CUDA_FIXUPS=1 (素のノード用。既定 0):
  pip wheel の CUDA toolchain だけで flashinfer の JIT を通すための補正。
  /usr/local/cuda に toolkit 一式があるコンテナでは 0 のまま触らない。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _cuda_fixups() -> bool:
    return os.environ.get("SG_CUDA_FIXUPS", "0") == "1"


def _pip(python: Path, *args: str) -> None:
    subprocess.run([str(python), "-m", "pip", *args], check=True)


def setup(venv: Path) -> None:
    if (venv / "bin/sglang").is_file():
        return

    # ensurepip が無い環境がある (素の Ubuntu 24.04 は python3-venv 未導入、
    # pytorch コンテナも同様)。ある環境では標準の venv を使う。
    has_ensurepip = subprocess.run(
        [sys.executable, "-c", "import ensurepip"],
        capture_output=True, check=False,
    ).returncode == 0
    if has_ensurepip:
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--break-system-packages", "--no-cache-dir", "virtualenv"],
            check=True,
        )
        subprocess.run([sys.executable, "-m", "virtualenv", str(venv)], check=True)

    python = venv / "bin/python"
    _pip(python, "install", "--no-cache-dir", "--upgrade", "pip")
    _pip(python, "install", "--no-cache-dir",
         os.environ.get("SG_PACKAGE", "sglang==0.5.18"))

    if _cuda_fixups():
        # 素の依存解決では nvcc が 13.3.73 まで上がり、13.0.96 のままの
        # nvidia-cuda-runtime とズレて flashinfer の JIT が全滅する
        # ("CUDA compiler and CUDA toolkit headers are incompatible")。
        # cuda-toolkit 13.0.3.0 が要求する組み合わせへ戻す。
        _pip(python, "install", "--no-cache-dir",
             "nvidia-cuda-nvcc==13.0.88.*",
             "nvidia-cuda-crt==13.0.88.*",
             "nvidia-nvvm==13.0.88.*")


def env(venv: Path) -> Path:
    """PATH 等を整え、site-packages のパスを返す。"""
    pyver = subprocess.run(
        [str(venv / "bin/python"), "-c",
         'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")'],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    site = venv / "lib" / pyver / "site-packages"

    # ninja / nvcc は JIT から subprocess で呼ばれるので PATH に要る
    paths = [str(venv / "bin")]

    if _cuda_fixups():
        cu = site / "nvidia/cu13"
        paths.append(str(cu / "bin"))
        os.environ["CUDA_HOME"] = str(cu)
        os.environ["LD_LIBRARY_PATH"] = (
            f"{cu / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        )

        # JIT は -L$CUDA_HOME/lib64 -lcudart / -L$CUDA_HOME/lib64/stubs -lcuda で
        # リンクするが、pip wheel には lib64/ も libcudart.so も libcuda.so も無い
        lib = cu / "lib"
        if lib.is_dir():
            stubs = cu / "lib64/stubs"
            stubs.mkdir(parents=True, exist_ok=True)
            for so in lib.glob("*.so.*"):
                link = cu / "lib64" / (so.name.split(".so.")[0] + ".so")
                _force_symlink(Path("../lib") / so.name, link)
            for cand in [Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
                         Path("/usr/local/cuda/lib64/stubs/libcuda.so")]:
                if cand.exists():
                    _force_symlink(cand, stubs / "libcuda.so")
                    break

    os.environ["PATH"] = ":".join(paths + [os.environ.get("PATH", "")])
    return site


def _force_symlink(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def tidy(venv: Path, site: Path) -> None:
    python = venv / "bin/python"

    # torchcodec は sglang の依存だが、FFmpeg (libavutil.so.56-60) が無い環境では
    # ロードに必ず失敗し、プロセスごとに 100 行超の traceback を stderr へ吐く
    # (tokenizer + TP4 の 5 プロセス x 2 モジュールで約 1600 行/起動)。
    # audio/video 入力は使わず、sglang 側も torchcodec が無ければ decord /
    # soundfile+torchaudio に落ちる実装 (srt/utils/video_decoder.py,
    # srt/utils/common.py) なので、FFmpeg が無いなら消す。
    if (site / "torchcodec").is_dir() and not _has_system_ffmpeg():
        _pip(python, "uninstall", "-y", "-q", "torchcodec")
        print("removed torchcodec (no system FFmpeg; sglang falls back to decord/soundfile)",
              flush=True)

    patch_sarashina(site)

    subprocess.run(
        [str(python), "-c",
         'import sglang, torch; print(f"sglang {sglang.__version__} / '
         'torch {torch.__version__} (cuda {torch.version.cuda})")'],
        check=True,
    )
    if shutil.which("nvcc"):
        out = subprocess.run(["nvcc", "--version"], capture_output=True,
                             text=True, check=False).stdout.strip().splitlines()
        if out:
            print(out[-1], flush=True)


def _has_system_ffmpeg() -> bool:
    out = subprocess.run(["ldconfig", "-p"], capture_output=True,
                         text=True, check=False).stdout
    return "libavutil.so" in out


# sglang 0.5.18 のバグ: models/sarashina2_vision.py が MultimodalDataItem を
# mm_utils から import しているが、実体は schedule_batch にある
# (mm_utils が再エクスポートしているのは MultimodalInputs だけ)。
# 起動のたびに ImportError の警告が 5 プロセス分出る (sglang 自身は握りつぶす
# ので実害は無い)。他の VL モデル (qwen2_vl 等) は schedule_batch から
# import しているので、そちらに揃える。冪等で、upstream が直せば no-op。
_BAD = "from sglang.srt.managers.mm_utils import (\n    MultimodalDataItem,\n"
_GOOD = "from sglang.srt.managers.mm_utils import (\n"
_ANCHOR = "from sglang.srt.model_executor.forward_batch_info import ForwardBatch"
_ADDED = "from sglang.srt.managers.schedule_batch import MultimodalDataItem\n"


def patch_sarashina(site: Path) -> None:
    target = site / "sglang/srt/models/sarashina2_vision.py"
    if not target.is_file():
        return
    src = target.read_text(encoding="utf-8")
    if _BAD not in src or _ANCHOR not in src:
        # upstream が直したか、構造が変わった。触らない。
        return
    src = src.replace(_BAD, _GOOD, 1).replace(_ANCHOR, _ADDED + _ANCHOR, 1)
    target.write_text(src, encoding="utf-8")
    print("patched sarashina2_vision.py (MultimodalDataItem <- schedule_batch)",
          flush=True)
