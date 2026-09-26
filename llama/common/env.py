"""Load llama/.env (KEY=VALUE per line, # comments) into os.environ."""

import os
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / ".env"


def load_env():
    if ENV.stat().st_mode & 0o077:
        raise RuntimeError(f"{ENV} must be readable only by you (chmod 600)")
    for line in ENV.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and not name.strip().startswith("#"):
            os.environ[name.strip()] = value.strip()
    if not os.environ.get("LLAMA_API_KEY"):
        raise RuntimeError(f"LLAMA_API_KEY is not set in {ENV}")
    return os.environ
