#!/usr/bin/env python3
"""Download a pinned file from Hugging Face and check its LFS SHA-256.

    python3 common/fetch.py --repo OWNER/NAME --revision COMMIT --dest DIR FILE
"""

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dest", type=lambda p: ROOT / p, required=True)
    parser.add_argument("file")
    args = parser.parse_args()

    api = f"https://huggingface.co/api/models/{args.repo}/paths-info/{args.revision}"
    body = urllib.parse.urlencode({"paths": args.file, "expand": "true"}).encode()
    with urllib.request.urlopen(api, data=body, timeout=60) as response:
        expected = json.load(response)[0]["lfs"]["oid"]

    target = args.dest / args.file
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{args.file}"
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out, 8 << 20)
        part.rename(target)

    digest = hashlib.sha256()
    with target.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(f"SHA-256 mismatch, delete and retry: {target}")
    print(f"Verified: {target}", flush=True)


if __name__ == "__main__":
    main()
