#!/usr/bin/env python3
"""Download from the Hub via the huggingface_hub library (avoids the `hf`/`huggingface-cli`
launchers, which can be broken/removed on some boxes).

Usage:
  python _hf_download.py <repo_id> <local_dir> [--include PATTERN] [--revision REV]
"""
import argparse, os
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("repo_id")
ap.add_argument("local_dir")
ap.add_argument("--include", default=None, help="allow_patterns glob, e.g. 'gsm8k/*'")
ap.add_argument("--revision", default=None)
a = ap.parse_args()

path = snapshot_download(
    repo_id=a.repo_id,
    local_dir=a.local_dir,
    allow_patterns=[a.include] if a.include else None,
    revision=a.revision,
    token=os.environ.get("HF_TOKEN") or None,
)
print(f"downloaded {a.repo_id} ({a.include or 'all'}) -> {path}")
