#!/usr/bin/env python3
"""Create the bulk-artifact directories from the resolved config (idempotent).

CPU-only; safe to run locally or on the server. The directories themselves are
gitignored (generations/, compressed/, weights/, eval_dumps/); this script just
materializes them under the configured data_root so downstream (server) scripts
have somewhere to write.

Usage:
    python scripts/bootstrap_dirs.py
"""
from __future__ import annotations

import pathlib
import sys

# Allow running from source without `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import ensure_dirs, get_paths  # noqa: E402


def main() -> int:
    paths = get_paths()
    created = ensure_dirs(paths)
    print(f"data_root: {paths.data_root}")
    for directory in paths.artifact_dirs:
        flag = "created" if directory in created else "exists"
        print(f"  [{flag}] {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
