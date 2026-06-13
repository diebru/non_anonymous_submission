#!/usr/bin/env python3
"""Print the resolved project configuration, paths, and frozen constants.

CPU-only sanity check; safe to run locally or on the server. Does not create or
write anything.

Usage:
    python scripts/show_config.py
"""
from __future__ import annotations

import pathlib
import sys

# Allow running from source without `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc import constants as C  # noqa: E402
from tsmc.config import get_paths, load_config  # noqa: E402


def main() -> int:
    cfg = load_config()
    paths = get_paths(cfg)
    print(f"config file  : {cfg.get('_config_path')}")
    print(f"repo_root    : {paths.repo_root}")
    print(f"data_root    : {paths.data_root}")
    print(f"mceval_dir   : {paths.mceval_dir}")
    print(f"mceval_data  : {paths.mceval_data_dir}")
    print(f"manifest     : {paths.manifest_path}")
    print("artifact dirs:")
    for directory in paths.artifact_dirs:
        state = "present" if directory.exists() else "absent"
        print(f"  - {directory}  ({state})")
    print(f"sentinel     : {C.SENTINEL}")
    print(f"gamma grid   : {list(C.GAMMA_GRID)}  (n={len(C.GAMMA_GRID)})")
    print(f"seed         : {C.SEED}   num_runs: {C.NUM_RUNS}")
    print(f"models       : {list(C.MODEL_IDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
