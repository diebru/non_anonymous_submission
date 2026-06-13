#!/usr/bin/env python3
"""Build (or check) the split manifest (roadmap s6). CPU-only; runnable locally.

Writes the committed CSV at paths.manifest_path (default
<repo>/manifest/split_manifest.csv), then runs the distributional balance gate.

Usage:
    python3 scripts/build_manifest.py            # build + validate + write
    python3 scripts/build_manifest.py --check    # validate the existing file only
    python3 scripts/build_manifest.py --dry-run  # build + validate, do not write
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc import manifest as MAN  # noqa: E402
from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import SEED  # noqa: E402


def _print_summary(rows: list[dict[str, str]], paths) -> None:
    s = MAN.summarize(rows, paths)
    print(f"  total base problems : {s['total']}")
    print(f"  split               : {s['by_split']}")
    print(f"  languages           : {s['n_languages']}")
    print(f"  membership          : {s['by_membership']}")
    print(f"  difficulty_source   : {s['by_difficulty_source']}")
    print(f"  difficulty          : {s['by_difficulty']}")
    print("  row-level per task (train / test):")
    for task, d in s["task_rows"].items():
        print(f"     {task:11s}: {d['train_problems']} / {d['test_problems']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate existing manifest, no write")
    parser.add_argument("--dry-run", action="store_true", help="build + validate, do not write")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    paths = get_paths()
    out = paths.manifest_path

    if args.check:
        if not out.is_file():
            print(f"ERROR: manifest not found at {out}")
            return 1
        rows = MAN.read_manifest(out)
        print(f"Checking existing manifest: {out}")
    else:
        rows = MAN.build_manifest_rows(paths, seed=args.seed)
        print(f"Built manifest (seed={args.seed})")

    _print_summary(rows, paths)

    errors = MAN.validate_manifest(rows)
    print()
    if errors:
        print(f"RESULT: FAIL -- distributional gate: {len(errors)} issue(s)")
        for e in errors[:20]:
            print(f"  - {e}")
        return 1
    print("RESULT: PASS -- distributional balance gate holds")

    if not args.check and not args.dry_run:
        MAN.write_manifest(rows, out)
        print(f"\nWrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
