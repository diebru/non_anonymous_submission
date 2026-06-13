#!/usr/bin/env python3
"""Per-gamma CoT vs code token split in the answer. CPU-only.

For each gamma, reports the average CoT and code token counts and their percentage
of the content (CoT + code). Computed over WELL-FORMED generations (outcome in
pass/exec_fail) where the sentinel splits CoT from code cleanly -- the degenerate
runaways (no sentinel -> whole output counted as "CoT", code=0) are excluded, since
including them fakes a ~100% "CoT" share at low gamma. The ALL-generations split is
also printed so you can see that artifact.

Usage (server):
    python3 scripts/cot_code_split.py --run-id sft01
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS  # noqa: E402

WELL_FORMED = ("pass", "exec_fail")  # sentinel emitted -> cot/code split is real


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    args = ap.parse_args()

    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split
    print(f"CoT vs code split | {args.model} {args.task}/{args.split} run={args.run_id}")
    print(f"  (well-formed = outcome in {WELL_FORMED}; %% of CoT+code tokens)")
    print(f"{'gamma':>6} {'n_wf':>5} {'avg_cot':>8} {'avg_code':>9} {'%CoT':>6} {'%code':>6} "
          f"| {'%CoT(all)':>9}")
    for g in sorted(set(args.gammas), reverse=True):
        recs_dir = base / f"gamma{g:g}" / "records"
        if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
            continue
        rows = []
        for f in sorted(recs_dir.glob("*.jsonl")):
            rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]

        wf = [r for r in rows if r.get("outcome") in WELL_FORMED]
        cot_wf = sum(r.get("cot_token_count", 0) for r in wf)
        code_wf = sum(r.get("code_token_count", 0) for r in wf)
        denom = cot_wf + code_wf
        pct_cot = 100 * cot_wf / denom if denom else 0
        pct_code = 100 * code_wf / denom if denom else 0
        avg_cot = cot_wf / len(wf) if wf else 0
        avg_code = code_wf / len(wf) if wf else 0

        # all-generations split (includes the runaway artifact) for contrast
        cot_all = sum(r.get("cot_token_count", 0) for r in rows)
        code_all = sum(r.get("code_token_count", 0) for r in rows)
        pct_cot_all = 100 * cot_all / (cot_all + code_all) if (cot_all + code_all) else 0

        print(f"{g:>6g} {len(wf):>5} {avg_cot:>8.1f} {avg_code:>9.1f} "
              f"{pct_cot:>5.1f}% {pct_code:>5.1f}% | {pct_cot_all:>8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
