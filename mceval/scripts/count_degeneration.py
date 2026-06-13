#!/usr/bin/env python3
"""Count neural-text-degeneration (repetition-loop) generations per gamma. CPU-only.

Truncation (hit max_tokens) is a proxy for degeneration; this measures it directly
via the distinct-4-gram ratio of raw_full_output (a repetition loop -> almost all
4-grams identical -> ratio near 0). A generation is flagged DEGENERATE if it has
enough words and its distinct-4-gram ratio < --thresh. Reports, per gamma:
truncated, repetition-degenerate, and the union (either signal).

Usage (server):
    python3 scripts/count_degeneration.py --run-id sft01            # all 12 gammas
    python3 scripts/count_degeneration.py --run-id sft01 --gammas 0.1 0.5 1.0 --examples 2
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS  # noqa: E402


def distinct_ngram_ratio(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n + 1:
        return 1.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def _trunc(r):
    return bool(r.get("extraction_status", {}).get("truncated"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--thresh", type=float, default=0.30,
                    help="distinct-4-gram ratio below this = repetition-degenerate")
    ap.add_argument("--min-words", type=int, default=50,
                    help="only judge outputs with at least this many words")
    ap.add_argument("--examples", type=int, default=0, help="print N degenerate problem_ids/gamma")
    args = ap.parse_args()

    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split
    print(f"degeneration count | {args.model} {args.task}/{args.split} run={args.run_id} "
          f"(distinct-4gram < {args.thresh})")
    print(f"{'gamma':>6} {'n':>5} {'truncated':>10} {'rep_degen':>10} {'either':>8} {'either%':>8}")
    for g in sorted(set(args.gammas), reverse=True):
        recs_dir = base / f"gamma{g:g}" / "records"
        if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
            recs_dir = base / f"gamma{g:g}" / "trajectories"
        if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
            continue
        rows = []
        for f in sorted(recs_dir.glob("*.jsonl")):
            rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        n = len(rows)
        trunc = sum(1 for r in rows if _trunc(r))
        degen, either, ex = 0, 0, []
        for r in rows:
            raw = r.get("raw_full_output", "") or ""
            is_rep = (len(raw.split()) >= args.min_words
                      and distinct_ngram_ratio(raw, 4) < args.thresh)
            if is_rep:
                degen += 1
                if len(ex) < args.examples:
                    ex.append(r.get("problem_id"))
            if is_rep or _trunc(r):
                either += 1
        pct = 100 * either / n if n else 0
        print(f"{g:>6g} {n:>5} {trunc:>10} {degen:>10} {either:>8} {pct:>7.1f}%"
              + (f"   e.g. {ex}" if ex else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
