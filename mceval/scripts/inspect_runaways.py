#!/usr/bin/env python3
"""Inspect the raw generations behind the energy numbers. CPU-only.

For one (model, run, task, split, gamma), reads the per-language records (or
trajectories) and prints:
  * the aggregate that build_curves used (outcome + parser_branch counts, mean
    cot/code/output tokens, truncated count) -- a cross-check on the curve table;
  * a few RUNAWAY examples (truncated -> hit max_tokens) showing the tail of
    raw_full_output so you can SEE the rambling, plus their cot/code/output token
    split and parser branch;
  * a few WELL-FORMED examples for contrast.

This is what confirms the low-gamma energy rise is real model behavior (the model
decodes ~2048 tokens of ramble) and not a counting/parse bug.

Usage (server):
    python3 scripts/inspect_runaways.py --run-id sft01 --gamma 0.1 --n 3
    python3 scripts/inspect_runaways.py --run-id sft01 --gamma 1.0 --n 3   # baseline
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import MODEL_IDS  # noqa: E402


def _out_tokens(r):
    t = (r.get("_provenance") or {}).get("timing", {}).get("n_output_tokens")
    return sum(t) if isinstance(t, list) else (t or 0)


def _trunc(r):
    return bool(r.get("extraction_status", {}).get("truncated"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=3, help="examples of each kind to show")
    ap.add_argument("--chars", type=int, default=600, help="chars of raw output tail to show")
    args = ap.parse_args()

    paths = get_paths()
    gdir = paths.generations_dir / args.model / args.run_id / args.task / args.split / f"gamma{args.gamma:g}"
    src = gdir / "records"
    if not (src.is_dir() and any(src.glob("*.jsonl"))):
        src = gdir / "trajectories"
    if not (src.is_dir() and any(src.glob("*.jsonl"))):
        print(f"No records/ or trajectories/ under {gdir}")
        return 1

    rows = []
    for f in sorted(src.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]

    cot = [r.get("cot_token_count", 0) for r in rows]
    code = [r.get("code_token_count", 0) for r in rows]
    out = [_out_tokens(r) for r in rows]
    trunc = [r for r in rows if _trunc(r)]
    at_cap = sum(1 for o in out if o >= 2048)

    print("=" * 74)
    print(f"INSPECT {args.model} {args.task}/{args.split} gamma={args.gamma:g}  (source: {src.name}/)")
    print(f"dir: {gdir}")
    print("=" * 74)
    print(f"n rows                : {len(rows)}")
    print(f"outcome counts        : {dict(Counter(r.get('outcome', 'n/a') for r in rows))}")
    print(f"parser_branch counts  : {dict(Counter(r.get('extraction_status', {}).get('parser_branch') for r in rows))}")
    print(f"truncated (hit cap)   : {len(trunc)}   | output tokens >=2048: {at_cap}")
    print(f"cot_token_count       : median {statistics.median(cot):.0f}  mean {statistics.mean(cot):.1f}  max {max(cot)}")
    print(f"code_token_count      : median {statistics.median(code):.0f}  mean {statistics.mean(code):.1f}  max {max(code)}")
    print(f"output tokens (decode): median {statistics.median(out):.0f}  mean {statistics.mean(out):.1f}  max {max(out)}  sum {sum(out)}")

    def _show(r, tag):
        es = r.get("extraction_status", {})
        raw = r.get("raw_full_output", "") or ""
        print("\n" + "-" * 74)
        print(f"[{tag}] {r.get('problem_id')} ({r.get('lang')})  outcome={r.get('outcome')} "
              f"branch={es.get('parser_branch')} truncated={es.get('truncated')} "
              f"fence={es.get('fence_found')}")
        print(f"   cot_token_count={r.get('cot_token_count')}  code_token_count={r.get('code_token_count')}  "
              f"output_tokens={_out_tokens(r)}  raw_chars={len(raw)}")
        print(f"   ...raw_full_output TAIL ({args.chars} chars)...")
        print("   " + raw[-args.chars:].replace("\n", "\n   "))

    print("\n" + "#" * 74 + "\n#  RUNAWAY examples (truncated -> hit max_tokens)\n" + "#" * 74)
    for r in trunc[: args.n]:
        _show(r, "RUNAWAY")
    if not trunc:
        print("  (none truncated at this gamma)")

    wf = [r for r in rows if not _trunc(r) and r.get("outcome") in ("pass", "exec_fail")]
    print("\n" + "#" * 74 + "\n#  WELL-FORMED examples (for contrast)\n" + "#" * 74)
    for r in wf[: args.n]:
        _show(r, "WELL-FORMED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
