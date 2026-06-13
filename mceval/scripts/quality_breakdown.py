#!/usr/bin/env python3
"""Per-gamma generation-quality breakdown. CPU-only (matplotlib for the plot).

Every generation is classified on two independent axes:
  * CoT good   = coherent reasoning -- NOT degenerate (not truncated at max_tokens and
                 not a repetition loop, distinct-4-gram ratio >= --thresh).
  * code good  = a valid, executable code block was produced (outcome in pass/exec_fail;
                 format_fail = no usable code).
and the report gives, per gamma, the counts/percentages of:
  * good CoT
  * good code
  * BAD both  = CoT degenerate AND no valid code (the runaways)
  * union good = good CoT OR good code (= total - bad-both)
plus, for reference, `correct` = outcome==pass (the accuracy numerator).

Run AFTER scripts/score_generations.py (needs records/ with `outcome`).

Usage (server):
    python3 scripts/quality_breakdown.py --run-id sft01
    python3 scripts/quality_breakdown.py --run-id sft01 --gammas 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS  # noqa: E402

VALID_CODE = ("pass", "exec_fail")   # a usable code block was extracted and executed


def distinct4(text: str) -> float:
    w = text.split()
    if len(w) < 5:
        return 1.0
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(g)) / len(g)


def classify(r: dict, thresh: float, min_words: int) -> tuple[bool, bool]:
    """(cot_good, code_good) for one record."""
    raw = r.get("raw_full_output", "") or ""
    truncated = bool(r.get("extraction_status", {}).get("truncated"))
    repeating = len(raw.split()) >= min_words and distinct4(raw) < thresh
    cot_good = not (truncated or repeating)
    code_good = r.get("outcome") in VALID_CODE
    return cot_good, code_good


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--thresh", type=float, default=0.30, help="distinct-4gram below = degenerate CoT")
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split
    G, rowsout = [], []
    for g in sorted(set(args.gammas), reverse=True):
        recs_dir = base / f"gamma{g:g}" / "records"
        if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
            continue
        recs = []
        for f in sorted(recs_dir.glob("*.jsonl")):
            recs += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        n = len(recs)
        cot_good = code_good = both_bad = correct = 0
        for r in recs:
            cg, kg = classify(r, args.thresh, args.min_words)
            cot_good += cg
            code_good += kg
            both_bad += (not cg and not kg)
            correct += (r.get("outcome") == "pass")
        union_good = n - both_bad
        G.append(g)
        rowsout.append(dict(n=n, cot_good=cot_good, code_good=code_good,
                            both_bad=both_bad, union_good=union_good, correct=correct))

    if not G:
        print(f"No records under {base} (run the sweep + score_generations first).")
        return 1

    def pc(x, n):
        return f"{100*x/n:4.1f}%" if n else "  -  "
    print(f"Generation-quality breakdown | {args.model} {args.task}/{args.split} run={args.run_id}")
    print(f"{'gamma':>6} {'n':>4} {'good_CoT':>13} {'good_code':>13} {'BAD_both':>13} "
          f"{'union_good':>13} {'correct':>13}")
    for g, d in zip(G, rowsout):
        n = d["n"]
        print(f"{g:>6g} {n:>4} {d['cot_good']:>5} {pc(d['cot_good'],n):>7} "
              f"{d['code_good']:>5} {pc(d['code_good'],n):>7} "
              f"{d['both_bad']:>5} {pc(d['both_bad'],n):>7} "
              f"{d['union_good']:>5} {pc(d['union_good'],n):>7} "
              f"{d['correct']:>5} {pc(d['correct'],n):>7}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6.5))
        series = [("good_CoT", "cot_good", "tab:blue", "o-"),
                  ("good_code", "code_good", "tab:green", "s-"),
                  ("union_good (CoT or code)", "union_good", "tab:gray", "^--"),
                  ("BAD both (degenerate)", "both_bad", "tab:red", "x-")]
        for label, key, col, fmt in series:
            ys = [100 * d[key] / d["n"] for d in rowsout]
            ax.plot(G, ys, fmt, color=col, lw=2, ms=7, label=label)
        ax.set_xlabel("γ  (compression ratio = fraction of CoT retained)", fontsize=12)
        ax.set_ylabel("% of generations", fontsize=12)
        ax.set_title(f"Generation-quality breakdown vs γ — {args.model} {args.task}/{args.split}",
                     fontsize=11)
        ax.invert_xaxis(); ax.grid(True, alpha=0.3); ax.legend(fontsize=10, loc="center left")
        fig.tight_layout()
        out = base / "quality_breakdown_vs_gamma.png"
        fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
