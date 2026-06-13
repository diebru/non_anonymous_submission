#!/usr/bin/env python3
"""Average CoT and code token length, split by McEval outcome, vs gamma. CPU-only.

Publication-ready figure: x = gamma, y = average token length, dashed lines with
markers. Seven series (averaged per gamma over the matching generations):

  CoT length (cot_token_count, code excluded):
    1. CoT — all              (every generation)
    2. CoT — pass             (CoTs that led to a PASS code)
    3. CoT — exec_fail        (CoTs that led to an exec_fail code)
    4. CoT — format_fail      (CoTs that led to a format_fail code)
  code length (code_token_count):
    5. code — pass
    6. code — exec_fail
    7. code — format_fail

Consistent styling: colour = outcome (all=black, pass=green, exec_fail=orange,
format_fail=red); marker = component (CoT = circle, code = square); all dashed.

Run AFTER scripts/score_generations.py (needs records/ with `outcome`).

Usage (server):
    python3 scripts/plot_cot_code_by_outcome.py --run-id sft01
    python3 scripts/plot_cot_code_by_outcome.py --run-id sft01 \
        --gammas 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
    python3 scripts/plot_cot_code_by_outcome.py --run-id sft01 --stat median
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS  # noqa: E402

# (label, field, outcome filter or None=all, colour, marker)
SERIES = [
    ("CoT — all",          "cot_token_count", None,          "black",      "o"),
    ("CoT — pass",         "cot_token_count", "pass",        "tab:green",  "o"),
    ("CoT — exec_fail",    "cot_token_count", "exec_fail",   "tab:orange", "o"),
    ("CoT — format_fail",  "cot_token_count", "format_fail", "tab:red",    "o"),
    ("code — pass",        "code_token_count", "pass",        "tab:green",  "s"),
    ("code — exec_fail",   "code_token_count", "exec_fail",   "tab:orange", "s"),
    ("code — format_fail", "code_token_count", "format_fail", "tab:red",    "s"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--stat", choices=("mean", "median"), default="mean")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = statistics.mean if args.stat == "mean" else statistics.median
    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split

    G = []
    vals = {lbl: [] for (lbl, *_2) in SERIES}
    ns = {lbl: [] for (lbl, *_2) in SERIES}
    for g in sorted(set(args.gammas), reverse=True):
        recs_dir = base / f"gamma{g:g}" / "records"
        if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
            continue
        recs = []
        for f in sorted(recs_dir.glob("*.jsonl")):
            recs += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        G.append(g)
        for (lbl, field, oc, _c, _m) in SERIES:
            sub = [r[field] for r in recs
                   if (oc is None or r.get("outcome") == oc) and isinstance(r.get(field), int)]
            vals[lbl].append(agg(sub) if sub else None)
            ns[lbl].append(len(sub))

    if not G:
        print(f"No records under {base} (run the sweep + score_generations first).")
        return 1

    # --- table ---
    print(f"avg token length ({args.stat}) by outcome | {args.model} {args.task}/{args.split}")
    print("gamma".rjust(6) + "".join(f"{lbl.replace(' ', ''):>20}" for (lbl, *_2) in SERIES))
    for i, g in enumerate(G):
        row = f"{g:>6g}"
        for (lbl, *_2) in SERIES:
            v = vals[lbl][i]
            row += f"{(f'{v:.0f}(n={ns[lbl][i]})' if v is not None else '-'):>20}"
        print(row)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(11.5, 7))
    for (lbl, _field, _oc, col, mk) in SERIES:
        xs = [g for g, v in zip(G, vals[lbl]) if v is not None]
        ys = [v for v in vals[lbl] if v is not None]
        ax.plot(xs, ys, linestyle="--", marker=mk, color=col, lw=1.8, ms=8,
                markerfacecolor=col, markeredgecolor="black", markeredgewidth=0.4, label=lbl)
    ax.set_xlabel(r"compression ratio  $\gamma$  (fraction of CoT retained)", fontsize=13)
    ax.set_ylabel(f"average token length  ({args.stat})", fontsize=13)
    ax.set_title(f"CoT and code length by outcome vs $\\gamma$ — {args.model}, {args.task}/{args.split}",
                 fontsize=12)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2, title="series  (○ = CoT, □ = code)", framealpha=0.95)
    fig.tight_layout()
    out = base / "cot_code_length_by_outcome.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
