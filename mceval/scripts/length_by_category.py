#!/usr/bin/env python3
"""Average answer length per inference, by outcome category, vs compression ratio.

One plot: y = average total output length per generation (tokens = CoT + code +
everything decoded), x = gamma. Five lines, each the average over a subset defined
by the McEval `outcome`:
  * good result = solved correctly (pass)
  * bad result  = not solved (exec_fail + format_fail)
  * good code   = valid executable code block (pass + exec_fail)
  * bad code    = no usable code (format_fail) -- where degenerate runaways land
  * all         = every generation (the full answer, no good/bad distinction)

Length = decode tokens from _provenance.timing.n_output_tokens (the real generated
length, including any runaway ramble). Run AFTER scripts/score_generations.py.

Usage (server):
    python3 scripts/length_by_category.py --run-id sft01
    python3 scripts/length_by_category.py --run-id sft01 --gammas 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
    python3 scripts/length_by_category.py --run-id sft01 --stat median
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

# subset name -> set of outcomes it includes ("*" = all)
CATEGORIES = {
    "good result (pass)": {"pass"},
    "bad result (exec+format fail)": {"exec_fail", "format_fail"},
    "good code (pass+exec_fail)": {"pass", "exec_fail"},
    "bad code (format_fail)": {"format_fail"},
    "all (full answer)": "*",
}
STYLE = {
    "good result (pass)": ("tab:green", "o-"),
    "bad result (exec+format fail)": ("tab:red", "v-"),
    "good code (pass+exec_fail)": ("tab:blue", "s-"),
    "bad code (format_fail)": ("tab:orange", "x-"),
    "all (full answer)": ("black", "D--"),
}


def _out_tokens(r: dict) -> int:
    t = (r.get("_provenance") or {}).get("timing", {}).get("n_output_tokens")
    return sum(t) if isinstance(t, list) else (int(t) if isinstance(t, int) else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--stat", choices=("mean", "median"), default="mean")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    agg = statistics.mean if args.stat == "mean" else statistics.median
    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split

    G = []
    series = {name: [] for name in CATEGORIES}      # name -> list of avg length per gamma
    counts = {name: [] for name in CATEGORIES}
    for g in sorted(set(args.gammas), reverse=True):
        recs_dir = base / f"gamma{g:g}" / "records"
        if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
            continue
        recs = []
        for f in sorted(recs_dir.glob("*.jsonl")):
            recs += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        G.append(g)
        for name, outs in CATEGORIES.items():
            sub = [_out_tokens(r) for r in recs if outs == "*" or r.get("outcome") in outs]
            series[name].append(agg(sub) if sub else None)
            counts[name].append(len(sub))

    if not G:
        print(f"No records under {base} (run the sweep + score_generations first).")
        return 1

    # table
    names = list(CATEGORIES)
    print(f"avg answer length per inference ({args.stat} output tokens) | "
          f"{args.model} {args.task}/{args.split}")
    hdr = "gamma".rjust(6) + "".join(f"{n.split('(')[0].strip()[:14]:>16}" for n in names)
    print(hdr)
    for i, g in enumerate(G):
        line = f"{g:>6g}"
        for n in names:
            v = series[n][i]
            line += f"{(f'{v:.0f} (n={counts[n][i]})' if v is not None else '-'):>16}"
        print(line)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for name in names:
            xs = [g for g, v in zip(G, series[name]) if v is not None]
            ys = [v for v in series[name] if v is not None]
            col, fmt = STYLE[name]
            lw = 3 if name.startswith("all") else 2
            ax.plot(xs, ys, fmt, color=col, lw=lw, ms=7, label=name)
        ax.set_xlabel("γ  (compression ratio = fraction of CoT retained)", fontsize=12)
        ax.set_ylabel(f"avg answer length per inference ({args.stat} output tokens)", fontsize=12)
        ax.set_title(f"Answer length by outcome category vs γ — {args.model} {args.task}/{args.split}",
                     fontsize=11)
        ax.invert_xaxis(); ax.grid(True, alpha=0.3); ax.legend(fontsize=10, loc="upper center")
        fig.tight_layout()
        out = base / "length_by_category_vs_gamma.png"
        fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
