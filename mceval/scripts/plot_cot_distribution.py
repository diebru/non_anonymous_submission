#!/usr/bin/env python3
"""Full distribution of CoT (or code / total) length vs compression ratio. CPU-only.

For each gamma, over WELL-FORMED generations (outcome pass/exec_fail -- the sentinel
splits CoT from code; degenerate runaways excluded so their 2048-token rambles don't
pollute the CoT count), it shows the whole length distribution, not just the mean:
  * median line, mean line,
  * 25-75th percentile band (IQR), 10-90th percentile band,
  * optional min..max whiskers.

Default metric is CoT (reasoning, code excluded); --metric code or total available.

Run AFTER scripts/score_generations.py (needs records/ with `outcome`).

Usage (server):
    python3 scripts/plot_cot_distribution.py --run-id sft01
    python3 scripts/plot_cot_distribution.py --run-id sft01 \
        --gammas 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1     # 0.1-step grid
    python3 scripts/plot_cot_distribution.py --run-id sft01 --metric code
    python3 scripts/plot_cot_distribution.py --run-id sft01 --box   # box-plots instead of bands
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

WELL_FORMED = ("pass", "exec_fail")


def _metric(r: dict, which: str) -> int | None:
    cot = r.get("cot_token_count")
    code = r.get("code_token_count")
    if which == "cot":
        return cot if isinstance(cot, int) else None
    if which == "code":
        return code if isinstance(code, int) else None
    if isinstance(cot, int) and isinstance(code, int):
        return cot + code
    return None


def _pct(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[i])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--metric", choices=("cot", "code", "total"), default="cot")
    ap.add_argument("--box", action="store_true", help="box-plots instead of percentile bands")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split

    G, VALS = [], []
    for g in sorted(set(args.gammas)):  # ascending; x-axis inverted at the end
        recs_dir = base / f"gamma{g:g}" / "records"
        if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
            recs_dir = base / f"gamma{g:g}" / "trajectories"
        if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
            continue
        vals = []
        for f in sorted(recs_dir.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("outcome") in WELL_FORMED:
                    v = _metric(r, args.metric)
                    if v is not None:
                        vals.append(v)
        if vals:
            G.append(g); VALS.append(sorted(vals))

    if not G:
        print(f"No records under {base} (run the sweep + score_generations first).")
        return 1

    mean = [statistics.mean(v) for v in VALS]
    p50 = [_pct(v, 0.50) for v in VALS]
    p25 = [_pct(v, 0.25) for v in VALS]
    p75 = [_pct(v, 0.75) for v in VALS]
    p10 = [_pct(v, 0.10) for v in VALS]
    p90 = [_pct(v, 0.90) for v in VALS]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    label = {"cot": "CoT (reasoning)", "code": "code", "total": "CoT + code"}[args.metric]
    if args.box:
        ax.boxplot(VALS, positions=G, widths=0.025, showmeans=True, manage_ticks=False,
                   flierprops=dict(marker=".", ms=3, alpha=0.3))
    else:
        ax.fill_between(G, p10, p90, color="tab:blue", alpha=0.12, label="10–90th pct")
        ax.fill_between(G, p25, p75, color="tab:blue", alpha=0.28, label="25–75th pct (IQR)")
        ax.plot(G, p50, "o-", color="tab:blue", lw=2.5, ms=7, label="median")
        ax.plot(G, mean, "D--", color="tab:red", lw=2, ms=6, label="mean")
        for g, y in zip(G, mean):
            ax.annotate(f"{y:.0f}", (g, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color="tab:red")

    ax.set_xlabel("γ  (compression ratio = fraction of CoT retained)", fontsize=12)
    ax.set_ylabel(f"{label} length (tokens), well-formed generations", fontsize=12)
    ax.set_title(f"{label} length distribution vs γ — {args.model} {args.task}/{args.split}",
                 fontsize=11)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper center")
    fig.tight_layout()
    out = base / f"{args.metric}_distribution_vs_gamma.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"{'gamma':>6} {'n':>5} {'mean':>7} {'p10':>5} {'p25':>5} {'p50':>5} {'p75':>5} {'p90':>5}")
    for i, g in enumerate(G):
        print(f"{g:>6g} {len(VALS[i]):>5} {mean[i]:>7.1f} {p10[i]:>5.0f} {p25[i]:>5.0f} "
              f"{p50[i]:>5.0f} {p75[i]:>5.0f} {p90[i]:>5.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
