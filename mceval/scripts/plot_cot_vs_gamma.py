#!/usr/bin/env python3
"""Plot average CoT and average code token length vs gamma. CPU-only (needs matplotlib).

Reads the scored long-format records and, per gamma, computes the average CoT and
average code token counts over WELL-FORMED generations (outcome pass/exec_fail, where
the sentinel cleanly splits CoT from code; the degenerate runaways -- no sentinel, so
their whole output is mis-counted as CoT -- are excluded). Plots both vs gamma, plus
their sum (the answer content length).

Shows the structural result: CoT shrinks monotonically under compression while code
stays roughly flat, so the full answer barely shortens -- the reason CoT compression
moves total tokens / energy so little on code generation.

Run AFTER scripts/score_generations.py (needs records/ with the `outcome` field).

Usage (server):
    python3 scripts/plot_cot_vs_gamma.py --run-id sft01
    # 0.1-step grid only (drop 0.95/0.85):
    python3 scripts/plot_cot_vs_gamma.py --run-id sft01 \
        --gammas 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
    # use medians instead of means:
    python3 scripts/plot_cot_vs_gamma.py --run-id sft01 --stat median
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

WELL_FORMED = ("pass", "exec_fail")  # sentinel emitted -> CoT/code split is real


def _read_records(recs_dir: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(recs_dir.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--stat", choices=("mean", "median"), default="mean",
                    help="average type over well-formed generations (default: mean)")
    ap.add_argument("--no-sum", action="store_true", help="hide the CoT+code total line")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = statistics.mean if args.stat == "mean" else statistics.median
    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split

    G, COT, CODE, N = [], [], [], []
    for g in sorted(set(args.gammas), reverse=True):
        recs_dir = base / f"gamma{g:g}" / "records"
        if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
            recs_dir = base / f"gamma{g:g}" / "trajectories"
        if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
            continue
        wf = [r for r in _read_records(recs_dir) if r.get("outcome") in WELL_FORMED]
        cots = [r["cot_token_count"] for r in wf if isinstance(r.get("cot_token_count"), int)]
        codes = [r["code_token_count"] for r in wf if isinstance(r.get("code_token_count"), int)]
        if not cots or not codes:
            continue
        G.append(g); COT.append(agg(cots)); CODE.append(agg(codes)); N.append(len(wf))

    if not G:
        print(f"No records under {base} (run the sweep + score_generations first).")
        return 1

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(G, COT, "o-", color="tab:blue", lw=2.5, ms=8, label=f"{args.stat} CoT (reasoning)")
    ax.plot(G, CODE, "s-", color="tab:green", lw=2.5, ms=7, label=f"{args.stat} code")
    if not args.no_sum:
        tot = [c + k for c, k in zip(COT, CODE)]
        ax.plot(G, tot, "^--", color="tab:gray", lw=1.5, ms=6, alpha=0.8,
                label=f"{args.stat} CoT + code")
    for g, y in zip(G, COT):
        ax.annotate(f"{y:.0f}", (g, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, fontweight="bold", color="tab:blue")
    for g, y in zip(G, CODE):
        ax.annotate(f"{y:.0f}", (g, y), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, fontweight="bold", color="tab:green")

    ax.set_xlabel("γ  (compression ratio = fraction of CoT retained)", fontsize=12)
    ax.set_ylabel(f"{args.stat} token length (well-formed generations)", fontsize=12)
    ax.set_title(f"CoT vs code length vs γ — {args.model} {args.task}/{args.split}", fontsize=11)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper center")
    fig.tight_layout()
    out = base / "cot_vs_code_vs_gamma.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"{'gamma':>6} {'n_wf':>5} {args.stat+'_cot':>10} {args.stat+'_code':>11}")
    for g, c, k, n in zip(G, COT, CODE, N):
        print(f"{g:>6g} {n:>5} {c:>10.1f} {k:>11.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
