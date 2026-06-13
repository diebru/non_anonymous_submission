#!/usr/bin/env python3
"""CoT/code breakdown by McEval OUTCOME at a single gamma. CPU-only.

For one ``generations/<model>/<run>/<task>/<split>/gamma<g>/records/`` directory --
every trajectory, all outcomes, exactly as produced (no compression) -- it groups by
outcome (PASS / exec_fail / format_fail / ALL) and writes two figures:

  cot_boxplot_by_outcome.png    box plot of CoT length per outcome (median, IQR,
                                fliers); dashed line marks max_tokens (the runaway
                                ceiling). Same style as cot_boxplot_by_gamma, but the
                                x-axis is outcome instead of gamma (this data is 1 gamma).
  cot_code_bars_by_outcome.png  stacked avg CoT(solid)+code(faded) bars per outcome,
                                token-share % inside, count n on top. Same style as
                                cot_code_bars_avgtokens, one gamma.

Default target is the raw Phase-1 train inference (--split train --run-id run01
--gamma 1.0): the base model's full train pool, before the correct-only filter / any
compression -- use it to see whether the wrong-code trajectories already carry long /
runaway CoTs.

Usage (server):
  python3 scripts/plot_outcome_breakdown.py --model qwen2.5-14b-instruct \
      --split train --run-id run01 --gamma 1.0
  python3 scripts/plot_outcome_breakdown.py --model qwen2.5-coder-3b-instruct \
      --split train --run-id run01 --gamma 1.0
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import MODEL_IDS  # noqa: E402

# (label, outcome filter or None=ALL, colour)
OUTCOMES = [("PASS", "pass", "tab:green"), ("exec_fail", "exec_fail", "tab:orange"),
            ("format_fail", "format_fail", "tab:red"), ("ALL", None, "0.45")]


def _read_rows(d: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def _subset(records: list[dict], oc: str | None) -> list[dict]:
    return records if oc is None else [r for r in records if r.get("outcome") == oc]


def _ints(rows: list[dict], key: str) -> list[int]:
    return [r[key] for r in rows if isinstance(r.get(key), int)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", default="train")
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048, help="ceiling line + runaway threshold")
    args = ap.parse_args()

    paths = get_paths()
    recs_dir = (paths.generations_dir / args.model / args.run_id / args.task
                / args.split / f"gamma{args.gamma:g}" / "records")
    if not (recs_dir.is_dir() and any(recs_dir.glob("*.jsonl"))):
        print(f"No records under {recs_dir} (run inference + score_generations first).")
        return 1
    records = _read_rows(recs_dir)
    runaway = int(args.max_tokens * 0.9)

    # per-outcome aggregates
    rows = []
    for label, oc, color in OUTCOMES:
        sub = _subset(records, oc)
        cots = _ints(sub, "cot_token_count")
        codes = _ints(sub, "code_token_count")
        if not cots:
            rows.append({"label": label, "color": color, "n": len(sub), "cots": [],
                         "avg_cot": 0.0, "avg_code": 0.0})
            continue
        q = statistics.quantiles(cots, n=4) if len(cots) > 1 else [cots[0]] * 3
        rows.append({
            "label": label, "color": color, "n": len(sub), "cots": cots,
            "median": int(statistics.median(cots)), "q1": int(q[0]), "q3": int(q[2]),
            "max": max(cots), "n_runaway": sum(1 for c in cots if c >= runaway),
            "avg_cot": statistics.mean(cots), "avg_code": statistics.mean(codes) if codes else 0.0,
        })

    # stats table
    print(f"CoT/code by outcome | {args.model} {args.task}/{args.split} "
          f"gamma={args.gamma:g} run={args.run_id}  (uncompressed)")
    print(f"{'outcome':>12} {'n':>6} {'median':>7} {'q1':>6} {'q3':>6} {'max':>6} "
          f"{'n>=' + str(runaway):>9} {'avg_cot':>8} {'avg_code':>9}")
    for r in rows:
        if r["n"] and r["cots"]:
            print(f"{r['label']:>12} {r['n']:>6} {r['median']:>7} {r['q1']:>6} {r['q3']:>6} "
                  f"{r['max']:>6} {r['n_runaway']:>9} {r['avg_cot']:>8.1f} {r['avg_code']:>9.1f}")
        else:
            print(f"{r['label']:>12} {r['n']:>6} {'-':>7} {'-':>6} {'-':>6} {'-':>6} "
                  f"{'-':>9} {'-':>8} {'-':>9}")

    out_dir = recs_dir.parent  # the gamma<g> dir
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(no PNGs: matplotlib unavailable: {e}; stats printed above)")
        return 0

    have = [r for r in rows if r["cots"]]
    tag = f"{args.task}/{args.split} gamma={args.gamma:g}"

    # --- 1. box plot of CoT length by outcome ---
    fig, ax = plt.subplots(figsize=(9, 6))
    pos = list(range(len(have)))
    bp = ax.boxplot([r["cots"] for r in have], positions=pos, widths=0.6, showfliers=True,
                    patch_artist=True, flierprops=dict(marker=".", markersize=4,
                                                       markerfacecolor="black", alpha=0.4))
    for box, r in zip(bp["boxes"], have):
        box.set(facecolor=r["color"], alpha=0.4)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.5)
    for i, r in enumerate(have):
        ax.annotate(f"med {r['median']}\nn={r['n']}", (pos[i], r["median"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.axhline(args.max_tokens, ls="--", color="tab:red", lw=1,
               label=f"max_tokens = {args.max_tokens} (runaway ceiling)")
    ax.set_xticks(pos); ax.set_xticklabels([r["label"] for r in have])
    ax.set_xlabel("McEval outcome"); ax.set_ylabel("CoT length (tokens)")
    ax.set_title(f"CoT length by outcome — {args.model}\n{tag} (uncompressed)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    p1 = out_dir / "cot_boxplot_by_outcome.png"
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"\nwrote {p1}")

    # --- 2. stacked avg CoT(solid)+code(faded) bars by outcome ---
    def _faded(c):
        r, g, b = mcolors.to_rgb(c)
        return (r + (1 - r) * 0.55, g + (1 - g) * 0.55, b + (1 - b) * 0.55)

    fig, ax = plt.subplots(figsize=(9, 6))
    pos = list(range(len(rows)))
    for i, r in enumerate(rows):
        cm, dm = r["avg_cot"], r["avg_code"]
        ax.bar(i, cm, 0.6, color=r["color"])
        ax.bar(i, dm, 0.6, bottom=cm, color=_faded(r["color"]))
        tot = cm + dm
        if tot > 0:
            if cm > tot * 0.05:
                ax.text(i, cm / 2, f"{cm / tot * 100:.0f}%", ha="center", va="center",
                        fontsize=8, color="white")
            if dm > tot * 0.05:
                ax.text(i, cm + dm / 2, f"{dm / tot * 100:.0f}%", ha="center", va="center",
                        fontsize=8, color="black")
        ax.text(i, tot, f"n={r['n']}", ha="center", va="bottom", fontsize=8, color=r["color"])
    ax.set_xticks(pos); ax.set_xticklabels([r["label"] for r in rows])
    ax.set_xlabel("McEval outcome"); ax.set_ylabel("average tokens")
    ax.set_title(f"Average CoT + code tokens by outcome — {args.model}\n{tag} "
                 "(uncompressed)  ·  solid = CoT, faded = code; % = token share")
    ax.legend(handles=[mpatches.Patch(color="0.5", label="CoT (solid)"),
                       mpatches.Patch(color="0.85", label="code (faded)")],
              loc="upper right", fontsize=9)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.08)
    fig.tight_layout()
    p2 = out_dir / "cot_code_bars_by_outcome.png"
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
