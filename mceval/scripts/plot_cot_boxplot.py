#!/usr/bin/env python3
"""Box plot of the CoT-length distribution per gamma. CPU-only.

Answers: as gamma shrinks (more compression), does the CoT really get shorter --
and where do the >2000-token "CoTs" come from?

Two data sources (--source):

  records    -- what the SFT'd model PRODUCES at inference (the gamma sweep, default).
                A minority of low-gamma generations DEGENERATE: they ramble to
                max_tokens WITHOUT ever emitting the sentinel, so the parser has no
                CoT/code boundary and counts the whole ~2048-token output as "CoT".
                Those appear as high outliers above an otherwise-shrinking box (and
                are exactly the format_fail rows -- see --wellformed to drop them).

  compressed -- what we FINE-TUNED ON (the Phase-2 compressed training corpus). Use
                this to confirm we did NOT train on degenerate >2000-token CoTs: the
                boxes should shrink monotonically with NO ceiling outliers. If they
                don't, the training data is wrong.

One box per gamma (median, IQR, whiskers, fliers); a dashed line marks max_tokens.
Also prints a per-gamma stats table (n, median, q1/q3, p95, max, n_runaway).

Usage (server):
  # what the model produces at test time (where the >2000 CoTs show up):
  python3 scripts/plot_cot_boxplot.py --model qwen2.5-14b-instruct \
      --source records --run-id sft01 --split test
  # what we actually trained on (verify the training data is clean):
  python3 scripts/plot_cot_boxplot.py --model qwen2.5-14b-instruct \
      --source compressed --run-id run01 --split train
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


def _read_rows(d: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def _gamma_dir(paths, args, g: float) -> pathlib.Path:
    """records/ live under generations/<run>; the compressed corpus is flat per gamma."""
    if args.source == "records":
        return (paths.generations_dir / args.model / args.run_id / args.task
                / args.split / f"gamma{g:g}" / "records")
    return (paths.compressed_dir / args.model / args.run_id / args.task
            / args.split / f"gamma{g:g}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--source", choices=("records", "compressed"), default="records",
                    help="records = model output at inference; compressed = training corpus")
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", default="test", help="records: test (default); compressed: train")
    ap.add_argument("--run-id", default="sft01", help="records: sft01; compressed corpus: run01")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--max-tokens", type=int, default=2048, help="ceiling line + runaway threshold")
    ap.add_argument("--wellformed", action="store_true",
                    help="records only: keep pass/exec_fail (drop the format_fail runaways)")
    ap.add_argument("--logy", action="store_true", help="log y-axis (boxes vs ceiling fliers)")
    args = ap.parse_args()

    paths = get_paths()
    runaway = int(args.max_tokens * 0.9)  # near the ceiling -> a degenerate generation

    G: list[float] = []
    data: list[list[int]] = []
    stats: list[dict] = []
    for g in sorted(set(args.gammas), reverse=True):
        d = _gamma_dir(paths, args, g)
        if not (d.is_dir() and any(d.glob("*.jsonl"))):
            continue
        rows = _read_rows(d)
        if args.source == "records" and args.wellformed:
            rows = [r for r in rows if r.get("outcome") in ("pass", "exec_fail")]
        cots = [r["cot_token_count"] for r in rows if isinstance(r.get("cot_token_count"), int)]
        if not cots:
            continue
        G.append(g)
        data.append(cots)
        q = statistics.quantiles(cots, n=4) if len(cots) > 1 else [cots[0]] * 3
        p95 = statistics.quantiles(cots, n=20)[-1] if len(cots) > 1 else cots[0]
        stats.append({
            "gamma": g, "n": len(cots), "median": int(statistics.median(cots)),
            "q1": int(q[0]), "q3": int(q[2]), "p95": int(p95),
            "max": max(cots), "n_runaway": sum(1 for c in cots if c >= runaway),
        })

    if not G:
        print(f"No {args.source} data found for {args.model} {args.task}/{args.split} "
              f"run={args.run_id}. (records need the sweep; compressed needs Phase 2.)")
        return 1

    print(f"CoT length by gamma | {args.model} [{args.source}] {args.task}/{args.split} "
          f"run={args.run_id}{' wellformed' if args.wellformed else ''}")
    print(f"{'gamma':>6} {'n':>6} {'median':>7} {'q1':>6} {'q3':>6} {'p95':>6} {'max':>6} "
          f"{'n>=' + str(runaway):>9}")
    for s in stats:
        print(f"{s['gamma']:>6g} {s['n']:>6} {s['median']:>7} {s['q1']:>6} {s['q3']:>6} "
              f"{s['p95']:>6} {s['max']:>6} {s['n_runaway']:>9}")

    # output dir: alongside the data (records -> split dir; compressed -> corpus root)
    if args.source == "records":
        out_dir = paths.generations_dir / args.model / args.run_id / args.task / args.split
    else:
        out_dir = paths.compressed_dir / args.model / args.run_id / args.task / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.source}{'_wf' if args.wellformed else ''}"
    out = out_dir / f"cot_boxplot_by_gamma_{tag}.png"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(no PNG: matplotlib unavailable: {e}; stats printed above)")
        return 0

    fig, ax = plt.subplots(figsize=(13, 6.5))
    pos = list(range(len(G)))
    bp = ax.boxplot(data, positions=pos, widths=0.6, showfliers=True,
                    patch_artist=True, flierprops=dict(marker=".", markersize=4,
                                                       markerfacecolor="tab:red", alpha=0.4))
    for box in bp["boxes"]:
        box.set(facecolor="tab:blue", alpha=0.35)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.5)
    for i, s in enumerate(stats):
        ax.annotate(str(s["median"]), (pos[i], s["median"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7)
    ax.axhline(args.max_tokens, ls="--", color="tab:red", lw=1,
               label=f"max_tokens = {args.max_tokens} (runaway ceiling)")
    ax.set_xticks(pos); ax.set_xticklabels([f"{g:g}" for g in G])
    ax.set_xlabel(r"$\gamma$  (compression ratio; -> = more compression)")
    ax.set_ylabel("CoT length (tokens)" + (" — log" if args.logy else ""))
    src = ("model output at inference" if args.source == "records"
           else "training corpus (what we fine-tuned on)")
    ax.set_title(f"CoT length distribution by gamma — {args.model}\n{src} · {args.task}/{args.split}"
                 + ("  (well-formed only)" if args.wellformed else ""))
    if args.logy:
        ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
