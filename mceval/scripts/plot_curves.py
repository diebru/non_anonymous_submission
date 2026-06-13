#!/usr/bin/env python3
"""Phase-4 accuracy plots from a scored sweep. CPU-only; needs matplotlib.

Two plots (one PNG each), read from the build_curves output ``curves.csv``:
  1. accuracy (%) vs AVERAGE CoT length (tokens) -- with a gold star at the training
     phase's average CoT length (auto-computed from the correct-CoT corpus, or
     --train-avg-cot). CoT length is reasoning ONLY (code excluded): the default
     x-column is `wf_mean` = mean cot_token_count over well-formed generations (where
     cot_text is strictly the pre-sentinel reasoning). `--x-cot mean` uses the raw
     average instead (folds back at low gamma due to the no-sentinel runaways).
  2. accuracy (%) vs PDU energy (J).
Each point is labelled with its gamma.

Run AFTER scripts/build_curves.py (which writes curves.csv). Server or local.

Usage (server):
    python3 scripts/build_curves.py  --task generation --split test --run-id sft01
    python3 scripts/plot_curves.py   --task generation --split test --run-id sft01
    # the training avg CoT is read from corpus/<model>/run01/generation/train; override:
    #   --train-avg-cot 179      (or --corpus-run-id <id>)

    # overlay the two gamma-scaled bases (+ the fixed baseline) on one figure:
    python3 scripts/plot_curves.py --task generation --split test \
        --run-id sft01_sg2048 --overlay sft01_sg1024 sft01
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import MODEL_IDS  # noqa: E402

_XCOT = {"wf_mean": "wellformed_mean_cot", "mean": "mean_cot"}
# output filename suffix per --x-cot, so the well-formed (default) and "all answers"
# (mean = includes format_fail runaways) variants don't overwrite each other.
_XCOT_SUFFIX = {"wf_mean": "", "mean": "_all"}


def _read_curves(csv_path: pathlib.Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for d in csv.DictReader(fh):
            def num(k):
                v = d.get(k, "")
                return float(v) if v not in ("", "None", None) else None
            rows.append({"gamma": num("gamma"), "acc": num("healthy_accuracy"),
                         "wellformed_mean_cot": num("wellformed_mean_cot"),
                         "mean_cot": num("mean_cot"), "median_cot": num("median_cot"),
                         "pdu_energy_j": num("pdu_energy_j"), "gpu_energy_j": num("gpu_energy_j")})
    return [r for r in rows if r["gamma"] is not None]


def train_avg_cot(paths, model: str, corpus_run_id: str) -> float | None:
    """Mean cot_token_count over the correct-CoT training corpus (gamma=1.0,
    reasoning-only by construction). Returns None if the corpus isn't present."""
    d = paths.compressed_dir.parent / "corpus" / model / corpus_run_id / "generation" / "train"
    if not d.is_dir():
        return None
    vals = []
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line).get("cot_token_count")
                if isinstance(c, int):
                    vals.append(c)
    return (sum(vals) / len(vals)) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--corpus-run-id", default="run01",
                    help="run-id of the correct-CoT corpus for the train-avg-CoT star")
    ap.add_argument("--x-cot", choices=list(_XCOT), default="wf_mean",
                    help="avg-CoT column for plot 1 (wf_mean = code-excluded + monotone)")
    ap.add_argument("--train-avg-cot", type=float, default=None,
                    help="override the auto-computed training avg CoT (star x-position)")
    ap.add_argument("--gammas", type=float, nargs="+", default=None,
                    help="restrict to these gammas (default: all in curves.csv), "
                         "e.g. --gammas 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 (drops 0.95/0.85)")
    ap.add_argument("--overlay", nargs="*", default=[], metavar="RUN_ID",
                    help="extra run-ids to overlay for a budget/config comparison, e.g. "
                         "--run-id sft01_sg2048 --overlay sft01_sg1024 sft01. Writes "
                         "*_compare.png alongside the primary run's plots; one line per "
                         "run-id (legend = run-id), gamma labels on the primary line.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split
    csv_path = base / "curves.csv"
    if not csv_path.is_file():
        print(f"No {csv_path}. Run scripts/build_curves.py --run-id {args.run_id} first.")
        return 1
    rows = sorted(_read_curves(csv_path), key=lambda r: r["gamma"], reverse=True)
    if args.gammas:
        keep = list(args.gammas)
        rows = [r for r in rows if any(abs(r["gamma"] - g) < 1e-9 for g in keep)]

    xcol = _XCOT[args.x_cot]
    tcot = args.train_avg_cot if args.train_avg_cot is not None \
        else train_avg_cot(paths, args.model, args.corpus_run_id)

    def _label_points(ax, xs, ys):
        for r, x, y in zip(rows, xs, ys):
            if x is not None and y is not None:
                ax.annotate(f"γ={r['gamma']:g}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")

    # ---- Plot 1: accuracy (%) vs avg CoT length ----
    p1 = [(r[xcol], r["acc"] * 100) for r in rows if r[xcol] is not None and r["acc"] is not None]
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    if p1:
        xs, ys = zip(*p1)
        ax1.plot(xs, ys, "o-", color="tab:blue", lw=2, ms=7, zorder=3)
        _label_points(ax1, [r[xcol] for r in rows], [r["acc"] * 100 if r["acc"] else None for r in rows])
        if tcot is not None:
            acc_g1 = next((r["acc"] * 100 for r in rows if r["gamma"] == 1.0 and r["acc"]), max(ys))
            ax1.scatter([tcot], [acc_g1], marker="*", s=420, color="gold",
                        edgecolor="black", zorder=5,
                        label=f"train avg CoT ≈ {tcot:.0f} tok")
            ax1.axvline(tcot, color="gold", ls=":", lw=1.5, zorder=1)
    ax1.set_xlabel(f"average CoT length (tokens, code excluded) [{args.x_cot}]", fontsize=12)
    ax1.set_ylabel("accuracy (%)", fontsize=12)
    ax1.set_title(f"Accuracy vs avg CoT length — {args.model} {args.task}/{args.split}", fontsize=11)
    ax1.grid(True, alpha=0.3)
    if tcot is not None:
        ax1.legend(fontsize=11)
    fig1.tight_layout()
    out1 = base / f"acc_vs_cot{_XCOT_SUFFIX[args.x_cot]}.png"
    fig1.savefig(out1, dpi=130)

    # ---- Plot 2: accuracy (%) vs PDU energy (J) ----
    p2 = [(r["pdu_energy_j"], r["acc"] * 100) for r in rows
          if r["pdu_energy_j"] is not None and r["acc"] is not None]
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    if p2:
        xs2, ys2 = zip(*p2)
        ax2.plot(xs2, ys2, "s-", color="tab:red", lw=2, ms=7, zorder=3)
        _label_points(ax2, [r["pdu_energy_j"] for r in rows],
                      [r["acc"] * 100 if r["acc"] else None for r in rows])
    ax2.set_xlabel("PDU energy (joules)", fontsize=12)
    ax2.set_ylabel("accuracy (%)", fontsize=12)
    ax2.set_title(f"Accuracy vs PDU energy — {args.model} {args.task}/{args.split}", fontsize=11)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = base / "acc_vs_pdu_energy.png"
    fig2.savefig(out2, dpi=130)

    # ---- Optional overlay: compare multiple run-ids (e.g. the two scaled bases ----
    # vs the fixed-budget baseline) on shared axes. Each run-id's curves.csv is read
    # from its own split dir; the comparison PNGs land under the PRIMARY run's dir.
    out3 = out4 = None
    if args.overlay:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        series = []  # (run_id, rows)
        for rid in [args.run_id, *args.overlay]:
            cpath = paths.generations_dir / args.model / rid / args.task / args.split / "curves.csv"
            if not cpath.is_file():
                print(f"(overlay: no {cpath} — skipping {rid}; build_curves it first)")
                continue
            srows = sorted(_read_curves(cpath), key=lambda r: r["gamma"], reverse=True)
            if args.gammas:
                srows = [r for r in srows if any(abs(r["gamma"] - g) < 1e-9 for g in args.gammas)]
            series.append((rid, srows))

        # compare 1: accuracy (%) vs avg CoT length
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        for i, (rid, srows) in enumerate(series):
            pts = [(r[xcol], r["acc"] * 100) for r in srows
                   if r.get(xcol) is not None and r["acc"] is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax3.plot(xs, ys, "o-", color=colors[i % len(colors)], lw=2, ms=6, label=rid)
            if i == 0:  # gamma labels on the primary line only (keeps the overlay readable)
                for r, x, y in zip([r for r in srows if r.get(xcol) is not None and r["acc"] is not None],
                                   xs, ys):
                    ax3.annotate(f"γ={r['gamma']:g}", (x, y), textcoords="offset points",
                                 xytext=(0, 8), ha="center", fontsize=7)
        if tcot is not None:
            ax3.axvline(tcot, color="gold", ls=":", lw=1.5, label=f"train avg CoT ≈ {tcot:.0f}")
        ax3.set_xlabel(f"average CoT length (tokens, code excluded) [{args.x_cot}]", fontsize=12)
        ax3.set_ylabel("accuracy (%)", fontsize=12)
        ax3.set_title(f"Accuracy vs avg CoT — {args.model} {args.task}/{args.split} (compare)", fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        fig3.tight_layout()
        out3 = base / f"acc_vs_cot{_XCOT_SUFFIX[args.x_cot]}_compare.png"
        fig3.savefig(out3, dpi=130)

        # compare 2: accuracy (%) vs PDU energy (J) -- the budget tradeoff
        fig4, ax4 = plt.subplots(figsize=(9, 6))
        for i, (rid, srows) in enumerate(series):
            pts = [(r["pdu_energy_j"], r["acc"] * 100) for r in srows
                   if r.get("pdu_energy_j") is not None and r["acc"] is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax4.plot(xs, ys, "s-", color=colors[i % len(colors)], lw=2, ms=6, label=rid)
        ax4.set_xlabel("PDU energy (joules)", fontsize=12)
        ax4.set_ylabel("accuracy (%)", fontsize=12)
        ax4.set_title(f"Accuracy vs PDU energy — {args.model} {args.task}/{args.split} (compare)", fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        fig4.tight_layout()
        out4 = base / "acc_vs_pdu_energy_compare.png"
        fig4.savefig(out4, dpi=130)

    print(f"train avg CoT (star) = {tcot}")
    print(f"wrote {out1}")
    print(f"wrote {out2}")
    if out3:
        print(f"wrote {out3}")
    if out4:
        print(f"wrote {out4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
