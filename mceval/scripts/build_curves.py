#!/usr/bin/env python3
"""Build the Phase-4 curves from a scored, energy-joined gamma sweep. CPU-only.

Reads each gamma-run's ``records/`` (+ ``energy/energy_summary.json``) and emits one
row per gamma with the curve quantities:
  * accuracy-vs-cot  : healthy_accuracy vs MEASURED median cot_token_count (concavity)
  * energy-vs-cot    : gpu_energy_j / per-output-token vs median cot_token_count (Goal 2/3)
  * format_fail-vs-gamma : the confound diagnostic (a rising format_fail with gamma is
    an artifact, never a reasoning failure -- it's reported separately, never folded
    into accuracy).
Accuracy + cot are over the HEALTHY-language scored rows (tsmc.eval.language_health),
so broken McEval scorers never pollute the curve.

Writes ``curves.csv`` + ``curves.json`` under the split dir and prints the table.
An optional PNG is written if matplotlib is importable (never a hard dependency).

Usage:
    python3 scripts/build_curves.py --model qwen2.5-coder-3b-instruct \
        --task generation --split test --run-id run01
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS  # noqa: E402
from tsmc.eval.language_health import is_healthy  # noqa: E402

_SCORED = ("pass", "exec_fail", "format_fail")


def _out_tokens(r: dict) -> int:
    """Decode tokens for a record (explanation two-pass stores a list)."""
    t = (r.get("_provenance") or {}).get("timing", {}).get("n_output_tokens")
    if isinstance(t, list):
        return sum(int(x or 0) for x in t)
    return int(t) if isinstance(t, int) else 0


def _truncated(r: dict) -> bool:
    return bool(r.get("extraction_status", {}).get("truncated"))


def summarize_gamma(records: list[dict], energy: dict | None = None) -> dict:
    """One curve row from a gamma-run's records + its energy summary (pure).

    Curve quantities (accuracy, median cot/code) are over HEALTHY scored rows; the
    token/energy attribution is over ALL rows (energy was measured over the whole
    run). ``wellformed_energy_j`` token-weights the run energy by the non-truncated
    share -> the "clean compression" energy with the runaway tail removed.
    """
    healthy = [r for r in records if is_healthy(r.get("lang", ""))]
    scored = [r for r in healthy if r.get("outcome") in _SCORED]
    n = len(scored)
    npass = sum(1 for r in scored if r.get("outcome") == "pass")
    nff = sum(1 for r in scored if r.get("outcome") == "format_fail")
    cots = [r["cot_token_count"] for r in scored if isinstance(r.get("cot_token_count"), int)]
    codes = [r["code_token_count"] for r in scored if isinstance(r.get("code_token_count"), int)]
    # mean over WELL-FORMED generations only (pass/exec_fail): excludes format_fail
    # runaways whose cot_token_count is a parse artifact (whole rambling output). This
    # is a meaningful AND monotone "mean reasoning length".
    wf_cots = [r["cot_token_count"] for r in scored
               if r.get("outcome") in ("pass", "exec_fail") and isinstance(r.get("cot_token_count"), int)]

    # token/energy attribution over ALL run rows (the energy curve is run-level)
    total_out = sum(_out_tokens(r) for r in records)
    wf_out = sum(_out_tokens(r) for r in records if not _truncated(r))
    trunc_count = sum(1 for r in records if _truncated(r))
    e = energy or {}
    gpu_j = e.get("gpu_energy_j")
    wf_energy = (gpu_j * wf_out / total_out) if (gpu_j and total_out) else None

    return {
        "n_scored": n,
        "healthy_accuracy": (npass / n) if n else None,
        "format_fail_rate": (nff / n) if n else None,
        "median_cot": statistics.median(cots) if cots else None,
        "mean_cot": round(statistics.mean(cots), 1) if cots else None,
        "wellformed_mean_cot": round(statistics.mean(wf_cots), 1) if wf_cots else None,
        "median_code": statistics.median(codes) if codes else None,
        "mean_code": round(statistics.mean(codes), 1) if codes else None,
        "trunc_count": trunc_count,
        "total_out_tokens": total_out,
        "wellformed_out_tokens": wf_out,
        "wellformed_energy_j": wf_energy,
        "gpu_energy_j": e.get("gpu_energy_j"),
        "gpu_energy_per_output_token_j": e.get("gpu_energy_per_output_token_j"),
        "gpu_mean_power_w": e.get("gpu_mean_power_w"),
        "run_duration_s": e.get("run_duration_s"),
        "pdu_energy_j": e.get("pdu_energy_j"),
        "n_output_tokens": e.get("n_output_tokens"),
    }


def _read_rows(d: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


_COLS = ("gamma", "n_scored", "median_cot", "mean_cot", "wellformed_mean_cot",
         "median_code", "mean_code", "healthy_accuracy",
         "format_fail_rate", "gpu_energy_j", "wellformed_energy_j",
         "gpu_energy_per_output_token_j", "gpu_mean_power_w", "run_duration_s",
         "trunc_count", "total_out_tokens", "wellformed_out_tokens",
         "pdu_energy_j", "n_output_tokens")

# x-axis choices for the curve plot (--x-axis); all are in the CSV either way.
_XAXES = {"median": "median_cot", "mean": "mean_cot", "wf_mean": "wellformed_mean_cot"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--x-axis", choices=list(_XAXES), default="mean",
                    help="plot x-axis: mean (avg cot, default -- matches TokenSkip's "
                         "avg_cot_length, but folds back at low gamma because runaways' "
                         "cot_token_count is their whole output); wf_mean (same average "
                         "over well-formed gens only -- de-polluted + monotone); median (robust)")
    args = ap.parse_args()

    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split
    rows: list[dict] = []
    for g in sorted(set(args.gammas), reverse=True):
        gdir = base / f"gamma{g:g}"
        recs_dir = gdir / "records"
        if not recs_dir.is_dir() or not any(recs_dir.glob("*.jsonl")):
            continue
        energy = None
        esum = gdir / "energy" / "energy_summary.json"
        if esum.is_file():
            energy = json.loads(esum.read_text(encoding="utf-8"))
        row = {"gamma": g, **summarize_gamma(_read_rows(recs_dir), energy)}
        rows.append(row)

    if not rows:
        print(f"No scored gamma-runs under {base} (run the sweep + scoring first).")
        return 1

    # write csv + json
    csv_path = base / "curves.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _COLS})
    (base / "curves.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # print the table
    def _f(x, nd=4):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
    print(f"\nCurves | {args.model} {args.task}/{args.split} (healthy-language scored)")
    print(f"{'gamma':>6} {'n':>5} {'med_cot':>8} {'mean_cot':>9} {'wf_mean':>8} {'acc':>7} "
          f"{'ffail':>7} {'gpu_J':>10} {'J/tok':>8} {'meanW':>7} {'dur_s':>7}")
    for r in rows:
        print(f"{r['gamma']:>6g} {_f(r['n_scored']):>5} {_f(r['median_cot']):>8} "
              f"{_f(r['mean_cot'],1):>9} {_f(r['wellformed_mean_cot'],1):>8} "
              f"{_f(r['healthy_accuracy']):>7} {_f(r['format_fail_rate']):>7} "
              f"{_f(r['gpu_energy_j'],1):>10} {_f(r['gpu_energy_per_output_token_j'],4):>8} "
              f"{_f(r['gpu_mean_power_w'],0):>7} {_f(r['run_duration_s'],1):>7}")

    # token decomposition: where do the output tokens (and thus energy) go, and how
    # much of the low-gamma energy is the runaway tail vs the well-formed generations?
    print(f"\nToken decomposition | {args.task}/{args.split}")
    print(f"{'gamma':>6} {'med_cot':>8} {'med_code':>9} {'tot_tok':>9} {'wf_tok':>9} "
          f"{'trunc':>6} {'energy_J':>9} {'wf_J':>9}")
    for r in rows:
        print(f"{r['gamma']:>6g} {_f(r['median_cot']):>8} {_f(r['median_code']):>9} "
              f"{_f(r['total_out_tokens']):>9} {_f(r['wellformed_out_tokens']):>9} "
              f"{_f(r['trunc_count']):>6} {_f(r['gpu_energy_j'],0):>9} "
              f"{_f(r['wellformed_energy_j'],0):>9}")
    print(f"\nwrote {csv_path}\nwrote {base / 'curves.json'}")

    # optional plot (never a hard dep)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xkey = _XAXES[args.x_axis]
        xlabel = f"{args.x_axis} cot_token_count"
        pts = [r for r in rows if r.get(xkey) is not None]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        x = [r[xkey] for r in pts]
        ax[0].plot(x, [r["healthy_accuracy"] for r in pts], "o-")
        ax[0].set(xlabel=xlabel, ylabel="healthy accuracy",
                  title=f"{args.task}: accuracy vs CoT")
        ye = [r["gpu_energy_j"] for r in pts]
        if any(v is not None for v in ye):
            ax[1].plot(x, ye, "s-", color="tab:red")
            ax[1].set(xlabel=xlabel, ylabel="GPU energy (J)",
                      title=f"{args.task}: energy vs CoT")
        fig.tight_layout()
        png = base / "curves.png"
        fig.savefig(png, dpi=120)
        print(f"wrote {png}")
    except Exception as e:  # noqa: BLE001
        print(f"(no PNG: {e}; curves.csv/json written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
