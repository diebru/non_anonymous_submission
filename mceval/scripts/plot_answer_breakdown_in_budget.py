#!/usr/bin/env python3
"""In-budget variant of plot_answer_breakdown.py. CPU-only.

Same inputs (records/ + energy/energy_summary.json) and same plots/CSV/dumps as
plot_answer_breakdown, but BEFORE any averaging, every generation whose measured
cot_token_count exceeds its gamma's CoT budget (budget_base * gamma, default
budget_base=1024) is dropped. This removes low-gamma "runaway" generations -- and
any other CoT that blew past its target compression -- from every downstream
number (averages, accuracy, n_scored, the per-outcome bar chart, and the
cot_only/code_only dumps), not just the well-formed-token averages.

Caveat: gpu_energy_j/pdu_energy_j are the unfiltered run-level totals (straight
from energy_summary.json), but wellformed_energy_j is re-derived from the
in-budget subset's own output-token total, so it is NOT directly comparable to
the same column in plot_answer_breakdown's CSV.

Outputs go to a sibling 'in_budget/' subdirectory so they don't clobber
plot_answer_breakdown's outputs.

Usage (server, after the sweep + build_curves):
    python3 scripts/plot_answer_breakdown_in_budget.py --model qwen2.5-14b-instruct \
        --task generation --split test --run-id sft01
    # custom budget base (per-gamma budget = budget_base * gamma):
    python3 scripts/plot_answer_breakdown_in_budget.py ... --budget-base 2048
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from plot_answer_breakdown import _CSV_COLS, _read_rows, make_plots, summarize  # noqa: E402
from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--budget-base", type=float, default=1024.0,
                    help="CoT token budget at gamma=1.0; per-gamma budget = budget_base * gamma")
    ap.add_argument("--no-dump", action="store_true", help="skip the cot_only/code_only jsonl dumps")
    args = ap.parse_args()

    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split
    out = base / "in_budget"

    rows: list[dict] = []
    records_by_gamma: list[tuple[float, list[dict]]] = []
    cot_lines: list[str] = []
    code_lines: list[str] = []
    for g in sorted(set(args.gammas), reverse=True):
        gdir = base / f"gamma{g:g}"
        recs_dir = gdir / "records"
        if not recs_dir.is_dir() or not any(recs_dir.glob("*.jsonl")):
            continue
        records = _read_rows(recs_dir)
        n_raw = len(records)
        budget = args.budget_base * g
        records = [r for r in records if (r.get("cot_token_count") or 0) <= budget]
        print(f"gamma={g:g}: budget={budget:.1f} cot tokens -> kept {len(records)}/{n_raw} records")
        esum = gdir / "energy" / "energy_summary.json"
        energy = json.loads(esum.read_text(encoding="utf-8")) if esum.is_file() else None
        rows.append({"gamma": g, **summarize(records, energy)})
        records_by_gamma.append((g, records))
        if not args.no_dump:
            for r in records:
                key = {"gamma": g, "lang": r.get("lang"), "problem_id": r.get("problem_id"),
                       "outcome": r.get("outcome")}
                cot_lines.append(json.dumps({**key, "cot_token_count": r.get("cot_token_count"),
                                             "cot_text": r.get("cot_text")}))
                code_lines.append(json.dumps({**key, "code_token_count": r.get("code_token_count"),
                                              "code_snippet": r.get("code_snippet")}))

    if not rows:
        print(f"No scored gamma-runs under {base} (run the sweep + build_curves first).")
        return 1

    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "answer_breakdown.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _CSV_COLS})
    print(f"wrote {csv_path}")
    print(f"{'gamma':>6} {'avg_cot':>8} {'avg_code':>9} {'avg_full':>9} {'acc':>7} "
          f"{'gpu_J':>10} {'pdu_J':>10} {'dur_s':>8}")
    for r in rows:
        f = lambda x, nd=1: "-" if x is None else f"{x:.{nd}f}"
        print(f"{r['gamma']:>6g} {f(r['avg_cot']):>8} {f(r['avg_code']):>9} {f(r['avg_full']):>9} "
              f"{f(r['accuracy'],4):>7} {f(r['gpu_energy_j']):>10} {f(r['pdu_energy_j']):>10} "
              f"{f(r['run_duration_s']):>8}")

    title_tag = f"{args.model} {args.task}/{args.split}, in-budget (cot_token_count <= {args.budget_base:g}*gamma)"
    for p in make_plots(rows, records_by_gamma, out, title_tag):
        print(f"wrote {p}")

    if not args.no_dump:
        (out / "cot_only.jsonl").write_text("\n".join(cot_lines) + "\n", encoding="utf-8")
        (out / "code_only.jsonl").write_text("\n".join(code_lines) + "\n", encoding="utf-8")
        print(f"wrote {out / 'cot_only.jsonl'}  ({len(cot_lines)} rows)")
        print(f"wrote {out / 'code_only.jsonl'} ({len(code_lines)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
