#!/usr/bin/env python3
"""Per-gamma CoT/code/answer-length + energy plots, and raw CoT/code dumps. CPU-only.

Reads a scored, energy-joined sweep (the same ``records/`` + ``energy/energy_summary
.json`` that build_curves uses) and writes, under the split dir:

  Plots (averages are over WELL-FORMED gens = pass/exec_fail, so the low-gamma runaway
  tail -- whose cot_token_count is a parse artifact -- never pollutes the average):
    1. cot_code_bars_avgtokens.png          per-outcome (PASS/exec_fail/format_fail/TOTAL)
                                            stacked CoT(solid)+code(faded) bars per gamma,
                                            token-share % inside + count n on top
    2. energy_vs_avgtokens_full_answer.png  measured GPU + PDU energy vs avg full-answer tokens (CoT+code)
    3. accuracy_vs_avgtokens_only_cot.png   accuracy (%) vs avg CoT tokens
    4. cot_length_vs_gamma.png              avg CoT length vs gamma (compression factor)
    5. full_answer_length_vs_gamma.png      avg full-answer length (CoT+code) vs gamma, well-formed
    5b.full_answer_length_vs_gamma_all.png  same, but over ALL answers (pass+exec_fail+format_fail)
    6. inference_time_vs_gamma.png          generation wall-time (s) vs gamma
  Data: answer_breakdown.csv (the per-gamma numbers behind the plots).
  Dumps (every generated row, all languages, so you can read what the model produced):
    cot_only.jsonl   {gamma, lang, problem_id, outcome, cot_token_count, cot_text}
    code_only.jsonl  {gamma, lang, problem_id, outcome, code_token_count, code_snippet}

gamma = fraction of CoT retained (1.0 = no compression; lower = more compression).

Usage (server, after the sweep + build_curves):
    python3 scripts/plot_answer_breakdown.py --model qwen2.5-14b-instruct \
        --task generation --split test --run-id sft01
    # one gamma's text only:  --gammas 1.0
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS  # noqa: E402
from tsmc.eval.language_health import is_healthy  # noqa: E402

_SCORED = ("pass", "exec_fail", "format_fail")
_WELLFORMED = ("pass", "exec_fail")


def _read_rows(d: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def _out_tokens(r: dict) -> int:
    t = (r.get("_provenance") or {}).get("timing", {}).get("n_output_tokens")
    if isinstance(t, list):
        return sum(int(x or 0) for x in t)
    return int(t) if isinstance(t, int) else 0


def _truncated(r: dict) -> bool:
    return bool(r.get("extraction_status", {}).get("truncated"))


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), int)]
    return round(sum(vals) / len(vals), 1) if vals else None


def summarize(records: list[dict], energy: dict | None) -> dict:
    """Per-gamma averages over well-formed healthy gens + run-level energy."""
    healthy = [r for r in records if is_healthy(r.get("lang", ""))]
    scored = [r for r in healthy if r.get("outcome") in _SCORED]
    wf = [r for r in scored if r.get("outcome") in _WELLFORMED]
    n = len(scored)
    npass = sum(1 for r in scored if r.get("outcome") == "pass")
    avg_cot = _avg(wf, "cot_token_count")
    avg_code = _avg(wf, "code_token_count")
    avg_full = round(avg_cot + avg_code, 1) if (avg_cot is not None and avg_code is not None) else None
    # ALL-outcome averages (pass + exec_fail + format_fail): includes the low-gamma
    # runaways (cot_token_count = whole ~2048 output), so avg_full_all folds back up.
    avg_cot_all = _avg(scored, "cot_token_count")
    avg_code_all = _avg(scored, "code_token_count")
    avg_full_all = (round(avg_cot_all + avg_code_all, 1)
                    if (avg_cot_all is not None and avg_code_all is not None) else None)
    total_out = sum(_out_tokens(r) for r in records)
    wf_out = sum(_out_tokens(r) for r in records if not _truncated(r))
    e = energy or {}
    gpu_j = e.get("gpu_energy_j")
    pdu_j = e.get("pdu_energy_j")
    wf_j = (gpu_j * wf_out / total_out) if (gpu_j and total_out) else None
    return {
        "n_scored": n,
        "accuracy": round(npass / n, 4) if n else None,
        "avg_cot": avg_cot, "avg_code": avg_code, "avg_full": avg_full,
        "avg_cot_all": avg_cot_all, "avg_code_all": avg_code_all, "avg_full_all": avg_full_all,
        "gpu_energy_j": round(gpu_j, 1) if gpu_j is not None else None,
        "pdu_energy_j": round(pdu_j, 1) if pdu_j is not None else None,
        "wellformed_energy_j": round(wf_j, 1) if wf_j is not None else None,
        "run_duration_s": e.get("run_duration_s"),
    }


_CSV_COLS = ("gamma", "n_scored", "avg_cot", "avg_code", "avg_full",
             "avg_cot_all", "avg_code_all", "avg_full_all",
             "accuracy", "gpu_energy_j", "pdu_energy_j", "wellformed_energy_j",
             "run_duration_s")


def make_plots(rows: list[dict], records_by_gamma: list[tuple[float, list[dict]]],
               base: pathlib.Path, title_tag: str) -> list[str]:
    """rows are gamma-descending. Returns the filenames written (or [] if no mpl)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # noqa: BLE001
        print(f"(no plots: matplotlib/numpy unavailable: {e})")
        return []

    g = [r["gamma"] for r in rows]
    cot = [r["avg_cot"] for r in rows]
    code = [r["avg_code"] for r in rows]
    full = [r["avg_full"] for r in rows]
    cot_all = [r["avg_cot_all"] for r in rows]
    code_all = [r["avg_code_all"] for r in rows]
    full_all = [r["avg_full_all"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    gpu = [r["gpu_energy_j"] for r in rows]
    pdu = [r["pdu_energy_j"] for r in rows]
    dur = [r["run_duration_s"] for r in rows]
    written: list[str] = []

    def save(fig, name):
        p = base / name
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

    # 1. per-outcome stacked CoT(solid)+code(faded) bars: PASS / exec_fail / format_fail
    #    / TOTAL, with token-share % inside each segment and count n on top. Over ALL
    #    records (every language) so TOTAL n = the full split; format_fail's tall CoT is
    #    the runaway artifact (a truncated gen has no sentinel -> whole output = "CoT").
    def _mean(rs, key):
        v = [r[key] for r in rs if isinstance(r.get(key), int)]
        return sum(v) / len(v) if v else 0.0

    def _faded(c):
        r, gg, b = mcolors.to_rgb(c)
        return (r + (1 - r) * 0.55, gg + (1 - gg) * 0.55, b + (1 - b) * 0.55)

    outcomes = [("PASS", "pass", "tab:green"), ("exec_fail", "exec_fail", "tab:orange"),
                ("format_fail", "format_fail", "tab:red"), ("TOTAL", None, "0.45")]
    Gx = [gv for gv, _ in records_by_gamma]
    x = np.arange(len(Gx)); width = 0.2
    fig, ax = plt.subplots(figsize=(16, 8))
    for j, (label, oc, color) in enumerate(outcomes):
        xs = x + (j - (len(outcomes) - 1) / 2) * width
        cots, codes, cnts = [], [], []
        for _gv, recs in records_by_gamma:
            sub = recs if oc is None else [r for r in recs if r.get("outcome") == oc]
            cots.append(_mean(sub, "cot_token_count"))
            codes.append(_mean(sub, "code_token_count"))
            cnts.append(len(sub))
        ax.bar(xs, cots, width, color=color)
        ax.bar(xs, codes, width, bottom=cots, color=_faded(color))
        for xi, cm, dm, n in zip(xs, cots, codes, cnts):
            tot = cm + dm
            if cm > tot * 0.05:
                ax.text(xi, cm / 2, f"{cm / tot * 100:.0f}%", ha="center", va="center",
                        fontsize=6, color="white")
            if dm > tot * 0.05:
                ax.text(xi, cm + dm / 2, f"{dm / tot * 100:.0f}%", ha="center", va="center",
                        fontsize=6, color="black")
            ax.text(xi, tot, str(n), ha="center", va="bottom", fontsize=6, color=color, rotation=90)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.08)  # headroom so the count-n labels aren't clipped
    ax.set_xticks(x); ax.set_xticklabels([f"{gv:g}" for gv in Gx])
    ax.set(xlabel=r"$\gamma$  (compression ratio)", ylabel="average answer length (tokens)",
           title=f"Average CoT + code tokens by outcome and $\\gamma$ — {title_tag}\n"
                 "solid = CoT, faded = code;  % inside = token share;  number on top = count n")
    handles = [mpatches.Patch(color=c, label=l) for (l, _o, c) in outcomes]
    handles += [mpatches.Patch(color="0.5", label="CoT (solid)"),
                mpatches.Patch(color="0.85", label="code (faded)")]
    ax.legend(handles=handles, ncol=6, fontsize=9, loc="upper left", framealpha=0.95)
    save(fig, "cot_code_bars_avgtokens.png")

    # 2. measured energy (GPU + PDU) vs avg full-answer length (CoT + code, well-formed)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(full, gpu, "s-", color="tab:red", label="GPU energy (J)")
    if any(v is not None for v in pdu):
        ax.plot(full, pdu, "o-", color="tab:purple", label="PDU energy (J)")
    for xf, yg, gv in zip(full, gpu, g):
        if xf is not None and yg is not None:
            ax.annotate(f"{gv:g}", (xf, yg), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set(xlabel="avg full-answer tokens (CoT + code, well-formed)",
           ylabel="energy (J)", title="Measured GPU & PDU energy vs average full-answer length")
    ax.legend()
    save(fig, "energy_vs_avgtokens_full_answer.png")

    # 3. accuracy (%) vs avg CoT length
    fig, ax = plt.subplots(figsize=(7, 5))
    acc_pct = [a * 100 if a is not None else None for a in acc]
    ax.plot(cot, acc_pct, "o-", color="tab:blue")
    for xc, ya, gv in zip(cot, acc_pct, g):
        if xc is not None and ya is not None:
            ax.annotate(f"{gv:g}", (xc, ya), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set(xlabel="avg CoT tokens (well-formed)", ylabel="accuracy (%)",
           title="Accuracy vs average CoT length")
    save(fig, "accuracy_vs_avgtokens_only_cot.png")

    # 4. avg CoT length vs gamma
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g, cot, "o-", color="tab:blue")
    ax.invert_xaxis()  # 1.0 (no compression) on the left -> more compression to the right
    ax.set(xlabel="gamma (CoT retained; -> = more compression)",
           ylabel="avg CoT tokens", title="CoT length vs compression factor")
    save(fig, "cot_length_vs_gamma.png")

    # 5. avg full-answer length vs gamma (WELL-FORMED: pass + exec_fail) -- unchanged
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g, full, "o-", color="tab:purple", label="full answer (CoT + code)")
    ax.plot(g, cot, "o--", color="tab:blue", alpha=0.7, label="CoT")
    ax.plot(g, code, "o--", color="tab:orange", alpha=0.7, label="code")
    ax.invert_xaxis()
    ax.set(xlabel="gamma (CoT retained; -> = more compression)",
           ylabel="avg tokens", title="Full-answer length vs compression factor")
    ax.legend()
    save(fig, "full_answer_length_vs_gamma.png")

    # 5b. SAME plot but over ALL answers (pass + exec_fail + format_fail). The format_fail
    #     runaways carry the whole ~2048-tok output as "CoT", so this folds back up at low gamma.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g, full_all, "o-", color="tab:purple", label="full answer (CoT + code)")
    ax.plot(g, cot_all, "o--", color="tab:blue", alpha=0.7, label="CoT")
    ax.plot(g, code_all, "o--", color="tab:orange", alpha=0.7, label="code")
    ax.invert_xaxis()
    ax.set(xlabel="gamma (CoT retained; -> = more compression)",
           ylabel="avg tokens",
           title="Full-answer length vs compression factor (ALL answers)")
    ax.legend()
    save(fig, "full_answer_length_vs_gamma_all.png")

    # 6. inference (generation) wall-time vs gamma (x ascending 0.1 -> 1.0)
    if any(v is not None for v in dur):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(g, dur, "o-", color="tab:red")
        ax.set(xlabel="gamma (compression ratio; <- = more compression)",
               ylabel="inference time (s)",
               title="Inference time vs compression factor")
        save(fig, "inference_time_vs_gamma.png")

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--no-dump", action="store_true", help="skip the cot_only/code_only jsonl dumps")
    args = ap.parse_args()

    paths = get_paths()
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split

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

    # numbers behind the plots
    csv_path = base / "answer_breakdown.csv"
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

    for p in make_plots(rows, records_by_gamma, base, f"{args.model} {args.task}/{args.split}"):
        print(f"wrote {p}")

    if not args.no_dump:
        (base / "cot_only.jsonl").write_text("\n".join(cot_lines) + "\n", encoding="utf-8")
        (base / "code_only.jsonl").write_text("\n".join(code_lines) + "\n", encoding="utf-8")
        print(f"wrote {base / 'cot_only.jsonl'}  ({len(cot_lines)} rows)")
        print(f"wrote {base / 'code_only.jsonl'} ({len(code_lines)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
