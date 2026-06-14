#!/usr/bin/env python3
"""Aggregate reproduced runs into an inference_summary.json compatible with
plot_new/ and plots/ (one row per Model x Task x Ratio, averaged over repeats).

Scans the reproduce_from_hub tree:
  <repo>/<bench>/outputs_hubrepro/qwen2.5-<size>/<bench>/tok<T>/run<k>/
      <size>/Original/test/samples/metrics.json        (ratio 1.0)
      <size>/TokenSkip/<r>/samples/metrics.json        (ratio < 1.0)
      <size>_<bench>_tok<T>_ratio<r>_run<k>_{gpu,pdu}.json

Energy = trapezoid integral of power over the monitor trace (TOTAL J per run, like the
reference summary). Multiple repeats are averaged; multiple token budgets are kept
separate unless --token selects one.

Usage:
  python _aggregate.py --repo-root ~/non_anonymous_submission --token 512 \
         --out ~/non_anonymous_submission/plot_new/data_repro/inference_summary.json
Then cross-check:  SUMMARY_DIR=<dir of that file> python generate_savings_table.py
"""
import argparse, glob, json, os, re
from collections import defaultdict
from datetime import datetime

MODEL_LABEL = {"3b": "Qwen2.5_3b", "7b": "Qwen2.5_7b", "14b": "Qwen2.5_14b", "8b": "Llama3.1_8b"}


def energy_j(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if len(d) < 2:
        return None
    e = 0.0
    for a, b in zip(d, d[1:]):
        dt = (datetime.fromisoformat(b["timestamp"]) - datetime.fromisoformat(a["timestamp"])).total_seconds()
        e += 0.5 * (a["power_draw"] + b["power_draw"]) * dt
    return e


def find_energy(run_dir, ratio, run, kind):
    # filename: <size>_<bench>_tok<T>_ratio<r>_run<k>_<kind>.json
    hits = glob.glob(os.path.join(run_dir, f"*ratio{ratio}_run{run}_{kind}.json"))
    return energy_j(hits[0]) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.path.expanduser("~/non_anonymous_submission"))
    ap.add_argument("--token", default=None, help="keep only this max_new_tokens budget (e.g. 512)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(a.repo_root, "plot_new", "data_repro", "inference_summary.json")
    metrics = glob.glob(os.path.join(a.repo_root, "*", "outputs_hubrepro", "**", "metrics.json"), recursive=True)

    # group[(model,task,ratio)] = list of per-run dicts
    group = defaultdict(list)
    for m in metrics:
        parts = m.split(os.sep)
        try:
            i = parts.index("outputs_hubrepro")
        except ValueError:
            continue
        model_dir = parts[i + 1]                 # qwen2.5-<size>
        size = model_dir.split("-")[-1]
        bench = parts[i + 2]
        tok = next((p[3:] for p in parts if p.startswith("tok")), None)
        ri = next((j for j, p in enumerate(parts) if re.fullmatch(r"run\d+", p)), None)
        run = parts[ri][3:] if ri is not None else "1"
        if a.token and tok != str(a.token):
            continue
        ratio = 1.0 if "Original" in m else float(m.split("TokenSkip" + os.sep)[1].split(os.sep)[0])

        md = json.load(open(m))
        run_dir = os.sep.join(parts[:ri + 1]) if ri is not None else os.path.dirname(m)  # holds *_gpu/_pdu.json
        rec = {
            "n": md.get("n_samples") or 0,
            "acc": md.get("accuracy"),
            "cot": md.get("avg_cot_length"),
            "time": md.get("total_inference_time"),
            "gpu": find_energy(run_dir, ratio, run, "gpu"),
            "pdu": find_energy(run_dir, ratio, run, "pdu"),
        }
        group[(size, bench, ratio)].append(rec)

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    rows = []
    for (size, bench, ratio), runs in sorted(group.items()):
        rows.append({
            "Model": MODEL_LABEL.get(size, f"Qwen2.5_{size}"),
            "Task": bench,
            "Ratio": ratio,
            "N_Samples": max((r["n"] for r in runs), default=0),
            "Accuracy": avg([r["acc"] for r in runs]),
            "Avg_COT_Length": avg([r["cot"] for r in runs]),
            "Inference_Time_s": avg([r["time"] for r in runs]),
            "GPU_Energy_J": avg([r["gpu"] for r in runs]),
            "PDU_Energy_J": avg([r["pdu"] for r in runs]),
            "n_repeats": len(runs),
        })

    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"wrote {len(rows)} rows -> {out}")
    for r in rows:
        acc = f"{r['Accuracy']*100:.2f}%" if r["Accuracy"] is not None else "n/a"
        gpu = f"{r['GPU_Energy_J']:.0f}" if r["GPU_Energy_J"] is not None else "n/a"
        print(f"  {r['Model']:<12} {r['Task']:<7} g={r['Ratio']:<4} acc={acc:>8} cot={r['Avg_COT_Length']:.1f} gpuJ={gpu} (x{r['n_repeats']})")


if __name__ == "__main__":
    main()
