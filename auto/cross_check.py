#!/usr/bin/env python3
"""Live cross-validation: reproduced runs vs the committed reference, with PASS/WARN flags.

Read-only. Scans <repo>/*/outputs_hubrepro for completed runs, looks up the matching
reference row in plots/data/inference_summary.json, and prints per (bench, gamma):
  reproduced vs reference accuracy (delta in pp) and CoT (delta in %), with a flag.

Flags (accuracy + CoT are what match exactly when correct; energy is informational
because our whole-process window runs ~10-20% high vs the reference):
  OK     within tolerance
  WARN-A accuracy off by > --acc-tol pp
  WARN-C CoT off by > --cot-tol %

Run in a 2nd terminal:
  source config.env && python3 cross_check.py            # snapshot
  source config.env && python3 cross_check.py --loop 60  # refresh
"""
import argparse, glob, json, os, time
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
    return sum(0.5 * (a["power_draw"] + b["power_draw"]) *
               (datetime.fromisoformat(b["timestamp"]) - datetime.fromisoformat(a["timestamp"])).total_seconds()
               for a, b in zip(d, d[1:]))


def reproduced(repo_root):
    grp = defaultdict(list)
    for m in glob.glob(os.path.join(repo_root, "*", "outputs_hubrepro", "**", "metrics.json"), recursive=True):
        try:
            md = json.load(open(m))
        except Exception:
            continue
        parts = m.split(os.sep); i = parts.index("outputs_hubrepro")
        size = parts[i + 1].split("-")[-1]; bench = parts[i + 2]
        ratio = 1.0 if "Original" in m else float(m.split("TokenSkip" + os.sep)[1].split(os.sep)[0])
        ri = next((j for j, p in enumerate(parts) if p.startswith("run") and p[3:].isdigit()), None)
        run_dir = os.sep.join(parts[:ri + 1]) if ri is not None else os.path.dirname(m)
        run = parts[ri][3:] if ri is not None else "1"
        gp = glob.glob(os.path.join(run_dir, f"*ratio{ratio}_run{run}_pdu.json"))
        grp[(size, bench, ratio)].append((md.get("accuracy"), md.get("avg_cot_length"),
                                          md.get("n_samples"), energy_j(gp[0]) if gp else None))
    return grp


def reference(path):
    ref = {}
    for r in json.load(open(path)):
        n = r.get("N_Samples") or 0
        ref[(r.get("Model"), r.get("Task"), r.get("Ratio"))] = {
            "acc": r.get("Accuracy"), "cot": r.get("Avg_COT_Length"),
            "pdu_ps": (r["PDU_Energy_J"] / n) if r.get("PDU_Energy_J") and n else None,
        }
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.path.expanduser("~/non_anonymous_submission"))
    ap.add_argument("--reference", default=None)
    ap.add_argument("--acc-tol", type=float, default=2.0, help="accuracy tolerance in pp")
    ap.add_argument("--cot-tol", type=float, default=8.0, help="CoT tolerance in %")
    ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    ref_path = a.reference or os.path.join(a.repo_root, "plots", "data", "inference_summary.json")
    ref = reference(ref_path)

    while True:
        rep = reproduced(a.repo_root)
        if os.environ.get("CLEAR") != "0":
            print("\033[2J\033[H", end="")
        print(f"[{datetime.now():%H:%M:%S}] reproduced vs reference   (acc tol +/-{a.acc_tol}pp, CoT tol +/-{a.cot_tol}%)\n")
        hdr = f"{'bench':<7}{'g':>5}{'rep':>4} | {'acc_r':>7}{'acc_ref':>8}{'dpp':>7} | {'cot_r':>7}{'cot_ref':>8}{'d%':>7} | {'pduJ/s':>8}{'ref/s':>7} | flag"
        print(hdr); print("-" * len(hdr))
        warns = 0
        for (size, bench, ratio) in sorted(rep, key=lambda k: (k[1], -k[2])):
            runs = rep[(size, bench, ratio)]
            accs = [x[0] for x in runs if x[0] is not None]
            cots = [x[1] for x in runs if x[1] is not None]
            ns = max((x[2] or 0) for x in runs)
            pjs = [x[3] / x[2] for x in runs if x[3] and x[2]]
            ar = sum(accs) / len(accs) if accs else None
            cr = sum(cots) / len(cots) if cots else None
            pr = sum(pjs) / len(pjs) if pjs else None
            rf = ref.get((MODEL_LABEL.get(size, size), bench, ratio), {})
            aref, cref, pref = rf.get("acc"), rf.get("cot"), rf.get("pdu_ps")
            dpp = (ar - aref) * 100 if ar is not None and aref is not None else None
            dco = (cr - cref) / cref * 100 if cr is not None and cref else None
            flag = "OK"
            if dpp is not None and abs(dpp) > a.acc_tol:
                flag = "WARN-A"; warns += 1
            elif dco is not None and abs(dco) > a.cot_tol:
                flag = "WARN-C"; warns += 1
            def f(v, p="{:.2f}"):
                return p.format(v) if v is not None else "-"
            print(f"{bench:<7}{ratio:>5}{len(runs):>4} | "
                  f"{f(ar*100 if ar is not None else None):>7}{f(aref*100 if aref is not None else None):>8}{f(dpp,'{:+.2f}'):>7} | "
                  f"{f(cr,'{:.0f}'):>7}{f(cref,'{:.0f}'):>8}{f(dco,'{:+.1f}'):>7} | "
                  f"{f(pr,'{:.1f}'):>8}{f(pref,'{:.1f}'):>7} | {flag}")
        print(f"\n{'='*40}\n{'ALL OK' if warns == 0 else str(warns)+' WARNING(S) — investigate before trusting'}")
        if a.loop <= 0:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
