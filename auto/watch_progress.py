#!/usr/bin/env python3
"""Live progress of a running reproduce_from_hub sweep (read-only; safe mid-run).

Scans <repo>/*/outputs_hubrepro for completed runs and prints, per (model,bench,ratio):
  runs done / expected, mean accuracy, mean CoT, mean GPU energy.
Plus a live nvidia-smi power line and overall progress.

Run in a 2nd terminal:
  source config.env && python3 watch_progress.py            # one snapshot
  source config.env && python3 watch_progress.py --loop 30  # refresh every 30s
Expected totals come from the SWEEP_*/PUBLISH_SIZES/BENCHMARKS_TO_RUN env vars
(so `source config.env` first); without them it just shows what's done.
"""
import argparse, glob, json, os, subprocess, time
from collections import defaultdict
from datetime import datetime


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


def gpu_now():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,power.draw,utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        return " | ".join(f"GPU{l.split(',')[0].strip()} {float(l.split(',')[1]):.0f}W "
                          f"{float(l.split(',')[2]):.0f}% {float(l.split(',')[3])/1024:.1f}GB"
                          for l in out.stdout.strip().splitlines())
    except Exception:
        return "nvidia-smi n/a"


def snapshot(repo_root):
    grp = defaultdict(list)
    for m in glob.glob(os.path.join(repo_root, "*", "outputs_hubrepro", "**", "metrics.json"), recursive=True):
        try:
            md = json.load(open(m))
        except Exception:
            continue                                   # being written right now
        parts = m.split(os.sep); i = parts.index("outputs_hubrepro")
        size = parts[i + 1].split("-")[-1]; bench = parts[i + 2]
        ratio = 1.0 if "Original" in m else float(m.split("TokenSkip" + os.sep)[1].split(os.sep)[0])
        run_dir = m.split(os.sep + ("Original" if ratio == 1.0 else "TokenSkip"))[0]
        run = next((p[3:] for p in parts if p.startswith("run")), "1")
        g = glob.glob(os.path.join(run_dir, f"*ratio{ratio}_run{run}_gpu.json"))
        grp[(size, bench, ratio)].append((md.get("accuracy"), md.get("avg_cot_length"),
                                          energy_j(g[0]) if g else None))
    return grp


def expected_per_cell():
    toks = len((os.environ.get("SWEEP_TOKENS") or "").split()) or 1
    reps = int(os.environ.get("SWEEP_REPEATS") or 1)
    return toks * reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.path.expanduser("~/non_anonymous_submission"))
    ap.add_argument("--loop", type=int, default=0, help="refresh interval in seconds (0 = once)")
    a = ap.parse_args()
    exp = expected_per_cell()
    while True:
        grp = snapshot(a.repo_root)
        done = sum(len(v) for v in grp.values())
        if os.environ.get("CLEAR") != "0":
            print("\033[2J\033[H", end="")
        print(f"[{datetime.now():%H:%M:%S}] {gpu_now()}")
        print(f"completed runs: {done}   (expected {exp}/cell)\n")
        print(f"{'model':<8} {'bench':<7} {'gamma':>5} {'done':>6} {'acc':>8} {'cot':>7} {'gpuJ':>9}")
        for (size, bench, ratio), runs in sorted(grp.items()):
            accs = [a for a, _, _ in runs if a is not None]
            cots = [c for _, c, _ in runs if c is not None]
            gj = [e for _, _, e in runs if e is not None]
            acc = f"{sum(accs)/len(accs)*100:.2f}%" if accs else "-"
            cot = f"{sum(cots)/len(cots):.1f}" if cots else "-"
            g = f"{sum(gj)/len(gj):.0f}" if gj else "-"
            print(f"{size:<8} {bench:<7} {ratio:>5} {len(runs):>3}/{exp:<2} {acc:>8} {cot:>7} {g:>9}")
        if a.loop <= 0:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
