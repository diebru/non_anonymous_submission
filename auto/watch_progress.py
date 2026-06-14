#!/usr/bin/env python3
"""Live progress of a running reproduce_from_hub sweep (read-only; safe mid-run).

Shows, per (model,bench,ratio): runs done/expected, mean accuracy, mean CoT, mean GPU
energy, mean PDU energy. Plus a live nvidia-smi power line and a live PDU power reading.

Run in a 2nd terminal:
  source config.env && python3 watch_progress.py            # one snapshot
  source config.env && python3 watch_progress.py --loop 30  # refresh every 30s
`source config.env` first so the expected totals and the PDU SNMP settings are in env.
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


def pdu_now():
    if os.environ.get("ENABLE_PDU", "1") != "1":
        return "PDU off"
    host = os.environ.get("PDU_HOST") or os.environ.get("PDU_IP") or "192.0.2.1"
    comm = os.environ.get("PDU_SNMP_COMMUNITY", "public")
    oid = os.environ.get("PDU_OID", "PowerNet-MIB::ePDUPhaseStatusActivePower.1")
    try:
        out = subprocess.run(["snmpget", "-v2c", "-c", comm, host, oid],
                             capture_output=True, text=True, timeout=5)
        import re
        m = re.search(r"(-?\d+(?:\.\d+)?)", out.stdout.split("=", 1)[1]) if "=" in out.stdout else None
        return f"PDU {float(m.group(1)):.0f}W" if m else f"PDU n/a ({out.stderr.strip()[:40]})"
    except Exception as e:
        return f"PDU err ({e})"


def snapshot(repo_root):
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
        run_dir = os.sep.join(parts[:ri + 1]) if ri is not None else os.path.dirname(m)  # holds *_gpu/_pdu.json
        run = parts[ri][3:] if ri is not None else "1"
        g = glob.glob(os.path.join(run_dir, f"*ratio{ratio}_run{run}_gpu.json"))
        p = glob.glob(os.path.join(run_dir, f"*ratio{ratio}_run{run}_pdu.json"))
        grp[(size, bench, ratio)].append((md.get("accuracy"), md.get("avg_cot_length"),
                                          energy_j(g[0]) if g else None,
                                          energy_j(p[0]) if p else None))
    return grp


def expected_per_cell():
    # one base-token budget per benchmark now -> expected runs/cell = repeats
    return int(os.environ.get("SWEEP_REPEATS") or 1)


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
        print(f"[{datetime.now():%H:%M:%S}] {gpu_now()}  ||  {pdu_now()}")
        print(f"completed runs: {done}   (expected {exp}/cell)\n")
        print(f"{'model':<8} {'bench':<7} {'gamma':>5} {'done':>6} {'acc':>8} {'cot':>7} {'gpuJ':>9} {'pduJ':>9}")
        for (size, bench, ratio), runs in sorted(grp.items()):
            accs = [x[0] for x in runs if x[0] is not None]
            cots = [x[1] for x in runs if x[1] is not None]
            gj = [x[2] for x in runs if x[2] is not None]
            pj = [x[3] for x in runs if x[3] is not None]
            acc = f"{sum(accs)/len(accs)*100:.2f}%" if accs else "-"
            cot = f"{sum(cots)/len(cots):.1f}" if cots else "-"
            g = f"{sum(gj)/len(gj):.0f}" if gj else "-"
            p = f"{sum(pj)/len(pj):.0f}" if pj else "-"
            print(f"{size:<8} {bench:<7} {ratio:>5} {len(runs):>3}/{exp:<2} {acc:>8} {cot:>7} {g:>9} {p:>9}")
        if a.loop <= 0:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
