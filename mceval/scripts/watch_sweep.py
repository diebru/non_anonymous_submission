#!/usr/bin/env python3
"""Live watcher for the energy sweep. SERVER-ONLY for the GPU read (read-only).

A refreshing dashboard that tells you WHAT IS HAPPENING while run_energy_sweep.py
runs, without touching its files:
  * live GPU power/util/mem/temp for the dedicated card (+ optional PDU watts) -> is
    it decoding (high W/util) or scoring in Docker (GPU idle)?
  * per-gamma progress from the filesystem: pending / generating / inferred / scored
    / done;
  * each finished gamma's result (healthy_acc, format_fail_rate) + energy
    (gpu_energy_j, mean_W, dur_s, J/tok) from score_summary.json / energy_summary.json;
  * the current orchestrator step + live throughput by tailing the sweep log
    (a low toks/s during [infer] is the runaway/destabilization signal).

Usage (server):
    python3 scripts/watch_sweep.py --run-id sft01 --log sweep_sft01.log
    python3 scripts/watch_sweep.py --run-id sft01 --once        # single snapshot
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS  # noqa: E402

_STEP_RE = re.compile(r"^\[(gamma|infer|score|join|energy|done)\b")
_TOKS_RE = re.compile(r"output:\s*([\d.]+)\s*toks/s")
_PROG_RE = re.compile(r"(\d+)/(\d+)\s*\[")


def gpu_snapshot(gpu_index: int) -> dict | None:
    q = "index,power.draw,power.limit,utilization.gpu,memory.used,memory.total,temperature.gpu,pstate"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        for line in out.stdout.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 8 and int(float(f[0])) == gpu_index:
                return {"power": float(f[1]), "plimit": float(f[2]), "util": float(f[3]),
                        "mem_used": float(f[4]), "mem_total": float(f[5]),
                        "temp": float(f[6]), "pstate": f[7]}
    except Exception:  # noqa: BLE001
        return None
    return None


def pdu_snapshot(host: str, community: str, oid: str) -> float | None:
    try:
        out = subprocess.run(["snmpget", "-v2c", "-c", community, host, oid],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0 or "=" not in out.stdout:
            return None
        return float(out.stdout.split("=")[1].replace("INTEGER:", "").strip())
    except Exception:  # noqa: BLE001
        return None


def gamma_status(gdir: pathlib.Path) -> str:
    if not gdir.is_dir():
        return "pending"
    if (gdir / "energy" / "energy_summary.json").is_file():
        return "done"
    if (gdir / "records").is_dir() and any((gdir / "records").glob("*.jsonl")):
        return "scored"
    if (gdir / "result").is_dir() and any((gdir / "result").glob("*.jsonl")):
        return "inferred"
    return "generating"


def gamma_row(gdir: pathlib.Path) -> dict:
    d: dict = {}
    ss = gdir / "score_summary.json"
    if ss.is_file():
        s = json.loads(ss.read_text())
        d["acc"] = s.get("healthy_accuracy")
        d["ffail"] = s.get("healthy_format_fail_rate")
    es = gdir / "energy" / "energy_summary.json"
    if es.is_file():
        e = json.loads(es.read_text())
        d["energy_j"] = e.get("gpu_energy_j")
        d["mean_w"] = e.get("gpu_mean_power_w")
        d["dur_s"] = e.get("run_duration_s")
        d["j_tok"] = e.get("gpu_energy_per_output_token_j")
    return d


def tail_status(log: pathlib.Path | None) -> tuple[str, str]:
    """(last orchestrator step line, last 'X/Y .. toks/s' progress) from the log."""
    if not log or not log.is_file():
        return "", ""
    lines = log.read_text(errors="ignore").splitlines()[-400:]
    step = next((ln for ln in reversed(lines) if _STEP_RE.match(ln.strip())), "")
    prog = ""
    for ln in reversed(lines):
        m = _TOKS_RE.search(ln)
        if m:
            pm = _PROG_RE.search(ln)
            prog = f"{pm.group(0).rstrip(' [') if pm else '?'} @ {m.group(1)} toks/s"
            break
    return step.strip(), prog


def _f(x, nd=3):
    return "  -  " if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render(args, paths, gammas) -> bool:
    base = paths.generations_dir / args.model / args.run_id / args.task / args.split
    g = gpu_snapshot(args.gpu_index)
    pdu = pdu_snapshot(args.pdu_host, args.pdu_community, args.pdu_oid) if args.pdu else None
    step, prog = tail_status(pathlib.Path(args.log) if args.log else None)

    lines = []
    lines.append(f"=== sweep watch | {args.model} {args.task}/{args.split} run={args.run_id} "
                 f"=== {datetime.now().strftime('%H:%M:%S')}")
    if g:
        busy = "GENERATING" if (g["util"] >= 50 or g["power"] >= 0.4 * g["plimit"]) else "idle"
        lines.append(f"GPU{args.gpu_index}  {g['power']:.0f}/{g['plimit']:.0f} W   util {g['util']:.0f}%   "
                     f"mem {g['mem_used'] / 1024:.1f}/{g['mem_total'] / 1024:.0f} GB   "
                     f"{g['temp']:.0f}C  {g['pstate']}   [{busy}]")
    else:
        lines.append(f"GPU{args.gpu_index}  (nvidia-smi unavailable)")
    if args.pdu:
        lines.append(f"PDU   {pdu:.0f} W" if pdu is not None else "PDU   (no reading)")
    if step:
        lines.append(f"step  {step}" + (f"   |  {prog}" if prog else ""))
    lines.append("")
    lines.append(f"{'gamma':>6} {'status':>10} {'acc':>7} {'ffail':>7} "
                 f"{'energy_J':>10} {'mean_W':>7} {'dur_s':>7} {'J/tok':>7}")
    ndone = 0
    for gam in gammas:
        gdir = base / f"gamma{gam:g}"
        st = gamma_status(gdir)
        if st == "done":
            ndone += 1
        r = gamma_row(gdir) if st in ("scored", "done") else {}
        lines.append(f"{gam:>6g} {st:>10} {_f(r.get('acc')):>7} {_f(r.get('ffail')):>7} "
                     f"{_f(r.get('energy_j'), 0):>10} {_f(r.get('mean_w'), 0):>7} "
                     f"{_f(r.get('dur_s'), 1):>7} {_f(r.get('j_tok'), 4):>7}")
    lines.append("")
    lines.append(f"done {ndone}/{len(gammas)}")
    sys.stdout.write(("\033[2J\033[H" if not args.once else "") + "\n".join(lines) + "\n")
    sys.stdout.flush()
    return ndone >= len(gammas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="sft01")
    ap.add_argument("--task", default="generation")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--log", default=None, help="sweep log to tail for step + throughput")
    ap.add_argument("--pdu", action="store_true", help="also poll the PDU each refresh (slower)")
    ap.add_argument("--pdu-host", default="192.0.2.1")
    ap.add_argument("--pdu-community", default="public")
    ap.add_argument("--pdu-oid", default="PowerNet-MIB::ePDUPhaseStatusActivePower.1")
    args = ap.parse_args()

    paths = get_paths()
    gammas = sorted(set(args.gammas), reverse=True)
    if args.once:
        render(args, paths, gammas)
        return 0
    try:
        while True:
            finished = render(args, paths, gammas)
            if finished:
                print("\nsweep complete.")
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n(stopped watching; the sweep keeps running)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
