#!/usr/bin/env python3
"""GPU power/metrics poller for energy measurement. SERVER-ONLY (needs nvidia-smi).

Samples ``nvidia-smi --query-gpu=power.draw,...`` at a fixed interval and writes a
JSON array of per-sample dicts to ``<output-dir>/<run-name>_gpu.json``.
``tsmc.energy.core.integrate_power`` then integrates ``power_draw``(t) over the
recorded ``generate()`` window -> Joules.

Hardened vs the reference ``example_energy/monitor_gpu.py``: that version did
``stdout.strip().split(',')`` on the WHOLE nvidia-smi output, so with >1 GPU it
mixed rows and only ever half-worked. This one parses **per line** and records only
the ``--gpu-index`` row, so the energy sweep can pin one dedicated card and ignore
the rest.

Run wrapped around INFERENCE ONLY -- never around the McEval Docker eval: the
accuracy control must stay outside the energy window.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

# Column order requested from nvidia-smi (18 fields).
QUERY = (
    "timestamp,index,name,power.draw,power.limit,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,temperature.gpu,clocks.current.graphics,"
    "clocks.current.memory,clocks.max.graphics,clocks.max.memory,fan.speed,"
    "compute_mode,driver_version,pstate"
)
_NAMES = QUERY.split(",")
# field name -> (output key, caster); 'timestamp'/'index'/'name'/strings handled below
_FLOATS = {
    "power.draw": "power_draw", "power.limit": "power_limit",
    "utilization.gpu": "gpu_utilization", "utilization.memory": "memory_utilization",
    "memory.used": "memory_used", "memory.total": "memory_total",
    "temperature.gpu": "temperature", "clocks.current.graphics": "graphics_clock",
    "clocks.current.memory": "memory_clock", "clocks.max.graphics": "max_graphics_clock",
    "clocks.max.memory": "max_memory_clock", "fan.speed": "fan_speed",
}
_STRINGS = {"name": "name", "compute_mode": "compute_mode",
            "driver_version": "driver_version", "pstate": "pstate"}


class GPUMonitor:
    def __init__(self, output_dir, gpu_index):
        self.output_dir = output_dir
        self.gpu_index = gpu_index
        self.running = False
        self.stats = []
        self.run_name = None
        os.makedirs(output_dir, exist_ok=True)

    def get_gpu_stats(self):
        """One sample for the configured --gpu-index (physical nvidia-smi index)."""
        try:
            cmd = ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print(f"[GPU] nvidia-smi error: {result.stderr}", file=sys.stderr)
                return None
            for line in result.stdout.strip().splitlines():       # one row per GPU
                fields = [f.strip() for f in line.split(",")]
                if len(fields) < len(_NAMES):
                    continue
                row = dict(zip(_NAMES, fields))
                if int(float(row["index"])) != self.gpu_index:
                    continue
                out = {"timestamp": datetime.now().isoformat(), "index": self.gpu_index}
                for src, dst in _FLOATS.items():
                    try:
                        out[dst] = float(row[src])
                    except (ValueError, KeyError):
                        out[dst] = None
                for src, dst in _STRINGS.items():
                    out[dst] = row.get(src)
                return out
            print(f"[GPU] gpu-index {self.gpu_index} not present in nvidia-smi output",
                  file=sys.stderr)
            return None
        except Exception as e:                                    # noqa: BLE001
            print(f"[GPU] Exception: {e}", file=sys.stderr)
            return None

    def start_monitoring(self, run_name, interval):
        self.running = True
        self.stats = []
        self.run_name = run_name
        print(f"[GPU] Starting monitoring: {run_name} (gpu-index {self.gpu_index}, "
              f"interval {interval}s)", file=sys.stderr)
        try:
            while self.running:
                s = self.get_gpu_stats()
                if s:
                    self.stats.append(s)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("[GPU] Received interrupt", file=sys.stderr)
        finally:
            self.stop_monitoring()

    def stop_monitoring(self):
        self.running = False
        print(f"[GPU] Stopping... collected {len(self.stats)} samples", file=sys.stderr)
        if self.stats and self.run_name:
            path = os.path.join(self.output_dir, f"{self.run_name}_gpu.json")
            with open(path, "w") as f:
                json.dump(self.stats, f, indent=2)
            print(f"GPU stats saved to {path}")
        else:
            print(f"[GPU] No data to save (stats={len(self.stats)})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Sample GPU power for energy measurement")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--gpu-index", type=int, default=0,
                    help="physical nvidia-smi index of the dedicated GPU (== CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    monitor = GPUMonitor(args.output_dir, args.gpu_index)

    def _stop(signum, frame):
        print(f"[GPU] Received signal {signum}", file=sys.stderr)
        monitor.running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    monitor.start_monitoring(args.run_name, args.interval)


if __name__ == "__main__":
    main()
