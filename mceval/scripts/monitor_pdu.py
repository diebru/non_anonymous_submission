#!/usr/bin/env python3
"""PDU active-power poller for energy measurement. SERVER-ONLY (needs snmpget).

Samples the rack PDU's active power via SNMP at a fixed interval and writes a JSON
array of ``{timestamp, power_draw}`` dicts to ``<output-dir>/<run-name>_pdu.json``.
``tsmc.energy.core`` integrates it as the **secondary** (node-level) energy signal;
GPU power.draw is the primary, per-card one. On a single-tenant node the PDU is a
useful whole-machine cross-check.

Hardened vs the reference ``example_energy/monitor_pdu.py``: the SNMP target
(host / community / OID) is configurable instead of hardcoded -- defaults reproduce
the reference monitor exactly (192.0.2.1 / public / ePDUPhaseStatusActivePower.1).

Run wrapped around INFERENCE ONLY -- never around the McEval Docker eval.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime


class PDUMonitor:
    def __init__(self, output_dir, host, community, oid):
        self.output_dir = output_dir
        self.host = host
        self.community = community
        self.oid = oid
        self.running = False
        self.stats = []
        self.run_name = None
        self.error_count = 0
        os.makedirs(output_dir, exist_ok=True)

    def get_pdu_stats(self):
        try:
            cmd = ["snmpget", "-v2c", "-c", self.community, self.host, self.oid]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                self.error_count += 1
                if self.error_count <= 3:
                    print(f"[PDU] SNMP error: {result.stderr.strip()}", file=sys.stderr)
                return None
            response = result.stdout.strip()
            if not response:
                print("[PDU] Empty SNMP response", file=sys.stderr)
                return None
            # e.g. "PowerNet-MIB::ePDUPhaseStatusActivePower.1 = INTEGER: 73"
            value_str = response.split("=")[1].strip()
            power_value = float(value_str.replace("INTEGER:", "").strip())
            if self.error_count > 0:
                print(f"[PDU] Recovered after {self.error_count} errors", file=sys.stderr)
                self.error_count = 0
            return {"timestamp": datetime.now().isoformat(), "power_draw": power_value}
        except subprocess.TimeoutExpired:
            self.error_count += 1
            if self.error_count == 1:
                print("[PDU] SNMP timeout - PDU not responding", file=sys.stderr)
            return None
        except Exception as e:                                    # noqa: BLE001
            self.error_count += 1
            if self.error_count <= 3:
                print(f"[PDU] Exception: {e}", file=sys.stderr)
            return None

    def start_monitoring(self, run_name, interval):
        self.running = True
        self.stats = []
        self.run_name = run_name
        print(f"[PDU] Starting monitoring: {run_name} ({self.host} {self.oid}, "
              f"interval {interval}s)", file=sys.stderr)
        probe = self.get_pdu_stats()
        if probe is None:
            print("[PDU] WARNING: initial PDU query failed - will keep trying", file=sys.stderr)
        else:
            print(f"[PDU] Initial reading: {probe['power_draw']}W", file=sys.stderr)
        try:
            while self.running:
                s = self.get_pdu_stats()
                if s:
                    self.stats.append(s)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("[PDU] Received interrupt", file=sys.stderr)
        finally:
            self.stop_monitoring()

    def stop_monitoring(self):
        self.running = False
        print(f"[PDU] Stopping... collected {len(self.stats)} samples", file=sys.stderr)
        if self.error_count > 0:
            print(f"[PDU] Total SNMP errors: {self.error_count}", file=sys.stderr)
        if self.stats and self.run_name:
            path = os.path.join(self.output_dir, f"{self.run_name}_pdu.json")
            with open(path, "w") as f:
                json.dump(self.stats, f, indent=2)
            print(f"PDU stats saved to {path}")
        else:
            print(f"[PDU] No data to save (stats={len(self.stats)})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Sample PDU active power for energy measurement")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--host", default="192.0.2.1", help="PDU SNMP host/IP")
    ap.add_argument("--community", default="public", help="SNMP v2c community")
    ap.add_argument("--oid", default="PowerNet-MIB::ePDUPhaseStatusActivePower.1",
                    help="active-power OID")
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()

    monitor = PDUMonitor(args.output_dir, args.host, args.community, args.oid)

    def _stop(signum, frame):
        print(f"[PDU] Received signal {signum}", file=sys.stderr)
        monitor.running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    monitor.start_monitoring(args.run_name, args.interval)


if __name__ == "__main__":
    main()
