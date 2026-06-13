"""Context manager that wraps an inference window with the GPU + PDU pollers.

SERVER-ONLY (spawns the pollers, which need nvidia-smi / snmpget). Used by the
Step-3 sweep orchestrator to bracket exactly the ``run_inference`` call for one
gamma: start the pollers, run inference, then SIGINT the pollers so they flush
their JSON. The McEval Docker eval runs AFTER the ``with`` block closes, so the
accuracy control stays outside the energy window. ``join_energy`` then integrates
the saved curves over the recorded generate() window.

Lazy/stdlib only (subprocess) -> importable on CPU; nothing runs until ``__enter__``.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tsmc.config import find_repo_root

DEFAULT_PDU_HOST = "192.0.2.1"
DEFAULT_PDU_COMMUNITY = "public"
DEFAULT_PDU_OID = "PowerNet-MIB::ePDUPhaseStatusActivePower.1"


class EnergyMonitors:
    """Spawn ``monitor_gpu.py`` (+ ``monitor_pdu.py``) for the duration of a ``with``.

    ``output_dir`` receives ``<run_name>_gpu.json`` / ``<run_name>_pdu.json``. On
    exit the pollers are SIGINT'd (graceful flush), then waited on / escalated.
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str | os.PathLike[str],
        *,
        gpu_index: int = 0,
        interval: float = 0.5,
        pdu: bool = True,
        pdu_host: str = DEFAULT_PDU_HOST,
        pdu_community: str = DEFAULT_PDU_COMMUNITY,
        pdu_oid: str = DEFAULT_PDU_OID,
        python_exe: str | None = None,
        scripts_dir: str | os.PathLike[str] | None = None,
        settle_s: float = 1.0,
        stop_timeout_s: float = 15.0,
    ):
        self.run_name = run_name
        self.output_dir = str(output_dir)
        self.gpu_index = gpu_index
        self.interval = interval
        self.pdu = pdu
        self.pdu_host = pdu_host
        self.pdu_community = pdu_community
        self.pdu_oid = pdu_oid
        self.python = python_exe or sys.executable
        self.scripts_dir = Path(scripts_dir) if scripts_dir else find_repo_root() / "scripts"
        self.settle_s = settle_s
        self.stop_timeout_s = stop_timeout_s
        self._procs: list[tuple[str, subprocess.Popen]] = []

    def __enter__(self) -> "EnergyMonitors":
        os.makedirs(self.output_dir, exist_ok=True)
        gpu_cmd = [
            self.python, str(self.scripts_dir / "monitor_gpu.py"),
            "--run-name", self.run_name, "--output-dir", self.output_dir,
            "--gpu-index", str(self.gpu_index), "--interval", str(self.interval),
        ]
        self._procs.append(("gpu", subprocess.Popen(gpu_cmd)))
        if self.pdu:
            pdu_cmd = [
                self.python, str(self.scripts_dir / "monitor_pdu.py"),
                "--run-name", self.run_name, "--output-dir", self.output_dir,
                "--host", self.pdu_host, "--community", self.pdu_community,
                "--oid", self.pdu_oid, "--interval", str(self.interval),
            ]
            self._procs.append(("pdu", subprocess.Popen(pdu_cmd)))
        if self.settle_s:
            time.sleep(self.settle_s)   # let the first samples bracket the work
        return self

    def __exit__(self, *exc: Any) -> bool:
        # graceful stop first (the pollers flush their JSON on SIGINT)
        for _name, p in self._procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for _name, p in self._procs:
            try:
                p.wait(timeout=self.stop_timeout_s)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        return False  # never suppress an exception from the wrapped block

    def output_files(self) -> dict[str, str]:
        base = Path(self.output_dir)
        return {
            "gpu": str(base / f"{self.run_name}_gpu.json"),
            "pdu": str(base / f"{self.run_name}_pdu.json") if self.pdu else None,
        }
