"""Energy accounting for the TokenSkip x McEval sweep (roadmap s7/s8).

CPU core (``core``) integrates monitor power time-series into the run-level energy
dict stamped onto each long-format record's reserved ``energy`` field. The GPU/PDU
pollers and the ``join_energy`` driver are server-side; this package's core is
stdlib-only and unit-testable locally.
"""
from __future__ import annotations

from tsmc.energy.core import integrate_power, parse_timestamp, summarize_run

__all__ = ["integrate_power", "summarize_run", "parse_timestamp"]
