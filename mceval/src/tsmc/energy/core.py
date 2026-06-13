"""Energy accounting: integrate monitor power time-series into run-level energy.

CPU-only, stdlib-only -> unit-testable locally with synthetic samples (no GPU, no
PDU). The pollers (``scripts/monitor_gpu.py`` / ``scripts/monitor_pdu.py``) write a
JSON array of ``{timestamp, power_draw, ...}`` samples while inference runs; this
module integrates power(t) -> energy (Joules) over the recorded ``generate()``
window and builds the run-level summary stamped onto each long-format record's
reserved ``energy`` field (roadmap s7).

Why **run-level** (per model x gamma x task x split), not per-request: vLLM
continuous-batching runs many generations concurrently, so the sampled power can't
be cleanly attributed to one request. Goals 2/3 are per-gamma claims anyway, so we
integrate the whole run and join on the run's aggregate ``cot_token_count``.
GPU energy is primary (per-GPU, attributable); PDU is node-level context.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence


def parse_timestamp(ts: str | int | float) -> float:
    """Monitor timestamps are ``datetime.now().isoformat()`` strings; epoch
    floats are also accepted (so synthetic tests can pass plain numbers)."""
    if isinstance(ts, (int, float)):
        return float(ts)
    return datetime.fromisoformat(ts).timestamp()


def _interp(points: list[tuple[float, float]], t: float) -> float:
    """Power at time ``t`` by linear interpolation; clamps outside the range."""
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    for (ta, pa), (tb, pb) in zip(points, points[1:]):
        if ta <= t <= tb:
            return pa if tb == ta else pa + (pb - pa) * (t - ta) / (tb - ta)
    return points[-1][1]


def integrate_power(
    samples: Sequence[dict[str, Any]],
    *,
    t0: float | None = None,
    t1: float | None = None,
    ts_key: str = "timestamp",
    power_key: str = "power_draw",
) -> dict[str, Any]:
    """Trapezoidal integral of power(t) over ``[t0, t1]`` -> energy + diagnostics.

    ``samples``: dicts each carrying a timestamp + a power reading (W). ``t0``/``t1``
    clip the integration window (epoch seconds); ``None`` -> first/last sample time.
    Window endpoints are linearly interpolated from neighbouring samples so a window
    landing between samples integrates correctly.

    Returns ``energy_j``, ``mean_power_w``, ``peak_power_w``, ``duration_s``,
    ``n_samples`` (raw samples used), ``max_gap_s`` (largest inter-sample gap -- a
    monitor-stall signal), and ``coverage`` (fraction of the window actually spanned
    by samples). Empty input -> zero energy with null stats.
    """
    points = sorted(
        (parse_timestamp(s[ts_key]), float(s[power_key]))
        for s in samples
        if s.get(ts_key) is not None and s.get(power_key) is not None
    )
    if not points:
        return {"energy_j": 0.0, "mean_power_w": None, "peak_power_w": None,
                "duration_s": 0.0, "n_samples": 0, "max_gap_s": None, "coverage": 0.0}

    lo = points[0][0] if t0 is None else float(t0)
    hi = points[-1][0] if t1 is None else float(t1)
    if hi < lo:
        lo, hi = hi, lo

    # Clip to the window with interpolated endpoints (so partial edges count).
    inner = [(t, p) for (t, p) in points if lo < t < hi]
    series = [(lo, _interp(points, lo)), *inner, (hi, _interp(points, hi))]

    energy = 0.0
    max_gap = 0.0
    for (ta, pa), (tb, pb) in zip(series, series[1:]):
        dt = tb - ta
        if dt <= 0:
            continue
        energy += 0.5 * (pa + pb) * dt   # trapezoid: W * s = J
        max_gap = max(max_gap, dt)

    duration = hi - lo
    peak = max(p for _, p in series)
    # coverage: span between the first and last ACTUAL samples inside the window.
    covered = (inner[-1][0] - inner[0][0]) if inner else 0.0
    coverage = (covered / duration) if duration > 0 else 0.0
    return {
        "energy_j": energy,
        "mean_power_w": (energy / duration) if duration > 0 else None,
        "peak_power_w": peak,
        "duration_s": duration,
        "n_samples": len(points),
        "max_gap_s": max_gap if len(series) > 1 else None,
        "coverage": min(coverage, 1.0),
    }


def summarize_run(
    *,
    gpu_samples: Sequence[dict[str, Any]] | None = None,
    pdu_samples: Sequence[dict[str, Any]] | None = None,
    t0: float | None = None,
    t1: float | None = None,
    n_requests: int | None = None,
    n_output_tokens: int | None = None,
    run_name: str | None = None,
    monitor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the run-level energy dict joined onto every record of one gamma-run.

    Integrates the GPU (primary) and PDU (secondary) curves over the generate()
    window ``[t0, t1]`` and derives per-request / per-output-token energy (the run
    total divided by counts -- a convenience, since attribution is run-level).
    ``granularity`` is always ``"run"``: the same dict is stamped on every row of
    the run, and the curve for Goals 2/3 is built by aggregating per gamma.
    """
    gpu = integrate_power(gpu_samples or [], t0=t0, t1=t1)
    has_pdu = bool(pdu_samples)
    pdu = integrate_power(pdu_samples or [], t0=t0, t1=t1) if has_pdu else None

    out: dict[str, Any] = {
        "granularity": "run",
        "run_name": run_name,
        "window": [t0, t1],
        "run_duration_s": gpu["duration_s"] or (pdu["duration_s"] if pdu else 0.0),
        "n_requests": n_requests,
        "n_output_tokens": n_output_tokens,
        "gpu_energy_j": gpu["energy_j"],
        "gpu_mean_power_w": gpu["mean_power_w"],
        "gpu_peak_power_w": gpu["peak_power_w"],
        "gpu_samples": gpu["n_samples"],
        "gpu_max_gap_s": gpu["max_gap_s"],
        "gpu_coverage": gpu["coverage"],
        "pdu_energy_j": pdu["energy_j"] if pdu else None,
        "pdu_mean_power_w": pdu["mean_power_w"] if pdu else None,
        "pdu_samples": pdu["n_samples"] if pdu else 0,
        "monitor": monitor,
    }
    if n_requests:
        out["gpu_energy_per_request_j"] = gpu["energy_j"] / n_requests
        if pdu:
            out["pdu_energy_per_request_j"] = pdu["energy_j"] / n_requests
    if n_output_tokens:
        out["gpu_energy_per_output_token_j"] = gpu["energy_j"] / n_output_tokens
    return out
