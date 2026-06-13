"""Unit tests for the energy CPU core (CPU-only, no GPU/PDU needed)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from tsmc.energy.core import integrate_power, parse_timestamp, summarize_run


def _samples(powers, *, start=0.0, step=1.0, key="power_draw"):
    """Synthetic monitor samples at fixed cadence, epoch-float timestamps."""
    return [{"timestamp": start + i * step, key: p} for i, p in enumerate(powers)]


# --- integrate_power ---------------------------------------------------------

def test_constant_power_is_exact():
    # 100 W sampled every 1 s for 10 s (t=0..10) -> 1000 J.
    out = integrate_power(_samples([100.0] * 11, step=1.0))
    assert out["energy_j"] == pytest.approx(1000.0)
    assert out["mean_power_w"] == pytest.approx(100.0)
    assert out["peak_power_w"] == 100.0
    assert out["duration_s"] == pytest.approx(10.0)
    assert out["n_samples"] == 11


def test_linear_ramp_is_triangle_area():
    # power 0 -> 100 linearly over 10 s -> area of triangle = 500 J.
    out = integrate_power(_samples([10.0 * i for i in range(11)], step=1.0))
    assert out["energy_j"] == pytest.approx(500.0)
    assert out["peak_power_w"] == 100.0


def test_window_clips_with_interpolation():
    # constant 100 W over [0,10]; integrate only [2,8] -> 600 J.
    out = integrate_power(_samples([100.0] * 11, step=1.0), t0=2.0, t1=8.0)
    assert out["energy_j"] == pytest.approx(600.0)
    assert out["duration_s"] == pytest.approx(6.0)


def test_window_interpolates_ramp_at_fractional_bounds():
    # ramp 0->100 over 10 s; window [1.0, 3.0]. Power(1)=10, Power(3)=30.
    # trapezoid of a straight line = average(10,30)*2 = 40 J.
    out = integrate_power(_samples([10.0 * i for i in range(11)], step=1.0), t0=1.0, t1=3.0)
    assert out["energy_j"] == pytest.approx(40.0)


def test_empty_samples():
    out = integrate_power([])
    assert out["energy_j"] == 0.0
    assert out["mean_power_w"] is None
    assert out["n_samples"] == 0
    assert out["coverage"] == 0.0


def test_single_sample_zero_duration():
    out = integrate_power(_samples([123.0]))
    assert out["energy_j"] == 0.0
    assert out["duration_s"] == 0.0


def test_iso_timestamps_parse_and_integrate():
    base = datetime(2026, 6, 1, 19, 0, 0)
    samples = [{"timestamp": (base + timedelta(seconds=i)).isoformat(), "power_draw": 200.0}
               for i in range(6)]  # 200 W over 5 s -> 1000 J
    out = integrate_power(samples)
    assert out["energy_j"] == pytest.approx(1000.0)
    assert out["n_samples"] == 6


def test_max_gap_detects_stall():
    # samples at t=0,1,2, then a 5 s gap, then t=7,8.
    samples = ([{"timestamp": float(t), "power_draw": 50.0} for t in (0, 1, 2)]
               + [{"timestamp": float(t), "power_draw": 50.0} for t in (7, 8)])
    out = integrate_power(samples)
    assert out["max_gap_s"] == pytest.approx(5.0)


def test_unsorted_samples_are_sorted():
    out = integrate_power(_samples([0.0, 100.0])[::-1])  # reversed input
    assert out["energy_j"] == pytest.approx(50.0)  # trapezoid avg(0,100)*1


def test_missing_power_fields_skipped():
    samples = [{"timestamp": 0.0, "power_draw": 100.0},
               {"timestamp": 1.0},  # no power -> skipped
               {"timestamp": 2.0, "power_draw": 100.0}]
    out = integrate_power(samples)
    assert out["n_samples"] == 2
    assert out["energy_j"] == pytest.approx(200.0)  # 100W over the 2s span


def test_parse_timestamp_accepts_float_and_iso():
    assert parse_timestamp(12.5) == 12.5
    assert parse_timestamp(datetime(2026, 6, 1).isoformat()) == datetime(2026, 6, 1).timestamp()


# --- summarize_run -----------------------------------------------------------

def test_summarize_run_gpu_primary_pdu_secondary():
    gpu = _samples([300.0] * 11, step=1.0)   # 300 W * 10 s = 3000 J
    pdu = _samples([500.0] * 11, step=1.0)    # 500 W * 10 s = 5000 J
    out = summarize_run(gpu_samples=gpu, pdu_samples=pdu, t0=0.0, t1=10.0,
                        n_requests=100, n_output_tokens=6000, run_name="run01")
    assert out["granularity"] == "run"
    assert out["gpu_energy_j"] == pytest.approx(3000.0)
    assert out["pdu_energy_j"] == pytest.approx(5000.0)
    assert out["gpu_energy_per_request_j"] == pytest.approx(30.0)
    assert out["gpu_energy_per_output_token_j"] == pytest.approx(0.5)
    assert out["run_name"] == "run01"


def test_summarize_run_without_pdu():
    gpu = _samples([100.0] * 11, step=1.0)
    out = summarize_run(gpu_samples=gpu, t0=0.0, t1=10.0, n_requests=10)
    assert out["pdu_energy_j"] is None
    assert out["pdu_samples"] == 0
    assert out["gpu_energy_per_request_j"] == pytest.approx(100.0)


def test_summarize_run_no_requests_no_division():
    gpu = _samples([100.0] * 11, step=1.0)
    out = summarize_run(gpu_samples=gpu, t0=0.0, t1=10.0)
    assert "gpu_energy_per_request_j" not in out
    assert "gpu_energy_per_output_token_j" not in out
