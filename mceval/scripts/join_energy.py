#!/usr/bin/env python3
"""Join measured energy onto the scored long-format records. CPU-only, re-runnable.

For a (model, run, task, split, gamma), this:
  1. reads the gamma-run's power curves ``energy/<run>_{gpu,pdu}.json`` (written by
     the pollers) + the ``generate_window`` from ``run_meta.json``;
  2. integrates power(t) over that window -> the run-level energy summary
     (tsmc.energy.summarize_run), written to ``energy/energy_summary.json``;
  3. stamps each ``records/<Lang>.jsonl`` row's reserved schema ``energy`` field with
     that summary (granularity="run").

Run AFTER scripts/score_generations.py (which produces ``records/``). The energy is
integrated over the generate() window ONLY, so the McEval Docker scoring -- run in a
separate step with the monitors already stopped -- never contaminates it.

Usage (server, after the monitored inference + scoring of one gamma):
    python3 scripts/join_energy.py --task generation --split test \
        --model qwen2.5-coder-3b-instruct --gamma 0.5
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_BASELINE, MODEL_IDS  # noqa: E402
from tsmc.energy.core import summarize_run  # noqa: E402

TASKS = ("generation", "explanation", "completion")


def _load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _read_rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _count_output_tokens(row: dict) -> int:
    """Decode tokens for a row (the explanation two-pass stores a list)."""
    timing = (row.get("_provenance") or {}).get("timing") or {}
    n = timing.get("n_output_tokens")
    if isinstance(n, list):
        return sum(int(x or 0) for x in n)
    return int(n) if isinstance(n, int) else 0


def load_monitor_meta(paths) -> dict | None:
    """Energy-monitor provenance (interval/gpu_index/PDU target) from run_metadata."""
    for name in ("run_metadata.yaml", "run_metadata.example.yaml"):
        meta = paths.configs_dir / name
        if meta.is_file():
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            if data.get("energy"):
                return data["energy"]
    return None


def join_one(gdir: pathlib.Path, monitor: dict | None = None) -> dict:
    """Integrate a gamma-run's curves over its generate_window and stamp records.

    Returns the energy summary. Raises if records/ or run_meta.json are missing
    (scoring/inference must have run first)."""
    run_meta = _load_json(gdir / "run_meta.json")
    if run_meta is None:
        raise SystemExit(f"no run_meta.json in {gdir} (run inference first)")
    window = run_meta.get("generate_window") or [None, None]

    records_dir = gdir / "records"
    rec_files = sorted(records_dir.glob("*.jsonl")) if records_dir.is_dir() else []
    if not rec_files:
        raise SystemExit(f"no records/ in {gdir} (run score_generations first)")

    rows_by_file: dict[pathlib.Path, list[dict]] = {}
    n_requests = 0
    n_output_tokens = 0
    for f in rec_files:
        rows = _read_rows(f)
        rows_by_file[f] = rows
        n_requests += len(rows)
        n_output_tokens += sum(_count_output_tokens(r) for r in rows)

    energy_dir = gdir / "energy"
    gpu_files = sorted(energy_dir.glob("*_gpu.json")) if energy_dir.is_dir() else []
    pdu_files = sorted(energy_dir.glob("*_pdu.json")) if energy_dir.is_dir() else []
    if not gpu_files:
        raise SystemExit(
            f"no *_gpu.json in {energy_dir} (run the monitors around inference first)")
    gpu_samples = _load_json(gpu_files[0]) or []
    pdu_samples = (_load_json(pdu_files[0]) or []) if pdu_files else []
    run_name = gpu_files[0].name[: -len("_gpu.json")]

    summary = summarize_run(
        gpu_samples=gpu_samples, pdu_samples=pdu_samples,
        t0=window[0], t1=window[1],
        n_requests=n_requests or None, n_output_tokens=n_output_tokens or None,
        run_name=run_name, monitor=monitor,
    )
    energy_dir.mkdir(parents=True, exist_ok=True)
    (energy_dir / "energy_summary.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")
    # stamp the reserved schema `energy` field on every record (same run-level dict)
    for f, rows in rows_by_file.items():
        for r in rows:
            r["energy"] = summary
        with open(f, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=(*TASKS, "all"), default="generation")
    ap.add_argument("--split", choices=("train", "test", "both"), default="test")
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--gamma", type=float, default=GAMMA_BASELINE)
    ap.add_argument("--run-id", default="run01")
    args = ap.parse_args()

    paths = get_paths()
    monitor = load_monitor_meta(paths)
    tasks = list(TASKS) if args.task == "all" else [args.task]
    splits = ["train", "test"] if args.split == "both" else [args.split]

    rc = 0
    for task in tasks:
        for split in splits:
            gdir = (paths.generations_dir / args.model / args.run_id / task / split
                    / f"gamma{args.gamma:g}")
            if not gdir.is_dir():
                print(f"[skip] {task}/{split}: no {gdir}")
                continue
            summ = join_one(gdir, monitor=monitor)
            print(f"[energy] {task}/{split} gamma={args.gamma:g}: "
                  f"gpu_energy={summ['gpu_energy_j']:.1f} J "
                  f"(mean {summ['gpu_mean_power_w']:.0f} W over {summ['run_duration_s']:.1f} s, "
                  f"{summ['gpu_samples']} samples) "
                  f"per_req={summ.get('gpu_energy_per_request_j')} "
                  f"per_tok={summ.get('gpu_energy_per_output_token_j')}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
