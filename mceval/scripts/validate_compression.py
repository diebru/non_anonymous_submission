#!/usr/bin/env python3
"""Phase-2 gate: validate the written compressed corpus (roadmap Phase 2). CPU.

Independent re-check of whatever scripts/compress_corpus.py wrote, so the
completion criteria can be re-run after the fact:

  monotonicity   per-trajectory measured cot_token_count NON-INCREASING as gamma
                 falls (ties allowed; strict increases counted), and the per-gamma
                 MEDIAN series strictly non-increasing (the aggregate headline).
  scaffolding    code_snippet identical across all gammas of a trajectory, no
                 sentinel inside any cot_text, every variant passes validate_record,
                 cot_origin == original iff gamma == 1.0.

Reads only the gitignored compressed/ tree -- no Docker, no tokenizer. Exit code is
non-zero if the aggregate-monotonicity or scaffolding gate fails (per-trajectory
ties/inversions on short CoTs are reported, not fatal).

Usage:
    python3 scripts/validate_compression.py --model qwen2.5-coder-3b-instruct
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.compression.corpus import (  # noqa: E402
    aggregate_monotonic,
    aggregate_token_medians,
    trajectory_monotonic,
)
from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import MODEL_IDS, SENTINEL  # noqa: E402
from tsmc.schema import validate_record  # noqa: E402

# Per-trajectory non-increase is soft (re-tokenized short CoTs legitimately tie or
# wobble near gamma=1.0); flag only if the corpus-wide fraction drops below this.
TRAJ_MONO_FLOOR = 0.80


def trajectory_key(r: dict) -> tuple:
    return (r.get("problem_id"), r.get("task_type"), r.get("completion_subtype"), r.get("run_id"))


def load_task_split(out_root: pathlib.Path, task: str, split: str) -> list[dict]:
    base = out_root / task / split
    if not base.is_dir():
        return []
    rows: list[dict] = []
    for gdir in sorted(base.glob("gamma*")):
        for f in sorted(gdir.glob("*.jsonl")):
            rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def validate_task_split(rows: list[dict]) -> dict:
    by_traj: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_traj[trajectory_key(r)].append(r)

    schema_errors = scaffolding_errors = origin_errors = 0
    traj_total = traj_nonincreasing = 0
    tokens_by_gamma: dict[float, list[int]] = defaultdict(list)
    examples: list[str] = []

    for key, variants in by_traj.items():
        codes = {v.get("code_snippet") for v in variants}
        if len(codes) > 1:
            scaffolding_errors += 1
            if len(examples) < 8:
                examples.append(f"code_snippet differs across gammas for {key}")
        for v in variants:
            tokens_by_gamma[float(v["gamma"])].append(int(v["cot_token_count"]))
            if SENTINEL in (v.get("cot_text") or ""):
                scaffolding_errors += 1
                if len(examples) < 8:
                    examples.append(f"sentinel in cot_text for {key} g={v['gamma']:g}")
            is_baseline = float(v["gamma"]) >= 1.0
            if v.get("cot_origin") != ("original" if is_baseline else "compressed"):
                origin_errors += 1
                if len(examples) < 8:
                    examples.append(f"cot_origin wrong for {key} g={v['gamma']:g}")
            errs = validate_record(v)
            if errs:
                schema_errors += 1
                if len(examples) < 8:
                    examples.append(f"schema {key} g={v['gamma']:g}: {errs[:2]}")

        traj_total += 1
        if trajectory_monotonic(variants)["monotonic"]:
            traj_nonincreasing += 1

    medians = aggregate_token_medians(tokens_by_gamma)
    agg = aggregate_monotonic(medians)
    traj_frac = (traj_nonincreasing / traj_total) if traj_total else None

    return {
        "n_records": len(rows),
        "n_trajectories": traj_total,
        "schema_errors": schema_errors,
        "scaffolding_errors": scaffolding_errors,
        "cot_origin_errors": origin_errors,
        "median_tokens_by_gamma": [[g, m] for g, m in medians],
        "aggregate_monotonic": agg["monotonic"],
        "aggregate_violations": agg["violations"],
        "trajectory_monotonic_fraction": round(traj_frac, 4) if traj_frac is not None else None,
        "examples": examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--tasks", nargs="+", default=["generation", "explanation"])
    ap.add_argument("--split", choices=("train", "test", "both"), default="train")
    args = ap.parse_args()
    paths = get_paths()
    out_root = paths.compressed_dir / args.model / args.run_id
    splits = ["train", "test"] if args.split == "both" else [args.split]

    print("=" * 70)
    print(f"Phase-2 validation | {out_root}")
    print("=" * 70)

    report: dict = {"model_id": args.model, "run_id": args.run_id, "tasks": {}}
    gate_ok = True
    any_data = False
    for task in args.tasks:
        for split in splits:
            rows = load_task_split(out_root, task, split)
            if not rows:
                continue
            any_data = True
            res = validate_task_split(rows)
            report["tasks"][f"{task}/{split}"] = res

            hard_ok = (res["schema_errors"] == 0 and res["scaffolding_errors"] == 0
                       and res["cot_origin_errors"] == 0 and res["aggregate_monotonic"])
            traj_frac = res["trajectory_monotonic_fraction"]
            traj_ok = traj_frac is None or traj_frac >= TRAJ_MONO_FLOOR
            gate_ok = gate_ok and hard_ok and traj_ok

            print(f"\n[{task}/{split}] records={res['n_records']} trajectories={res['n_trajectories']}")
            print("  median cot_tokens by gamma (desc): "
                  + ", ".join(f"{g:g}:{m:.0f}" for g, m in res["median_tokens_by_gamma"]))
            print(f"  aggregate monotonic : {'OK' if res['aggregate_monotonic'] else 'VIOLATED ' + str(res['aggregate_violations'])}")
            print(f"  per-traj non-increasing fraction: {traj_frac} "
                  f"(floor {TRAJ_MONO_FLOOR}) -> {'OK' if traj_ok else 'LOW'}")
            print(f"  schema_errors={res['schema_errors']} scaffolding_errors={res['scaffolding_errors']} "
                  f"cot_origin_errors={res['cot_origin_errors']}")
            if res["examples"]:
                print("  examples:")
                for e in res["examples"]:
                    print(f"    - {e}")

    if not any_data:
        print("\nNo compressed records found -- run scripts/compress_corpus.py first.")
        return 2

    report["gate_pass"] = gate_ok
    (out_root / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nGATE: {'PASS' if gate_ok else 'FAIL'}  (report -> {out_root / 'validation_report.json'})")
    print("=" * 70)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
