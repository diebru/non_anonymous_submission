#!/usr/bin/env python3
"""Energy-instrumented gamma sweep (Phase 4, Step 3). SERVER-ONLY (GPU + Docker).

For each gamma in the grid, on ONE dedicated GPU:
  1. wrap INFERENCE with the GPU + PDU pollers (tsmc.energy.EnergyMonitors) and run
     scripts/run_inference.py on the MERGED model at that gamma;
  2. stop the monitors, THEN run scripts/score_generations.py (McEval Docker) -- the
     accuracy control runs with the monitors already stopped, so it stays outside the
     energy window;
  3. run scripts/join_energy.py to integrate the curves over the generate() window and
     stamp each record's energy field.

Reload-per-gamma (a fresh run_inference subprocess each time, like the reference
eval_inference_example.sh) keeps each gamma's energy clean; model-load is excluded by
the generate_window anyway. The sweep is generation-only (that's the SFT'd task;
explanation is post-hoc on the base model, completion has no lever).

Usage (server, tokenskip_env; do a smoke first, then the full grid):
    # 1-gamma trio smoke through the whole chain
    python3 scripts/run_energy_sweep.py --gammas 1.0 --trio-only --limit 5 \
        --model-path "$PWD/weights/qwen2.5-coder-3b-instruct/merged_sft_run01"

    # full 12-gamma sweep on the dedicated GPU
    python3 scripts/run_energy_sweep.py \
        --model-path "$PWD/weights/qwen2.5-coder-3b-instruct/merged_sft_run01"

    # faithful TokenSkip budget (per-gamma cap = int(base * gamma)); one base per run-id.
    # NOTE: this is the SCALED axis -- distinct from the FIXED-budget _mt1024/_mt512 cells.
    python3 scripts/run_energy_sweep.py --scale-by-gamma --max-tokens 2048 --run-id sft01_sg2048 \
        --model-path "$PWD/weights/qwen2.5-coder-3b-instruct/merged_sft_run01"
    python3 scripts/run_energy_sweep.py --scale-by-gamma --max-tokens 1024 --run-id sft01_sg1024 \
        --model-path "$PWD/weights/qwen2.5-coder-3b-instruct/merged_sft_run01"
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS  # noqa: E402
from tsmc.eval import docker  # noqa: E402
from tsmc.energy.monitors import (  # noqa: E402
    DEFAULT_PDU_COMMUNITY, DEFAULT_PDU_HOST, DEFAULT_PDU_OID, EnergyMonitors,
)

HERE = pathlib.Path(__file__).resolve().parent


def _energy_cfg(paths) -> dict:
    """Energy block from run_metadata.yaml (else the .example), else {}."""
    for name in ("run_metadata.yaml", "run_metadata.example.yaml"):
        meta = paths.configs_dir / name
        if meta.is_file():
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            if data.get("energy"):
                return data["energy"]
    return {}


def _gdir(paths, model, run_id, task, split, gamma):
    return paths.generations_dir / model / run_id / task / split / f"gamma{gamma:g}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--model-path", default=None,
                    help="merged checkpoint dir (default: weights/<model>/merged_sft_run01)")
    ap.add_argument("--task", default="generation",
                    help="SFT'd task to sweep (generation; the energy curve task)")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--trio-only", action="store_true", help="Python/C/Rust only (smoke)")
    ap.add_argument("--limit", type=int, default=0, help="problems per language (0=all)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--scale-by-gamma", action="store_true",
                    help="Faithful TokenSkip budget: per-gamma output cap = int(--max-tokens * gamma), "
                         "so --max-tokens is the BASE cap (the gamma=1.0 budget). Default OFF = fixed cap "
                         "across gammas. Use a base-tagged --run-id per base (e.g. run01_b2048).")
    # decoding-config matrix knobs (docs/EXPERIMENTS.md); default OFF -> baseline behavior.
    ap.add_argument("--frequency-penalty", type=float, default=0.0,
                    help="vLLM frequency_penalty (count-scaled; >0 throttles runaway loops)")
    ap.add_argument("--presence-penalty", type=float, default=0.0, help="vLLM presence_penalty")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="vLLM repetition_penalty (1.0 = off)")
    ap.add_argument("--system", default=None,
                    help="pinned system prompt (must match the model's SFT contract; e.g. 7B reason-first)")
    # energy / GPU pinning (defaults from run_metadata energy block)
    ap.add_argument("--gpu-index", type=int, default=None, help="physical GPU id to dedicate")
    ap.add_argument("--interval", type=float, default=None, help="poller sample interval (s)")
    ap.add_argument("--no-pdu", action="store_true", help="GPU energy only (skip SNMP/PDU)")
    # scoring
    ap.add_argument("--digest", default=None, help="McEval image sha256 (else run_metadata)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a gamma whose records/ already exist (resume a sweep)")
    ap.add_argument("--force", action="store_true",
                    help="allow writing into a run-id that already has records (overwrite)")
    args = ap.parse_args()

    paths = get_paths()
    ecfg = _energy_cfg(paths)
    gpu_index = args.gpu_index if args.gpu_index is not None else int(ecfg.get("gpu_index", 0))
    interval = args.interval if args.interval is not None else float(ecfg.get("sample_interval_s", 0.5))
    pdu_host = ecfg.get("pdu_host", DEFAULT_PDU_HOST)
    pdu_community = ecfg.get("snmp_community", DEFAULT_PDU_COMMUNITY)
    pdu_oid = ecfg.get("snmp_oid", DEFAULT_PDU_OID)

    model_path = args.model_path or str(paths.weights_dir / args.model / "merged_sft_run01")
    if not pathlib.Path(model_path, "config.json").is_file():
        print(f"ERROR: no merged checkpoint at {model_path} (config.json missing). "
              f"Pass --model-path or run scripts/merge_lora.py first.", file=sys.stderr)
        return 2

    # Resolve the McEval digest UP FRONT -> abort before any (expensive) inference if
    # scoring would fail for lack of it (this is what wasted the first smoke).
    digest = args.digest or docker.load_digest_from_metadata(paths)
    if not digest:
        print("ERROR: no McEval Docker digest. Pass --digest sha256:... or set "
              "mceval.docker_digest in configs/run_metadata.yaml. Aborting before "
              "inference so a 12-gamma sweep is not wasted on unscored runs.", file=sys.stderr)
        return 2

    gammas = sorted(set(args.gammas), reverse=True)  # 1.0 -> 0.1

    # DATA-SAFETY GUARD: never silently overwrite a populated run-id (e.g. the finished
    # sft01 baseline). A new decoding-config cell MUST use a new --run-id; --skip-existing
    # resumes, --force overwrites on purpose. (docs/EXPERIMENTS.md run-id convention.)
    if not (args.skip_existing or args.force):
        def _has_records(g):
            rd = _gdir(paths, args.model, args.run_id, args.task, args.split, g) / "records"
            return rd.is_dir() and any(rd.glob("*.jsonl"))
        clobber = [g for g in gammas if _has_records(g)]
        if clobber:
            print(f"ERROR: run-id '{args.run_id}' already has records for gammas "
                  f"{[f'{g:g}' for g in clobber]} ({args.model} {args.task}/{args.split}). "
                  f"Use a NEW --run-id (or --skip-existing to resume / --force to overwrite).",
                  file=sys.stderr)
            return 2

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index))
    print("=" * 72)
    print(f"Energy sweep | model={args.model} task={args.task} split={args.split}")
    print(f"merged={model_path}")
    print(f"gammas={gammas} gpu_index={gpu_index} interval={interval}s pdu={not args.no_pdu}")
    print(f"trio_only={args.trio_only} limit={args.limit} run_id={args.run_id}")
    if args.scale_by_gamma:
        budgets = {f"{g:g}": max(1, int(args.max_tokens * g)) for g in gammas}
        print(f"budget=faithful int(base*gamma) base={args.max_tokens} -> {budgets}")
    else:
        print(f"budget=fixed max_tokens={args.max_tokens}")
    print("=" * 72)

    statuses: list[dict] = []
    for g in gammas:
        gdir = _gdir(paths, args.model, args.run_id, args.task, args.split, g)
        if args.skip_existing and (gdir / "records").is_dir() and any((gdir / "records").glob("*.jsonl")):
            print(f"\n[gamma {g:g}] skip (records/ exist)")
            statuses.append({"gamma": g, "status": "skipped"})
            continue
        run_name = f"{args.model}_{args.task}_{args.split}_gamma{g:g}"
        energy_dir = gdir / "energy"
        max_tokens = max(1, int(args.max_tokens * g)) if args.scale_by_gamma else args.max_tokens
        print(f"\n{'-' * 72}\n[gamma {g:g}] run_name={run_name} max_tokens={max_tokens}\n{'-' * 72}")

        # 1) monitored inference (monitors wrap ONLY this subprocess) -----------
        infer_cmd = [sys.executable, str(HERE / "run_inference.py"),
                     "--task", args.task, "--split", args.split, "--model", args.model,
                     "--model-path", model_path, "--gamma", f"{g:g}",
                     "--run-id", args.run_id, "--max-tokens", str(max_tokens)]
        if args.trio_only:
            infer_cmd.append("--trio-only")
        if args.limit:
            infer_cmd += ["--limit", str(args.limit)]
        if args.frequency_penalty:
            infer_cmd += ["--frequency-penalty", str(args.frequency_penalty)]
        if args.presence_penalty:
            infer_cmd += ["--presence-penalty", str(args.presence_penalty)]
        if args.repetition_penalty != 1.0:
            infer_cmd += ["--repetition-penalty", str(args.repetition_penalty)]
        if args.system:
            infer_cmd += ["--system", args.system]
        print("[infer] " + " ".join(infer_cmd))
        with EnergyMonitors(run_name, energy_dir, gpu_index=gpu_index, interval=interval,
                            pdu=not args.no_pdu, pdu_host=pdu_host,
                            pdu_community=pdu_community, pdu_oid=pdu_oid):
            rc = subprocess.run(infer_cmd, env=env).returncode
        if rc != 0:
            print(f"[gamma {g:g}] FAIL: run_inference rc={rc} -- aborting sweep")
            statuses.append({"gamma": g, "status": f"infer_fail({rc})"})
            break

        # 2) accuracy control (monitors already stopped -> outside energy window)
        score_cmd = [sys.executable, str(HERE / "score_generations.py"),
                     "--task", args.task, "--split", args.split, "--model", args.model,
                     "--gamma", f"{g:g}", "--run-id", args.run_id, "--digest", digest]
        print("[score] " + " ".join(score_cmd))
        if subprocess.run(score_cmd).returncode != 0:
            print(f"[gamma {g:g}] WARN: scoring failed (generations saved; re-score later)")
            statuses.append({"gamma": g, "status": "score_fail"})
            continue

        # 3) energy join (integrate over generate_window, stamp records) --------
        join_cmd = [sys.executable, str(HERE / "join_energy.py"),
                    "--task", args.task, "--split", args.split, "--model", args.model,
                    "--gamma", f"{g:g}", "--run-id", args.run_id]
        print("[join] " + " ".join(join_cmd))
        jrc = subprocess.run(join_cmd).returncode
        statuses.append({"gamma": g, "status": "ok" if jrc == 0 else "join_fail"})

    print("\n" + "=" * 72)
    print("Sweep summary:")
    for s in statuses:
        print(f"  gamma={s['gamma']:<5g} {s['status']}")
    print("Next: python3 scripts/build_curves.py "
          f"--model {args.model} --task {args.task} --split {args.split} --run-id {args.run_id}")
    print("=" * 72)
    return 0 if all(s["status"] in ("ok", "skipped") for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
