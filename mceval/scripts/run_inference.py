#!/usr/bin/env python3
"""Phase-1 inference driver (roadmap Phase 1). SERVER for real runs; --dry-run on CPU.

Generates baseline (gamma=1.0) CoT+code with vLLM for one model over one or more
(task, split) pairs, writing per-language McEval result files + our long-format
trajectories under the gitignored generations dir. The model is loaded ONCE and
reused across all requested tasks/splits.

Workflow (docs/WORKFLOW.md): write/commit/push locally; on the server `git pull`,
`conda activate tokenskip_env`, then run. Validate prompts first with --dry-run
(CPU, no model), then do a trio smoke, then the full run.

Examples:
    # CPU: preview prompts for the trio (no GPU, no model download)
    python3 scripts/run_inference.py --task all --split both --trio-only --dry-run

    # server: trio smoke (5/lang) generation, train+test
    python3 scripts/run_inference.py --task generation --split both --trio-only --limit 5

    # server: full generation baseline for Coder-3B (train + test)
    python3 scripts/run_inference.py --task generation --split both \
        --model qwen2.5-coder-3b-instruct
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
from tsmc.constants import GAMMA_BASELINE, MODEL_IDS  # noqa: E402
from tsmc.inference import (  # noqa: E402
    HarnessConfig,
    RunnerConfig,
    VLLMRunner,
    merge_shards,
    plan_task,
    run_task,
)

TASKS = ("generation", "explanation", "completion")


def resolve_model(model_id: str, model_path: str | None, paths) -> tuple[str, str | None]:
    """Return (model_path_or_repo, revision). --model-path wins; else read the HF
    repo + commit from configs/run_metadata.yaml (the committed .example as fallback,
    so a fresh server pull still resolves the repo)."""
    if model_path:
        return model_path, None
    for name in ("run_metadata.yaml", "run_metadata.example.yaml"):
        meta = paths.configs_dir / name
        if not meta.is_file():
            continue
        data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        entry = (data.get("models") or {}).get(model_id) or {}
        repo = entry.get("hf_repo")
        if repo:
            commit = entry.get("commit")
            rev = commit if commit and not str(commit).startswith("TBD") else None
            return repo, rev
    raise SystemExit(
        f"No model path for {model_id!r}: pass --model-path or set models.{model_id}"
        ".hf_repo in configs/run_metadata.yaml"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=(*TASKS, "all"), default="generation")
    ap.add_argument("--split", choices=("train", "test", "both"), default="both")
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--gamma", type=float, default=GAMMA_BASELINE)
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--system", default=None, help="pinned system prompt (default: chat-template default)")
    ap.add_argument("--trio-only", action="store_true", help="restrict to Python/C/Rust")
    ap.add_argument("--limit", type=int, default=0, help="max problems per language (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="CPU: build + preview prompts, no model")
    # runner (server)
    ap.add_argument("--model-path", default=None, help="local model dir or HF repo (overrides metadata)")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="GPUs per engine (use for 14B; for 3B/7B prefer --shards for DP throughput)")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=2048, help="output token budget")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="gpu_memory_utilization")
    # decoding-time repetition controls (default OFF -> unchanged behavior). See docs/EXPERIMENTS.md.
    ap.add_argument("--frequency-penalty", type=float, default=0.0,
                    help="vLLM frequency_penalty (count-scaled; >0 throttles runaway repetition loops)")
    ap.add_argument("--presence-penalty", type=float, default=0.0, help="vLLM presence_penalty (flat)")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="vLLM repetition_penalty (1.0 = off; >1 penalizes any repeat)")
    # data-parallel sharding (server): one worker per GPU, then auto-merge
    ap.add_argument("--shards", type=int, default=1,
                    help="data-parallel workers (e.g. 2 = one engine per GPU, ~2x faster on 3B)")
    ap.add_argument("--shard-id", type=int, default=-1, help="internal: worker index (orchestrator sets it)")
    ap.add_argument("--gpus", default="0,1", help="GPU ids to spread shards over (comma list)")
    args = ap.parse_args()

    paths = get_paths()
    tasks = list(TASKS) if args.task == "all" else [args.task]
    splits = ["train", "test"] if args.split == "both" else [args.split]
    cfg = HarnessConfig(model_id=args.model, gamma=args.gamma, run_id=args.run_id, system=args.system)

    # --- orchestrator: spawn one worker process per GPU, wait, then merge --------
    if args.shards > 1 and args.shard_id < 0 and not args.dry_run:
        return orchestrate(args, cfg, paths, tasks, splits)

    print("=" * 64)
    worker_tag = f" shard {args.shard_id}/{args.shards}" if args.shard_id >= 0 else ""
    print(f"Phase-1 inference | model={args.model} gamma={args.gamma:g} run={args.run_id}{worker_tag}")
    print(f"tasks={tasks} splits={splits} trio_only={args.trio_only} limit={args.limit}"
          f"{' DRY-RUN' if args.dry_run else ''}")
    print("=" * 64)

    if args.dry_run:
        for task in tasks:
            for split in splits:
                s = plan_task(task, split, cfg, paths, args.trio_only, args.limit)
                print(f"[plan] {task}/{split}: {s['n_units']} units, "
                      f"{len(s['by_language'])} langs -> {s['preview']}")
                print(f"       by_language={s['by_language']}")
        return 0

    shards = max(1, args.shards)
    shard_id = args.shard_id if args.shard_id >= 0 else 0
    # A sharded worker owns exactly one visible GPU (orchestrator set CUDA_VISIBLE_DEVICES).
    tp = 1 if shards > 1 else args.tensor_parallel_size

    model_path, revision = resolve_model(args.model, args.model_path, paths)
    rcfg = RunnerConfig(
        model_path=model_path, revision=revision,
        tensor_parallel_size=tp, max_model_len=args.max_model_len,
        max_tokens=args.max_tokens, gpu_memory_utilization=args.gpu_mem,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"Loading vLLM: {model_path}{f' @ {revision}' if revision else ''} "
          f"(tp={tp}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}) ...")
    runner = VLLMRunner(rcfg).load()

    for task in tasks:
        for split in splits:
            s = run_task(task, split, runner, cfg, paths, args.trio_only, args.limit,
                         shards=shards, shard_id=shard_id)
            print(f"[done] {task}/{split}: {s['n_units']} units in {s['elapsed_sec']}s "
                  f"-> {s['out_dir']}")
            print("       " + json.dumps(s["by_language"]))
    return 0


def orchestrate(args, cfg, paths, tasks, splits) -> int:
    """Spawn one worker per shard (pinned to a GPU), wait, then merge shard outputs."""
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if len(gpus) < args.shards:
        raise SystemExit(f"--shards {args.shards} needs >= {args.shards} --gpus (got {gpus})")
    base = [a for a in sys.argv[1:] if not a.startswith("--shard-id")]
    procs = []
    print("=" * 64)
    print(f"Phase-1 inference (data-parallel) | {args.shards} shards over GPUs {gpus[:args.shards]}")
    print(f"model={args.model} tasks={tasks} splits={splits}")
    print("=" * 64)
    for k in range(args.shards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[k])
        cmd = [sys.executable, sys.argv[0], *base, "--shard-id", str(k)]
        print(f"  -> shard {k} on GPU {gpus[k]}: {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        print(f"FAIL: worker exit codes {codes}")
        return 1
    print("\nAll shards done; merging ...")
    for task in tasks:
        for split in splits:
            m = merge_shards(task, split, cfg, paths)
            print(f"[merge] {task}/{split}: {m['merged_shards']} shards -> {m['out_dir']}")
            print("        " + json.dumps(m["by_language"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
