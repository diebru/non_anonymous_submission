#!/usr/bin/env python3
"""Phase-2: LLMLingua-2 multi-gamma compression of the correct-CoT corpus. SERVER.

Reads the Phase-1 correct-CoT corpus (built by scripts/build_corpus.py) and writes,
per trajectory, the 12-gamma family of compressed-CoT variants -- the SFT raw
material Phase 3 converts and Phase 4 fine-tunes on. Compresses ONLY the CoT region
(generation reasoning; explanation stage-1 descriptions, post-hoc); completion is
SKIPPED (Decision #5 gate = no lever, read from phase1_gates.json). Compression is
TokenSkip-qwen faithful: bare ``compress_prompt(text, rate=gamma)``.

Layout (roadmap Phase 2: model/task/gamma/language; split kept as in the rest of
the repo; gitignored compressed_dir):
    compressed/<model>/<run>/<task>/<split>/gamma<g>/<lang>.jsonl
    compressed/<model>/<run>/compression_summary.json

The real run needs tokenskip_env + the pinned checkpoint (GPU). ``--dry-run`` swaps
in a deterministic word-drop mock + whitespace counter so the orchestration/IO can
be smoke-tested locally with no GPU or heavy deps (docs/WORKFLOW.md s2).

Usage (server, after build_corpus):
    python3 scripts/compress_corpus.py --model qwen2.5-coder-3b-instruct
    python3 scripts/compress_corpus.py --model qwen2.5-coder-3b-instruct --dry-run   # local
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from tsmc.compression.corpus import (  # noqa: E402
    CompressionParams,
    CompressionResult,
    aggregate_monotonic,
    aggregate_token_medians,
    check_scaffolding_intact,
    compress_record,
    trajectory_monotonic,
)
from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS  # noqa: E402

# Phase 2 compresses the two tasks with a CoT lever; completion is gated out.
DEFAULT_TASKS = ("generation", "explanation")
ALL_TASKS = ("generation", "explanation", "completion")


def corpus_root(paths, model: str, run_id: str) -> pathlib.Path:
    # build_corpus.py writes corpus next to compressed_dir.
    return paths.compressed_dir.parent / "corpus" / model / run_id


def gamma_tag(gamma: float) -> str:
    return f"gamma{gamma:g}"  # matches the Phase-1 generations layout (gamma1, gamma0.5)


def load_corpus_lang_files(root: pathlib.Path, task: str, split: str):
    d = root / task / split
    if not d.is_dir():
        return []
    return sorted(d.glob("*.jsonl"))


def resolve_tokenizer_repo(model_id: str, model_path: str | None, paths) -> str | None:
    if model_path:
        return model_path
    for name in ("run_metadata.yaml", "run_metadata.example.yaml"):
        meta = paths.configs_dir / name
        if meta.is_file():
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            repo = ((data.get("models") or {}).get(model_id) or {}).get("hf_repo")
            if repo:
                return repo
    return None


def completion_skipped(paths, model_id: str, run_id: str) -> dict | None:
    """Return the completion gate block from phase1_gates.json, if present."""
    gates = paths.generations_dir / model_id / run_id / "phase1_gates.json"
    if not gates.is_file():
        return None
    data = json.loads(gates.read_text(encoding="utf-8"))
    return data.get("completion_gate") or None


# --- dry-run mock (local smoke; no GPU / no llmlingua) -------------------------

def mock_compress_fn(text: str, rate: float) -> CompressionResult:
    """Deterministic stand-in: keep the leading ceil(rate*N) whitespace tokens.

    Monotonic in ``rate`` by construction, so the dry run exercises the validators."""
    words = text.split()
    keep = max(1, math.ceil(rate * len(words))) if words else 0
    compressed = " ".join(words[:keep])
    return CompressionResult(
        compressed_text=compressed,
        origin_tokens=len(words),
        compressed_tokens=keep,
        rate=f"{(keep / len(words) * 100):.1f}%" if words else "0%",
    )


def whitespace_count_fn(text: str) -> int:
    return len(text.split())


# --- main ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--task", choices=(*ALL_TASKS, "all"), default="all",
                    help="'all' = generation+explanation (completion is gated out)")
    ap.add_argument("--split", choices=("train", "test", "both"), default="train")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--checkpoint", default=None, help="LLMLingua-2 checkpoint (else pinned default)")
    ap.add_argument("--model-path", default=None, help="tokenizer source (else run_metadata hf_repo)")
    ap.add_argument("--dry-run", action="store_true", help="mock compressor + whitespace counter (local, no GPU)")
    ap.add_argument("--limit", type=int, default=0, help="cap trajectories per language (smoke)")
    args = ap.parse_args()
    paths = get_paths()

    tasks = DEFAULT_TASKS if args.task == "all" else (args.task,)
    splits = ["train", "test"] if args.split == "both" else [args.split]
    src_root = corpus_root(paths, args.model, args.run_id)
    out_root = paths.compressed_dir / args.model / args.run_id

    print("=" * 70)
    print(f"Phase-2 compression | model={args.model} run={args.run_id} "
          f"tasks={list(tasks)} splits={splits} dry_run={args.dry_run}")
    print(f"gammas={args.gammas}")
    print("=" * 70)

    # --- wire heavy callables (or mocks) ---
    if args.dry_run:
        compress_fn, count_fn = mock_compress_fn, whitespace_count_fn
        params = CompressionParams(checkpoint="MOCK(dry-run)", checkpoint_sha=None,
                                   extra_kwargs={"_mock": True})
    else:
        from tsmc.compression.llmlingua import (  # server-only deps
            DEFAULT_LLMLINGUA2_CHECKPOINT, DEFAULT_LLMLINGUA2_SHA,
            Lingua2Compressor, make_token_counter, resolve_checkpoint_sha,
        )
        checkpoint = args.checkpoint or DEFAULT_LLMLINGUA2_CHECKPOINT
        repo = resolve_tokenizer_repo(args.model, args.model_path, paths)
        if not repo:
            print("ERROR: could not resolve a tokenizer repo (set models.<id>.hf_repo "
                  "in configs/run_metadata.yaml or pass --model-path).", file=sys.stderr)
            return 2
        print(f"compressor: {checkpoint}")
        print(f"token counter (Qwen): {repo}")
        compressor = Lingua2Compressor(checkpoint=checkpoint).load()
        compress_fn = compressor.compress_fn
        count_fn = make_token_counter(repo)
        sha = resolve_checkpoint_sha(checkpoint) or DEFAULT_LLMLINGUA2_SHA
        params = CompressionParams(checkpoint=checkpoint, checkpoint_sha=sha)

    # --- completion gate awareness ---
    if "completion" in tasks:
        gate = completion_skipped(paths, args.model, args.run_id)
        decisions = {s: g.get("gate_decision") for s, g in (gate or {}).items()}
        if gate is None or all(d == "skipped_no_lever" for d in decisions.values()):
            print(f"completion: SKIPPED (gate={decisions or 'no gate file -> default skip'}) -- "
                  "no compression lever (Decision #5).")
            tasks = tuple(t for t in tasks if t != "completion")

    summary: dict = {"model_id": args.model, "run_id": args.run_id,
                     "gammas": list(args.gammas), "params": params.to_dict(),
                     "dry_run": args.dry_run, "tasks": {}}
    scaffolding_errors = 0

    for task in tasks:
        for split in splits:
            files = load_corpus_lang_files(src_root, task, split)
            if not files:
                continue
            # measured tokens per gamma, pooled across the (task, split) for the
            # aggregate monotonicity headline.
            tokens_by_gamma: dict[float, list[int]] = defaultdict(list)
            traj_violations = n_traj = 0
            n_written = 0

            for f in files:
                lang = f.name[: -len(".jsonl")]
                records = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
                if args.limit:
                    records = records[: args.limit]
                # buffer variants per gamma so each gamma dir gets one file/lang
                per_gamma: dict[float, list[dict]] = defaultdict(list)
                for rec in records:
                    variants = compress_record(rec, args.gammas, compress_fn, count_fn, params)
                    for v in variants:
                        errs = check_scaffolding_intact(rec, v)
                        if errs:
                            scaffolding_errors += len(errs)
                            print(f"  !! scaffolding [{task}/{split}/{lang} {rec.get('problem_id')}"
                                  f" g={v['gamma']:g}]: {errs}", file=sys.stderr)
                        per_gamma[float(v["gamma"])].append(v)
                        tokens_by_gamma[float(v["gamma"])].append(int(v["cot_token_count"]))
                    mono = trajectory_monotonic(variants)
                    n_traj += 1
                    if not mono["monotonic"]:
                        traj_violations += 1

                for gamma, rows in per_gamma.items():
                    gdir = out_root / task / split / gamma_tag(gamma)
                    gdir.mkdir(parents=True, exist_ok=True)
                    with open(gdir / f"{lang}.jsonl", "w", encoding="utf-8") as fh:
                        for r in rows:
                            fh.write(json.dumps(r) + "\n")
                        n_written += len(rows)

            medians = aggregate_token_medians(tokens_by_gamma)
            agg = aggregate_monotonic(medians)
            key = f"{task}/{split}"
            summary["tasks"][key] = {
                "n_trajectories": n_traj,
                "n_variants_written": n_written,
                "median_tokens_by_gamma": [[g, m] for g, m in medians],
                "aggregate_monotonic": agg["monotonic"],
                "aggregate_violations": agg["violations"],
                "trajectory_monotonic_fraction": round(1 - traj_violations / n_traj, 4) if n_traj else None,
                "n_trajectory_violations": traj_violations,
            }
            print(f"\n[{key}] trajectories={n_traj} variants={n_written}")
            print(f"  median cot_tokens by gamma (desc): "
                  + ", ".join(f"{g:g}:{m:.0f}" for g, m in medians))
            print(f"  aggregate monotonic: {'OK' if agg['monotonic'] else 'VIOLATED ' + str(agg['violations'])}")
            print(f"  per-trajectory non-increasing: "
                  f"{summary['tasks'][key]['trajectory_monotonic_fraction']} "
                  f"({traj_violations}/{n_traj} have a strict-increase step)")

    summary["scaffolding_errors"] = scaffolding_errors
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "compression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nCompressed corpus -> {out_root}")
    print(f"scaffolding errors: {scaffolding_errors} (must be 0)")
    print("=" * 70)
    return 1 if scaffolding_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
