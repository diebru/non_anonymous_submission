#!/usr/bin/env python3
"""Phase-3: build the LLaMA-Factory SFT dataset from the Phase-2 generation corpus.

Re-joins each compressed *generation* variant to its McEval problem (to recover the
``instruction`` / ``entry_point`` the corpus does not carry), renders the gamma-control
ShareGPT example with the FROZEN inference assembler, runs the decontamination gate
against the manifest, and emits the LLaMA-Factory dataset + ``dataset_info.json``.

Scope is GENERATION ONLY (P3-1 / Decision #3): explanation is post-hoc at test time
on the un-SFT'd base model, completion has no lever -- neither is fine-tuned here.

The re-join uses the SAME ``tsmc.inference.prompts.select_units`` Phase-1/4 inference
use, indexed by ``mceval_task_id`` (carried in ``_provenance`` through Phase 2), and
the SAME ``reasoning_user_text`` -- so the user turn (and the gamma marker) is
byte-identical to inference by construction, not by re-derivation (roadmap s8 freeze).

Outputs (gitignored ``sft_dir``; LLaMA-Factory reads dataset_info.json via ``dataset_dir``):
    sft/<model>/generation_train.jsonl   ShareGPT messages (one example per line)
    sft/<model>/dataset_info.json        LLaMA-Factory registration
    sft/<model>/build_summary.json       coverage / drops / decontam / length stats

CPU-only (no GPU, no llmlingua). ``--count-tokens`` lazily loads the Qwen tokenizer
(transformers) to report the true cutoff_len input; without it, lengths are in
characters only. Canonical run on the server, where ``compressed/`` exists.

Usage:
    python3 scripts/build_sft_dataset.py --model qwen2.5-coder-3b-instruct
    python3 scripts/build_sft_dataset.py --model qwen2.5-coder-3b-instruct --count-tokens
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_GRID, MODEL_IDS, SEED, family_of  # noqa: E402
from tsmc.inference.prompts import select_units  # noqa: E402
from tsmc.manifest import read_manifest  # noqa: E402
from tsmc.sft import build_example, decontaminate, select_variants  # noqa: E402


def gamma_tag(gamma: float) -> str:
    return f"gamma{gamma:g}"  # matches the Phase-1/2 layout (gamma1, gamma0.5)


def dataset_name(model: str) -> str:
    return f"tsmc_{model}_generation"


def dataset_info(model: str, data_filename: str) -> dict:
    """LLaMA-Factory ShareGPT registration with explicit role/content message tags."""
    return {
        dataset_name(model): {
            "file_name": data_filename,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }


def load_gen_variants(paths, model: str, run_id: str, split: str, gammas) -> list[dict]:
    """Load every compressed generation variant across all gamma dirs for a split."""
    root = paths.compressed_dir / model / run_id / "generation" / split
    records: list[dict] = []
    for g in gammas:
        gdir = root / gamma_tag(g)
        if not gdir.is_dir():
            continue
        for f in sorted(gdir.glob("*.jsonl")):
            records += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return records


def resolve_tokenizer_repo(model_id: str, model_path: str | None, paths) -> str | None:
    """Qwen tokenizer source: explicit --model-path, else run_metadata models.<id>.hf_repo."""
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


def load_token_counter(repo: str):
    """Lazy model-tokenizer token counter (server/transformers): tokens of a rendered
    example. Loads whatever tokenizer ``repo`` names (Qwen or Llama), so counts are
    in the SFT model's own tokenization."""
    from transformers import AutoTokenizer  # heavy; only when --count-tokens

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False).input_ids)

    return count


def _stats(values: list[int]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    pct = lambda p: s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": pct(50), "p95": pct(95),
            "p100": s[-1], "mean": round(statistics.mean(s), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--task", choices=("generation",), default="generation",
                    help="generation only (explanation/completion are out of SFT scope, P3-1)")
    ap.add_argument("--split", choices=("train",), default="train",
                    help="train only (test is held out for the Phase-4 curves)")
    ap.add_argument("--gamma-sampling", choices=("all", "random-k"), default="all",
                    help="all 12 gamma per trajectory (default), or k random gamma (ablation)")
    ap.add_argument("--k", type=int, default=6, help="gamma per trajectory when --gamma-sampling random-k")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gammas", type=float, nargs="+", default=list(GAMMA_GRID))
    ap.add_argument("--count-tokens", action="store_true",
                    help="report true Qwen token lengths (loads transformers; sets cutoff_len)")
    ap.add_argument("--model-path", default=None, help="tokenizer source (else run_metadata hf_repo)")
    ap.add_argument("--system", default=None,
                    help="pinned system prompt baked into every example (MUST match inference --system)")
    args = ap.parse_args()
    paths = get_paths()

    print("=" * 70)
    print(f"Phase-3 SFT build | model={args.model} run={args.run_id} task={args.task} "
          f"split={args.split} gamma_sampling={args.gamma_sampling}"
          + (f"(k={args.k})" if args.gamma_sampling == "random-k" else ""))
    print("=" * 70)

    # --- inputs: compressed variants + the unit lookup for the re-join ---
    records = load_gen_variants(paths, args.model, args.run_id, args.split, args.gammas)
    if not records:
        print(f"ERROR: no compressed generation variants under "
              f"{paths.compressed_dir / args.model / args.run_id / 'generation' / args.split}. "
              "Run scripts/compress_corpus.py first (server).", file=sys.stderr)
        return 2
    units = select_units(args.task, args.split, paths)
    unit_by_taskid = {u.mceval_task_id: u for u in units}
    print(f"loaded {len(records)} compressed variants; {len(unit_by_taskid)} McEval generation units")

    # --- gamma sampling (P3-2) ---
    selected = select_variants(records, policy=args.gamma_sampling, k=args.k, seed=args.seed)

    # optional true-token counter
    counter = None
    if args.count_tokens:
        repo = resolve_tokenizer_repo(args.model, args.model_path, paths)
        if not repo:
            print("ERROR: --count-tokens needs a tokenizer repo (set models.<id>.hf_repo in "
                  "configs/run_metadata.yaml or pass --model-path).", file=sys.stderr)
            return 2
        print(f"token counter ({family_of(args.model)}): {repo}")
        counter = load_token_counter(repo)

    # --- build examples ---
    examples: list[dict] = []
    drops: Counter = Counter()
    no_unit = 0
    problem_ids: set[str] = set()
    by_gamma: Counter = Counter()
    cell_problems: dict = defaultdict(set)  # (lang, difficulty) -> distinct problem_ids
    char_lens: list[int] = []
    tok_lens: list[int] = []

    for rec in selected:
        unit = unit_by_taskid.get((rec.get("_provenance") or {}).get("mceval_task_id"))
        if unit is None:
            no_unit += 1
            continue
        res = build_example(rec, unit, system=args.system, family=family_of(args.model))
        if not res.ok:
            drops[res.reason] += 1
            continue
        examples.append({"messages": res.messages})
        problem_ids.add(rec["problem_id"])
        by_gamma[float(rec["gamma"])] += 1
        cell_problems[(rec.get("lang", "?"), rec.get("difficulty", "?"))].add(rec["problem_id"])
        rendered = res.messages[0]["content"] + "\n" + res.messages[1]["content"]
        char_lens.append(len(rendered))
        if counter is not None:
            tok_lens.append(counter(rendered))

    # --- decontamination gate (required) ---
    manifest_rows = read_manifest(paths.manifest_path)
    decon = decontaminate(problem_ids, manifest_rows)

    # --- write outputs ---
    out_dir = paths.sft_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    data_filename = "generation_train.jsonl"
    with open(out_dir / data_filename, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info(args.model, data_filename), indent=2), encoding="utf-8")

    char_stats = _stats(char_lens)
    tok_stats = _stats(tok_lens)
    cutoff_rec = None
    if tok_stats:
        cutoff_rec = ((tok_stats["p100"] // 512) + 1) * 512  # next 512 multiple above max

    summary = {
        "model_id": args.model, "run_id": args.run_id, "task": args.task, "split": args.split,
        "gamma_sampling": args.gamma_sampling, "k": args.k if args.gamma_sampling == "random-k" else None,
        "seed": args.seed,
        "n_variants_in": len(records), "n_selected": len(selected),
        "n_examples": len(examples), "n_problems": len(problem_ids),
        "n_no_unit": no_unit, "drops": dict(drops),
        "by_gamma": {f"{g:g}": c for g, c in sorted(by_gamma.items(), reverse=True)},
        # distinct PROBLEMS per language x difficulty cell (not examples: all-gamma
        # inflates each problem to 12 examples, so example-counts would never be thin).
        "thin_cells": sorted(f"{l}:{d}({len(ps)}p)" for (l, d), ps in cell_problems.items() if len(ps) < 3),
        "length_chars": char_stats, "length_tokens": tok_stats,
        "cutoff_len_recommendation": cutoff_rec,
        "decontamination": decon,
        "dataset_name": dataset_name(args.model),
        "dataset_dir": str(out_dir),
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- report ---
    print(f"\nexamples: {len(examples)} from {len(problem_ids)} problems "
          f"(selected {len(selected)}/{len(records)} variants)")
    print("  by gamma (desc): " + ", ".join(f"{g}:{c}" for g, c in summary["by_gamma"].items()))
    if no_unit:
        print(f"  WARNING: {no_unit} variants had no matching McEval unit (skipped)")
    if drops:
        print(f"  dropped (round-trip/marker): {dict(drops)}")
    if summary["thin_cells"]:
        print(f"  thin cells (<3 problems): {len(summary['thin_cells'])} e.g. {summary['thin_cells'][:8]}")
    print(f"  length (chars): {char_stats}")
    if tok_stats:
        print(f"  length ({family_of(args.model)} tokens): {tok_stats}  -> set cutoff_len >= {cutoff_rec}")
    else:
        print("  (run with --count-tokens to get the true cutoff_len input)")

    ok = decon["ok"]
    print(f"\ndecontamination: {'PASS' if ok else 'FAIL'} "
          f"(test-leak={decon['n_test_leak']}, not-in-train={decon['n_not_in_train']})")
    if not ok:
        print(f"  leaked (first): {decon['leaked']}", file=sys.stderr)
        print(f"  not-in-train (first): {decon['not_in_train']}", file=sys.stderr)
    print(f"\nDataset -> {out_dir}  (name='{dataset_name(args.model)}', dataset_dir for LLaMA-Factory)")
    print("=" * 70)

    if not examples:
        print("ERROR: no examples emitted.", file=sys.stderr)
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
