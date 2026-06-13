#!/usr/bin/env python3
"""Re-parse saved Phase-1 generation trajectories with the re-frozen contract. CPU.

The contract parser gained ``presentinel_salvage`` (recover the code from the LAST
fenced block in the CoT when the model coded inside its reasoning and emitted a
bare/empty trailing sentinel) and ``three_way_outcome`` now treats a fenced-branch
parse with no fence as ``format_fail``. This script re-applies the parser to the
ALREADY-SAVED ``raw_full_output`` -- so the fix lands WITHOUT GPU re-inference --
and rewrites, per language:

    trajectories/<Lang>.jsonl   updated code_snippet / cot_text / extraction_status
                                (pass reset to provisional false)
    result/<Lang>.jsonl         McEval input rebuilt from the recovered code

Then re-run the normal Phase-1.2 / 1.3 chain (score_generations -> phase1_gates ->
build_corpus) so McEval re-executes the recovered code and the corpus is rebuilt
from trustworthy verdicts. Generation only (the implicated task; completion is
gate-skipped, explanation parses stage-2 separately).

Usage (server, tokenskip_env):
    python3 scripts/reparse_trajectories.py --model qwen2.5-coder-3b-instruct
    #   add --count-tokens to recount cot/code tokens with the Qwen tokenizer.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import MODEL_IDS  # noqa: E402
from tsmc.contract import parse_generation  # noqa: E402
from tsmc.eval import results as R  # noqa: E402
from tsmc.inference.prompts import select_units  # noqa: E402


def resolve_tokenizer_repo(model_id: str, override: str | None, paths) -> str | None:
    if override:
        return override
    for name in ("run_metadata.yaml", "run_metadata.example.yaml"):
        meta = paths.configs_dir / name
        if meta.is_file():
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            repo = ((data.get("models") or {}).get(model_id) or {}).get("hf_repo")
            if repo:
                return repo
    return None


def load_qwen_counter(repo: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    return lambda text: len(tok(text or "", add_special_tokens=False).input_ids)


def _finish_reason(row: dict) -> str | None:
    timing = (row.get("_provenance") or {}).get("timing") or {}
    fr = timing.get("finish_reason") or [None]
    return fr[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--split", choices=("train", "test"), default="train")
    ap.add_argument("--gamma", type=float, default=1.0, help="baseline dir (Phase-1 = 1.0)")
    ap.add_argument("--count-tokens", action="store_true", help="recount cot/code via Qwen tokenizer")
    ap.add_argument("--model-path", default=None, help="tokenizer source (else run_metadata hf_repo)")
    args = ap.parse_args()
    paths = get_paths()
    task = "generation"

    gdir = paths.generations_dir / args.model / args.run_id / task / args.split / f"gamma{args.gamma:g}"
    traj_dir, result_dir = gdir / "trajectories", gdir / "result"
    if not traj_dir.is_dir():
        print(f"ERROR: no trajectories at {traj_dir}", file=sys.stderr)
        return 2

    units = select_units(task, args.split, paths)
    unit_by_taskid = {u.mceval_task_id: u for u in units}

    counter = None
    if args.count_tokens:
        repo = resolve_tokenizer_repo(args.model, args.model_path, paths)
        if not repo:
            print("ERROR: --count-tokens needs a tokenizer repo.", file=sys.stderr)
            return 2
        counter = load_qwen_counter(repo)

    print("=" * 70)
    print(f"Re-parse | model={args.model} {task}/{args.split} gamma={args.gamma:g}")
    print("=" * 70)

    branches: Counter = Counter()
    n_total = n_recovered = n_changed = n_no_unit = n_still_empty = 0

    for traj_file in sorted(traj_dir.glob("*.jsonl")):
        lang = traj_file.name[: -len(".jsonl")]
        rows = [json.loads(x) for x in traj_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        result_items = []
        for row in rows:
            n_total += 1
            tid = (row.get("_provenance") or {}).get("mceval_task_id")
            unit = unit_by_taskid.get(tid)
            if unit is None:
                n_no_unit += 1
            entry_point = unit.entry_point if unit else None
            old_code = row.get("code_snippet") or ""

            pr = parse_generation(row.get("raw_full_output") or "", entry_point=entry_point,
                                  finish_reason=_finish_reason(row))
            branches[pr.status.parser_branch] += 1
            if not old_code.strip() and pr.code_snippet.strip():
                n_recovered += 1
            if pr.code_snippet != old_code:
                n_changed += 1
            if not pr.code_snippet.strip():
                n_still_empty += 1

            row["code_snippet"] = pr.code_snippet
            row["cot_text"] = pr.cot_text
            row["extraction_status"] = pr.status.to_dict()
            row["pass"] = False  # provisional; score_generations re-stamps it
            if counter is not None:
                row["cot_token_count"] = counter(pr.cot_text)
                row["code_token_count"] = counter(pr.code_snippet)

            if unit is not None:
                result_items.append(R.build_result_item(unit.record, R.wrap_code(pr.code_snippet, lang)))

        with open(traj_file, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        if result_items:
            (result_dir).mkdir(parents=True, exist_ok=True)
            with open(result_dir / f"{lang}.jsonl", "w", encoding="utf-8") as fh:
                for item in result_items:
                    fh.write(json.dumps(item) + "\n")

    summary = {
        "model_id": args.model, "task": task, "split": args.split, "gamma": args.gamma,
        "n_trajectories": n_total, "n_code_changed": n_changed,
        "n_recovered_from_empty": n_recovered, "n_still_empty": n_still_empty,
        "n_no_unit": n_no_unit, "parser_branches": dict(branches),
        "recounted_tokens": bool(counter),
    }
    (gdir / "reparse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"trajectories re-parsed: {n_total}")
    print(f"  code recovered from empty: {n_recovered}")
    print(f"  code changed (any):        {n_changed}")
    print(f"  STILL empty (format_fail): {n_still_empty}")
    if n_no_unit:
        print(f"  WARNING: {n_no_unit} rows had no McEval unit (result not rewritten for those)")
    print(f"  parser branches: {dict(branches)}")
    print(f"\nrewrote {traj_dir} and {result_dir}")
    print("NEXT (server): score_generations -> phase1_gates -> build_corpus -> "
          "compress_corpus -> build_sft_dataset -> check_sft_dataset")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
