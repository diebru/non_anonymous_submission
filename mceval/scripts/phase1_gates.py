#!/usr/bin/env python3
"""Phase-1 gates (roadmap Phase 1). CPU; tokenizer only for an optional backfill.

Run AFTER scoring (records/ populated). Computes, per the scored long-format records:
  - the behavioral ±3% gate (healthy train vs test accuracy) per task -> if all
    within tolerance, the manifest is CONFIRM-FROZEN (roadmap s6);
  - the completion induced-CoT gate per subtype -> gate_decision (Decision #5).

Older runs predate ``code_token_count``; we backfill it from ``code_snippet`` with
the model tokenizer (transformers, CPU) so the cot/code ratio is exact. Pass
--no-backfill to skip (ratio then uses 0 -> conservative skip).

Usage (server, after score_generations):
    python3 scripts/phase1_gates.py --model qwen2.5-coder-3b-instruct
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
from tsmc.eval.gates import behavioral_gate, completion_gate  # noqa: E402

TASKS = ("generation", "explanation", "completion")


def load_records(paths, model, run_id, task, split, gamma) -> list[dict]:
    d = paths.generations_dir / model / run_id / task / split / f"gamma{gamma:g}" / "records"
    if not d.is_dir():
        return []
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


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


def backfill_code_tokens(records: list[dict], repo: str | None) -> int:
    need = [r for r in records if r.get("code_token_count") is None]
    if not need or not repo:
        return 0
    from transformers import AutoTokenizer  # CPU; tokenizer only
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    for r in need:
        r["code_token_count"] = len(tok(r.get("code_snippet") or "", add_special_tokens=False).input_ids)
    return len(need)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--gamma", type=float, default=GAMMA_BASELINE)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--model-path", default=None, help="tokenizer source for backfill (else metadata repo)")
    ap.add_argument("--no-backfill", action="store_true", help="skip code_token_count backfill")
    args = ap.parse_args()
    paths = get_paths()

    print("=" * 64)
    print(f"Phase-1 gates | model={args.model} gamma={args.gamma:g} run={args.run_id}")
    print("=" * 64)

    # --- behavioral gate per task ---
    print("\n[Behavioral ±%.0f%% gate] healthy train vs test accuracy" % (args.tol * 100))
    behavioral: dict[str, dict] = {}
    all_within = True
    any_task = False
    for task in TASKS:
        train = load_records(paths, args.model, args.run_id, task, "train", args.gamma)
        test = load_records(paths, args.model, args.run_id, task, "test", args.gamma)
        if not train and not test:
            print(f"  {task}: no records (score it first) -> skip")
            continue
        any_task = True
        g = behavioral_gate(train, test, args.tol)
        behavioral[task] = g
        tr, te, d = g["train_accuracy"], g["test_accuracy"], g["abs_delta"]
        if g["within_tol"]:
            verdict = "WITHIN"
        elif d is None:
            verdict = "INCOMPLETE"; all_within = False
        else:
            verdict = "OUTSIDE"; all_within = False
        fmt = lambda x: "n/a" if x is None else f"{x:.4f}"
        print(f"  {task:11} train={fmt(tr)} test={fmt(te)} |Δ|={fmt(d)} "
              f"[{verdict}]  (scored {g['train_scored']}/{g['test_scored']})")

    manifest_ok = any_task and all_within
    print(f"\n  MANIFEST: {'CONFIRM-FREEZE OK (all tasks within ±3%)' if manifest_ok else 'NOT yet confirm-frozen'}")

    # --- completion induced-CoT gate (Decision #5) ---
    print("\n[Completion induced-CoT gate] per subtype (Decision #5)")
    compl = load_records(paths, args.model, args.run_id, "completion", "train", args.gamma)
    completion: dict[str, dict] = {}
    if compl:
        if not args.no_backfill:
            repo = resolve_tokenizer_repo(args.model, args.model_path, paths)
            n = backfill_code_tokens(compl, repo)
            if n:
                print(f"  (backfilled code_token_count for {n} records via {repo})")
        completion = completion_gate(compl)
        for sub in ("single", "multi", "span"):
            if sub in completion:
                c = completion[sub]
                print(f"  {sub:6} n={c['n']:5} median_cot={c['median_cot_tokens']:.0f} "
                      f"median_cot/code={c['median_cot_code_ratio']:.2f} -> {c['gate_decision']}")
    else:
        print("  no completion/train records -> skip")

    out = {"model_id": args.model, "run_id": args.run_id, "gamma": args.gamma,
           "behavioral": behavioral, "manifest_confirm_frozen": manifest_ok,
           "completion_gate": completion}
    out_path = paths.generations_dir / args.model / args.run_id / "phase1_gates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
