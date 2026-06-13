#!/usr/bin/env python3
"""Build the per-model correct-CoT corpus (roadmap Phase 1). CPU-only.

Filters the SCORED long-format records to verified-correct trajectories
(outcome=='pass' on a healthy language) -- the SFT raw material that Phase 2
compresses and Phase 3 fine-tunes on. Writes per (task, language) jsonl plus a
coverage summary (per language×difficulty cell counts), so thin strata are visible
before training.

By default it uses the TRAIN split only (test is held out for the Phase-4 curves);
pass --split both to inspect test coverage too. Excluded/soft languages
(F#/Java/R/SQL/Rust) are dropped -- their pass verdict isn't trustworthy.

Usage (after score_generations):
    python3 scripts/build_corpus.py --model qwen2.5-coder-3b-instruct
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import GAMMA_BASELINE, MODEL_IDS  # noqa: E402
from tsmc.eval.gates import cell_counts, filter_correct_report  # noqa: E402

TASKS = ("generation", "explanation", "completion")


def load_records(paths, model, run_id, task, split, gamma) -> list[dict]:
    d = paths.generations_dir / model / run_id / task / split / f"gamma{gamma:g}" / "records"
    if not d.is_dir():
        return []
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--run-id", default="run01")
    ap.add_argument("--gamma", type=float, default=GAMMA_BASELINE)
    ap.add_argument("--split", choices=("train", "test", "both"), default="train")
    args = ap.parse_args()
    paths = get_paths()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    corpus_root = paths.compressed_dir.parent / "corpus" / args.model / args.run_id
    print("=" * 64)
    print(f"Phase-1 corpus | model={args.model} run={args.run_id} gamma={args.gamma:g} splits={splits}")
    print("=" * 64)

    summary: dict[str, dict] = {}
    for task in TASKS:
        for split in splits:
            recs = load_records(paths, args.model, args.run_id, task, split, args.gamma)
            if not recs:
                continue
            correct, drop = filter_correct_report(recs)
            # write per-language jsonl
            out_dir = corpus_root / task / split
            out_dir.mkdir(parents=True, exist_ok=True)
            by_lang: dict[str, list[dict]] = {}
            for r in correct:
                by_lang.setdefault(r.get("lang", "?"), []).append(r)
            for lang, rows in by_lang.items():
                with open(out_dir / f"{lang}.jsonl", "w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
            cells = cell_counts(correct)
            by_diff = Counter(r.get("difficulty", "?") for r in correct)
            summary[f"{task}/{split}"] = {
                "n_scored": len(recs),
                "n_correct": len(correct),
                "yield": round(len(correct) / len(recs), 4) if recs else None,
                "n_languages": len(by_lang),
                "by_difficulty": dict(by_diff),
                "sentinel_leak_dropped": drop["sentinel_leak_dropped"],
                "thin_cells": sorted(f"{l}:{d}" for (l, d), c in cells.items() if c < 3),
            }
            s = summary[f"{task}/{split}"]
            print(f"  {task}/{split}: correct={s['n_correct']}/{s['n_scored']} "
                  f"(yield={s['yield']}) langs={s['n_languages']} diff={s['by_difficulty']}")
            if drop["sentinel_leak_dropped"]:
                print(f"      dropped {drop['sentinel_leak_dropped']} pass+healthy traj "
                      f"with a leaked sentinel in cot_text (corrupted boundary)")
            if s["thin_cells"]:
                print(f"      thin cells (<3): {len(s['thin_cells'])} e.g. {s['thin_cells'][:8]}")

    (corpus_root).mkdir(parents=True, exist_ok=True)
    (corpus_root / "corpus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nCorpus -> {corpus_root}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
