#!/usr/bin/env python
"""Per-problem McEval scoring. RUNS INSIDE the pinned McEval container only.

McEval's stock ``eval_all.py`` writes only aggregate accuracy (its per-problem
``detail_scores`` write is commented out), but Phase 1 needs the per-``task_id``
verdict to filter CORRECT trajectories. Rather than fork McEval, this thin shim
IMPORTS ``eval_all`` and reuses its ``calculate_accuracy`` / scaffolding verbatim,
adding a ``<name>_detail.jsonl`` (one ``{task_id, pass}`` per line) alongside the
same ``<name>.jsonl`` aggregate eval_all would produce.

It mirrors ``eval_all.eval``: copy the auxiliary data into the hardcoded tmp dir
(while CWD is the eval dir so ``../data`` resolves), chdir there, then score each
language. SQL is excluded exactly as upstream. Invoked by
``tsmc.eval.docker.run_detail_eval`` (mounted at /work/detail_eval.py).
"""
import argparse
import json
import os
import sys

EVAL_DIR = "/workspace/MMCodeEval/eval"
TMP_DIR = os.path.join(EVAL_DIR, "tmp")
EXCLUDE_LANGS = ["sql"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_path", required=True)
    ap.add_argument("--save_path", required=True)
    args = ap.parse_args()

    sys.path.insert(0, EVAL_DIR)
    os.chdir(EVAL_DIR)  # so eval_all's ``../data`` and tmp scaffolding resolve
    import eval_all

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    name = os.path.basename(args.result_path.rstrip("/"))
    agg_path = os.path.join(args.save_path, name + ".jsonl")
    detail_path = os.path.join(args.save_path, name + "_detail.jsonl")
    for stale in (agg_path, detail_path):
        if os.path.exists(stale):
            os.remove(stale)

    langs = [f[:-len(".jsonl")] for f in os.listdir(args.result_path) if f.endswith(".jsonl")]
    langs = [l for l in langs if l.lower() not in EXCLUDE_LANGS]

    eval_all.prepare_tempdir_context(TMP_DIR)  # must run from EVAL_DIR (../data)
    os.chdir(TMP_DIR)
    for lang in langs:
        # ROBUST per language: McEval's eval_all has no try/except around excute(),
        # so a single missing toolchain (e.g. `go` not on PATH) would abort the whole
        # eval and lose every other language. We isolate each language: on error we
        # record it and move on, so the rest still score (those rows stay unscored in
        # the join, never counted as failures).
        try:
            score, detail = eval_all.calculate_accuracy(args, lang, TMP_DIR)
        except Exception as exc:  # noqa: BLE001 - toolchain/exec error -> isolate
            err = f"{type(exc).__name__}: {exc}"
            with open(agg_path, "a") as f:
                f.write(lang + "\t" + json.dumps(
                    {"accuracy": None, "total_count": 0, "correct": 0, "error": err}) + "\n")
            print(f"[detail_eval] {lang}: ERROR {err}", file=sys.stderr)
            os.chdir(TMP_DIR)  # restore CWD in case the failure left it elsewhere
            continue
        with open(agg_path, "a") as f:
            f.write(lang + "\t" + json.dumps(score) + "\n")
        with open(detail_path, "a") as f:
            for d in detail:
                f.write(json.dumps({"lang": lang, "task_id": d["task_id"], "pass": d["pass"]}) + "\n")
        print(f"[detail_eval] {lang}: acc={score.get('accuracy')} n={score.get('total_count')}")


if __name__ == "__main__":
    main()
