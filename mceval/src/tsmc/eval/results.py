"""Build McEval evaluation result files (CPU-only; runnable locally).

McEval's ``eval_all.py`` reads ``<result_path>/<Lang>.jsonl`` where each line is
the original problem record plus a ``raw_generation`` list; it then runs its own
``extract(raw_generation[0], item, lang)`` and executes. So a result file is just
the rich problem record with our output attached at ``raw_generation[0]``.

This module builds those files. Two ``raw_generation[0]`` shapes:
  - GOLD: a fenced block of the reference program (prompt + canonical_solution),
    for the Docker harness verification (expected ~100% pass).
  - CONTRACT: our parser's ``code_snippet`` re-wrapped in a single fenced block,
    for the contract<->extractor smoke test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# Phase-0 validation trio: Python + Rust (Family A) and C (Family B).
TRIO: tuple[str, ...] = ("Python", "C", "Rust")

# Execution gate policy. Python (Family A) + C (Family B) confirm the pipeline
# executes; both must clear the threshold. Rust is a SOFT check: McEval's executor
# does `rm -rf target` + `cargo test` per problem under a ~35s timeout, so every
# problem cold-recompiles crate deps and slow ones time out -- a known McEval
# limitation, not a fault in our pipeline (our Rust extraction matches gold).
EXEC_REQUIRED: tuple[str, ...] = ("Python", "C")
EXEC_SOFT: tuple[str, ...] = ("Rust",)


def report_trio_execution(scores: dict[str, Any], threshold: float) -> bool:
    """Print per-language accuracy; return overall pass (required langs only)."""
    failed_required = 0
    for lang in TRIO:
        acc = scores.get(lang, {}).get("accuracy")
        ok = acc is not None and acc >= threshold
        if lang in EXEC_SOFT:
            note = "  (soft: McEval cold-recompiles Rust deps per problem under a timeout)"
            print(f"  [{'PASS' if ok else 'warn'}] {lang}: accuracy={acc}{note}")
        else:
            failed_required += int(not ok)
            print(f"  [{'PASS' if ok else 'FAIL'}] {lang}: accuracy={acc}")
    return failed_required == 0


def discover_generation_languages(paths: Any) -> list[str]:
    """All McEval generation language file stems (the 40 languages), e.g. 'Python',
    'C#', 'Common Lisp'. The stem IS the lang key McEval's extractor dispatches on."""
    return sorted(p.name[:-len(".jsonl")] for p in paths.mceval_data_dir.glob("*.jsonl"))


def classify_gold(scores: dict[str, Any], langs: list[str], threshold: float
                  ) -> dict[str, tuple[str, Any]]:
    """Per-language gold verdict: ok / low / zero / nodata.

    Run on REFERENCE solutions, this maps McEval's per-language scoring HEALTH:
    'zero' = the language's extractor/executor can't even score gold (scoring-broken,
    needs a handler, like C with real model output); 'ok' = scoring is sound and acc
    is the per-language ceiling; 'nodata' = no rich gold items (e.g. reduced AWK)."""
    out: dict[str, tuple[str, Any]] = {}
    for lang in langs:
        s = scores.get(lang)
        acc = s.get("accuracy") if s else None
        if s and s.get("error"):
            out[lang] = ("error", None)            # toolchain/exec crash (e.g. `go` missing)
        elif not s or not s.get("total_count") or acc is None:
            out[lang] = ("nodata", acc)
        elif acc >= threshold:
            out[lang] = ("ok", acc)
        elif acc == 0:
            out[lang] = ("zero", acc)
        else:
            out[lang] = ("low", acc)
    return out


def report_gold_languages(scores: dict[str, Any], langs: list[str], threshold: float
                          ) -> dict[str, tuple[str, Any]]:
    """Print the per-language gold map + buckets; return the classification."""
    cls = classify_gold(scores, langs, threshold)
    tags = {"ok": "OK  ", "low": "LOW ", "zero": "ZERO", "error": "ERR ", "nodata": "----"}
    for lang in langs:
        verdict, acc = cls[lang]
        extra = f"  ({scores[lang]['error']})" if verdict == "error" and scores.get(lang) else ""
        print(f"  [{tags[verdict]}] {lang}: accuracy={acc}{extra}")
    buckets: dict[str, list[str]] = {"ok": [], "low": [], "zero": [], "error": [], "nodata": []}
    for lang in langs:
        buckets[cls[lang][0]].append(lang)
    print(f"\n  OK={len(buckets['ok'])}  LOW={len(buckets['low'])}  ZERO={len(buckets['zero'])}  "
          f"ERROR={len(buckets['error'])}  NODATA={len(buckets['nodata'])}  (threshold={threshold})")
    if buckets["error"]:
        print("  ERROR (toolchain/exec crash -> fix PATH or toolchain):", buckets["error"])
    if buckets["zero"]:
        print("  ZERO (gold can't be scored -> scoring-broken, need a handler):", buckets["zero"])
    if buckets["low"]:
        print("  LOW  (below threshold):", [f"{l}={cls[l][1]:.2f}" for l in buckets["low"]])
    if buckets["nodata"]:
        print("  NODATA (no rich gold items):", buckets["nodata"])
    return cls


def fence_tag(lang: str) -> str:
    """Markdown fence tag the model would emit for a language (e.g. Python->python)."""
    return lang.lower()


def reference_program(item: dict[str, Any]) -> str:
    """Reference solution = prompt + canonical_solution with a clean newline join.

    McEval prompts don't always end in a newline (they can end at the docstring's
    closing ``\"\"\"``), so a bare concatenation would glue the body onto that line
    and break syntax. Force exactly one newline between them.
    """
    return item["prompt"].rstrip("\n") + "\n" + item["canonical_solution"]


def gold_raw_generation(item: dict[str, Any], lang: str) -> str:
    """A perfect model output: the reference program in one fenced block."""
    return f"```{fence_tag(lang)}\n{reference_program(item)}\n```"


def wrap_code(code_snippet: str, lang: str) -> str:
    """Wrap our parser's clean ``code_snippet`` as one canonical fenced block."""
    return f"```{fence_tag(lang)}\n{code_snippet}\n```"


def synthetic_contract_output(item: dict[str, Any], lang: str, with_distractors: bool = True) -> str:
    """Simulate a model's contract-format output for the reference solution.

    Used by the contract<->extractor smoke test / regression tests: CoT (optionally
    with a scratch fence + entry_point distractor that the LAST-sentinel / first-fence
    rules must ignore) -> sentinel -> the real fenced reference program.
    """
    from tsmc.constants import SENTINEL  # local import keeps results import-light

    tag = fence_tag(lang)
    body = reference_program(item)
    cot = "Let me reason about the approach."
    if with_distractors:
        cot += (
            f"\nScratch attempt (ignore this):\n```{tag}\n"
            f"WRONG {item['entry_point']} draft\n```\nNow the final version."
        )
    return f"{cot}\n{SENTINEL}\n```{tag}\n{body}\n```"


def build_result_item(item: dict[str, Any], raw_generation_text: str) -> dict[str, Any]:
    """Original problem record + ``raw_generation: [text]`` (McEval's input shape)."""
    out = dict(item)
    out["raw_generation"] = [raw_generation_text]
    return out


def write_result_dir(items_by_lang: dict[str, Iterable[dict[str, Any]]], out_dir: Path) -> list[Path]:
    """Write one ``<Lang>.jsonl`` per language. Returns the files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for lang, items in items_by_lang.items():
        path = out_dir / f"{lang}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item) + "\n")
        written.append(path)
    return written
