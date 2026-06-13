"""Phase-1 gates + corpus filtering (roadmap Phase 1, Decisions #5/#7). CPU-only.

Three analyses over the SCORED long-format records (after the eval join):

  behavioral_gate   train vs test HEALTHY accuracy per task, |Δ| <= 3% pooled
                    across languages -> confirm-freeze the manifest (roadmap s6).
                    Computed on healthy languages only (tsmc.eval.language_health)
                    so broken-scoring languages never move the gate.

  completion_gate   per (subtype) median induced cot_token_count and cot/code token
                    ratio -> gate_decision applied / skipped_no_lever (Decision #5);
                    `single` is expected to skip ("no lever").

  filter_correct    keep outcome=="pass" on healthy languages -> the per-model
                    correct-CoT corpus that feeds Phase-2 compression / Phase-3 SFT.

Stdlib + tsmc only; fully unit-testable without Docker or a tokenizer.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from tsmc.constants import (
    COMPLETION_GATE_MIN_COT_CODE_RATIO,
    COMPLETION_GATE_MIN_COT_TOKENS,
    SENTINEL,
)
from tsmc.eval import language_health as H

BEHAVIORAL_TOL = 0.03  # roadmap s6: ±3% train vs test, pooled


# --- behavioral gate -----------------------------------------------------------

def _healthy_pass_scored(records: Iterable[dict[str, Any]]) -> tuple[int, int]:
    """(pass, scored) over healthy languages (excludes EXCLUDED + SOFT + unscored)."""
    passed = scored = 0
    for r in records:
        if not H.is_healthy(r.get("lang", "")):
            continue
        outcome = r.get("outcome")
        if outcome == "pass":
            passed += 1
            scored += 1
        elif outcome in ("exec_fail", "format_fail"):
            scored += 1
    return passed, scored


def behavioral_gate(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    tol: float = BEHAVIORAL_TOL,
) -> dict[str, Any]:
    """Compare healthy train vs test accuracy for ONE task. |Δ| <= tol -> within."""
    tp, ts = _healthy_pass_scored(train_records)
    ep, es = _healthy_pass_scored(test_records)
    train_acc = (tp / ts) if ts else None
    test_acc = (ep / es) if es else None
    delta = abs(train_acc - test_acc) if (train_acc is not None and test_acc is not None) else None
    within = delta is not None and delta <= tol
    return {
        "train_accuracy": train_acc, "test_accuracy": test_acc,
        "abs_delta": delta, "tol": tol, "within_tol": within,
        "train_scored": ts, "test_scored": es,
    }


# --- completion induced-CoT gate (Decision #5) ---------------------------------

def _ratio(cot: int, code: int) -> float:
    return (cot / code) if code else 0.0


def completion_gate(
    records: Iterable[dict[str, Any]],
    min_cot: int = COMPLETION_GATE_MIN_COT_TOKENS,
    min_ratio: float = COMPLETION_GATE_MIN_COT_CODE_RATIO,
) -> dict[str, dict[str, Any]]:
    """Per completion subtype: median induced cot_token_count + median cot/code ratio
    -> applied / skipped_no_lever. Reads ``code_token_count`` (backfilled by the
    caller for older runs that predate the field)."""
    by_sub: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in records:
        if r.get("task_type") != "completion":
            continue
        sub = r.get("completion_subtype")
        cot = int(r.get("cot_token_count") or 0)
        code = int(r.get("code_token_count") or 0)
        by_sub[sub].append((cot, _ratio(cot, code)))
    out: dict[str, dict[str, Any]] = {}
    for sub, vals in by_sub.items():
        cots = [c for c, _ in vals]
        ratios = [rr for _, rr in vals]
        med_cot = statistics.median(cots) if cots else 0.0
        med_ratio = statistics.median(ratios) if ratios else 0.0
        applied = med_cot >= min_cot and med_ratio >= min_ratio
        out[sub] = {
            "n": len(vals),
            "median_cot_tokens": med_cot,
            "median_cot_code_ratio": round(med_ratio, 3),
            "gate_decision": "applied" if applied else "skipped_no_lever",
            "min_cot": min_cot, "min_ratio": min_ratio,
        }
    return out


# --- correct-trajectory corpus -------------------------------------------------

def cot_scaffolding_clean(record: dict[str, Any]) -> bool:
    """True iff the reasoning region carries no leaked sentinel. A ``cot_text`` that
    contains the SENTINEL has a corrupted CoT/code boundary: the model emitted the
    delimiter more than once, so split-on-LAST-sentinel left a stray copy inside the
    reasoning. Such a trajectory must not seed the SFT corpus (we can't trust which
    span is reasoning vs code) and would trip the Phase-2 per-variant scaffolding
    gate. Enforced at corpus admission only -- the frozen prompt/parser are untouched."""
    return SENTINEL not in (record.get("cot_text") or "")


def filter_correct_report(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """:func:`filter_correct` plus a count of admission drops, so a corrupted
    trajectory is logged rather than silently lost. Returns (kept, drop_stats)."""
    kept: list[dict[str, Any]] = []
    sentinel_leak = 0
    for r in records:
        if r.get("outcome") != "pass" or not H.is_healthy(r.get("lang", "")):
            continue
        if not cot_scaffolding_clean(r):
            sentinel_leak += 1
            continue
        kept.append(r)
    return kept, {"sentinel_leak_dropped": sentinel_leak}


def filter_correct(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep verified-correct, well-scaffolded trajectories: outcome=='pass' on a
    healthy language AND no leaked sentinel in the CoT. (Excludes EXCLUDED/SOFT
    languages whose pass verdict is untrustworthy.)"""
    return filter_correct_report(records)[0]


def cell_counts(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str], int]:
    """Counts per (language, difficulty) stratum -- the corpus coverage map."""
    counter: Counter = Counter()
    for r in records:
        counter[(r.get("lang", "?"), r.get("difficulty", "?"))] += 1
    return dict(counter)
