"""Join inference trajectories with McEval pass/fail -> final long-format records.

Phase-1 inference writes ``trajectories/<Lang>.jsonl`` with a PROVISIONAL
``pass=false``; the McEval detail eval (``detail_eval.py`` in the container)
yields a per-``task_id`` verdict. This module merges the two and stamps the
three-way ``outcome`` (format_fail / exec_fail / pass), keeping extraction
failures distinct from reasoning failures (the central concavity confound,
roadmap s4). CPU-only and stdlib + tsmc only, so the whole join is unit-testable
without Docker.

SQL (and any language McEval excludes from execution) has no verdict -> those
rows are marked ``outcome="unscored"`` and dropped from accuracy, never silently
counted as failures.
"""
from __future__ import annotations

from typing import Any, Iterable

from tsmc.contract import three_way_outcome
from tsmc.eval import language_health as H
from tsmc.schema import ExtractionStatus

UNSCORED = "unscored"  # not one of constants.OUTCOMES: exec-excluded (e.g. SQL)


def _task_id(rec: dict[str, Any]) -> str | None:
    prov = rec.get("_provenance") or {}
    return prov.get("mceval_task_id")


def finalize_record(rec: dict[str, Any], passed: bool | None) -> dict[str, Any]:
    """Return a copy with ``pass`` set and an ``outcome`` field added.

    ``passed is None`` (no McEval verdict, i.e. execution-excluded) -> the row is
    kept but marked ``unscored`` and ``pass=false``.
    """
    out = dict(rec)
    status = ExtractionStatus.from_dict(rec["extraction_status"])
    if passed is None:
        out["pass"] = False
        out["outcome"] = UNSCORED
    else:
        out["pass"] = bool(passed)
        out["outcome"] = three_way_outcome(status, bool(passed))
    return out


def join_language(
    traj_rows: Iterable[dict[str, Any]], pass_by_task_id: dict[str, bool]
) -> list[dict[str, Any]]:
    """Finalize all trajectory rows of one language against the verdict map."""
    return [finalize_record(r, pass_by_task_id.get(_task_id(r))) for r in traj_rows]


def outcome_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Tally outcomes: pass / exec_fail / format_fail / unscored."""
    counts = {"pass": 0, "exec_fail": 0, "format_fail": 0, UNSCORED: 0}
    for r in records:
        counts[r.get("outcome", UNSCORED)] = counts.get(r.get("outcome", UNSCORED), 0) + 1
    return counts


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Outcome breakdown + accuracy/format-fail rate over the SCORED rows.

    accuracy = pass / (pass + exec_fail + format_fail); format_fail and unscored
    are reported separately so a rising format_fail (a contract artifact) can never
    masquerade as a reasoning failure in the accuracy number.
    """
    counts = outcome_counts(records)
    scored = counts["pass"] + counts["exec_fail"] + counts["format_fail"]
    by_lang: dict[str, dict[str, int]] = {}
    for r in records:
        lang = r.get("lang", "?")
        d = by_lang.setdefault(lang, {"pass": 0, "exec_fail": 0, "format_fail": 0, UNSCORED: 0})
        d[r.get("outcome", UNSCORED)] = d.get(r.get("outcome", UNSCORED), 0) + 1

    # Health-aware view: accuracy over languages whose McEval scoring is trustworthy
    # (drops EXCLUDED like F#/Java/R/SQL and SOFT like Rust). This is the headline the
    # behavioral gate reads -- broken-scoring languages must not pollute it.
    hp = he = hf = 0
    for r in records:
        if not H.is_healthy(r.get("lang", "")):
            continue
        o = r.get("outcome", UNSCORED)
        if o == "pass": hp += 1
        elif o == "exec_fail": he += 1
        elif o == "format_fail": hf += 1
    healthy_scored = hp + he + hf
    return {
        "n_records": len(records),
        "counts": counts,
        "scored": scored,
        "accuracy": (counts["pass"] / scored) if scored else None,
        "format_fail_rate": (counts["format_fail"] / scored) if scored else None,
        "healthy_scored": healthy_scored,
        "healthy_accuracy": (hp / healthy_scored) if healthy_scored else None,
        "healthy_format_fail_rate": (hf / healthy_scored) if healthy_scored else None,
        "by_language": by_lang,
    }
