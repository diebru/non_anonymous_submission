"""CPU tests for the Phase-1.2 eval join (no Docker).

Verifies that {task_id: pass} verdicts merge onto trajectories with the correct
three-way outcome, that execution-excluded rows (no verdict) become ``unscored``
rather than silent failures, and that the summary keeps format_fail out of the
accuracy denominator's numerator (accuracy = pass / scored).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsmc.eval.join import (
    UNSCORED,
    finalize_record,
    join_language,
    outcome_counts,
    summarize,
)
from tsmc.schema import ExtractionStatus, validate_record


def _traj(task_id, branch, truncated=False, lang="python", subtype=None, task="generation"):
    """A minimal provisional trajectory row (as inference would emit)."""
    status = ExtractionStatus(
        fence_found=True, entry_point_found=True, truncated=truncated, parser_branch=branch
    )
    return {
        "problem_id": f"{lang}/1", "task_type": task, "completion_subtype": subtype,
        "model_id": "qwen2.5-coder-3b-instruct", "gamma": 1.0, "run_id": "run01",
        "raw_full_output": "x", "cot_text": "reason", "code_snippet": "code",
        "cot_token_count": 1, "compression_ratio": 1.0, "pass": False,
        "extraction_status": status.to_dict(), "cot_origin": "original",
        "compression_method": "model_side" if task != "explanation" else "post_hoc",
        "gate_decision": None, "gate_measured_median": None,
        "split": "train_problems", "lang": lang, "difficulty": "easy",
        "difficulty_source": "level_propagated",
        "_provenance": {"mceval_task_id": task_id, "mceval_lang": lang.capitalize()},
    }


class TestFinalize(unittest.TestCase):
    def test_pass_branch(self):
        rec = finalize_record(_traj("Python/1", "sentinel"), True)
        self.assertEqual(rec["pass"], True)
        self.assertEqual(rec["outcome"], "pass")
        self.assertEqual(validate_record(rec), [])  # still schema-valid

    def test_exec_fail(self):
        rec = finalize_record(_traj("Python/1", "sentinel"), False)
        self.assertEqual(rec["outcome"], "exec_fail")

    def test_format_fail_overrides_pass(self):
        # truncated -> format_fail even if McEval somehow passed
        rec = finalize_record(_traj("Python/1", "sentinel", truncated=True), True)
        self.assertEqual(rec["outcome"], "format_fail")
        self.assertEqual(rec["pass"], True)  # raw verdict preserved; outcome is the gate

    def test_fallback_branch_is_format_fail(self):
        rec = finalize_record(_traj("Python/1", "fallback"), True)
        self.assertEqual(rec["outcome"], "format_fail")

    def test_no_verdict_is_unscored(self):
        rec = finalize_record(_traj("sql/1", "sentinel"), None)
        self.assertEqual(rec["outcome"], UNSCORED)
        self.assertEqual(rec["pass"], False)


class TestJoinAndSummary(unittest.TestCase):
    def test_join_maps_by_task_id(self):
        rows = [_traj("Python/1", "sentinel"), _traj("Python/2", "sentinel"),
                _traj("Python/3", "none")]
        verdicts = {"Python/1": True, "Python/2": False}  # /3 has no verdict
        recs = join_language(rows, verdicts)
        outcomes = [r["outcome"] for r in recs]
        self.assertEqual(outcomes, ["pass", "exec_fail", UNSCORED])

    def test_summary_accuracy_excludes_format_and_unscored(self):
        recs = [
            finalize_record(_traj("Python/1", "sentinel"), True),    # pass
            finalize_record(_traj("Python/2", "sentinel"), False),   # exec_fail
            finalize_record(_traj("Python/3", "none"), True),        # format_fail
            finalize_record(_traj("sql/1", "sentinel"), None),       # unscored
        ]
        s = summarize(recs)
        self.assertEqual(s["counts"],
                         {"pass": 1, "exec_fail": 1, "format_fail": 1, UNSCORED: 1})
        self.assertEqual(s["scored"], 3)               # excludes unscored
        self.assertAlmostEqual(s["accuracy"], 1 / 3)   # pass / scored
        self.assertAlmostEqual(s["format_fail_rate"], 1 / 3)

    def test_outcome_counts_empty(self):
        self.assertEqual(outcome_counts([]),
                         {"pass": 0, "exec_fail": 0, "format_fail": 0, UNSCORED: 0})

    def test_healthy_accuracy_excludes_broken_languages(self):
        # python (healthy) pass; java (EXCLUDED) pass; rust (SOFT) pass.
        recs = [
            finalize_record(_traj("Python/1", "sentinel", lang="python"), True),
            finalize_record(_traj("Java/1", "sentinel", lang="java"), True),
            finalize_record(_traj("Rust/1", "sentinel", lang="rust"), True),
            finalize_record(_traj("Python/2", "sentinel", lang="python"), False),
        ]
        s = summarize(recs)
        # all-language accuracy counts every scored row (3 pass / 4)
        self.assertAlmostEqual(s["accuracy"], 3 / 4)
        # healthy accuracy only counts python (1 pass / 2) -- java + rust dropped
        self.assertEqual(s["healthy_scored"], 2)
        self.assertAlmostEqual(s["healthy_accuracy"], 1 / 2)


if __name__ == "__main__":
    unittest.main()
