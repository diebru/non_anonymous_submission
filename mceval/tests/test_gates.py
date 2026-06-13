"""CPU tests for the Phase-1 gates (behavioral, completion, corpus filter)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsmc.constants import SENTINEL
from tsmc.eval.gates import (
    behavioral_gate,
    cell_counts,
    completion_gate,
    filter_correct,
    filter_correct_report,
)


def rec(lang="python", outcome="pass", task="generation", subtype=None,
        cot=0, code=10, difficulty="easy", cot_text="clean reasoning"):
    return {"lang": lang, "outcome": outcome, "task_type": task,
            "completion_subtype": subtype, "cot_token_count": cot,
            "code_token_count": code, "difficulty": difficulty,
            "cot_text": cot_text}


class TestBehavioralGate(unittest.TestCase):
    def test_within_tol_excludes_broken_langs(self):
        # python (healthy): train 8/10=0.8 ; java (EXCLUDED) all pass -> ignored
        train = [rec("python", "pass")] * 8 + [rec("python", "exec_fail")] * 2 \
            + [rec("java", "pass")] * 5
        test = [rec("python", "pass")] * 8 + [rec("python", "exec_fail")] * 2  # 0.8
        g = behavioral_gate(train, test)
        self.assertAlmostEqual(g["train_accuracy"], 0.8)
        self.assertAlmostEqual(g["test_accuracy"], 0.8)
        self.assertEqual(g["abs_delta"], 0.0)
        self.assertTrue(g["within_tol"])
        self.assertEqual(g["train_scored"], 10)  # java excluded from denom

    def test_outside_tol(self):
        train = [rec("python", "pass")] * 8 + [rec("python", "exec_fail")] * 2  # 0.8
        test = [rec("python", "pass")] * 6 + [rec("python", "exec_fail")] * 4   # 0.6
        g = behavioral_gate(train, test)
        self.assertAlmostEqual(g["abs_delta"], 0.2)
        self.assertFalse(g["within_tol"])

    def test_format_fail_counts_in_denominator_not_numerator(self):
        train = [rec("python", "pass")] * 5 + [rec("python", "format_fail")] * 5  # 0.5
        test = [rec("python", "pass")] * 5 + [rec("python", "format_fail")] * 5
        g = behavioral_gate(train, test)
        self.assertAlmostEqual(g["train_accuracy"], 0.5)


class TestCompletionGate(unittest.TestCase):
    def test_subtype_decisions(self):
        records = (
            [rec(task="completion", subtype="single", cot=0, code=10)] * 5 +    # ratio 0
            [rec(task="completion", subtype="multi", cot=40, code=10)] * 5 +    # cot>=30, ratio 4
            [rec(task="completion", subtype="span", cot=5, code=100)] * 5       # cot<30
        )
        g = completion_gate(records)
        self.assertEqual(g["single"]["gate_decision"], "skipped_no_lever")
        self.assertEqual(g["multi"]["gate_decision"], "applied")
        self.assertEqual(g["span"]["gate_decision"], "skipped_no_lever")
        self.assertEqual(g["multi"]["median_cot_tokens"], 40)

    def test_ignores_non_completion(self):
        g = completion_gate([rec(task="generation", cot=99, code=1)])
        self.assertEqual(g, {})


class TestCorpusFilter(unittest.TestCase):
    def test_keeps_only_correct_healthy(self):
        records = [
            rec("python", "pass"),       # keep
            rec("python", "exec_fail"),  # drop (not pass)
            rec("java", "pass"),         # drop (EXCLUDED)
            rec("rust", "pass"),         # drop (SOFT)
            rec("go", "pass"),           # keep (healthy)
        ]
        kept = filter_correct(records)
        self.assertEqual({r["lang"] for r in kept}, {"python", "go"})

    def test_drops_sentinel_leak_in_cot(self):
        # a pass+healthy trajectory whose cot_text carries the sentinel has a
        # corrupted CoT/code boundary -> dropped at admission, counted, not silent.
        records = [
            rec("python", "pass"),                                            # keep
            rec("markdown", "pass", cot_text=f"reason {SENTINEL} oops"),       # drop: leak
            rec("java", "pass", cot_text=f"{SENTINEL}"),                       # drop: unhealthy, NOT counted as leak
        ]
        kept, drop = filter_correct_report(records)
        self.assertEqual({r["lang"] for r in kept}, {"python"})
        self.assertEqual(drop["sentinel_leak_dropped"], 1)

    def test_cell_counts(self):
        records = [rec("python", difficulty="easy"), rec("python", difficulty="easy"),
                   rec("c", difficulty="hard")]
        self.assertEqual(cell_counts(records),
                         {("python", "easy"): 2, ("c", "hard"): 1})


if __name__ == "__main__":
    unittest.main()
