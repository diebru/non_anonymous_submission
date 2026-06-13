"""CPU tests for the Phase-2 compression core (variant construction + validators).

Uses injected mock callables (a deterministic word-drop compressor + a whitespace
token counter) so nothing here needs llmlingua, transformers, or a GPU.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsmc.compression.corpus import (
    CompressionParams,
    CompressionResult,
    aggregate_monotonic,
    aggregate_token_medians,
    check_scaffolding_intact,
    compress_record,
    trajectory_monotonic,
)
from tsmc.constants import SENTINEL
from tsmc.schema import validate_record

GAMMAS = (1.0, 0.9, 0.8, 0.5, 0.2)
PARAMS = CompressionParams(checkpoint="mock", checkpoint_sha="deadbeef")

COT = "First we read the integer n then we loop and accumulate the sum and return it"


def word_drop(text: str, rate: float) -> CompressionResult:
    """Keep the leading ceil(rate*N) whitespace tokens -- monotonic in rate."""
    words = text.split()
    keep = max(1, math.ceil(rate * len(words))) if words else 0
    return CompressionResult(
        compressed_text=" ".join(words[:keep]),
        origin_tokens=len(words),
        compressed_tokens=keep,
        rate=f"{keep}/{len(words)}",
    )


def ws_count(text: str) -> int:
    return len(text.split())


def source_record(task="generation", cot=COT, **over):
    method = "post_hoc" if task == "explanation" else "model_side"
    rec = {
        "problem_id": "python/1", "task_type": task, "completion_subtype": None,
        "model_id": "qwen2.5-coder-3b-instruct", "gamma": 1.0, "run_id": "run01",
        "raw_full_output": cot + "\n" + SENTINEL + "\n```python\ndef f():\n    return 1\n```",
        "cot_text": cot, "code_snippet": "def f():\n    return 1",
        "cot_token_count": ws_count(cot), "code_token_count": 6,
        "compression_ratio": 1.0, "pass": True,
        "extraction_status": {"fence_found": True, "entry_point_found": True,
                              "truncated": False, "parser_branch": "sentinel"},
        "cot_origin": "original", "compression_method": method,
        "gate_decision": None, "gate_measured_median": None,
        "split": "train_problems", "lang": "python",
        "difficulty": "easy", "difficulty_source": "level_propagated",
        "outcome": "pass", "_provenance": {"mceval_task_id": "Python/1"},
    }
    rec.update(over)
    return rec


class TestVariantConstruction(unittest.TestCase):
    def test_one_variant_per_gamma(self):
        variants = compress_record(source_record(), GAMMAS, word_drop, ws_count, PARAMS)
        self.assertEqual(len(variants), len(GAMMAS))
        self.assertEqual([v["gamma"] for v in variants], list(GAMMAS))
        for v in variants:
            self.assertEqual(v["compression_ratio"], v["gamma"])

    def test_baseline_is_passthrough(self):
        v = compress_record(source_record(), (1.0,), word_drop, ws_count, PARAMS)[0]
        self.assertEqual(v["cot_text"], COT)            # unchanged
        self.assertEqual(v["cot_origin"], "original")
        self.assertTrue(v["pass"])                       # carried (was executed)
        self.assertIsNone(v["_compression"]["llmlingua"])  # not compressed
        self.assertEqual(v["cot_token_count"], ws_count(COT))

    def test_compressed_variant(self):
        v = compress_record(source_record(), (0.5,), word_drop, ws_count, PARAMS)[0]
        self.assertEqual(v["cot_origin"], "compressed")
        self.assertFalse(v["pass"])                      # provisional until Phase 4
        self.assertTrue(v["_compression"]["source_pass"])
        self.assertIsNotNone(v["_compression"]["llmlingua"])
        self.assertLess(v["cot_token_count"], ws_count(COT))

    def test_cot_token_count_recounted(self):
        # re-counted with the injected counter, not LLMLingua-2's own number
        v = compress_record(source_record(), (0.5,), word_drop, ws_count, PARAMS)[0]
        self.assertEqual(v["cot_token_count"], ws_count(v["cot_text"]))

    def test_outcome_dropped_kept_in_provenance(self):
        v = compress_record(source_record(), (0.5,), word_drop, ws_count, PARAMS)[0]
        self.assertNotIn("outcome", v)
        self.assertEqual(v["_compression"]["source_outcome"], "pass")

    def test_compression_method_preserved(self):
        for task, expect in (("generation", "model_side"), ("explanation", "post_hoc")):
            v = compress_record(source_record(task=task), (0.5,), word_drop, ws_count, PARAMS)[0]
            self.assertEqual(v["compression_method"], expect)


class TestEdgeCases(unittest.TestCase):
    def test_empty_cot_skips_compressor(self):
        def boom(text, rate):  # must never be called on an empty CoT
            raise AssertionError("compressor called on empty CoT")

        variants = compress_record(source_record(cot="   "), GAMMAS, boom, ws_count, PARAMS)
        self.assertEqual(len(variants), len(GAMMAS))
        for v in variants:
            self.assertEqual(v["cot_token_count"], 0)
        # compressed (gamma<1) rows are flagged degenerate; baseline is just origin
        self.assertEqual(variants[-1]["_compression"]["degenerate"], "empty_cot")
        self.assertIsNone(variants[0]["_compression"]["degenerate"])


class TestScaffolding(unittest.TestCase):
    def test_intact_for_all_variants(self):
        src = source_record()
        for v in compress_record(src, GAMMAS, word_drop, ws_count, PARAMS):
            self.assertEqual(check_scaffolding_intact(src, v), [])

    def test_detects_code_change(self):
        src = source_record()
        v = compress_record(src, (0.5,), word_drop, ws_count, PARAMS)[0]
        v["code_snippet"] = "def g(): return 2"
        self.assertIn("code_snippet changed", check_scaffolding_intact(src, v))

    def test_detects_sentinel_leak(self):
        src = source_record()
        v = compress_record(src, (0.5,), word_drop, ws_count, PARAMS)[0]
        v["cot_text"] = v["cot_text"] + " " + SENTINEL
        self.assertIn("sentinel leaked into cot_text", check_scaffolding_intact(src, v))


class TestSchemaValid(unittest.TestCase):
    def test_every_variant_validates(self):
        for task in ("generation", "explanation"):
            for v in compress_record(source_record(task=task), GAMMAS, word_drop, ws_count, PARAMS):
                self.assertEqual(validate_record(v), [], f"{task} g={v['gamma']}")


class TestMonotonicity(unittest.TestCase):
    def test_trajectory_non_increasing(self):
        variants = compress_record(source_record(), GAMMAS, word_drop, ws_count, PARAMS)
        res = trajectory_monotonic(variants)
        self.assertTrue(res["monotonic"], res["violations"])
        # series is ordered gamma-descending
        self.assertEqual([g for g, _ in res["series"]], sorted(GAMMAS, reverse=True))

    def test_trajectory_violation_detected(self):
        variants = [
            {"gamma": 1.0, "cot_token_count": 5},
            {"gamma": 0.5, "cot_token_count": 9},  # strict increase as gamma falls
        ]
        res = trajectory_monotonic(variants)
        self.assertFalse(res["monotonic"])
        self.assertEqual(len(res["violations"]), 1)

    def test_aggregate_medians_monotonic(self):
        by_gamma = {1.0: [10, 12], 0.5: [6, 6], 0.2: [2, 3]}
        medians = aggregate_token_medians(by_gamma)
        self.assertEqual([g for g, _ in medians], [1.0, 0.5, 0.2])
        self.assertTrue(aggregate_monotonic(medians)["monotonic"])

    def test_aggregate_violation_detected(self):
        by_gamma = {1.0: [5], 0.5: [9]}  # median goes up as gamma falls
        agg = aggregate_monotonic(aggregate_token_medians(by_gamma))
        self.assertFalse(agg["monotonic"])
        self.assertEqual(len(agg["violations"]), 1)


if __name__ == "__main__":
    unittest.main()
