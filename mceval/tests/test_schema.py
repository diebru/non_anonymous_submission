"""CPU-only tests for the long-format record schema + validator (roadmap s7)."""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc.schema import ExtractionStatus, LongFormatRecord, validate_record  # noqa: E402


def valid_generation_record() -> dict:
    return LongFormatRecord(
        problem_id="python/1",
        task_type="generation",
        model_id="qwen2.5-coder-3b-instruct",
        gamma=1.0,
        run_id="run01",
        raw_full_output="cot ... code",
        code_snippet="def solve():\n    return 1",
        cot_token_count=12,
        compression_ratio=1.0,
        passed=True,
        extraction_status=ExtractionStatus(True, True, False, "sentinel"),
        cot_origin="original",
        compression_method="model_side",
        split="train_problems",
        lang="python",
        difficulty="easy",
        difficulty_source="level_propagated",
        cot_text="reasoning here",
    ).to_dict()


class TestSchema(unittest.TestCase):
    def test_valid_record_passes(self):
        self.assertEqual(validate_record(valid_generation_record()), [])

    def test_roundtrip(self):
        d = valid_generation_record()
        r = LongFormatRecord.from_dict(d)
        self.assertEqual(r.to_dict(), d)
        self.assertTrue(r.passed)
        self.assertEqual(r.extraction_status.parser_branch, "sentinel")

    def test_missing_required_field(self):
        d = valid_generation_record()
        del d["code_snippet"]
        self.assertTrue(any("code_snippet" in e for e in validate_record(d)))

    def test_bad_enum(self):
        d = valid_generation_record()
        d["task_type"] = "translation"
        self.assertTrue(any("task_type" in e for e in validate_record(d)))

    def test_completion_subtype_required_for_completion(self):
        d = valid_generation_record()
        d["task_type"] = "completion"
        d["compression_method"] = "model_side"
        d["completion_subtype"] = None
        self.assertTrue(any("completion_subtype" in e for e in validate_record(d)))

    def test_completion_subtype_forbidden_for_generation(self):
        d = valid_generation_record()
        d["completion_subtype"] = "single"
        self.assertTrue(any("completion_subtype" in e for e in validate_record(d)))

    def test_valid_completion_record(self):
        d = valid_generation_record()
        d.update(
            task_type="completion",
            completion_subtype="single",
            compression_method="model_side",
            gate_decision="skipped_no_lever",
            cot_text="",
            cot_token_count=0,
        )
        d["extraction_status"]["parser_branch"] = "direct_fill"
        self.assertEqual(validate_record(d), [])

    def test_gate_decision_only_completion(self):
        d = valid_generation_record()
        d["gate_decision"] = "applied"
        self.assertTrue(any("gate_decision" in e for e in validate_record(d)))

    def test_gamma_out_of_range(self):
        d = valid_generation_record()
        d["gamma"] = 1.5
        self.assertTrue(any("gamma" in e for e in validate_record(d)))

    def test_gamma_zero_invalid(self):
        d = valid_generation_record()
        d["gamma"] = 0.0
        self.assertTrue(any("gamma" in e for e in validate_record(d)))

    def test_baseline_origin_must_be_original(self):
        d = valid_generation_record()
        d["gamma"] = 1.0
        d["cot_origin"] = "compressed"
        self.assertTrue(any("cot_origin" in e for e in validate_record(d)))

    def test_compressed_record_valid(self):
        d = valid_generation_record()
        d.update(gamma=0.5, compression_ratio=0.5, cot_origin="compressed")
        self.assertEqual(validate_record(d), [])

    def test_explanation_must_be_post_hoc(self):
        d = valid_generation_record()
        d["task_type"] = "explanation"
        d["compression_method"] = "model_side"  # wrong for explanation
        self.assertTrue(any("compression_method" in e for e in validate_record(d)))

    def test_explanation_post_hoc_ok(self):
        d = valid_generation_record()
        d.update(task_type="explanation", compression_method="post_hoc")
        self.assertEqual(validate_record(d), [])

    def test_bool_not_valid_number(self):
        d = valid_generation_record()
        d["gamma"] = True  # bool is a subclass of int; must be rejected
        self.assertTrue(any("gamma" in e for e in validate_record(d)))

    def test_bad_extraction_status_branch(self):
        d = valid_generation_record()
        d["extraction_status"]["parser_branch"] = "bogus"
        self.assertTrue(any("parser_branch" in e for e in validate_record(d)))

    def test_extraction_status_wrong_type(self):
        d = valid_generation_record()
        d["extraction_status"]["fence_found"] = "yes"
        self.assertTrue(any("fence_found" in e for e in validate_record(d)))


if __name__ == "__main__":
    unittest.main()
