"""CPU-only tests for the CoT/code separation contract (roadmap s4)."""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc import contract  # noqa: E402
from tsmc.constants import SENTINEL  # noqa: E402

S = SENTINEL


def gen_output(cot: str, lang: str, body: str) -> str:
    return f"{cot}\n{S}\n```{lang}\n{body}\n```"


class TestPromptScaffolding(unittest.TestCase):
    def test_gamma_marker_baseline_empty(self):
        self.assertEqual(contract.gamma_marker(1.0), "")
        self.assertEqual(contract.gamma_marker(1.5), "")

    def test_gamma_marker_compressed(self):
        self.assertEqual(contract.gamma_marker(0.5), "<|eot_id|>0.5<|eot_id|>")
        self.assertEqual(contract.gamma_marker(0.95), "<|eot_id|>0.95<|eot_id|>")

    def test_gamma_marker_per_family(self):
        # Default stays Qwen (byte-identical to the pre-cross-family contract).
        self.assertEqual(contract.gamma_marker(0.5, "qwen"), "<|eot_id|>0.5<|eot_id|>")
        # Llama uses the Llama-safe nonce delimiter (no <|eot_id|> special token).
        self.assertEqual(contract.gamma_marker(0.5, "llama3"),
                         "@@@GAMMA_7F3A9@@@0.5@@@GAMMA_7F3A9@@@")
        self.assertNotIn("<|eot_id|>", contract.gamma_marker(0.5, "llama3"))
        self.assertEqual(contract.gamma_marker(1.0, "llama3"), "")  # baseline omitted
        self.assertEqual(contract.gamma_delimiter("llama3"), "@@@GAMMA_7F3A9@@@")

    def test_assemble_reasoning_prompt_family_marker(self):
        p = contract.assemble_reasoning_prompt("Do X.", "python", "f", 0.5, "llama3")
        self.assertIn("@@@GAMMA_7F3A9@@@0.5@@@GAMMA_7F3A9@@@", p)
        self.assertNotIn("<|eot_id|>", p)
        # baseline carries no marker regardless of family
        self.assertNotIn("@@@GAMMA_7F3A9@@@",
                         contract.assemble_reasoning_prompt("Do X.", "python", "f", 1.0, "llama3"))

    def test_generation_directive_embeds_sentinel_and_fields(self):
        d = contract.generation_directive(lang="python", entry_point="solve")
        self.assertIn(S, d)
        self.assertIn("python", d)
        self.assertIn("solve", d)

    def test_explanation_stage2_is_cot_free_template(self):
        p = contract.explanation_stage2_prompt("python", "def f(x)", "do the thing")
        self.assertIn("do the thing", p)
        self.assertNotIn(S, p)  # stage 2 must be CoT-free / sentinel-free


class TestFenceAndSentinel(unittest.TestCase):
    def test_extract_fenced_blocks(self):
        text = "```python\nprint(1)\n```\nmid\n```\nplain\n```"
        blocks = contract.extract_fenced_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], ("python", "print(1)"))
        self.assertEqual(blocks[1][1], "plain")

    def test_split_on_last_sentinel(self):
        text = f"a {S} b {S} c"
        cot, code = contract.split_on_last_sentinel(text)
        self.assertEqual(cot, f"a {S} b ")
        self.assertEqual(code, " c")

    def test_split_no_sentinel(self):
        cot, code = contract.split_on_last_sentinel("no marker here")
        self.assertEqual(cot, "no marker here")
        self.assertIsNone(code)


class TestParseGeneration(unittest.TestCase):
    def test_clean(self):
        out = gen_output("Let me think...", "python", "def solve():\n    return 1")
        r = contract.parse_generation(out, entry_point="solve", finish_reason="stop")
        self.assertEqual(r.status.parser_branch, "sentinel")
        self.assertTrue(r.status.fence_found)
        self.assertTrue(r.status.entry_point_found)
        self.assertFalse(r.status.truncated)
        self.assertIn("def solve", r.code_snippet)
        self.assertIn("think", r.cot_text)
        self.assertEqual(contract.three_way_outcome(r.status, passed=True), "pass")
        self.assertEqual(contract.three_way_outcome(r.status, passed=False), "exec_fail")

    def test_last_sentinel_defuses_scratch(self):
        # A scratch sentinel + fence inside the CoT must not win.
        out = f"draft {S}\n```python\nWRONG\n```\nmore reasoning {S}\n```python\nRIGHT\n```"
        r = contract.parse_generation(out, entry_point=None)
        self.assertEqual(r.code_snippet, "RIGHT")
        self.assertEqual(r.status.parser_branch, "sentinel")

    def test_multi_fence_in_code_region(self):
        out = f"cot {S}\n```python\nA\n```\n```python\nB\n```"
        r = contract.parse_generation(out)
        self.assertEqual(r.status.parser_branch, "multi_fence")
        self.assertEqual(r.code_snippet, "A")  # first fence in code region

    def test_no_sentinel_salvage_fallback(self):
        out = "reasoning only\n```python\ndef solve():\n    return 1\n```"
        r = contract.parse_generation(out, entry_point="solve")
        self.assertEqual(r.status.parser_branch, "fallback")
        self.assertTrue(r.status.fence_found)
        self.assertEqual(contract.three_way_outcome(r.status, passed=True), "format_fail")

    def test_no_sentinel_no_fence_none(self):
        r = contract.parse_generation("just prose, no code")
        self.assertEqual(r.status.parser_branch, "none")
        self.assertFalse(r.status.fence_found)
        self.assertEqual(contract.three_way_outcome(r.status, passed=True), "format_fail")

    def test_truncated_is_format_fail(self):
        out = gen_output("cot", "python", "def solve(): return 1")
        r = contract.parse_generation(out, entry_point="solve", finish_reason="length")
        self.assertTrue(r.status.truncated)
        self.assertEqual(contract.three_way_outcome(r.status, passed=True), "format_fail")

    def test_entry_point_missing(self):
        out = gen_output("cot", "python", "def other():\n    return 1")
        r = contract.parse_generation(out, entry_point="solve")
        self.assertFalse(r.status.entry_point_found)

    def test_sentinel_without_fence(self):
        out = f"cot here {S}\nI forgot to fence the code"
        r = contract.parse_generation(out)
        self.assertFalse(r.status.fence_found)
        self.assertEqual(r.status.parser_branch, "sentinel")
        self.assertEqual(r.code_snippet, "")

    def test_sentinel_no_fence_anywhere_is_format_fail(self):
        # No extractable code on a fenced (generation) branch must never be a pass.
        out = f"cot here {S}\nI forgot to fence the code"
        r = contract.parse_generation(out)
        self.assertEqual(contract.three_way_outcome(r.status, passed=True), "format_fail")

    def test_presentinel_salvage_recovers_code_from_cot(self):
        # The 3B coded inside the reasoning, then a bare trailing sentinel (empty
        # code region). Recover the code from the CoT; the CoT loses the code block.
        out = f"Here is my solution:\n```python\ndef solve():\n    return 1\n```\n{S}"
        r = contract.parse_generation(out, entry_point="solve")
        self.assertEqual(r.status.parser_branch, "presentinel_salvage")
        self.assertTrue(r.status.fence_found)
        self.assertTrue(r.status.entry_point_found)
        self.assertEqual(r.code_snippet, "def solve():\n    return 1")
        self.assertNotIn("```", r.cot_text)
        self.assertIn("Here is my solution:", r.cot_text)
        self.assertEqual(contract.three_way_outcome(r.status, passed=True), "pass")

    def test_presentinel_salvage_takes_last_cot_fence(self):
        out = ("scratch\n```python\ndef bad():\n    pass\n```\nfinal:\n"
               f"```python\ndef solve():\n    return 2\n```\n\n{S}\n")
        r = contract.parse_generation(out, entry_point="solve")
        self.assertEqual(r.status.parser_branch, "presentinel_salvage")
        self.assertEqual(r.code_snippet, "def solve():\n    return 2")


class TestParseCompletion(unittest.TestCase):
    def test_direct_fill_expected(self):
        out = "```python\ndef solve():\n    return 42\n```"
        r = contract.parse_completion(out, entry_point="solve")
        self.assertEqual(r.status.parser_branch, "direct_fill")
        self.assertEqual(r.cot_text, "")
        self.assertTrue(r.status.fence_found)
        self.assertEqual(contract.three_way_outcome(r.status, passed=True), "pass")

    def test_direct_fill_unfenced_whole_text(self):
        out = "def solve():\n    return 42"
        r = contract.parse_completion(out)
        self.assertEqual(r.status.parser_branch, "direct_fill")
        self.assertFalse(r.status.fence_found)
        self.assertIn("return 42", r.code_snippet)

    def test_with_sentinel_parses_like_generation(self):
        out = gen_output("brief reason", "python", "def solve(): return 1")
        r = contract.parse_completion(out, entry_point="solve")
        self.assertEqual(r.status.parser_branch, "sentinel")
        self.assertIn("brief reason", r.cot_text)


class TestParseExplanationStage2(unittest.TestCase):
    def test_fence_first(self):
        out = "```python\ndef solve():\n    return 1\n```"
        r = contract.parse_explanation_stage2(out, entry_point="solve")
        self.assertEqual(r.status.parser_branch, "fence")
        self.assertEqual(r.cot_text, "")
        self.assertTrue(r.status.entry_point_found)

    def test_no_fence_none(self):
        r = contract.parse_explanation_stage2("just text")
        self.assertEqual(r.status.parser_branch, "none")
        self.assertFalse(r.status.fence_found)


if __name__ == "__main__":
    unittest.main()
