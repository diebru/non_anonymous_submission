"""CPU tests for the Phase-3 SFT format core (example build, round-trip, decontam).

No GPU / transformers / llmlingua: synthetic compressed records + a hand-built
ProblemUnit exercise the pure logic. The byte-identity-with-inference guard is the
critical one -- it pins the SFT user turn (and the gamma marker) to the same frozen
assembler Phase-4 inference uses.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsmc.constants import GAMMA_GRID, SENTINEL  # noqa: E402
from tsmc.contract import GAMMA_DELIMITER, assemble_reasoning_prompt, parse_generation  # noqa: E402
from tsmc.inference.prompts import ProblemUnit, reasoning_user_text  # noqa: E402
from tsmc.sft import (  # noqa: E402
    build_assistant_target,
    build_example,
    decontaminate,
    gamma_marker_consistent,
    select_variants,
)

INSTR = "Write a python function add(a, b) that returns their sum."
CODE = "def add(a, b):\n    return a + b"
COT = "We read a and b then return a plus b as the result of the function call"


def make_unit(task_id="Python/1", lang="python", entry_point="add", instruction=INSTR):
    return ProblemUnit(
        problem_id=task_id.split("/")[0].lower() + "/" + task_id.split("/")[1],
        task_type="generation", completion_subtype=None,
        mceval_task_id=task_id, mceval_lang=lang.capitalize(), fence_lang=lang,
        entry_point=entry_point, signature=None, difficulty="easy",
        difficulty_source="level_propagated", split="train_problems",
        membership="gen+expl+compl", prompt_instruction=instruction, record={},
    )


def make_record(gamma=0.5, cot=COT, code=CODE, task_id="Python/1", **over):
    rec = {
        "problem_id": task_id.split("/")[0].lower() + "/" + task_id.split("/")[1],
        "task_type": "generation", "gamma": gamma, "compression_ratio": gamma,
        "cot_text": cot, "code_snippet": code, "cot_token_count": len(cot.split()),
        "lang": "python", "difficulty": "easy",
        "cot_origin": "original" if gamma >= 1.0 else "compressed",
        "_provenance": {"mceval_task_id": task_id},
    }
    rec.update(over)
    return rec


class TestAssistantTarget(unittest.TestCase):
    def test_structure_and_roundtrip(self):
        target = build_assistant_target(COT, CODE, "python")
        self.assertIn(f"\n{SENTINEL}\n```python\n", target)
        pr = parse_generation(target, entry_point="add")
        self.assertEqual(pr.code_snippet, CODE)
        self.assertIn(pr.status.parser_branch, ("sentinel", "multi_fence"))

    def test_cot_is_rstripped(self):
        target = build_assistant_target(COT + "\n\n  ", CODE, "python")
        self.assertTrue(target.startswith(COT + "\n" + SENTINEL))


class TestBuildExample(unittest.TestCase):
    def test_ok_example_messages(self):
        res = build_example(make_record(gamma=0.5), make_unit())
        self.assertTrue(res.ok, res.reason)
        self.assertEqual([m["role"] for m in res.messages], ["user", "assistant"])
        # assistant round-trips to the same code on a clean branch
        pr = parse_generation(res.messages[1]["content"], entry_point="add")
        self.assertEqual(pr.code_snippet, CODE)
        self.assertIn(pr.status.parser_branch, ("sentinel", "multi_fence"))

    def test_byte_identity_with_inference(self):
        """The user turn MUST equal the frozen inference assembler (the Phase-3 freeze)."""
        unit = make_unit()
        res = build_example(make_record(gamma=0.5), unit)
        user = res.messages[0]["content"]
        self.assertEqual(user, reasoning_user_text(unit, 0.5))
        self.assertEqual(user, assemble_reasoning_prompt(INSTR, "python", "add", 0.5))

    def test_marker_present_when_compressed(self):
        res = build_example(make_record(gamma=0.5), make_unit())
        self.assertIn("<|eot_id|>0.5<|eot_id|>", res.messages[0]["content"])

    def test_marker_absent_at_baseline(self):
        res = build_example(make_record(gamma=1.0), make_unit())
        self.assertTrue(res.ok, res.reason)
        self.assertNotIn(GAMMA_DELIMITER, res.messages[0]["content"])

    def test_all_grid_gammas_build(self):
        unit = make_unit()
        for g in GAMMA_GRID:
            res = build_example(make_record(gamma=g), unit)
            self.assertTrue(res.ok, f"gamma={g}: {res.reason}")
            self.assertEqual(gamma_marker_consistent(res.messages[0]["content"], g), True)

    def test_empty_code_is_dropped(self):
        """An empty code_snippet (pre-fix corpus artifact) is never a valid target."""
        res = build_example(make_record(gamma=0.5, code="   "), make_unit())
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, "empty_code")

    def test_fence_in_code_is_dropped(self):
        """Code that contains a fence terminator breaks the round-trip -> dropped."""
        bad_code = 'x = """\n```\n"""'  # a ``` inside the body truncates the fence
        res = build_example(make_record(gamma=0.5, code=bad_code), make_unit())
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, "code_roundtrip_mismatch")


class TestGammaSampling(unittest.TestCase):
    def _corpus(self, n_traj=5):
        recs = []
        for t in range(n_traj):
            for g in GAMMA_GRID:
                recs.append(make_record(gamma=g, task_id=f"Python/{t}"))
        return recs

    def test_all_keeps_everything(self):
        recs = self._corpus(5)
        out = select_variants(recs, policy="all")
        self.assertEqual(len(out), len(recs))

    def test_random_k_caps_per_trajectory(self):
        recs = self._corpus(5)
        out = select_variants(recs, policy="random-k", k=4, seed=42)
        self.assertEqual(len(out), 5 * 4)
        # deterministic under the same seed
        out2 = select_variants(recs, policy="random-k", k=4, seed=42)
        self.assertEqual([r["gamma"] for r in out], [r["gamma"] for r in out2])

    def test_random_k_keeps_all_when_fewer(self):
        recs = [make_record(gamma=g, task_id="Python/1") for g in (1.0, 0.5)]
        out = select_variants(recs, policy="random-k", k=6)
        self.assertEqual(len(out), 2)

    def test_bad_policy_raises(self):
        with self.assertRaises(ValueError):
            select_variants([], policy="nope")


class TestDecontamination(unittest.TestCase):
    MANIFEST = [
        {"problem_id": "python/1", "split": "train_problems"},
        {"problem_id": "python/2", "split": "train_problems"},
        {"problem_id": "python/9", "split": "test_problems"},
    ]

    def test_clean(self):
        rep = decontaminate(["python/1", "python/2"], self.MANIFEST)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["n_test_leak"], 0)
        self.assertEqual(rep["n_not_in_train"], 0)

    def test_detects_test_leak(self):
        rep = decontaminate(["python/1", "python/9"], self.MANIFEST)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["n_test_leak"], 1)
        self.assertIn("python/9", rep["leaked"])

    def test_detects_not_in_train(self):
        rep = decontaminate(["python/1", "python/404"], self.MANIFEST)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["n_not_in_train"], 1)

    def test_canonical_normalization(self):
        # an upper-cased SQL-style id normalizes to the canonical train id
        rep = decontaminate(["Python/1"], self.MANIFEST)
        self.assertTrue(rep["ok"])


if __name__ == "__main__":
    unittest.main()
