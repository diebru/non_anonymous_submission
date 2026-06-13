"""CPU-only tests for the McEval loader + the Phase-0 base-problem invariants.

Runs against the vendored (git-tracked) McEval data, so it reproduces on any
checkout. Stdlib ``unittest`` (no pytest assumed).

Run locally:
    python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc import constants as C  # noqa: E402
from tsmc import mceval_data as M  # noqa: E402


class TestTaskIdHelpers(unittest.TestCase):
    def test_base_problem_id(self):
        self.assertEqual(M.base_problem_id("AWK/1"), "AWK/1")
        self.assertEqual(M.base_problem_id("VimScript/2-0-single"), "VimScript/2")
        self.assertEqual(M.base_problem_id("Python/10-3-span"), "Python/10")

    def test_canonical_base_id_normalizes_case(self):
        # The sql/SQL reconciliation: both map to the same canonical key.
        self.assertEqual(M.canonical_base_id("sql/5"), "sql/5")
        self.assertEqual(M.canonical_base_id("SQL/5-0-single"), "sql/5")
        self.assertEqual(M.canonical_base_id("Python/1"), "python/1")

    def test_completion_subtype(self):
        self.assertEqual(M.completion_subtype("VimScript/2-0-single"), "single")
        self.assertEqual(M.completion_subtype("Python/10-3-span"), "span")
        self.assertIsNone(M.completion_subtype("AWK/1"))  # generation/explanation


class TestMcEvalData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = M.load_generation()
        cls.expl = M.load_explanation()
        cls.compl = M.load_completion(subset="merge")

    def test_row_counts(self):
        self.assertEqual(len(self.gen), 2007)
        self.assertEqual(len(self.expl), 2066)
        self.assertEqual(len(self.compl), 10128)

    def test_completion_merge_equals_subtypes(self):
        total = sum(
            len(M.load_completion(subset=s)) for s in ("single", "multi", "span")
        )
        self.assertEqual(total, len(self.compl))

    def test_base_problem_overlap_normalized(self):
        gen_n = {M.canonical_base_id(r["task_id"]) for r in self.gen}
        expl_n = {M.canonical_base_id(r["task_id"]) for r in self.expl}
        compl_n = {M.canonical_base_id(r["task_id"]) for r in self.compl}
        # generation core is contained in both other tasks
        self.assertTrue(gen_n <= expl_n)
        self.assertTrue(gen_n <= compl_n)
        # normalized union is 2066; explanation and completion share the same set
        self.assertEqual(len(gen_n | expl_n | compl_n), 2066)
        self.assertEqual(expl_n, compl_n)
        self.assertEqual(len(gen_n), 2007)

    def test_raw_overlap_inflated_by_sql_casing(self):
        gen_r = {M.base_problem_id(r["task_id"]) for r in self.gen}
        expl_r = {M.base_problem_id(r["task_id"]) for r in self.expl}
        compl_r = {M.base_problem_id(r["task_id"]) for r in self.compl}
        # Raw (case-sensitive) union is inflated to 2125 by sql vs SQL.
        self.assertEqual(len(gen_r | expl_r | compl_r), 2125)
        expl_only = {b.split("/")[0] for b in (expl_r - gen_r - compl_r)}
        compl_only = {b.split("/")[0] for b in (compl_r - gen_r - expl_r)}
        self.assertEqual(expl_only, {"sql"})
        self.assertEqual(compl_only, {"SQL"})

    def test_difficulty_distribution(self):
        from collections import Counter

        diff = Counter(r.get("level") for r in self.gen)
        self.assertEqual(diff["easy"], 1221)
        self.assertEqual(diff["middle"], 401)
        self.assertEqual(diff["hard"], 385)

    def test_sentinel_absent(self):
        for rows in (self.gen, self.expl, self.compl):
            for row in rows:
                for _key, value in M.iter_string_fields(row):
                    self.assertNotIn(C.SENTINEL, value)


if __name__ == "__main__":
    unittest.main()
