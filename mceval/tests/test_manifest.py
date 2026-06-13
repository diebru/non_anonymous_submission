"""CPU-only tests for the split-manifest generator (roadmap s6).

Runs against the vendored McEval data; reproduces on any checkout. stdlib only.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc import manifest as MAN  # noqa: E402
from tsmc.constants import SPLITS  # noqa: E402

_TRAIN, _TEST = SPLITS


class TestManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = MAN.build_manifest_rows(seed=42)

    def test_shape_and_columns(self):
        self.assertEqual(len(self.rows), 2066)
        for r in self.rows[:5]:
            self.assertEqual(set(r), set(MAN.MANIFEST_COLUMNS))
        ids = [r["problem_id"] for r in self.rows]
        self.assertEqual(len(set(ids)), len(ids))  # unique

    def test_membership_and_difficulty_source(self):
        membership = Counter(r["membership"] for r in self.rows)
        self.assertEqual(membership["gen+expl+compl"], 2007)
        self.assertEqual(membership["expl+compl"], 59)
        source = Counter(r["difficulty_source"] for r in self.rows)
        self.assertEqual(source["level_propagated"], 2007)
        self.assertEqual(source["derived_proxy"], 59)

    def test_sql_is_canonical_lowercase(self):
        sql = [r for r in self.rows if r["language"] == "sql"]
        self.assertEqual(len(sql), 59)
        # all SQL ids are lower-cased (the casing fix); no "SQL/.." leaks through
        self.assertTrue(all(r["problem_id"].startswith("sql/") for r in sql))
        # derived proxy spreads SQL across all three difficulties
        self.assertEqual({r["difficulty"] for r in sql}, {"easy", "middle", "hard"})

    def test_global_split_8020(self):
        n_train = sum(1 for r in self.rows if r["split"] == _TRAIN)
        self.assertEqual(n_train, round(0.8 * 2066))  # 1653
        self.assertEqual(len(self.rows) - n_train, 413)

    def test_no_leakage_one_split_per_problem(self):
        train = {r["problem_id"] for r in self.rows if r["split"] == _TRAIN}
        test = {r["problem_id"] for r in self.rows if r["split"] == _TEST}
        self.assertEqual(train & test, set())
        self.assertEqual(len(train | test), len(self.rows))

    def test_validate_passes(self):
        self.assertEqual(MAN.validate_manifest(self.rows), [])

    def test_determinism_same_seed(self):
        again = MAN.build_manifest_rows(seed=42)
        self.assertEqual(self.rows, again)

    def test_different_seed_changes_split_only(self):
        other = MAN.build_manifest_rows(seed=7)
        self.assertEqual(MAN.validate_manifest(other), [])  # still balanced
        split42 = {r["problem_id"]: r["split"] for r in self.rows}
        split7 = {r["problem_id"]: r["split"] for r in other}
        self.assertNotEqual(split42, split7)  # assignment differs
        # membership/difficulty are seed-independent
        diff42 = {r["problem_id"]: r["difficulty"] for r in self.rows}
        diff7 = {r["problem_id"]: r["difficulty"] for r in other}
        self.assertEqual(diff42, diff7)

    def test_validate_catches_duplicate(self):
        bad = self.rows + [self.rows[0]]
        self.assertTrue(any("duplicate" in e for e in MAN.validate_manifest(bad)))

    def test_validate_catches_global_imbalance(self):
        bad = [dict(r) for r in self.rows]
        for r in bad:  # force everything to train -> breaks target
            r["split"] = _TRAIN
        self.assertTrue(any("train size" in e for e in MAN.validate_manifest(bad)))

    def test_roundtrip_write_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "m.csv"
            MAN.write_manifest(self.rows, path)
            back = MAN.read_manifest(path)
            self.assertEqual(back, self.rows)


if __name__ == "__main__":
    unittest.main()
