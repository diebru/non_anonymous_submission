"""CPU-only tests for the Task 0.1 skeleton: config resolution + frozen constants.

Uses the stdlib ``unittest`` (pytest is not assumed to be installed locally).

Run locally:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest

# Allow running from a fresh checkout without `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc import constants as C  # noqa: E402
from tsmc.config import (  # noqa: E402
    ENV_DATA_ROOT,
    ensure_dirs,
    find_repo_root,
    get_paths,
    load_config,
)


class TestConstants(unittest.TestCase):
    def test_gamma_grid_shape(self):
        self.assertEqual(len(C.GAMMA_GRID), 12)
        self.assertEqual(C.GAMMA_GRID[0], 1.0)
        self.assertTrue(all(0.0 < g <= 1.0 for g in C.GAMMA_GRID))
        # strictly descending, no duplicates
        self.assertEqual(list(C.GAMMA_GRID), sorted(C.GAMMA_GRID, reverse=True))
        self.assertEqual(len(set(C.GAMMA_GRID)), len(C.GAMMA_GRID))

    def test_sentinel(self):
        self.assertTrue(C.SENTINEL.startswith("@@@"))
        self.assertTrue(C.SENTINEL.endswith("@@@"))
        self.assertTrue(C.SENTINEL.isascii())
        self.assertNotIn(">>>", C.SENTINEL)  # the rejected-delimiter pitfall

    def test_seed_runs_and_enums(self):
        self.assertEqual(C.SEED, 42)
        self.assertEqual(C.NUM_RUNS, 1)
        self.assertEqual(C.TASK_TYPES, ("generation", "explanation", "completion"))
        self.assertEqual(C.COMPLETION_SUBTYPES, ("single", "multi", "span"))
        self.assertIn("format_fail", C.OUTCOMES)
        # non-code 3B/7B/14B ladder + Coder 3B/7B/14B (2026-06-02) + Llama cross-family (2026-06-08).
        self.assertEqual(len(C.MODEL_IDS), 7)
        for m in ("qwen2.5-7b-instruct", "qwen2.5-14b-instruct", "llama-3.1-8b-instruct"):
            self.assertIn(m, C.MODEL_IDS)

    def test_family_of(self):
        self.assertEqual(C.family_of("llama-3.1-8b-instruct"), "llama3")
        for m in ("qwen2.5-3b-instruct", "qwen2.5-coder-3b-instruct",
                  "qwen2.5-14b-instruct"):
            self.assertEqual(C.family_of(m), "qwen")


class TestConfig(unittest.TestCase):
    def test_repo_root_detected(self):
        root = find_repo_root()
        self.assertTrue((root / "configs" / "paths.example.yaml").is_file())

    def test_load_example_config(self):
        cfg = load_config()
        self.assertIn("_config_path", cfg)

    def test_paths_resolve_under_repo(self):
        paths = get_paths(load_config())
        self.assertEqual(paths.generations_dir, paths.data_root / "generations")
        self.assertEqual(paths.compressed_dir, paths.data_root / "compressed")
        self.assertEqual(paths.weights_dir, paths.data_root / "weights")
        self.assertEqual(paths.eval_dumps_dir, paths.data_root / "eval_dumps")
        self.assertEqual(paths.mceval_data_dir, paths.mceval_dir / "data")
        self.assertEqual(paths.sft_dir, paths.data_root / "sft")
        self.assertTrue(str(paths.manifest_path).endswith("split_manifest.csv"))
        self.assertEqual(len(paths.artifact_dirs), 5)

    def test_env_override_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_resolved = pathlib.Path(tmp).resolve()
            os.environ[ENV_DATA_ROOT] = tmp
            try:
                paths = get_paths(load_config())
                self.assertEqual(paths.data_root, tmp_resolved)
                self.assertEqual(paths.weights_dir, tmp_resolved / "weights")
            finally:
                del os.environ[ENV_DATA_ROOT]

    def test_ensure_dirs_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[ENV_DATA_ROOT] = tmp
            try:
                paths = get_paths(load_config())
                created_first = ensure_dirs(paths)
                created_second = ensure_dirs(paths)
                self.assertEqual(len(created_first), 5)
                self.assertEqual(created_second, [])
                for directory in paths.artifact_dirs:
                    self.assertTrue(directory.is_dir())
            finally:
                del os.environ[ENV_DATA_ROOT]


if __name__ == "__main__":
    unittest.main()
