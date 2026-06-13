"""CPU-only tests for the eval layer (roadmap Phase 0 / Task 0.5).

Covers the local-testable parts: result-file building, the Docker command
assembly (no Docker invoked), and -- via McEval's own pure-regex extract() --
the contract<->extractor equivalence guarantee for the trio. Execution itself
(Docker) is verified on the server by scripts/verify_mceval_docker.py.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tsmc import contract, mceval_data  # noqa: E402
from tsmc.config import get_paths  # noqa: E402
from tsmc.eval import docker, results  # noqa: E402
from tsmc.eval.mceval_adapter import get_mceval_extract, mceval_extract_available  # noqa: E402


class TestResultsBuilding(unittest.TestCase):
    item = {
        "task_id": "Python/1",
        "entry_point": "f",
        "prompt": "def f(x):\n",
        "canonical_solution": "    return x + 1\n",
        "test": "def check(f):\n    assert f(1) == 2\n",
    }

    def test_fence_tag(self):
        self.assertEqual(results.fence_tag("Python"), "python")
        self.assertEqual(results.fence_tag("Rust"), "rust")

    def test_gold_and_wrap(self):
        gold = results.gold_raw_generation(self.item, "Python")
        self.assertTrue(gold.startswith("```python\n"))
        self.assertIn("def f(x):", gold)
        self.assertEqual(results.wrap_code("X", "C"), "```c\nX\n```")

    def test_build_result_item_attaches_raw_generation(self):
        out = results.build_result_item(self.item, "RG")
        self.assertEqual(out["raw_generation"], ["RG"])
        self.assertEqual(out["task_id"], "Python/1")  # original fields preserved

    def test_synthetic_contract_output_parses_to_reference(self):
        out = results.synthetic_contract_output(self.item, "Python")
        parsed = contract.parse_generation(out, entry_point="f", finish_reason="stop")
        self.assertIn("def f(x):", parsed.code_snippet)
        self.assertIn("return x + 1", parsed.code_snippet)
        self.assertNotIn("WRONG", parsed.code_snippet)  # distractor stayed in the CoT

    def test_write_result_dir_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = pathlib.Path(tmp) / "res"
            items = {"Python": [results.build_result_item(self.item, "RG")]}
            written = results.write_result_dir(items, out_dir)
            self.assertEqual(len(written), 1)
            rows = [json.loads(x) for x in (out_dir / "Python.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["raw_generation"], ["RG"])


class TestDockerDriver(unittest.TestCase):
    def test_image_ref_digest_vs_tag(self):
        self.assertEqual(
            docker.DockerEvalConfig(digest="sha256:abc").image_ref(),
            "multilingualnlp/mceval@sha256:abc",
        )
        self.assertEqual(
            docker.DockerEvalConfig(digest="v1").image_ref(),
            "multilingualnlp/mceval:v1",
        )

    def test_default_python_is_conda(self):
        self.assertEqual(docker.DockerEvalConfig(digest="sha256:abc").python_exe, "/opt/conda/bin/python")

    def test_build_command_structure(self):
        cfg = docker.DockerEvalConfig(digest="sha256:abc", network="none")
        # host dir name must be preserved in the container path so the output file matches
        cmd = docker.build_command(cfg, pathlib.Path("/data/gold_trio"), pathlib.Path("/s"))
        self.assertEqual(cmd[:3], ["docker", "run", "--rm"])
        self.assertIn("/data/gold_trio:/work/gold_trio:ro", cmd)
        self.assertIn("--network", cmd)
        self.assertIn("multilingualnlp/mceval@sha256:abc", cmd)
        self.assertIn("export PATH=/opt/conda/bin:$PATH", cmd[-1])  # conda first for subprocs
        self.assertIn("/opt/conda/bin/python eval_all.py --result_path /work/gold_trio --save_path /work/save", cmd[-1])

    def test_report_trio_execution_rust_is_soft(self):
        # Python + C clear the bar, Rust does not -> still overall PASS (Rust soft).
        scores = {"Python": {"accuracy": 1.0}, "C": {"accuracy": 1.0}, "Rust": {"accuracy": 0.5}}
        self.assertTrue(results.report_trio_execution(scores, threshold=0.9))
        # A required language failing -> overall FAIL.
        scores["C"] = {"accuracy": 0.0}
        self.assertFalse(results.report_trio_execution(scores, threshold=0.9))

    def test_run_eval_rejects_placeholder_digest(self):
        with self.assertRaises(ValueError):
            docker.run_eval(docker.DockerEvalConfig(digest="sha256:TBD"),
                            pathlib.Path("/r"), pathlib.Path("/s"))

    def test_save_file_for(self):
        self.assertEqual(
            docker.save_file_for(pathlib.Path("/a/gold_trio"), pathlib.Path("/b")),
            pathlib.Path("/b/gold_trio.jsonl"),
        )

    def test_parse_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = pathlib.Path(tmp) / "s.jsonl"
            f.write_text('Python\t{"accuracy": 1.0, "correct": 5}\nC\t{"accuracy": 0.9}\n')
            scores = docker.parse_scores(f)
            self.assertEqual(scores["Python"]["accuracy"], 1.0)
            self.assertEqual(scores["C"]["accuracy"], 0.9)


@unittest.skipUnless(mceval_extract_available(), "McEval extractor not vendored")
class TestContractExtractorEquivalence(unittest.TestCase):
    """The Task 0.5 local guarantee: our contract output extracts identically to gold."""

    def _rich(self, items):
        return [
            it for it in items
            if it.get("entry_point") and it.get("test")
            and it.get("prompt") is not None and it.get("canonical_solution") is not None
        ]

    def test_trio_contract_equals_gold(self):
        paths = get_paths()
        # local var, not a class/instance attribute (a stored function would bind as a method)
        extract = get_mceval_extract(paths)
        for lang in results.TRIO:  # Python, C, Rust
            items = self._rich(mceval_data.load_generation_language(lang, paths))[:3]
            self.assertTrue(items, f"no rich problems for {lang}")
            for it in items:
                out = results.synthetic_contract_output(it, lang)
                parsed = contract.parse_generation(out, entry_point=it["entry_point"], finish_reason="stop")
                self.assertIn(parsed.status.parser_branch, ("sentinel", "multi_fence"))
                self.assertTrue(parsed.status.entry_point_found)
                contract_extract = extract(results.wrap_code(parsed.code_snippet, lang), it, lang)
                gold_extract = extract(results.gold_raw_generation(it, lang), it, lang)
                self.assertTrue(contract_extract)
                self.assertEqual(contract_extract, gold_extract)
                self.assertIn(it["entry_point"], contract_extract)


@unittest.skipUnless(mceval_extract_available(), "McEval extractor not vendored")
class TestPythonGoldExecutesLocally(unittest.TestCase):
    """Guard the gold/contract construction is EXECUTABLE, not just parseable.

    The prompt+canonical glue bug (missing newline) passed the parsing check but
    failed execution; Python runs locally, so we exercise the real reconstruction.
    """

    def _run(self, full_code: str) -> bool:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(full_code)
            path = fh.name
        try:
            proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
            return proc.returncode == 0
        finally:
            os.unlink(path)

    def test_python_gold_and_contract_execute(self):
        paths = get_paths()
        extract = get_mceval_extract(paths)
        items = [
            it for it in mceval_data.load_generation_language("Python", paths)
            if it.get("entry_point") and it.get("test")
        ][:8]
        self.assertTrue(items)
        gold_ok = sum(self._run(extract(results.gold_raw_generation(it, "Python"), it, "Python")) for it in items)
        # >=7/8 tolerates rare McEval extraction edge cases; the glue bug gave ~2/8.
        self.assertGreaterEqual(gold_ok, 7, f"gold executed {gold_ok}/{len(items)}")
        contract_ok = 0
        for it in items:
            parsed = contract.parse_generation(
                results.synthetic_contract_output(it, "Python"),
                entry_point=it["entry_point"], finish_reason="stop",
            )
            contract_ok += self._run(extract(results.wrap_code(parsed.code_snippet, "Python"), it, "Python"))
        self.assertGreaterEqual(contract_ok, 7, f"contract executed {contract_ok}/{len(items)}")


if __name__ == "__main__":
    unittest.main()
