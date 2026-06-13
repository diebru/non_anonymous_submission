"""CPU tests for the Phase-1 inference harness (no GPU / no vLLM).

Covers the frozen prompt assembly, manifest-driven selection, the striped shard
split, and the full orchestration (select -> render -> generate -> parse ->
records) via a fake runner, asserting every emitted trajectory is schema-valid
and the McEval result files are well-formed. The only untested layer is the vLLM
call itself (server-only), which the fake runner stands in for.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsmc import contract
from tsmc.config import get_paths
from tsmc.constants import SENTINEL
from tsmc.inference import harness, prompts
from tsmc.inference.harness import HarnessConfig, merge_shards, run_task, shard_units
from tsmc.inference.runner import GenOutput, RunnerConfig
from tsmc.schema import validate_record

PATHS = get_paths()


# --- a CPU stand-in for VLLMRunner --------------------------------------------

class FakeRunner:
    """Implements the 3 methods run_task uses; emits a contract-valid output."""

    def __init__(self, max_tokens: int = 2048):
        self.cfg = RunnerConfig(model_path="fake-model", max_tokens=max_tokens)

    def render(self, messages):
        return messages[-1]["content"]

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompts):
        body = "def stub():\n    return 1"
        text = f"reasoning here\n{SENTINEL}\n```python\n{body}\n```"
        return [GenOutput(text=text, finish_reason="stop", n_prompt_tokens=10,
                          n_output_tokens=12, arrival_time=1.0, finished_time=2.0)
                for _ in prompts]


# --- runner config (LoRA wiring is CPU-inspectable; vLLM itself is server-only) --

class TestRunnerConfig(unittest.TestCase):
    def test_base_model_has_no_lora(self):
        cfg = RunnerConfig(model_path="m")
        self.assertIsNone(cfg.lora_path)
        self.assertEqual(cfg.max_lora_rank, 16)  # >= our Phase-3 LoRA rank 8

    def test_adapter_path_carried(self):
        cfg = RunnerConfig(model_path="base", lora_path="/w/lora_sft_run01")
        self.assertEqual(cfg.lora_path, "/w/lora_sft_run01")


# --- prompt assembly (frozen contract) ----------------------------------------

class TestPromptAssembly(unittest.TestCase):
    def test_baseline_has_no_gamma_marker(self):
        p = contract.assemble_reasoning_prompt("Do X.", "python", "f", 1.0)
        self.assertTrue(p.startswith("Do X."))
        self.assertNotIn(contract.GAMMA_DELIMITER, p)
        self.assertIn(SENTINEL, p)  # directive carries the sentinel

    def test_compressed_includes_marker(self):
        p = contract.assemble_reasoning_prompt("Do X.", "python", "f", 0.5)
        self.assertIn("<|eot_id|>0.5<|eot_id|>", p)

    def test_explanation_stage_prompts(self):
        s1 = contract.explanation_stage1_prompt("Describe this code.  ")
        self.assertEqual(s1, "Describe this code.")
        s2 = contract.explanation_stage2_prompt("python", "def f():", "a description")
        self.assertIn("def f():", s2)
        self.assertIn("a description", s2)

    def test_chat_messages_system_optional(self):
        self.assertEqual(len(prompts.chat_messages("hi")), 1)
        msgs = prompts.chat_messages("hi", system="sys")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


# --- selection ----------------------------------------------------------------

class TestSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idx = prompts.manifest_index(PATHS)

    def test_generation_trio_units_populated(self):
        units = prompts.select_units("generation", "train", PATHS, trio_only=True, limit=4, idx=self.idx)
        self.assertTrue(units)
        langs = {u.mceval_lang for u in units}
        self.assertTrue(langs <= {"Python", "C", "Rust"})
        for u in units:
            self.assertEqual(u.task_type, "generation")
            self.assertEqual(u.split, "train_problems")
            self.assertIsNone(u.completion_subtype)
            self.assertTrue(u.entry_point)
            self.assertIs(u.record.get("test") is not None, True)

    def test_completion_has_subtype(self):
        units = prompts.select_units("completion", "test", PATHS, trio_only=True, limit=6, idx=self.idx)
        self.assertTrue(units)
        for u in units:
            self.assertIn(u.completion_subtype, ("single", "multi", "span"))

    def test_explanation_only_executable_core(self):
        units = prompts.select_units("explanation", "train", PATHS, trio_only=True, limit=4, idx=self.idx)
        self.assertTrue(units)
        for u in units:
            # result record is the generation record -> has a test + signature
            self.assertIsNotNone(u.record.get("test"))
            self.assertTrue(u.signature)

    def test_limit_is_per_language(self):
        units = prompts.select_units("generation", "train", PATHS, trio_only=True, limit=3, idx=self.idx)
        by_lang = prompts.group_by_language(units)
        for us in by_lang.values():
            self.assertLessEqual(len(us), 3)


# --- shard split --------------------------------------------------------------

class TestShardSplit(unittest.TestCase):
    def test_stripe_partitions_disjointly(self):
        items = list(range(23))
        a = shard_units(items, 2, 0)
        b = shard_units(items, 2, 1)
        self.assertEqual(sorted(a + b), items)
        self.assertEqual(set(a) & set(b), set())

    def test_single_shard_returns_all(self):
        items = list(range(5))
        self.assertEqual(shard_units(items, 1, 0), items)


# --- full orchestration via the fake runner -----------------------------------

class TestRunTaskCPU(unittest.TestCase):
    def _run(self, task, split, td):
        paths = get_paths()
        object.__setattr__(paths, "generations_dir", Path(td))
        cfg = HarnessConfig(model_id="qwen2.5-coder-3b-instruct")
        return run_task(task, split, FakeRunner(), cfg, paths, trio_only=True, limit=2), paths, cfg

    def test_generation_emits_valid_records_and_results(self):
        with tempfile.TemporaryDirectory() as td:
            summary, paths, cfg = self._run("generation", "train", td)
            self.assertGreater(summary["n_units"], 0)
            out = Path(summary["out_dir"])
            traj_files = list((out / "trajectories").glob("*.jsonl"))
            res_files = list((out / "result").glob("*.jsonl"))
            self.assertTrue(traj_files and res_files)
            for f in traj_files:
                for line in f.read_text().splitlines():
                    rec = json.loads(line)
                    self.assertEqual(rec["pass"], False)  # provisional pre-eval
                    self.assertEqual(rec["gamma"], 1.0)
                    self.assertEqual(rec["cot_origin"], "original")
                    self.assertEqual(rec["compression_method"], "model_side")
                    self.assertEqual(validate_record(rec), [])
            # result rows carry raw_generation for McEval
            row = json.loads(res_files[0].read_text().splitlines()[0])
            self.assertIn("raw_generation", row)
            self.assertTrue(row["raw_generation"][0].startswith("```"))

    def test_explanation_two_pass_records(self):
        with tempfile.TemporaryDirectory() as td:
            summary, paths, cfg = self._run("explanation", "train", td)
            out = Path(summary["out_dir"])
            recs = [json.loads(x) for f in (out / "trajectories").glob("*.jsonl")
                    for x in f.read_text().splitlines()]
            self.assertTrue(recs)
            for rec in recs:
                self.assertEqual(rec["compression_method"], "post_hoc")
                self.assertIn("STAGE 2", rec["raw_full_output"])
                self.assertEqual(validate_record(rec), [])

    def test_completion_records_have_subtype(self):
        with tempfile.TemporaryDirectory() as td:
            summary, paths, cfg = self._run("completion", "test", td)
            out = Path(summary["out_dir"])
            recs = [json.loads(x) for f in (out / "trajectories").glob("*.jsonl")
                    for x in f.read_text().splitlines()]
            self.assertTrue(recs)
            for rec in recs:
                self.assertIn(rec["completion_subtype"], ("single", "multi", "span"))
                self.assertEqual(validate_record(rec), [])


# --- sharded run + merge equals single run ------------------------------------

class TestShardMerge(unittest.TestCase):
    def test_two_shards_merge_to_full(self):
        cfg = HarnessConfig(model_id="qwen2.5-coder-3b-instruct")
        with tempfile.TemporaryDirectory() as td:
            paths = get_paths()
            object.__setattr__(paths, "generations_dir", Path(td))
            run_task("generation", "train", FakeRunner(), cfg, paths,
                     trio_only=True, limit=2, shards=2, shard_id=0)
            run_task("generation", "train", FakeRunner(), cfg, paths,
                     trio_only=True, limit=2, shards=2, shard_id=1)
            m = merge_shards("generation", "train", cfg, paths)
            merged = Path(m["out_dir"])
            total = sum(len(f.read_text().splitlines())
                        for f in (merged / "result").glob("*.jsonl"))
            self.assertEqual(total, sum(m["by_language"].values()))
            self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main()
