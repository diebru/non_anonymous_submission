"""vLLM generation harness (roadmap Phases 1 and 4).

Public API:
  prompts (CPU, tokenizer-free): select_units, ProblemUnit, reasoning_user_text,
    explanation_stage1_user, explanation_stage2_user, stage1_user_text,
    chat_messages, group_by_language, manifest_index, SPLIT_VALUE, TRIO_LOWER
  runner (SERVER/GPU): VLLMRunner, RunnerConfig, GenOutput
  harness: HarnessConfig, plan_task (CPU dry-run), run_task (GPU)

``vllm`` is imported lazily inside the runner so prompt construction / dry-runs
stay importable on CPU. Real generation is server-only (docs/WORKFLOW.md s2).
"""
from tsmc.inference.harness import (
    HarnessConfig,
    merge_shards,
    plan_task,
    run_task,
    shard_units,
)
from tsmc.inference.prompts import (
    SPLIT_VALUE,
    TRIO_LOWER,
    ProblemUnit,
    chat_messages,
    explanation_stage1_user,
    explanation_stage2_user,
    group_by_language,
    manifest_index,
    reasoning_user_text,
    select_units,
    stage1_user_text,
)
from tsmc.inference.runner import GenOutput, RunnerConfig, VLLMRunner

__all__ = [
    "select_units",
    "ProblemUnit",
    "reasoning_user_text",
    "explanation_stage1_user",
    "explanation_stage2_user",
    "stage1_user_text",
    "chat_messages",
    "group_by_language",
    "manifest_index",
    "SPLIT_VALUE",
    "TRIO_LOWER",
    "VLLMRunner",
    "RunnerConfig",
    "GenOutput",
    "HarnessConfig",
    "plan_task",
    "run_task",
    "merge_shards",
    "shard_units",
]
