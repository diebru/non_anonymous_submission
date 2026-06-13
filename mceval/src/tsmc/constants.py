"""Frozen project constants.

These values are pinned by the Phase-0 decision sheet (docs/PROJECT_ROADMAP.md s5)
and schema (s7). This module is the single Python source of truth; later modules
import from here rather than re-declaring values. Changing anything here is a
contract change: it must be reflected in the run-metadata prompt_template_hash
and re-frozen.
"""
from __future__ import annotations

# --- Decision #1: CoT/code separation sentinel ---------------------------------
# Own line; nonce pinned at init. All-ASCII + uppercase + @@@-delimiter -> negligible
# collision and tokenizer-stable; matched as a decoded-text string, never compressed.
# Rejected "<<<FINAL_CODE>>>": ">>>" appears in every Python REPL docstring; "<<"/">>"
# are shift/heredoc operators.
SENTINEL: str = "@@@FINAL_CODE_7F3A9@@@"

# --- Decision #2: gamma grid ---------------------------------------------------
# gamma = fraction of CoT tokens RETAINED (1.0 = full CoT; lower = more compression).
# Dense near 1.0 for the Goal-3 sweet spot; long tail for the collapse region.
# Curves are plotted vs the MEASURED cot_token_count, not gamma itself.
GAMMA_GRID: tuple[float, ...] = (
    1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
)
GAMMA_BASELINE: float = 1.0  # no compression; pre-SFT prompt carries no gamma marker

# --- Decision #7: reproducibility / decoding -----------------------------------
SEED: int = 42          # manifest + run provenance
NUM_RUNS: int = 1       # 1 run per (model x gamma x task) for now
TEMPERATURE: float = 0.0  # greedy (note: vLLM greedy is not bitwise-deterministic)
TOP_P: float = 0.95

# --- Split ratio (roadmap s6) --------------------------------------------------
TRAIN_FRACTION: float = 0.8  # ~1,653 train / ~413 test base problems

# --- Decision #5: completion gate (tentative; validate in Phase 1) -------------
COMPLETION_GATE_MIN_COT_TOKENS: int = 30
COMPLETION_GATE_MIN_COT_CODE_RATIO: float = 1.0

# --- Schema enums (roadmap s7) -------------------------------------------------
TASK_TYPES: tuple[str, ...] = ("generation", "explanation", "completion")
COMPLETION_SUBTYPES: tuple[str, ...] = ("single", "multi", "span")
DIFFICULTY_LEVELS: tuple[str, ...] = ("easy", "middle", "hard")
DIFFICULTY_SOURCES: tuple[str, ...] = ("level_propagated", "derived_proxy")
COMPRESSION_METHODS: tuple[str, ...] = ("model_side", "post_hoc")
COT_ORIGINS: tuple[str, ...] = ("original", "compressed")
SPLITS: tuple[str, ...] = ("train_problems", "test_problems")

# Three-way outcome: extraction failure must never be misread as a reasoning
# failure (the central confound for the concavity result).
OUTCOMES: tuple[str, ...] = ("format_fail", "exec_fail", "pass")

# Completion gate decision (Decision #5); null for non-completion rows.
GATE_DECISIONS: tuple[str, ...] = ("applied", "skipped_no_lever")

# extraction_status.parser_branch domain (roadmap s4.4).
# ``presentinel_salvage``: sentinel present but the post-sentinel region had no
# fence, so the code was recovered from the LAST fenced block in the CoT (the 3B
# often codes inside its reasoning and emits a bare/empty trailing sentinel). The
# text before that fence becomes the clean CoT. Added when the contract was
# re-frozen to recover these trajectories (was silently empty code_snippet).
PARSER_BRANCHES: tuple[str, ...] = (
    "sentinel", "fence", "direct_fill", "multi_fence", "fallback", "none",
    "presentinel_salvage",
)

# --- Model matrix (Decision #6; Qwen-only, shared ~151k tokenizer) -------------
# The non-code Qwen2.5-Instruct 3B/7B/14B ladder is the 2026-06-02 direction: a
# CoT-dominated model is hypothesised to show the aggregate energy descent the
# code-specialised Coder-3B did not (docs/PIPELINE_RUNBOOK.md). The Coder 7B/14B
# stay registered (earlier size axis); the new pipeline runs on the non-code ids.
MODEL_IDS: tuple[str, ...] = (
    "qwen2.5-3b-instruct",        # controlled pair: non-code
    "qwen2.5-coder-3b-instruct",  # controlled pair: code (same size/tokenizer)
    "qwen2.5-7b-instruct",        # non-code size axis (CoT-dominated hypothesis)
    "qwen2.5-14b-instruct",       # non-code size axis
    "qwen2.5-coder-7b-instruct",  # code size axis
    "qwen2.5-coder-14b-instruct",  # code size axis
    "llama-3.1-8b-instruct",      # cross-family robustness arm (2026-06-08); needs the
                                  # Llama-safe gamma marker (see family_of + tsmc.contract.prompt)
)


def family_of(model_id: str) -> str:
    """Model family for the family-specific gamma-marker delimiter + LlamaFactory
    template. Llama-3 needs a Llama-safe literal marker because its ``<|eot_id|>``
    is a REAL special token (the Qwen marker would collapse into control tokens);
    see ``tsmc.contract.prompt.GAMMA_DELIMITERS``.

    >>> family_of("qwen2.5-14b-instruct")
    'qwen'
    >>> family_of("llama-3.1-8b-instruct")
    'llama3'
    """
    return "llama3" if "llama" in model_id else "qwen"
