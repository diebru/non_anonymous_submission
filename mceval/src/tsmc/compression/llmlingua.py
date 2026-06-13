"""LLMLingua-2 compressor + model-tokenizer counter (roadmap Phase 2). SERVER-ONLY.

Thin wrappers around the two heavy, server-only dependencies that the CPU-testable
core (``tsmc.compression.corpus``) consumes as injected callables:

  * ``Lingua2Compressor`` -- the LLMLingua-2 ``PromptCompressor`` (TokenSkip-qwen
    faithful: bare ``compress_prompt(text, rate=gamma)``), returning a normalized
    ``CompressionResult``.
  * ``make_token_counter`` -- the Qwen tokenizer, reproducing the Phase-1 harness'
    ``count_tokens`` EXACTLY (``len(tok(text, add_special_tokens=False).input_ids)``)
    so ``cot_token_count`` stays in the model-token domain (not LLMLingua-2's XLM-R
    domain).

``llmlingua`` / ``transformers`` are imported lazily inside ``load()`` so this
module stays importable on a CPU-only machine; never run the real compressor
locally (docs/WORKFLOW.md s2). Requires ``tokenskip_env`` + the pinned checkpoint.
"""
from __future__ import annotations

from typing import Any, Callable

from tsmc.compression.corpus import CompressionResult

# Checkpoint pinned in committed code so the pin reaches the server via git even
# though configs/run_metadata.yaml is gitignored. TokenSkip's exact choice
# (TokenSkip/LLMLingua.py). SHA verified on the server (tokenskip_env), 2026-06-01.
DEFAULT_LLMLINGUA2_CHECKPOINT = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
DEFAULT_LLMLINGUA2_SHA = "ebaba9b0e874dadd3003ffcff828e4397e568089"


class Lingua2Compressor:
    """Lazily-constructed LLMLingua-2 compressor. Build on CPU, ``.load()`` on GPU."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_LLMLINGUA2_CHECKPOINT,
        use_llmlingua2: bool = True,
        device_map: str = "cuda",
        **compress_kwargs: Any,
    ):
        self.checkpoint = checkpoint
        self.use_llmlingua2 = use_llmlingua2
        self.device_map = device_map
        # Decision (this run): TokenSkip-qwen faithful -> no force_tokens / digit /
        # drop_consecutive. Any override flows through here and is recorded as
        # provenance by the driver.
        self.compress_kwargs = compress_kwargs
        self._pc: Any = None

    def load(self) -> "Lingua2Compressor":
        from llmlingua import PromptCompressor  # deferred: server-only

        self._pc = PromptCompressor(
            model_name=self.checkpoint,
            use_llmlingua2=self.use_llmlingua2,
            device_map=self.device_map,
        )
        return self

    def compress(self, text: str, rate: float) -> CompressionResult:
        """One ``compress_prompt`` call at target ``rate`` (= our gamma)."""
        if self._pc is None:
            raise RuntimeError("Compressor not loaded; call .load() first (server/GPU).")
        out = self._pc.compress_prompt(text, rate=rate, **self.compress_kwargs)
        return CompressionResult(
            compressed_text=out["compressed_prompt"],
            origin_tokens=out.get("origin_tokens"),
            compressed_tokens=out.get("compressed_tokens"),
            rate=out.get("rate"),
            ratio=out.get("ratio"),
            saving=out.get("saving"),
        )

    @property
    def compress_fn(self) -> Callable[[str, float], CompressionResult]:
        """Bound callable for injection into ``corpus.compress_record``."""
        return self.compress


def make_token_counter(repo: str, trust_remote_code: bool = True) -> Callable[[str], int]:
    """Return a Qwen token counter identical to the Phase-1 harness' ``count_tokens``.

    ``repo`` is the HF model id/path (from run_metadata ``models.<id>.hf_repo``).
    Loads the tokenizer once (CPU) and closes over it.
    """
    from transformers import AutoTokenizer  # deferred: server dep (CPU is fine)

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code)

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tok(text, add_special_tokens=False).input_ids)

    return count


def resolve_checkpoint_sha(checkpoint: str = DEFAULT_LLMLINGUA2_CHECKPOINT) -> str | None:
    """Best-effort HF commit SHA for the checkpoint (provenance). None on failure."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(checkpoint).sha
    except Exception:
        return None
