"""LLMLingua-2 multi-gamma compression (roadmap Phase 2).

Two layers:
  * ``corpus`` -- CPU-testable, dependency-light core: expand one verified-correct
    (gamma=1.0) trajectory into the 12-gamma family of compressed-CoT variants,
    compressing only the CoT region and holding the sentinel / fenced code / gamma
    marker out of compression by construction. Heavy work is injected as callables.
  * ``llmlingua`` -- SERVER-ONLY wrappers (lazy imports) for the LLMLingua-2
    ``PromptCompressor`` and the Qwen token counter. Requires tokenskip_env + the
    pinned checkpoint; never run the real compressor locally (docs/WORKFLOW.md s2).

The CPU core is importable anywhere; ``llmlingua`` only pulls in torch/llmlingua/
transformers when its ``Lingua2Compressor.load`` / ``make_token_counter`` run.
"""
from tsmc.compression.corpus import (
    CompressionParams,
    CompressionResult,
    aggregate_monotonic,
    aggregate_token_medians,
    check_scaffolding_intact,
    compress_record,
    trajectory_monotonic,
)

__all__ = [
    "CompressionParams",
    "CompressionResult",
    "compress_record",
    "check_scaffolding_intact",
    "trajectory_monotonic",
    "aggregate_token_medians",
    "aggregate_monotonic",
]
