"""CoT/code separation contract + parsing (roadmap s4). FROZEN in Phase 0.

Public API:
  prompt side (byte-identical SFT<->inference):
    gamma_marker, generation_directive, explanation_stage2_prompt,
    GAMMA_DELIMITER, GENERATION_DIRECTIVE, EXPLANATION_STAGE2_TEMPLATE
  parse side:
    parse_generation, parse_completion, parse_explanation_stage2,
    extract_fenced_blocks, split_on_last_sentinel, three_way_outcome,
    ParseResult

The sentinel itself lives in tsmc.constants.SENTINEL; ExtractionStatus lives in
tsmc.schema. CPU-only and tokenizer-free.
"""
from tsmc.contract.parser import (
    ParseResult,
    extract_fenced_blocks,
    parse_completion,
    parse_explanation_stage2,
    parse_generation,
    split_on_last_sentinel,
    three_way_outcome,
)
from tsmc.contract.prompt import (
    EXPLANATION_STAGE2_TEMPLATE,
    GAMMA_DELIMITER,
    GAMMA_DELIMITERS,
    GENERATION_DIRECTIVE,
    REGION_SEP,
    assemble_reasoning_prompt,
    explanation_stage1_prompt,
    explanation_stage2_prompt,
    gamma_delimiter,
    gamma_marker,
    generation_directive,
)

__all__ = [
    "ParseResult",
    "extract_fenced_blocks",
    "split_on_last_sentinel",
    "parse_generation",
    "parse_completion",
    "parse_explanation_stage2",
    "three_way_outcome",
    "gamma_marker",
    "gamma_delimiter",
    "generation_directive",
    "assemble_reasoning_prompt",
    "explanation_stage1_prompt",
    "explanation_stage2_prompt",
    "REGION_SEP",
    "GAMMA_DELIMITER",
    "GAMMA_DELIMITERS",
    "GENERATION_DIRECTIVE",
    "EXPLANATION_STAGE2_TEMPLATE",
]
