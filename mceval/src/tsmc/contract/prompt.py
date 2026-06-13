"""Prompt-side contract scaffolding (roadmap s4). FROZEN in Phase 0.

This is the single source for the structural prompt pieces that must be
**byte-identical between SFT (Phase 3) and inference (Phase 4)** or the model
will not honor gamma. Any change here is a contract change -> bump the
run-metadata ``prompt_template_hash`` and re-freeze.

gamma-control marker (grounded in vendored TokenSkip):
  TokenSkip injects the ratio as ``<|eot_id|>{ratio}<|eot_id|>`` appended to the
  user content, and OMITS it entirely at ratio 1.0 (baseline). See
  TokenSkip/get_llamafactory_input.py (train) and TokenSkip/evaluation.py
  (inference). For Qwen, ``<|eot_id|>`` is a *literal text* delimiter (not a
  special token), used the same way in both train and inference.

  The delimiter is therefore PER MODEL FAMILY (``GAMMA_DELIMITERS``): for Llama-3
  ``<|eot_id|>`` is a REAL special token (id 128009, end-of-turn), so the literal
  marker would be tokenized into control tokens at SFT/inference and corrupt both
  the prompt structure and gamma control. Llama uses a Llama-safe nonce delimiter
  (plain text for the Llama tokenizer; cf. the ``constants.SENTINEL`` style). The
  family is resolved from the model id via ``constants.family_of`` at every call
  site; ``family`` defaults to ``"qwen"`` so existing Qwen runs are byte-identical.

gamma convention (confirmed from TokenSkip/LLMLingua.py): the ratio is
LLMLingua-2's ``rate`` = fraction of CoT tokens RETAINED -> matches our
``constants.GAMMA_GRID`` (1.0 = full CoT).
"""
from __future__ import annotations

from tsmc.constants import GAMMA_BASELINE, SENTINEL

# Literal delimiter TokenSkip wraps the ratio in, PER MODEL FAMILY (see module
# docstring). Qwen keeps TokenSkip's original ``<|eot_id|>`` (literal text for the
# Qwen tokenizer); Llama-3 uses a nonce the Llama tokenizer treats as plain text
# (its ``<|eot_id|>`` is a real special token). Open == close, so the wrap form +
# the ``_MARKER_RE`` that parses it are identical across families.
GAMMA_DELIMITERS: dict[str, str] = {
    "qwen": "<|eot_id|>",
    "llama3": "@@@GAMMA_7F3A9@@@",
}
# Back-compat module constant: the Qwen delimiter. Existing imports of
# ``GAMMA_DELIMITER`` keep the (Qwen) default; family-aware code uses
# ``gamma_delimiter(family)``.
GAMMA_DELIMITER = GAMMA_DELIMITERS["qwen"]


def gamma_delimiter(family: str = "qwen") -> str:
    """The gamma-marker delimiter string for a model family (default Qwen)."""
    return GAMMA_DELIMITERS[family]


def gamma_marker(gamma: float, family: str = "qwen") -> str:
    """Return the gamma-control marker, or ``""`` at baseline (gamma >= 1.0).

    ``family`` selects the delimiter (default ``"qwen"`` -> byte-identical to the
    pre-cross-family contract); Llama uses the Llama-safe nonce delimiter.

    >>> gamma_marker(1.0)
    ''
    >>> gamma_marker(0.5)
    '<|eot_id|>0.5<|eot_id|>'
    >>> gamma_marker(0.5, "llama3")
    '@@@GAMMA_7F3A9@@@0.5@@@GAMMA_7F3A9@@@'
    """
    if gamma >= GAMMA_BASELINE:
        return ""
    d = GAMMA_DELIMITERS[family]
    return f"{d}{gamma}{d}"


# Output-format directive for GENERATION / COMPLETION (induced-CoT). Replaces
# TokenSkip's math "\boxed{}" answer format with our sentinel + fenced-code
# contract. {lang} is the McEval language tag; {entry_point} the required symbol.
GENERATION_DIRECTIVE = (
    "Reason step by step. When you are ready to give the final program, write the "
    "line\n{sentinel}\non its own, then output the complete solution as a single "
    "{lang} fenced code block that defines `{entry_point}`. Put no prose after the "
    "code block."
)

# Stage-2 of EXPLANATION is CoT-free and uses McEval's own template verbatim
# (roadmap s4.2); the compressed stage-1 description is substituted in.
EXPLANATION_STAGE2_TEMPLATE = (
    "Write a {lang} function {signature} to solve the following problem:\n{description}"
)


def generation_directive(lang: str, entry_point: str) -> str:
    """Frozen output-format directive for generation/completion induced-CoT."""
    return GENERATION_DIRECTIVE.format(sentinel=SENTINEL, lang=lang, entry_point=entry_point)


# Region separator between the three assembled prompt regions. A blank line keeps
# the McEval instruction, the (optional) gamma marker, and the output-format
# directive visually/structurally distinct without introducing any token the
# compressor would touch (the marker is structural scaffolding, never compressed).
REGION_SEP = "\n\n"


def assemble_reasoning_prompt(
    instruction: str, lang: str, entry_point: str, gamma: float = GAMMA_BASELINE,
    family: str = "qwen",
) -> str:
    """Assemble the full user message for GENERATION / COMPLETION (induced-CoT).

    Region order (roadmap s4.1): McEval ``instruction`` (structural) -> gamma
    marker (omitted at gamma=1.0) -> output-format directive. The model then emits
    CoT -> sentinel -> fenced code. This is the SINGLE assembler shared by Phase-1
    inference, Phase-4 inference, and Phase-3 SFT formatting, so the prompt is
    byte-identical across train and inference (changing it is a contract change ->
    bump ``prompt_template_hash``). The gamma-marker *placement* is confirmed
    jointly with Phase-4 knob validation; at gamma=1.0 the marker is absent so the
    Phase-1 baseline prompt is unaffected.

    >>> assemble_reasoning_prompt("Do X.", "python", "f", 1.0).startswith("Do X.")
    True
    >>> "<|eot_id|>0.5<|eot_id|>" in assemble_reasoning_prompt("Do X.", "python", "f", 0.5)
    True
    """
    regions = [instruction.rstrip()]
    marker = gamma_marker(gamma, family)
    if marker:
        regions.append(marker)
    regions.append(generation_directive(lang, entry_point))
    return REGION_SEP.join(regions)


def explanation_stage1_prompt(instruction: str) -> str:
    """Stage-1 of EXPLANATION: the model describes the given code.

    McEval's explanation ``instruction`` already embeds the code plus the request
    for a natural-language description, so stage 1 uses it verbatim. The stage-1
    output (the description) is the compressible CoT; it carries NO gamma marker --
    explanation compression is post-hoc (LLMLingua-2, Phase 2) on the description
    itself, not prompt-controlled (Decision #3).
    """
    return instruction.rstrip()


def explanation_stage2_prompt(lang: str, signature: str, description: str) -> str:
    """McEval stage-2 prompt; ``description`` is the (compressed) stage-1 text."""
    return EXPLANATION_STAGE2_TEMPLATE.format(lang=lang, signature=signature, description=description)
