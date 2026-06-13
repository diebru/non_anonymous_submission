"""CoT/code separation parsing and the three-way outcome (roadmap s4). FROZEN.

We own the separation with our sentinel and write only clean code into McEval's
``raw_generation[0]`` as one canonical fenced block; McEval's extractor is the
confirming second net. Every parse records an ``ExtractionStatus`` so extraction
failures are never misread as reasoning failures (the central concavity
confound).

CPU-only, stdlib only. Tokenizer-free: ``cot_token_count`` is computed later by
the inference harness with the model tokenizer.

RE-FROZEN (Phase-3 gate finding): added the ``presentinel_salvage`` branch and a
``three_way_outcome`` fence gate -- the 3B often codes inside its reasoning and
emits a bare/empty trailing sentinel, which used to yield an empty ``code_snippet``
that was still scored as a pass. The prompt side (``contract.prompt``) is
UNCHANGED, so the gamma-marker freeze is intact; only code extraction changed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from tsmc.constants import SENTINEL
from tsmc.schema import ExtractionStatus

# Markdown fenced block: ```lang\n ... \n``` (lang tag optional). Non-greedy body.
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


@dataclass
class ParseResult:
    """Output of a per-task parse: the split text plus diagnostics."""

    cot_text: str
    code_snippet: str
    status: ExtractionStatus


def extract_fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``[(lang_tag, body), ...]`` for every fenced block, in order."""
    blocks: list[tuple[str, str]] = []
    for match in _FENCE_RE.finditer(text):
        lang = match.group(1).strip()
        body = match.group(2)
        if body.endswith("\n"):
            body = body[:-1]
        blocks.append((lang, body))
    return blocks


def split_on_last_sentinel(text: str) -> tuple[str, str | None]:
    """Split on the LAST sentinel occurrence (defuses scratch code in the CoT).

    Returns ``(cot, code_region)``; ``code_region`` is ``None`` when no sentinel.
    """
    idx = text.rfind(SENTINEL)
    if idx == -1:
        return text, None
    return text[:idx], text[idx + len(SENTINEL):]


def _entry_point_ok(code: str, entry_point: str | None, require: bool) -> bool:
    """Whether the entry_point requirement is satisfied (True if not required)."""
    if not require or not entry_point:
        return True
    return entry_point in code


def parse_generation(
    raw_output: str,
    entry_point: str | None = None,
    finish_reason: str | None = None,
    require_entry_point: bool = True,
) -> ParseResult:
    """Generation: CoT -> sentinel -> fenced code. Last sentinel; first fence."""
    truncated = finish_reason == "length"
    cot, code_region = split_on_last_sentinel(raw_output)

    if code_region is None:
        # No sentinel -> format_fail. Instrumented salvage: last fence in the
        # WHOLE output (branch=fallback), which never counts as a clean pass.
        blocks = extract_fenced_blocks(raw_output)
        if blocks:
            code = blocks[-1][1]
            status = ExtractionStatus(
                fence_found=True,
                entry_point_found=_entry_point_ok(code, entry_point, require_entry_point),
                truncated=truncated,
                parser_branch="fallback",
            )
            return ParseResult(cot_text=raw_output, code_snippet=code, status=status)
        return ParseResult(
            cot_text=raw_output,
            code_snippet="",
            status=ExtractionStatus(False, False, truncated, "none"),
        )

    blocks = extract_fenced_blocks(code_region)
    if not blocks:
        # Sentinel present but no fenced code AFTER it. The 3B frequently puts the
        # code in a fenced block INSIDE its reasoning and emits a bare/empty trailing
        # sentinel. Salvage the LAST fenced block from the CoT as the code, and keep
        # the text before it as the clean CoT (so the reasoning region no longer
        # carries the code). Re-execution then confirms the recovered code.
        cot_matches = list(_FENCE_RE.finditer(cot))
        if cot_matches:
            last = cot_matches[-1]
            blocks_in_cot = extract_fenced_blocks(cot)
            code = blocks_in_cot[-1][1]
            status = ExtractionStatus(
                fence_found=True,
                entry_point_found=_entry_point_ok(code, entry_point, require_entry_point),
                truncated=truncated,
                parser_branch="presentinel_salvage",
            )
            return ParseResult(cot_text=cot[: last.start()], code_snippet=code, status=status)
        # No fence anywhere -> nothing to extract -> format_fail (gated below).
        return ParseResult(
            cot_text=cot,
            code_snippet="",
            status=ExtractionStatus(False, False, truncated, "sentinel"),
        )
    code = blocks[0][1]  # first fence in the code region
    branch = "multi_fence" if len(blocks) > 1 else "sentinel"
    status = ExtractionStatus(
        fence_found=True,
        entry_point_found=_entry_point_ok(code, entry_point, require_entry_point),
        truncated=truncated,
        parser_branch=branch,
    )
    return ParseResult(cot_text=cot, code_snippet=code, status=status)


def parse_completion(
    raw_output: str,
    entry_point: str | None = None,
    finish_reason: str | None = None,
    require_entry_point: bool = True,
) -> ParseResult:
    """Completion: sentinel optional. No sentinel is EXPECTED (not a failure).

    With a sentinel -> parse as generation. Without -> ``direct_fill``:
    ``cot_text=""``, code taken from the first fence (or whole text if unfenced,
    mirroring McEval's Family-B behavior).
    """
    if SENTINEL in raw_output:
        return parse_generation(raw_output, entry_point, finish_reason, require_entry_point)

    truncated = finish_reason == "length"
    blocks = extract_fenced_blocks(raw_output)
    if blocks:
        code, fence_found = blocks[0][1], True
    else:
        code, fence_found = raw_output.strip(), False
    status = ExtractionStatus(
        fence_found=fence_found,
        entry_point_found=_entry_point_ok(code, entry_point, require_entry_point),
        truncated=truncated,
        parser_branch="direct_fill",
    )
    return ParseResult(cot_text="", code_snippet=code, status=status)


def parse_explanation_stage2(
    raw_output: str,
    entry_point: str | None = None,
    finish_reason: str | None = None,
    require_entry_point: bool = True,
) -> ParseResult:
    """Explanation stage-2: CoT-free, fence-first (no sentinel to split on)."""
    truncated = finish_reason == "length"
    blocks = extract_fenced_blocks(raw_output)
    if not blocks:
        return ParseResult(
            cot_text="",
            code_snippet=raw_output.strip(),
            status=ExtractionStatus(
                fence_found=False,
                entry_point_found=_entry_point_ok(raw_output, entry_point, require_entry_point),
                truncated=truncated,
                parser_branch="none",
            ),
        )
    code = blocks[0][1]
    branch = "multi_fence" if len(blocks) > 1 else "fence"
    status = ExtractionStatus(
        fence_found=True,
        entry_point_found=_entry_point_ok(code, entry_point, require_entry_point),
        truncated=truncated,
        parser_branch=branch,
    )
    return ParseResult(cot_text="", code_snippet=code, status=status)


def three_way_outcome(status: ExtractionStatus, passed: bool) -> str:
    """Map (extraction_status, McEval pass) -> format_fail / exec_fail / pass.

    Format is judged from the parser branch and truncation -- the salvage
    (``fallback``) and ``none`` branches never count as a clean pass, and a
    truncated generation is a format failure regardless of execution. Fence /
    entry_point diagnostics are recorded but do not independently force
    format_fail (Family-B whole-text completions legitimately have no fence);
    McEval's own execution then resolves pass vs exec_fail.
    """
    if status.truncated:
        return "format_fail"
    if status.parser_branch in ("fallback", "none"):
        return "format_fail"
    # A fenced code block is mandatory on every branch EXCEPT completion's
    # ``direct_fill`` (Family-B whole-text fill, roadmap s4.3). A fenced-branch parse
    # that found NO fence is an extraction failure and must never count as a clean
    # pass -- otherwise extraction failures get misread as reasoning successes (the
    # central concavity confound). This catches the "code in the CoT, bare/empty
    # trailing sentinel" pattern that ``presentinel_salvage`` could not recover.
    if status.parser_branch != "direct_fill" and not status.fence_found:
        return "format_fail"
    return "pass" if passed else "exec_fail"
