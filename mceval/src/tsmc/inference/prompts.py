"""Problem selection + prompt assembly for the inference harness (CPU-only).

This is the tokenizer-free, GPU-free half of Phase-1 inference: it reads the
frozen manifest, picks the McEval problems for a (task, split) -- optionally the
validation trio or a per-language limit -- and builds the exact user-message text
for each, using the frozen ``tsmc.contract`` assemblers so the prompt is
byte-identical to what Phase-3 SFT / Phase-4 inference will use.

It is importable and fully testable locally; the actual model call lives in
``tsmc.inference.runner`` (server-only). The split between the two keeps prompt
construction unit-testable without vLLM.

Per-task wiring (see roadmap s4):
  generation  -> prompt = gen.instruction;       execute against the gen record.
  completion  -> prompt = completion.instruction; execute against the completion
                 record (rich for the trio; carries its own test).
  explanation -> TWO PASS. stage-1 prompt = explanation.instruction (describe the
                 code); stage-2 prompt = McEval template over the (Phase-1:
                 uncompressed) description. Explanation records carry no test /
                 signature, so we execute the reconstructed code against the
                 GENERATION record (joined by canonical_base_id) -- which also
                 means only the executable core (membership has ``gen``) is run;
                 the 59 SQL (expl+compl only) are unscored anyway (Task 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from tsmc import mceval_data as M
from tsmc.config import ProjectPaths, get_paths
from tsmc.contract import (
    assemble_reasoning_prompt,
    explanation_stage1_prompt,
    explanation_stage2_prompt,
)
from tsmc.eval import results as R
from tsmc.manifest import read_manifest

# CLI split name -> schema split value (roadmap s7 SPLITS).
SPLIT_VALUE = {"train": "train_problems", "test": "test_problems"}
# Canonical (lower-cased) languages of the validation trio.
TRIO_LOWER: tuple[str, ...] = tuple(lang.lower() for lang in R.TRIO)


@dataclass
class ProblemUnit:
    """One schedulable inference item (single-pass; explanation is two-pass)."""

    problem_id: str            # canonical base id (manifest key, e.g. "python/1")
    task_type: str             # generation / explanation / completion
    completion_subtype: str | None
    mceval_task_id: str        # original-case task_id (result-file row identity)
    mceval_lang: str           # original-case language (result-file name + extract lang)
    fence_lang: str            # lower-case tag used in the directive / fenced block
    entry_point: str | None
    signature: str | None
    difficulty: str
    difficulty_source: str
    split: str                 # schema value: train_problems / test_problems
    membership: str
    prompt_instruction: str    # instruction used to build the model prompt
    record: dict[str, Any]     # McEval record we execute against (the RESULT file row)


# --- manifest indexing ---------------------------------------------------------

def manifest_index(paths: ProjectPaths | None = None) -> dict[str, dict[str, str]]:
    """Map canonical ``problem_id`` -> its manifest row."""
    paths = paths or get_paths()
    return {row["problem_id"]: row for row in read_manifest(paths.manifest_path)}


def _keep(row: dict[str, str] | None, split: str, trio_only: bool) -> bool:
    if row is None or row["split"] != split:
        return False
    if trio_only and row["language"] not in TRIO_LOWER:
        return False
    return True


def _capped(units: list[ProblemUnit], limit: int) -> list[ProblemUnit]:
    """Keep at most ``limit`` units per original-case language (0 = no cap)."""
    if not limit:
        return units
    seen: dict[str, int] = {}
    out: list[ProblemUnit] = []
    for u in units:
        n = seen.get(u.mceval_lang, 0)
        if n < limit:
            out.append(u)
            seen[u.mceval_lang] = n + 1
    return out


# --- per-task selection --------------------------------------------------------

def generation_units(
    split: str,
    idx: dict[str, dict[str, str]],
    paths: ProjectPaths | None = None,
    trio_only: bool = False,
    limit: int = 0,
) -> list[ProblemUnit]:
    paths = paths or get_paths()
    target = SPLIT_VALUE[split]
    units: list[ProblemUnit] = []
    for rec in M.load_generation(paths):
        cid = M.canonical_base_id(rec["task_id"])
        row = idx.get(cid)
        if not _keep(row, target, trio_only):
            continue
        lang = M.split_language_number(rec["task_id"])[0]
        units.append(ProblemUnit(
            problem_id=cid, task_type="generation", completion_subtype=None,
            mceval_task_id=rec["task_id"], mceval_lang=lang, fence_lang=R.fence_tag(lang),
            entry_point=rec.get("entry_point"), signature=rec.get("signature"),
            difficulty=row["difficulty"], difficulty_source=row["difficulty_source"],
            split=target, membership=row["membership"],
            prompt_instruction=rec["instruction"], record=rec,
        ))
    return _capped(units, limit)


def completion_units(
    split: str,
    idx: dict[str, dict[str, str]],
    paths: ProjectPaths | None = None,
    trio_only: bool = False,
    limit: int = 0,
) -> list[ProblemUnit]:
    paths = paths or get_paths()
    target = SPLIT_VALUE[split]
    units: list[ProblemUnit] = []
    for rec in M.load_completion(paths, subset="merge"):
        cid = M.canonical_base_id(rec["task_id"])
        row = idx.get(cid)
        if not _keep(row, target, trio_only):
            continue
        lang = M.split_language_number(rec["task_id"])[0]
        units.append(ProblemUnit(
            problem_id=cid, task_type="completion",
            completion_subtype=M.completion_subtype(rec["task_id"]),
            mceval_task_id=rec["task_id"], mceval_lang=lang, fence_lang=R.fence_tag(lang),
            entry_point=rec.get("entry_point"), signature=rec.get("signature"),
            difficulty=row["difficulty"], difficulty_source=row["difficulty_source"],
            split=target, membership=row["membership"],
            prompt_instruction=rec["instruction"], record=rec,
        ))
    return _capped(units, limit)


def explanation_units(
    split: str,
    idx: dict[str, dict[str, str]],
    paths: ProjectPaths | None = None,
    trio_only: bool = False,
    limit: int = 0,
) -> list[ProblemUnit]:
    """Executable-core explanation units: stage-1 prompt from the explanation
    record, but the RESULT record (signature + test) from the generation record.
    Problems without a generation row (the 59 SQL) are skipped (unscored anyway).
    """
    paths = paths or get_paths()
    target = SPLIT_VALUE[split]
    gen_by_cid = {M.canonical_base_id(r["task_id"]): r for r in M.load_generation(paths)}
    units: list[ProblemUnit] = []
    for rec in M.load_explanation(paths):
        cid = M.canonical_base_id(rec["task_id"])
        row = idx.get(cid)
        if not _keep(row, target, trio_only):
            continue
        gen = gen_by_cid.get(cid)
        if gen is None:  # no executable generation record (SQL) -> skip
            continue
        lang = M.split_language_number(gen["task_id"])[0]
        units.append(ProblemUnit(
            problem_id=cid, task_type="explanation", completion_subtype=None,
            mceval_task_id=gen["task_id"], mceval_lang=lang, fence_lang=R.fence_tag(lang),
            entry_point=gen.get("entry_point"), signature=gen.get("signature"),
            difficulty=row["difficulty"], difficulty_source=row["difficulty_source"],
            split=target, membership=row["membership"],
            prompt_instruction=rec["instruction"], record=gen,
        ))
    return _capped(units, limit)


TASK_SELECTORS = {
    "generation": generation_units,
    "completion": completion_units,
    "explanation": explanation_units,
}


def select_units(
    task: str,
    split: str,
    paths: ProjectPaths | None = None,
    trio_only: bool = False,
    limit: int = 0,
    idx: dict[str, dict[str, str]] | None = None,
) -> list[ProblemUnit]:
    paths = paths or get_paths()
    idx = idx if idx is not None else manifest_index(paths)
    return TASK_SELECTORS[task](split, idx, paths, trio_only, limit)


# --- user-message text (the frozen contract assembly) --------------------------

def reasoning_user_text(unit: ProblemUnit, gamma: float, family: str = "qwen") -> str:
    """Generation / completion user message: instruction -> gamma -> directive.

    ``family`` picks the gamma-marker delimiter (default ``"qwen"``); resolve it
    from the model id with ``constants.family_of`` at the call site so SFT and
    inference stay byte-identical.
    """
    return assemble_reasoning_prompt(
        unit.prompt_instruction, unit.fence_lang, unit.entry_point or "", gamma, family
    )


def explanation_stage1_user(unit: ProblemUnit) -> str:
    """Explanation stage-1 user message (describe the code; no gamma marker)."""
    return explanation_stage1_prompt(unit.prompt_instruction)


def explanation_stage2_user(unit: ProblemUnit, description: str) -> str:
    """Explanation stage-2 user message over the (Phase-1: raw) description."""
    return explanation_stage2_prompt(unit.fence_lang, unit.signature or "", description)


def stage1_user_text(unit: ProblemUnit, gamma: float, family: str = "qwen") -> str:
    """First (and, except for explanation, only) user message for a unit."""
    if unit.task_type == "explanation":
        return explanation_stage1_user(unit)  # stage-1 carries no gamma marker
    return reasoning_user_text(unit, gamma, family)


def chat_messages(user_text: str, system: str | None = None) -> list[dict[str, str]]:
    """Wrap a user message as chat messages (optional pinned system prompt)."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_text})
    return messages


def group_by_language(units: Iterable[ProblemUnit]) -> dict[str, list[ProblemUnit]]:
    """Group units by original-case McEval language (result-file partitioning)."""
    out: dict[str, list[ProblemUnit]] = {}
    for u in units:
        out.setdefault(u.mceval_lang, []).append(u)
    return out
