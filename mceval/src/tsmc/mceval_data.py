"""Read-only access to the vendored McEval benchmark data (CPU-only).

Three tasks, loaded from the repo as shipped:

    generation  -> <mceval_dir>/data/*.jsonl                       (40 files)
    explanation -> <mceval_dir>/explanation/explaination_data.zip  (zip, in-memory)
    completion  -> <mceval_dir>/completion/completion_data.zip      (zip, in-memory)

`task_id` is the join key. Generation/explanation use ``Lang/N``; completion
expands each base into ``Lang/N-k-{single|multi|span}`` variants. The completion
zip carries five parallel views (merge / single / multi / span / light);
``merge`` is the canonical union of single+multi+span and is the default here.

IMPORTANT (Phase-0 empirical finding): the SQL problems are stored with
inconsistent language case across tasks -- ``sql/N`` in explanation but ``SQL/N``
in completion (same 59 problems; generation has no SQL). Always join base
problems on :func:`canonical_base_id`, which lower-cases the language, or the
split will leak SQL problems across train/test. See docs/phase0_findings.md.
"""
from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tsmc.config import ProjectPaths, get_paths

GENERATION = "generation"
EXPLANATION = "explanation"
COMPLETION = "completion"

# Upstream filenames/members (note the upstream "explaination" misspelling).
_EXPL_ZIP_REL = ("explanation", "explaination_data.zip")
_EXPL_MEMBER_PREFIX = "explaination_data/"
_COMPL_ZIP_REL = ("completion", "completion_data.zip")
_COMPL_SUBSETS = ("merge", "single", "multi", "span", "light")


# --- task_id helpers -----------------------------------------------------------

def base_problem_id(task_id: str) -> str:
    """``Lang/N-k-sub`` -> ``Lang/N`` (no-op for generation/explanation ids)."""
    return task_id.split("-", 1)[0]


def split_language_number(task_id: str) -> tuple[str, str]:
    """Split a base id ``Lang/N`` into ``(language, number)``."""
    language, _, number = base_problem_id(task_id).partition("/")
    return language, number


def normalize_language(language: str) -> str:
    """Canonical language key. Lower-cases to reconcile ``sql`` vs ``SQL``."""
    return language.lower()


def canonical_base_id(task_id: str) -> str:
    """Cross-task join key: base id with the language case-normalized.

    ``SQL/5-0-single`` and ``sql/5`` both map to ``sql/5``.
    """
    language, number = split_language_number(task_id)
    return f"{normalize_language(language)}/{number}"


def completion_subtype(task_id: str) -> str | None:
    """Return ``single``/``multi``/``span`` for a completion id, else ``None``."""
    parts = task_id.split("-")
    return parts[-1] if len(parts) >= 3 else None


def iter_string_fields(row: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield ``(key, value)`` for every string (or list-of-string) field.

    Used by the sentinel-collision scan to inspect all text the model could echo.
    """
    for key, value in row.items():
        if isinstance(value, str):
            yield key, value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    yield key, item


# --- loaders -------------------------------------------------------------------

def _iter_jsonl_text(text: str) -> Iterator[dict[str, Any]]:
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def load_generation(paths: ProjectPaths | None = None) -> list[dict[str, Any]]:
    """Load all generation rows from ``<mceval_dir>/data/*.jsonl``."""
    paths = paths or get_paths()
    data_dir = paths.mceval_data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"McEval generation data dir not found: {data_dir}")
    rows: list[dict[str, Any]] = []
    for jsonl in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as handle:
            rows.extend(_iter_jsonl_text(handle.read()))
    return rows


def load_generation_language(
    lang: str, paths: ProjectPaths | None = None
) -> list[dict[str, Any]]:
    """Load one generation language file ``<mceval_data_dir>/<lang>.jsonl``.

    These per-language files carry the rich problem fields (entry_point,
    signature, test, prompt, canonical_solution) that McEval's extractor needs.
    """
    paths = paths or get_paths()
    path = paths.mceval_data_dir / f"{lang}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"McEval generation file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        return list(_iter_jsonl_text(handle.read()))


def _load_zip(zip_path: Path, member_prefix: str) -> list[dict[str, Any]]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"McEval data zip not found: {zip_path}")
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(
            m for m in archive.namelist()
            if m.startswith(member_prefix) and m.endswith(".jsonl")
        )
        for member in members:
            text = archive.read(member).decode("utf-8")
            rows.extend(_iter_jsonl_text(text))
    return rows


def load_explanation(paths: ProjectPaths | None = None) -> list[dict[str, Any]]:
    """Load all explanation rows from the explanation zip (in-memory)."""
    paths = paths or get_paths()
    zip_path = paths.mceval_dir.joinpath(*_EXPL_ZIP_REL)
    return _load_zip(zip_path, _EXPL_MEMBER_PREFIX)


def load_completion(
    paths: ProjectPaths | None = None, subset: str = "merge"
) -> list[dict[str, Any]]:
    """Load completion rows from the completion zip (default canonical ``merge``)."""
    if subset not in _COMPL_SUBSETS:
        raise ValueError(f"subset must be one of {_COMPL_SUBSETS}, got {subset!r}")
    paths = paths or get_paths()
    zip_path = paths.mceval_dir.joinpath(*_COMPL_ZIP_REL)
    return _load_zip(zip_path, f"completion_data/{subset}/")


def load_all(paths: ProjectPaths | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load all three tasks (completion = canonical ``merge`` view)."""
    paths = paths or get_paths()
    return {
        GENERATION: load_generation(paths),
        EXPLANATION: load_explanation(paths),
        COMPLETION: load_completion(paths),
    }
