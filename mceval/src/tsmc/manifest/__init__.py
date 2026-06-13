"""Base-problem stratification and split-manifest generation (roadmap s6).

Atomic unit: the case-normalized base problem (`tsmc.mceval_data.canonical_base_id`
-- reconciling `sql`/`SQL`, the Task 0.2 finding). Stratum key: language x
difficulty. Proportional allocation with largest-remainder rounding to a global
80/20 split; every task variant sharing the canonical `Lang/N` prefix inherits
the same split label, which blocks cross-task leakage.

Difficulty:
  - 2,007 core problems  -> generation `level`        (difficulty_source=level_propagated)
  - 59 SQL (expl+compl)  -> rank tertiles of docstring length, internal to SQL
                            (difficulty_source=derived_proxy). Decision #4's
                            fallback signals both fail for SQL (Task 0.4: LOC=1
                            constant, #test-cases=2 constant); docstring length
                            is the only varying signal. See docs/phase0_findings.md.

CPU-only, stdlib + tsmc only. The committed CSV is the git exception; the FROZEN
manifest (not the seed) is authoritative.
"""
from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tsmc import mceval_data as M
from tsmc.config import ProjectPaths, get_paths
from tsmc.constants import (
    DIFFICULTY_LEVELS,
    DIFFICULTY_SOURCES,
    SEED,
    SPLITS,
    TRAIN_FRACTION,
)

MANIFEST_COLUMNS = (
    "problem_id",
    "split",
    "language",
    "difficulty",
    "difficulty_source",
    "membership",
)

_TRAIN, _TEST = SPLITS  # ("train_problems", "test_problems")


@dataclass(frozen=True)
class BaseProblem:
    problem_id: str  # canonical (lower-cased language) base id, e.g. "python/1"
    language: str  # canonical language, e.g. "python", "sql"
    difficulty: str  # easy / middle / hard
    difficulty_source: str  # level_propagated / derived_proxy
    membership: str  # e.g. "gen+expl+compl" or "expl+compl"

    @property
    def cell(self) -> tuple[str, str]:
        return (self.language, self.difficulty)


def _problem_sort_key(problem_id: str) -> tuple[str, int, str]:
    lang, _, num = problem_id.partition("/")
    try:
        return (lang, int(num), "")
    except ValueError:
        return (lang, 0, num)


def _docstring_words(row: dict[str, Any]) -> int:
    text = row.get("docstring") or row.get("instruction") or row.get("prompt") or ""
    return len(text.split())


def build_base_problems(paths: ProjectPaths | None = None) -> list[BaseProblem]:
    """Build the per-base-problem table (difficulty + membership), sorted."""
    paths = paths or get_paths()
    gen = M.load_generation(paths)
    expl = M.load_explanation(paths)
    compl = M.load_completion(paths, subset="merge")

    gen_ids = {M.canonical_base_id(r["task_id"]) for r in gen}
    expl_ids = {M.canonical_base_id(r["task_id"]) for r in expl}
    compl_ids = {M.canonical_base_id(r["task_id"]) for r in compl}
    all_ids = gen_ids | expl_ids | compl_ids

    # difficulty from generation level (the labeled core)
    level = {M.canonical_base_id(r["task_id"]): r["level"] for r in gen}

    # derived difficulty for unlabeled (SQL): rank tertiles of docstring length
    unlabeled = sorted(all_ids - set(level), key=_problem_sort_key)
    expl_by_id = {M.canonical_base_id(r["task_id"]): r for r in expl}
    ranked = sorted(
        unlabeled,
        key=lambda cid: (_docstring_words(expl_by_id.get(cid, {})), _problem_sort_key(cid)),
    )
    n = len(ranked)
    cut1, cut2 = n // 3, 2 * n // 3
    derived: dict[str, str] = {}
    for i, cid in enumerate(ranked):
        derived[cid] = "easy" if i < cut1 else ("middle" if i < cut2 else "hard")

    problems: list[BaseProblem] = []
    for cid in sorted(all_ids, key=_problem_sort_key):
        membership = "+".join(
            tag
            for tag, ids in (("gen", gen_ids), ("expl", expl_ids), ("compl", compl_ids))
            if cid in ids
        )
        if cid in level:
            difficulty, source = level[cid], "level_propagated"
        else:
            difficulty, source = derived[cid], "derived_proxy"
        language = cid.split("/")[0]
        problems.append(BaseProblem(cid, language, difficulty, source, membership))
    return problems


def assign_splits(
    problems: list[BaseProblem],
    train_fraction: float = TRAIN_FRACTION,
    seed: int = SEED,
) -> dict[str, str]:
    """Assign each base problem to a split via per-cell largest-remainder 80/20."""
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in problems:
        cells[p.cell].append(p.problem_id)

    # per-cell floor allocation + fractional remainder
    floor_alloc: dict[tuple[str, str], int] = {}
    remainder: dict[tuple[str, str], float] = {}
    for cell, ids in cells.items():
        ideal = train_fraction * len(ids)
        floor_alloc[cell] = math.floor(ideal)
        remainder[cell] = ideal - math.floor(ideal)

    target_train = round(train_fraction * len(problems))
    leftover = target_train - sum(floor_alloc.values())
    # hand the leftover to the cells with the largest remainder (tie-break by cell)
    for cell in sorted(cells, key=lambda c: (-remainder[c], c))[:leftover]:
        floor_alloc[cell] += 1

    rng = random.Random(seed)
    split: dict[str, str] = {}
    for cell in sorted(cells):
        ids = sorted(cells[cell], key=_problem_sort_key)
        rng.shuffle(ids)
        k = floor_alloc[cell]
        for i, pid in enumerate(ids):
            split[pid] = _TRAIN if i < k else _TEST
    return split


def build_manifest_rows(
    paths: ProjectPaths | None = None,
    train_fraction: float = TRAIN_FRACTION,
    seed: int = SEED,
) -> list[dict[str, str]]:
    """Build the full manifest (sorted by problem_id) as a list of row dicts."""
    problems = build_base_problems(paths)
    split = assign_splits(problems, train_fraction, seed)
    rows = [
        {
            "problem_id": p.problem_id,
            "split": split[p.problem_id],
            "language": p.language,
            "difficulty": p.difficulty,
            "difficulty_source": p.difficulty_source,
            "membership": p.membership,
        }
        for p in sorted(problems, key=lambda p: _problem_sort_key(p.problem_id))
    ]
    return rows


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def completion_variant_counts(paths: ProjectPaths | None = None) -> dict[str, int]:
    """Number of completion `merge` variants per canonical base id (for row counts)."""
    paths = paths or get_paths()
    counts: Counter = Counter()
    for r in M.load_completion(paths, subset="merge"):
        counts[M.canonical_base_id(r["task_id"])] += 1
    return dict(counts)


def summarize(rows: list[dict[str, str]], paths: ProjectPaths | None = None) -> dict[str, Any]:
    """Counts for the report + row-level per-task split sizes."""
    total = len(rows)
    by_split = Counter(r["split"] for r in rows)
    by_membership = Counter(r["membership"] for r in rows)
    by_source = Counter(r["difficulty_source"] for r in rows)
    by_difficulty = Counter(r["difficulty"] for r in rows)
    n_languages = len({r["language"] for r in rows})

    variants = completion_variant_counts(paths)
    task_rows: dict[str, dict[str, int]] = {
        t: {_TRAIN: 0, _TEST: 0} for t in ("generation", "explanation", "completion")
    }
    for r in rows:
        s = r["split"]
        m = r["membership"]
        if "gen" in m:
            task_rows["generation"][s] += 1
        if "expl" in m:
            task_rows["explanation"][s] += 1
        if "compl" in m:
            task_rows["completion"][s] += variants.get(r["problem_id"], 0)

    return {
        "total": total,
        "by_split": dict(by_split),
        "by_membership": dict(by_membership),
        "by_difficulty_source": dict(by_source),
        "by_difficulty": dict(by_difficulty),
        "n_languages": n_languages,
        "task_rows": task_rows,
    }


def validate_manifest(
    rows: list[dict[str, str]],
    train_fraction: float = TRAIN_FRACTION,
    expected_total: int = 2066,
) -> list[str]:
    """Distributional balance gate + structural checks. Returns error messages."""
    errors: list[str] = []

    if len(rows) != expected_total:
        errors.append(f"row count {len(rows)} != expected {expected_total}")

    ids = [r["problem_id"] for r in rows]
    if len(set(ids)) != len(ids):
        errors.append("duplicate problem_id rows")

    for r in rows:
        if set(r) != set(MANIFEST_COLUMNS):
            errors.append(f"{r.get('problem_id')}: bad columns {sorted(r)}")
            break
        if r["split"] not in SPLITS:
            errors.append(f"{r['problem_id']}: bad split {r['split']!r}")
        if r["difficulty"] not in DIFFICULTY_LEVELS:
            errors.append(f"{r['problem_id']}: bad difficulty {r['difficulty']!r}")
        if r["difficulty_source"] not in DIFFICULTY_SOURCES:
            errors.append(f"{r['problem_id']}: bad difficulty_source {r['difficulty_source']!r}")
        if not r["problem_id"].startswith(r["language"] + "/"):
            errors.append(f"{r['problem_id']}: language {r['language']!r} not the id prefix")

    # global split size matches the largest-remainder target
    target_train = round(train_fraction * len(rows))
    n_train = sum(1 for r in rows if r["split"] == _TRAIN)
    if n_train != target_train:
        errors.append(f"train size {n_train} != target {target_train}")

    # per-cell balance: train within +-1 of floor(fraction * cell size)
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        cells[(r["language"], r["difficulty"])].append(r["split"])
    for cell, splits in cells.items():
        n = len(splits)
        n_tr = splits.count(_TRAIN)
        ideal = math.floor(train_fraction * n)
        if abs(n_tr - ideal) > 1:
            errors.append(f"cell {cell}: train {n_tr} vs floor-ideal {ideal} (n={n}) exceeds +-1")

    return errors
