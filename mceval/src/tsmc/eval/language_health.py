"""Per-language McEval scoring health (derived from the gold scoring-health map).

Produced by ``scripts/verify_mceval_docker.py --langs all`` -- running McEval on the
REFERENCE solutions of every generation language inside the pinned image. A language
where McEval cannot score its OWN gold cannot be trusted to score model output, so we
bucket the 40 languages and let scoring / the behavioral gate report a principled
"healthy" accuracy (the same treatment SQL already gets: kept in the manifest, dropped
from accuracy).

Buckets (lower-cased keys, matching the long-format ``lang`` field):
  OK              gold >= threshold: scoring sound, ceiling ~1.0 -> trust model scores.
  SOFT            Rust: extraction matches gold but McEval cold-recompiles crates per
                  problem under a timeout (Phase 0) -> report, never gate on it.
  REDUCED_CEILING gold below threshold because McEval's own extractor mis-reconstructs
                  a fraction even of reference code (like Python's ~0.9) -> interpret
                  model accuracy against the per-language ceiling.
  EXCLUDED        McEval scores its own gold at 0.0 (F#/Java/R) or never executes the
                  language (SQL) -> verdicts unreliable; drop from accuracy until a
                  per-language handler exists.

Provenance below records the map this classification came from; re-run the verifier
(ideally a higher --limit for the REDUCED/EXCLUDED set) and update if it shifts.
"""
from __future__ import annotations

GOLD_MAP_PROVENANCE = {
    "digest": "sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5",
    "limit": 5,
    "date": "2026-06-01",
    "shell": "bash -ic",  # interactive: loads version-manager toolchains
}

# 27 languages McEval scores its reference solutions at ceiling.
OK: tuple[str, ...] = (
    "c", "c#", "cpp", "coffeescript", "common lisp", "dart", "elixir", "emacs lisp",
    "go", "groovy", "haskell", "javascript", "julia", "kotlin", "php", "perl",
    "powershell", "python", "racket", "ruby", "scala", "scheme", "shell", "swift",
    "tcl", "vimscript", "visual basic",
)

# Soft: report but never gate (cold-recompile timeout artifact).
SOFT: tuple[str, ...] = ("rust",)

# Reduced ceiling: gold < threshold (preliminary n=5 estimate; refine at higher limit).
REDUCED_CEILING: dict[str, float] = {
    "erlang": 0.60, "fortran": 0.40, "lua": 0.80, "pascal": 0.40, "typescript": 0.80,
    "python": 0.90,  # McEval mis-reconstructs e.g. Python/9 (Phase-0 finding)
}

# Excluded from accuracy (McEval cannot score gold, or never executes the language).
EXCLUDED: tuple[str, ...] = ("f#", "java", "r", "sql")


def _norm(lang: str) -> str:
    return lang.strip().lower()


def is_excluded(lang: str) -> bool:
    """Drop from accuracy (broken scoring or never executed)."""
    return _norm(lang) in EXCLUDED


def is_soft(lang: str) -> bool:
    """Report but do not gate (timeout artifact)."""
    return _norm(lang) in SOFT


def is_healthy(lang: str) -> bool:
    """True for languages whose accuracy is trustworthy enough to gate on
    (everything except EXCLUDED and SOFT). REDUCED_CEILING languages are healthy
    but carry a lower ceiling."""
    n = _norm(lang)
    return n not in EXCLUDED and n not in SOFT


def ceiling(lang: str) -> float | None:
    """Per-language gold ceiling if known to be below 1.0, else None (~1.0)."""
    return REDUCED_CEILING.get(_norm(lang))
