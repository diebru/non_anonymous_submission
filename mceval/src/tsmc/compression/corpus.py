"""Multi-gamma CoT compression of the correct-CoT corpus (roadmap Phase 2). CPU.

Pure, tokenizer-free, dependency-light core: turn one verified-correct (gamma=1.0)
trajectory into the 12-gamma family of compressed-CoT variants that Phase 3 turns
into SFT data. The heavy pieces -- the LLMLingua-2 compressor and the Qwen
tokenizer -- are injected as plain callables (``compress_fn`` / ``count_fn``), so
this module is fully unit-testable locally with mocks; the real server run wires
in ``tsmc.compression.llmlingua`` (docs/WORKFLOW.md s2).

What gets compressed: the ``cot_text`` region ONLY. The contract parser already
split the sentinel, the fenced code, and the entry_point out of ``cot_text`` back
in Phase 1, so the structural scaffolding is held out of compression *by
construction* -- ``code_snippet`` / ``extraction_status`` are copied verbatim and
``check_scaffolding_intact`` re-confirms it. The gamma-control marker is a
prompt-assembly artifact added in Phase 3, never stored here, so it cannot be
pruned either.

Per-variant semantics (roadmap s7 schema + Decision #3):
  - ``gamma`` / ``compression_ratio`` = the TARGET ratio (one of GAMMA_GRID);
    distinct from LLMLingua-2's *achieved* rate, which is noisy and kept only as
    provenance. Curves are plotted vs the MEASURED ``cot_token_count``.
  - ``cot_token_count`` = ``count_fn(compressed_text)`` -- re-counted with the
    MODEL (Qwen) tokenizer so it is comparable with the Phase-1 records and the
    energy join, NOT LLMLingua-2's own XLM-R token count.
  - gamma == 1.0 is a passthrough: ``cot_text`` unchanged, ``cot_origin=original``
    (required by ``validate_record``), ``pass`` carried from the source (it WAS
    executed). gamma < 1.0: ``cot_origin=compressed``, ``pass=False`` (provisional
    -- this variant has not been executed; Phase 4 re-evaluates). The source row's
    verified-correct status is preserved in ``_compression.source_pass``.
  - ``compression_method`` is carried from the source unchanged (generation =
    model_side, explanation = post_hoc); never merge curves across methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from tsmc.constants import GAMMA_BASELINE, SENTINEL

# Injected heavy callables (wired to tsmc.compression.llmlingua on the server).
CompressFn = Callable[[str, float], "CompressionResult"]
CountFn = Callable[[str], int]

# Keys overridden per variant; everything else on the source record is carried
# through. ``outcome`` (a join artifact, not a schema field) is dropped because it
# is stale for an un-executed compressed variant -- its source value lives in
# ``_compression.source_outcome``.
_DROP_KEYS = ("outcome",)


@dataclass
class CompressionResult:
    """Normalized return of one LLMLingua-2 ``compress_prompt`` call.

    ``compressed_text`` is the only field that flows into the variant's
    ``cot_text``; the token/rate fields are LLMLingua-2's *native* XLM-R
    measurements, stored as provenance (NOT used for ``cot_token_count``).
    """

    compressed_text: str
    origin_tokens: int | None = None       # LLMLingua-2 XLM-R token count of input
    compressed_tokens: int | None = None    # LLMLingua-2 XLM-R token count of output
    rate: str | float | None = None         # achieved rate as LLMLingua-2 reports it
    ratio: str | float | None = None
    saving: str | float | None = None

    def provenance(self) -> dict[str, Any]:
        return {
            "origin_tokens": self.origin_tokens,
            "compressed_tokens": self.compressed_tokens,
            "rate": self.rate,
            "ratio": self.ratio,
            "saving": self.saving,
        }


@dataclass
class CompressionParams:
    """Provenance of how the compressor was invoked (pinned per run)."""

    checkpoint: str
    checkpoint_sha: str | None = None
    use_llmlingua2: bool = True
    # TokenSkip-qwen faithful call is bare ``rate``; any extra kwargs recorded here.
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_sha": self.checkpoint_sha,
            "use_llmlingua2": self.use_llmlingua2,
            "extra_kwargs": dict(self.extra_kwargs),
        }


def _is_baseline(gamma: float) -> bool:
    return gamma >= GAMMA_BASELINE


def compress_record(
    record: dict[str, Any],
    gammas: Sequence[float],
    compress_fn: CompressFn,
    count_fn: CountFn,
    params: CompressionParams,
) -> list[dict[str, Any]]:
    """Expand one source (gamma=1.0) corpus record into one variant per gamma.

    ``compress_fn(text, gamma) -> CompressionResult`` is only called for gamma < 1.0
    and on a non-empty CoT; gamma == 1.0 and empty/whitespace CoTs are passthroughs
    (LLMLingua-2 is unstable on empty input -- roadmap Phase-2 risk). The source
    record is treated read-only.
    """
    source_cot = record.get("cot_text") or ""
    source_pass = bool(record.get("pass", False))
    source_outcome = record.get("outcome")
    empty_cot = not source_cot.strip()

    variants: list[dict[str, Any]] = []
    for gamma in gammas:
        baseline = _is_baseline(gamma)
        degenerate: str | None = None
        ll_prov: dict[str, Any] | None = None

        if baseline:
            compressed = source_cot
        elif empty_cot:
            compressed = source_cot
            degenerate = "empty_cot"
        else:
            result = compress_fn(source_cot, gamma)
            compressed = result.compressed_text
            ll_prov = result.provenance()

        variant = {k: v for k, v in record.items() if k not in _DROP_KEYS}
        variant["gamma"] = gamma
        variant["compression_ratio"] = gamma
        variant["cot_text"] = compressed
        variant["cot_token_count"] = count_fn(compressed)
        variant["cot_origin"] = "original" if baseline else "compressed"
        # pass is verified only for the executed baseline; compressed variants are
        # provisional until Phase 4 re-runs them.
        variant["pass"] = source_pass if baseline else False
        variant["_compression"] = {
            "params": params.to_dict(),
            "llmlingua": ll_prov,            # None for passthrough rows
            "source_pass": source_pass,
            "source_outcome": source_outcome,
            "source_cot_token_count": record.get("cot_token_count"),
            "degenerate": degenerate,        # e.g. "empty_cot"; None normally
        }
        variants.append(variant)
    return variants


# --- validation (Phase-2 completion criteria) ----------------------------------

def check_scaffolding_intact(source: dict[str, Any], variant: dict[str, Any]) -> list[str]:
    """Confirm compression touched only the CoT region (roadmap Phase-2 gate).

    The fenced code, the entry_point, and the parser diagnostics must survive
    untouched, and the sentinel must never appear inside the (compressible) CoT.
    """
    errors: list[str] = []
    if variant.get("code_snippet") != source.get("code_snippet"):
        errors.append("code_snippet changed")
    if variant.get("extraction_status") != source.get("extraction_status"):
        errors.append("extraction_status changed")
    if SENTINEL in (variant.get("cot_text") or ""):
        errors.append("sentinel leaked into cot_text")
    return errors


def _by_gamma_desc(variants: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(variants, key=lambda v: float(v["gamma"]), reverse=True)


def trajectory_monotonic(variants: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-trajectory check: as gamma DECREASES, measured cot_token_count must be
    NON-INCREASING. Ties are allowed (dense gammas near 1.0 often collide once the
    text is re-tokenized with the model tokenizer); a strict increase is a
    ``violation``. Returns the ordered token series + violation steps.
    """
    ordered = _by_gamma_desc(variants)
    series = [(float(v["gamma"]), int(v["cot_token_count"])) for v in ordered]
    violations: list[dict[str, Any]] = []
    for (g_hi, t_hi), (g_lo, t_lo) in zip(series, series[1:]):
        if t_lo > t_hi:
            violations.append({"gamma_hi": g_hi, "gamma_lo": g_lo,
                               "tokens_hi": t_hi, "tokens_lo": t_lo})
    return {"series": series, "violations": violations, "monotonic": not violations}


def aggregate_token_medians(
    variants_by_gamma: dict[float, list[int]]
) -> list[tuple[float, float]]:
    """Median measured cot_token_count per gamma, ordered by gamma descending.

    The corpus-level headline: this median series must be strictly non-increasing
    (the aggregate monotonicity gate), even where individual short CoTs tie or
    invert.
    """
    import statistics

    out = []
    for gamma in sorted(variants_by_gamma, reverse=True):
        toks = variants_by_gamma[gamma]
        out.append((gamma, statistics.median(toks) if toks else 0.0))
    return out


def aggregate_monotonic(medians: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Gate on the per-gamma median series: strictly non-increasing as gamma falls."""
    violations = [
        {"gamma_hi": g_hi, "gamma_lo": g_lo, "median_hi": m_hi, "median_lo": m_lo}
        for (g_hi, m_hi), (g_lo, m_lo) in zip(medians, medians[1:])
        if m_lo > m_hi
    ]
    return {"medians": list(medians), "violations": violations, "monotonic": not violations}
