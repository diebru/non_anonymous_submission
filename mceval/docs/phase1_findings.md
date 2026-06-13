# Phase-1 findings (inference + scoring)

Running notes from Phase 1 (baseline γ=1.0 inference + McEval scoring). Companion
to [`PHASE0_COMPLETE.md`](PHASE0_COMPLETE.md) and the roadmap.

## 1. Pipeline validated end-to-end (Qwen2.5-Coder-3B, trio smoke)

The first real generation→parse→McEval-execute→join ran on the validation trio
(Python/C/Rust, 5 problems × both splits). Result:

- **Python 9/10 pass (90%)** with the frozen contract — the assembler, Qwen chat
  template, sentinel parser, McEval execution, and the three-way-outcome join all
  work. `truncated=0` (the 2048-token output budget is enough), `train ≈ test`.
- **format_fail ≈ 10%**, all `fallback` branch: the base (pre-SFT) model sometimes
  skips our literal sentinel line and writes the fence directly. Expected before
  TokenSkip SFT; tracked, not counted as a reasoning failure.
- **C 0/10** in the smoke — see §3; it is a model-output/extractor interaction,
  **not** broken C scoring (gold C scores 1.0, §2).

## 2. McEval scoring-health map across all 40 languages (gold)

`scripts/verify_mceval_docker.py --langs all` runs McEval on the **reference
solutions** of every language inside the pinned image — if McEval can't score its
own gold, it can't be trusted to score model output. Two infra bugs were fixed to
get a clean map (both in `tsmc.eval.docker`):

1. **Toolchain PATH** — the image installs most compilers via version managers
   (rustup, SDKMAN, coursier, ghcup, nvm, Flutter, Julia, Go) that add to PATH only
   from `~/.bashrc`. A non-interactive shell skipped them (`go`/`cargo`/… not
   found). Fix: run the container with **`bash -ic`** (interactive). This alone
   moved 11 languages from ERROR → scored.
2. **One bad toolchain aborted the whole eval** — McEval's `eval_all.py` has its
   per-problem `try/except` commented out. `detail_eval.py` now isolates each
   language (error → record + continue), so one missing toolchain can't lose the
   rest.

**Result (digest `…4735…`, `--limit 5`, `bash -ic`): OK 27 · SOFT 1 · REDUCED 5 ·
EXCLUDED 3 · (NODATA 4 reduced langs).** Classification frozen in
`tsmc.eval.language_health`:

- **OK (27, gold ≈ 1.0):** C, C#, CPP, CoffeeScript, Common Lisp, Dart, Elixir,
  Emacs Lisp, Go, Groovy, Haskell, JavaScript, Julia, Kotlin, PHP, Perl,
  PowerShell, Python, Racket, Ruby, Scala, Scheme, Shell, Swift, Tcl, VimScript,
  Visual Basic. → trust model scores.
- **SOFT (1):** Rust — extraction matches gold but McEval cold-recompiles crates
  per problem under a timeout (Phase-0 known); report, never gate.
- **REDUCED_CEILING (5, gold < 0.9 at n=5):** Erlang 0.60, Fortran 0.40, Lua 0.80,
  Pascal 0.40, TypeScript 0.80 (plus Python ~0.90). McEval's own extractor
  mis-reconstructs a fraction even of reference code → interpret model accuracy
  against the per-language ceiling. *(Values are coarse at n=5; refine with a
  higher `--limit`.)*
- **EXCLUDED (3 + SQL):** F#, Java, R score **0.0 on gold** — McEval cannot run its
  own reference solution → verdicts unreliable. Dropped from accuracy like SQL
  (kept in the manifest), until a per-language handler exists.
- **NODATA (4):** AWK, HTML, JSON, Markdown — reduced languages with no rich gold
  fields; handled separately.

**Consequence for reporting:** scoring + the behavioral gate report a
**health-aware accuracy** (`tsmc.eval.join.summarize` → `healthy_accuracy`) over
the non-EXCLUDED/non-SOFT languages, so broken-scoring languages never pollute the
headline or the ±3% gate. Generation still runs for all 40 (outputs saved), and
scoring is decoupled and re-runnable, so EXCLUDED/REDUCED languages can be
recovered later without re-generating.

## 3. Open items carried forward

- **C model output 0/10** (gold C = 1.0): McEval's `extract_ccpp_code` re-stitches
  `includes + prompt[:-1] + model_code + test`; real model-style C (vs. the
  reference) breaks the stitch. Needs a C-output diagnostic (reconstruct + gcc) to
  decide: tune the directive for Family-B, adjust wrapping, or accept. Re-scorable.
- **Refine REDUCED/EXCLUDED ceilings** at a higher `--limit` (n=5 is coarse).
- **F#/Java/R handlers** (optional): McEval can't score their gold; investigate
  per-language or leave excluded with caveat (as SQL).

## 4. Realized γ=1.0 baseline (Qwen2.5-Coder-3B) — see PHASE1_COMPLETE.md for the full report

Healthy-language behavioral gate: generation 0.169/0.178 (Δ .009 ✅), completion
0.238/0.249 (Δ .010 ✅), explanation 0.440/0.404 (Δ .036 — 1.2 SE, sampling noise →
manifest confirm-frozen with a note). Generation `format_fail` ≈ 19% (pre-SFT
sentinel-compliance gap). Completion gate: all subtypes `skipped_no_lever` (median
induced CoT 81–114 tokens — the 3B *does* reason on completion — but cot/code ratio
<1, so the negative control holds). Correct-CoT corpus (train): generation 244,
explanation 634, completion 1700; generation is thin (32 langs, 30 cells <3).
Headline is depressed by (a) all-language averaging over broken scorers, (b) the 3B's
weakness on niche languages (Python ~90% but Emacs Lisp/VimScript/Racket/Tcl weak),
and (c) the format gap — all expected; SFT should lift compliance.
