# Phase 1 — Train-data generation: completion report (Qwen2.5-Coder-3B)

**Status: ✅ COMPLETE for Qwen2.5-Coder-3B-Instruct** (baseline γ=1.0, all 40
languages × 3 tasks × 2 splits, scored, gated, corpus built). Manifest
**CONFIRM-FROZEN with a note** (see §5). Test suite: **109 passing**.

Self-contained record so the work is recoverable without the build conversation.
Companions: [`PHASE0_COMPLETE.md`](PHASE0_COMPLETE.md) (foundations),
[`phase1_findings.md`](phase1_findings.md) (scoring-health map + empirical detail),
[`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md) (design source of truth).

> Scope note: Phase 1 was run for **one** model (Coder-3B). The other three
> (Qwen2.5-3B-Instruct, Coder-7B, Coder-14B) still need the same pass — see §7.

---

## 1. Goal of Phase 1
Produce the baseline (γ=1.0, **no SFT**) CoT+code on train+test per model; score
with McEval; run the behavioral ±3% gate (→ confirm-freeze the manifest); decide
the completion induced-CoT gate; and filter correct trajectories into the
per-model SFT corpus. No compression or SFT yet.

---

## 2. What was built (code, all pushed, 109 tests green)

| Task | Deliverable | Where |
|---|---|---|
| **1.1 Inference harness** | manifest-driven selection, frozen-contract prompt assembly, lazy vLLM runner (greedy, finish_reason, token counts, per-request timing), two-pass explanation, completion induced-CoT, **2-GPU data-parallel** (`--shards`, auto-spawn one worker/GPU + merge), `--dry-run` | `src/tsmc/inference/`, `scripts/run_inference.py` |
| **1.2 Eval join** | container-side detail-eval shim (per-`task_id` verdicts; robust per-language), Docker driver, CPU join → long-format records with three-way `outcome` | `src/tsmc/eval/{detail_eval,docker,join}.py`, `scripts/score_generations.py` |
| **Scoring hardening** | `bash -ic` toolchain PATH fix, robust per-language eval, 40-language gold scoring-health map, health-aware accuracy | `src/tsmc/eval/{docker,language_health,results}.py`, `scripts/verify_mceval_docker.py --langs all` |
| **1.3 Gates + corpus** | behavioral ±3% gate, completion induced-CoT gate (Decision #5), correct-trajectory corpus + per-cell counts | `src/tsmc/eval/gates.py`, `scripts/phase1_gates.py`, `scripts/build_corpus.py` |

### Pipeline (per model)
```
run_inference.py  --shards 2     -> generations/<model>/run01/<task>/<split>/gamma1/
                                      result/<Lang>.jsonl       (McEval input)
                                      trajectories/<Lang>.jsonl (long-format, pass provisional)
score_generations.py (Docker)    -> records/<Lang>.jsonl       (pass + three-way outcome)
                                      score_summary.json
phase1_gates.py                  -> behavioral gate + completion gate -> phase1_gates.json
build_corpus.py                  -> corpus/<model>/run01/<task>/<split>/<lang>.jsonl + summary
```

---

## 3. Results — Qwen2.5-Coder-3B-Instruct, γ=1.0

**Behavioral ±3% gate (healthy languages — the principled view):**

| Task | train acc | test acc | \|Δ\| | gate |
|---|---|---|---|---|
| generation | 0.1692 | 0.1783 | 0.0091 | ✅ within (scored 1442/359) |
| completion | 0.2382 | 0.2486 | 0.0104 | ✅ within (scored 7136/1786) |
| explanation | 0.4397 | 0.4039 | 0.0358 | ⚠️ mechanically outside → **noise** (§5) |

**All-language accuracy** (for reference, incl. broken-scoring langs): gen
0.154/0.160, compl 0.215/0.223, expl 0.435/0.397.

**`format_fail` (generation):** train **18.9%**, test 21.9% — the pre-SFT model
skips our sentinel and writes a bare fence (`fallback` branch). Tracked separately
(never counted as a reasoning failure); expected to shrink after SFT. Completion
`format_fail` ≈ 0.1% (`direct_fill` works); explanation ≈ 2.5–3.5% (fence-first).

**Completion induced-CoT gate (Decision #5)** — all subtypes `skipped_no_lever`:

| subtype | n | median CoT tokens | median cot/code | decision |
|---|---|---|---|---|
| single | 2396 | 95 | <1 | skipped_no_lever |
| multi | 2396 | 114 | <1 | skipped_no_lever |
| span | 3306 | 81 | <1 | skipped_no_lever |

Finding: the 3B **does** reason on completion (median 81–114 CoT tokens, not ~0 as
predicted), but the CoT is small relative to the full completed program (ratio <1),
so the negative control holds — completion has no compression lever.

**Correct-CoT corpus (train, verified-correct on healthy languages):**

| Task | correct/scored | yield | langs | easy/middle/hard | thin cells (<3) |
|---|---|---|---|---|---|
| generation | 244/1606 | 0.152 | 32 | 163/45/36 | 30 |
| explanation | 634/1606 | 0.395 | 35 | 493/97/44 | 38 |
| completion | 1700/8098 | 0.210 | 35 | 1039/362/299 | 20 |

---

## 4. Scoring-health map (gold, 40 languages) — frozen in `tsmc.eval.language_health`
McEval has 40 per-language extractors; some can't score even their own reference.
Running gold through the pinned image classified them (digest …4735…, n=5, `bash -ic`):
- **OK (27, gold ≈1.0):** C, C#, CPP, CoffeeScript, Common Lisp, Dart, Elixir, Emacs
  Lisp, Go, Groovy, Haskell, JavaScript, Julia, Kotlin, PHP, Perl, PowerShell,
  Python, Racket, Ruby, Scala, Scheme, Shell, Swift, Tcl, VimScript, Visual Basic.
- **SOFT (1):** Rust (cargo recompile-timeout per problem).
- **REDUCED_CEILING (5):** Erlang .6, Fortran .4, Lua .8, Pascal .4, TypeScript .8
  (+ Python ~.9). McEval's extractor mis-reconstructs a fraction of gold.
- **EXCLUDED:** F#, Java, R (gold 0.0 — McEval can't run its own reference) + SQL
  (never executed). Dropped from accuracy like SQL; kept in the manifest.

`join.summarize` reports `healthy_accuracy` over non-EXCLUDED/non-SOFT languages, so
broken scorers never pollute the headline or the gate. Everything is **re-scorable**
on saved generations (scoring is decoupled from generation).

---

## 5. Manifest decision: CONFIRM-FROZEN with a note
The explanation gate is mechanically OUTSIDE (Δ=0.0358) but **statistically
sampling noise**: at healthy test n=359 the difference SE ≈ 2.9%, so Δ ≈ **1.2 SE
(p ≈ 0.22)** — not significant. Generation and completion are comfortably within.
The roadmap (s6) pre-authorizes this: remedy #1 = "within sampling error → accept
with a note." **Decision: manifest is confirm-frozen**; re-drawing the seed over a
1.2-SE wobble would chase noise. Revisit only if a second model shows the same task
diverging in the same direction (systematic, not noise).

> **Re-confirmed after the Phase-3 contract re-freeze (2026-06-01).** Re-parsing +
> re-scoring generation with the corrected parser (`presentinel_salvage` + the
> fence-found outcome gate) raised honest generation accuracy and re-ran the gate:
> generation train **0.2933** / test **0.2618** (Δ=0.0315, **~1.2 SE**, p≈0.23),
> explanation 0.4397 / 0.4039 (Δ=0.0358, ~1.2 SE — unchanged; explanation was not
> reparsed), completion within. Both marginal misses are again sampling noise at the
> ~1-SE-wide ±3% gate, and the frozen split is already baked into the corpus /
> compressed / SFT artifacts, so re-drawing would discard Phases 1–3 to chase noise.
> **Manifest stays confirm-frozen** (user-confirmed, remedy #1).

---

## 6. Open items (carry into later phases; none block Phase 2)
- **C model-output stitch:** gold C = 1.0 but model-generated C scored ~0 — McEval's
  `extract_ccpp_code` re-stitches `includes + prompt[:-1] + code + test` and chokes
  on model-style C (vs. the reference). C is in the healthy set, so it depresses
  generation accuracy. Needs a reconstruct+gcc diagnostic; re-scorable.
- **Low 3B multilingual baseline** (gen ~17% healthy): Python ~90% but niche
  languages (Emacs Lisp, VimScript, Racket, Tcl, …) are weak, plus the 19%
  format_fail. Thin generation corpus (244, 30 thin cells) — watch for SFT richness.
- **Refine REDUCED/EXCLUDED ceilings** at a higher `--limit` (n=5 is coarse).
- **`code_token_count`** is native from this point on; this run's records were
  backfilled by `phase1_gates.py` via the model tokenizer.

---

## 7. What is NOT done
- **Phase 1 for the other 3 models** (Qwen2.5-3B-Instruct, Coder-7B, Coder-14B):
  rerun §2 pipeline per model (TokenSkip is per-model self-distillation). 7B/14B may
  want `--tensor-parallel-size 2` instead of `--shards`.
- **Phase 2 (compression):** LLMLingua-2 at the 12 γ on the correct-CoT corpus —
  generation CoT + explanation stage-1 descriptions (post-hoc); completion skipped
  (gate = no lever). `src/tsmc/compression/` is a placeholder.

---

## 8. Server run recipe (reproduce / next model)
```bash
# env: conda activate tokenskip_env ; repo <repo-root>
git pull
M=qwen2.5-coder-3b-instruct
DIGEST=sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5
python3 scripts/run_inference.py   --task all --split both --shards 2 --gpus 0,1 --model $M
python3 scripts/score_generations.py --task generation  --split both --model $M --digest $DIGEST
python3 scripts/score_generations.py --task completion  --split both --model $M --digest $DIGEST
python3 scripts/score_generations.py --task explanation --split both --model $M --digest $DIGEST
python3 scripts/phase1_gates.py  --model $M
python3 scripts/build_corpus.py  --model $M
```
Outputs live under the gitignored `generations/` and `corpus/`. Run exactly **one**
`score_generations` per task (duplicates for the same task collide on `result.jsonl`).
