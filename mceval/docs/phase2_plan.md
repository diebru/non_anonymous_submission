# Phase 2 — LLMLingua-2 compression: design + run recipe

Compress the Phase-1 correct-CoT corpus into the 12-γ family of compressed-CoT
variants that Phase 3 converts to SFT data.
**Status: ✅ COMPLETE & validated for Qwen2.5-Coder-3B (gate PASS, 2026-06-01).**
Companions: [`PHASE1_COMPLETE.md`](PHASE1_COMPLETE.md) (the corpus this consumes),
[`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md) §8 + Decision #3 (design source of truth),
[`phase3_plan.md`](phase3_plan.md) (what consumes this corpus next).

> **Server run result (Coder-3B, train, the server 2026-06-01):** generation 244 traj →
> 2,928 variants (median CoT **342 → 27** Qwen tokens across the 12 γ); explanation
> 634 → 7,608 (median **82 → 7**); completion skipped (no lever). Validator **GATE
> PASS**: aggregate-monotonic OK, per-trajectory non-increasing 1.0, scaffolding /
> schema / cot_origin errors 0. The XLM-R `577 > 512` warning is benign —
> LLMLingua-2 chunks long CoTs (verified on kotlin/23, 950 Qwen tokens, compressed
> head→tail). Corpus at gitignored `compressed/qwen2.5-coder-3b-instruct/run01/`.

## 1. Scope
Pure **text transformation + tokenization** — no Docker, no GPU *inference*, no
re-execution. Per verified-correct (γ=1.0) trajectory, emit one variant per γ in
`GAMMA_GRID` (12 values).

| Task | Phase-1 corpus (Coder-3B, train) | Phase 2 |
|---|---|---|
| generation | 244 | compress `cot_text` (reasoning) |
| explanation | 634 | compress `cot_text` (stage-1 description, post-hoc) |
| completion | 1700 | **SKIPPED** — gate `skipped_no_lever` (Decision #5), read from `phase1_gates.json` |

## 2. Two decisions (resolved 2026-06-01)
1. **Compression params:** TokenSkip-qwen *faithful* — bare
   `compress_prompt(cot_text, rate=γ)`, no `force_tokens`/`force_reserve_digit`/
   `drop_consecutive`. The NL-tuned-compressor-on-code risk is accepted and will be
   visible later in the explanation curve.
2. **`pass`/re-exec scope:** Phase 2 is pure compression. γ<1.0 variants carry
   `pass=False` (provisional, not executed) with `source_pass=True` in
   `_compression`; the γ=1.0 passthrough keeps `pass=True`. Explanation's
   information-bottleneck re-execution belongs to Phase 4.

## 3. Integration (grounded in TokenSkip + a the server smoke test)
- **Checkpoint (pinned):** `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`
  @ `ebaba9b0e874dadd3003ffcff828e4397e568089`. Pinned in committed code
  (`tsmc.compression.llmlingua.DEFAULT_LLMLINGUA2_*`) and documented in
  `configs/run_metadata.example.yaml`, so the pin reaches the server despite the
  gitignored working `run_metadata.yaml`.
- **Call:** `PromptCompressor(model_name=ckpt, use_llmlingua2=True)`, loaded once,
  reused across all γ/trajectories. Return dict keys observed on the server:
  `compressed_prompt, compressed_prompt_list, compressed_tokens, origin_tokens,
  rate, ratio, saving` — `rate` is the *achieved* fraction as a string (`"42.9%"`),
  not the target. We keep `origin_tokens/compressed_tokens/rate/ratio/saving` as
  provenance only.
- **Token-count authority:** `cot_token_count` is **re-counted with the Qwen
  tokenizer** (`len(tok(text, add_special_tokens=False).input_ids)` — identical to
  the Phase-1 harness `count_tokens`), NOT LLMLingua-2's XLM-R count, so the x-axis
  stays comparable with Phase 1 and the energy join. `gamma`/`compression_ratio`
  stay the **target** γ.
- **Where it runs:** server, `tokenskip_env` (GPU). Never local. `--dry-run` swaps a
  deterministic word-drop mock + whitespace counter for a no-GPU local smoke.

## 4. Scaffolding integrity (by construction)
The contract parser already split sentinel + fenced code + entry_point out of
`cot_text` in Phase 1, and the γ-marker is a Phase-3 prompt artifact never stored
here. So compression touches **only** `cot_text`; `code_snippet` /
`extraction_status` are copied verbatim. `check_scaffolding_intact` re-confirms
(code unchanged, no sentinel leaked into `cot_text`).

## 5. Output layout (gitignored `compressed/`)
```
compressed/<model>/<run>/<task>/<split>/gamma<g>/<lang>.jsonl
compressed/<model>/<run>/compression_summary.json
compressed/<model>/<run>/validation_report.json
```
Each variant = the source record with `gamma`/`compression_ratio`/`cot_text`/
`cot_token_count`/`cot_origin`/`pass` overridden, `code_snippet`+`extraction_status`
verbatim, plus a `_compression` provenance block (params, checkpoint SHA,
LLMLingua-2 native tokens/rate, `source_pass`, `degenerate`).

## 6. Completion criteria (roadmap Phase-2 gate → `validate_compression.py`)
- **Monotonic γ→tokens:** per-gamma **median** `cot_token_count` strictly
  non-increasing (aggregate headline, REQUIRED); per-trajectory non-increasing with
  ties allowed (short re-tokenized CoTs wobble near γ=1.0) — soft, flagged only if
  the corpus-wide non-increasing fraction < 0.80.
- **Scaffolding:** `code_snippet` identical across a trajectory's γ; no sentinel in
  any `cot_text`; every variant passes `validate_record`; `cot_origin==original`
  iff γ==1.0.

## 7. Code
- `src/tsmc/compression/corpus.py` — CPU core (`compress_record`, validators).
- `src/tsmc/compression/llmlingua.py` — server-only `Lingua2Compressor` +
  `make_token_counter` (lazy imports).
- `scripts/compress_corpus.py` — driver (gate-aware completion skip, `--dry-run`).
- `scripts/validate_compression.py` — the Phase-2 gate.
- `tests/test_compression.py` — 15 CPU tests (mock compressor/counter).

## 8. Server run recipe
```bash
# env: conda activate tokenskip_env ; repo <repo-root>
git pull
M=qwen2.5-coder-3b-instruct
python3 scripts/compress_corpus.py    --model $M --task all --split train   # gen+expl; completion auto-skipped
python3 scripts/validate_compression.py --model $M --split train            # GATE: PASS/FAIL
```
Outputs land under the gitignored `compressed/`. `--dry-run` (local) validates the
IO/orchestration without llmlingua or a GPU.
```bash
python3 scripts/compress_corpus.py --model $M --dry-run   # local smoke
```
