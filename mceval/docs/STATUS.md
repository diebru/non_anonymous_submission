# PROJECT STATUS — entry point after a chat reset

**TokenSkip × McEval** — controllable CoT compression (γ = fraction of CoT tokens
retained) applied to multilingual code generation, scored by McEval execution.
Goals: (1) accuracy-vs-CoT-tokens is **concave**; (2) energy drops as CoT shortens;
(3) slight compression preserves accuracy while saving energy.

> **Where we are (2026-06-08): the decoding-matrix study is DONE — read
> [`RESULTS.md`](RESULTS.md) FIRST.** Headline: **a `frequency_penalty=0.3` repetition
> penalty fixes the low-γ runaway tail** (the obstacle from the Coder-3B sweep), so
> **aggregate energy finally DESCENDS with compression** at preserved/better accuracy
> (14B @γ0.1: format_fail 0.086→0, GPU energy −59%, time −59%, acc 0.507→0.549). Valid
> models: **14B, 3B, Coder-3B** (full 6-cell penalty×max_tokens matrix on 14B+3B).
> **Qwen2.5-7B-Instruct emits no CoT on the default prompt** (genuine model behavior) →
> fixed with a reason-first **system prompt**, re-run under `run02`/`sft02`. Full detail +
> caveats + data layout: **[`RESULTS.md`](RESULTS.md)**; matrix tracker + commands:
> **[`EXPERIMENTS.md`](EXPERIMENTS.md)**.

> **Cross-family arm IN PROGRESS (2026-06-09): Llama-3.1-8B-Instruct.** The γ-marker was
> redesigned per-family (Llama's `<|eot_id|>` is a real special token → `@@@GAMMA_7F3A9@@@`),
> and Phase 1→4 + the knob are **DONE and validated** (knob: median CoT 204.5→43, 79%,
> monotonic). Remaining: the no-penalty energy-sweep matrix `{2048,1024,512} × {test,train}`.
> Full record + resume point: **[`LLAMA_CROSS_FAMILY.md`](LLAMA_CROSS_FAMILY.md)**.

**History:** Phases 0–4 first completed for Qwen2.5-Coder-3B ([`PHASE4_RESULTS.md`]
(PHASE4_RESULTS.md) — the original negative-for-aggregate-energy result that motivated the
penalty fix), then repeated on the non-code Qwen2.5-3B/7B/14B-Instruct. Per-model recipe:
**[`PIPELINE_RUNBOOK.md`](PIPELINE_RUNBOOK.md)**.

Read alongside: **[`RESULTS.md`](RESULTS.md)** (consolidated findings — START HERE),
**[`EXPERIMENTS.md`](EXPERIMENTS.md)** (decoding-matrix tracker + commands),
**[`PIPELINE_RUNBOOK.md`](PIPELINE_RUNBOOK.md)** (how to run the whole
pipeline on a model), **[`PHASE4_RESULTS.md`](PHASE4_RESULTS.md)** (the Phase-4 report —
results, mechanism, next steps), [`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md) (design
source of truth), [`PHASE1_COMPLETE.md`](PHASE1_COMPLETE.md),
[`phase2_plan.md`](phase2_plan.md), [`phase3_plan.md`](phase3_plan.md),
[`WORKFLOW.md`](WORKFLOW.md). The auto-memory `tokenskip-mceval-project.md` has the
full running log.

---

## 1. Phase status
| Phase | What | Status |
|---|---|---|
| 0 Foundations | repo, contracts/schema frozen, McEval Docker verified, manifest | ✅ complete |
| 1 Train-data gen | baseline CoT+code on train, eval, correct-CoT corpus | ✅ complete (Coder-3B) |
| 2 Compression | LLMLingua-2 multi-γ compression of correct CoTs | ✅ complete (Coder-3B) |
| 3 LLaMA-Factory | γ-control SFT dataset + decontam + LoRA config + gate | ✅ complete (Coder-3B) |
| 4 SFT + sweep + curves | LoRA SFT ✅ · knob ✅ · adapter merged ✅ · energy sweep + curves ✅ | ✅ complete (Coder-3B/generation) — see [`PHASE4_RESULTS.md`](PHASE4_RESULTS.md) |

Only **Qwen2.5-Coder-3B-Instruct / generation** has been carried through so far. Open
Phase-4 follow-ups (priority order in PHASE4_RESULTS §6): (1) fix the **runaway tail**
(repetition penalty / tighter cap / low-γ SFT) — the sole obstacle to aggregate energy
savings; (2) the **explanation** task (compressible region is a larger share → likely a
real aggregate energy descent); (3) the other 3 models (Qwen2.5-3B-Instruct controlled
pair, Coder-7B/14B — prefer `--tensor-parallel-size 2`), reusing `merge_lora.py` + the
sweep.

## 2. Headline numbers (Coder-3B)
- **Correct generation corpus: 423 trajectories** (after the §4 contract re-freeze;
  was 244). Honest generation accuracy γ=1.0 ≈ **0.293** healthy.
- **SFT dataset: 5,076 examples** (423 problems × 12 γ, balanced, zero drops,
  decontamination PASS). Templated token p100 = 1,211 (< cutoff_len 2,048).
- **SFT:** LoRA (r8/α16, all linear, qwen template, lr5e-5, 3 ep, bf16), eval_loss
  **0.2405**.
- **Knob validation (held-out test, 6 γ): median CoT 159.5 → 23 (86% shrink,
  monotonic) → PASS.** `generations/qwen2.5-coder-3b-instruct/knob_validation.json`.

## 3. Pinned versions / digests
- **Base model:** `Qwen/Qwen2.5-Coder-3B-Instruct`, HF snapshot
  `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- **LLMLingua-2:** `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` @
  `ebaba9b0e874dadd3003ffcff828e4397e568089`.
- **McEval Docker:** `multilingualnlp/mceval@sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5`
  (pass `--digest` to `score_generations.py`, or set `mceval.docker_digest` in the
  gitignored `configs/run_metadata.yaml`). Container MUST run via `bash -ic`.
- **γ grid (12):** 1.0, .95, .9, .85, .8, .7, .6, .5, .4, .3, .2, .1. **Sentinel:**
  `@@@FINAL_CODE_7F3A9@@@`. **γ marker:** `<|eot_id|>{γ}<|eot_id|>` (omitted at 1.0).

## 4. Why the corpus changed (contract re-freeze, important context)
The Phase-3 gate caught **1,812/2,928 SFT examples with empty `code_snippet`**: the
3B often codes *inside its reasoning* and emits a bare/empty trailing sentinel, so
the post-sentinel region was empty and McEval false-passed the empty code (extraction
confound). Fix (no GPU re-inference — raw outputs were saved):
- Parser **re-frozen**: `presentinel_salvage` branch recovers the last pre-sentinel
  fenced block as the code + the text before it as clean CoT; `three_way_outcome`
  now returns `format_fail` for any non-`direct_fill` branch with `fence_found=False`.
  **Prompt side unchanged → γ-marker freeze intact.**
- Rebuild chain: `reparse_trajectories.py` → `score_generations` → `phase1_gates` →
  `build_corpus` → `compress_corpus` → `build_sft_dataset` → `check_sft_dataset`.
- Behavioral ±3% gate re-confirmed (gen Δ0.0315, expl Δ0.0358 — both ~1.2 SE
  sampling noise at the ~1-SE-wide gate); **manifest stays confirm-frozen** (the
  split is baked into all downstream artifacts).

## 5. Server artifacts (gitignored; on `<repo-root>`)
- LoRA adapter: `weights/qwen2.5-coder-3b-instruct/lora_sft_run01/`
- SFT dataset:  `sft/qwen2.5-coder-3b-instruct/{generation_train.jsonl,dataset_info.json,build_summary.json}`
- Compressed corpus: `compressed/qwen2.5-coder-3b-instruct/run01/{generation,explanation}/train/gamma*/`
- Correct-CoT corpus: `corpus/qwen2.5-coder-3b-instruct/run01/`
- Baseline generations + scores: `generations/...`, `eval_dumps/...`
Envs: `tokenskip_env` (vLLM inference, LLMLingua-2, CPU prep) · `llamafactory_env`
(LoRA SFT) · McEval Docker (eval). Loop: edit locally → push → server pull → run.

## 6. NEXT — Phase 4 remaining work (the energy-instrumented test sweep)
**Do these in order; the first two are prerequisites the user flagged:**
1. ✅ **DONE — LoRA adapter MERGED** → standalone `weights/qwen2.5-coder-3b-instruct/
   merged_sft_run01` (bf16); merged-knob **PASS** (median CoT 160.5→22, matches
   base+adapter). Method = **`scripts/merge_lora.py`** (peft `merge_and_unload`,
   fp32 merge→bf16 save), run in **`tokenskip_env`**. ⚠ `llamafactory-cli export` is
   BROKEN on the server's transformers 5.2.0 (degenerate ~0-CoT merge) — do NOT use
   it; the env tokenizers also diverge (5.2.0 vs vLLM's ~4.46), so merge in
   `tokenskip_env`. Validate via `validate_knob.py --model-path <merged> --limit 3`.
2. ✅ **BUILT (pending server run) — energy instrumentation.** `src/tsmc/energy/`
   (`core.py` integrate_power/summarize_run — RUN-level energy, GPU primary / PDU
   secondary; `monitors.py` EnergyMonitors ctx), hardened `scripts/monitor_{gpu,pdu}.py`
   (GPU parses per-line + picks `--gpu-index`; PDU SNMP configurable), `scripts/
   join_energy.py` (integrate over the `generate_window` the harness now records in
   run_meta → stamp each record's reserved `energy` field). `configs/run_metadata
   .example.yaml` has an `energy:` block (gpu_index 0, 0.5s, PDU 192.0.2.1).
   Pollers smoke-tested on the server (GPU 22 W idle, PDU 240 W). **Accuracy control stays
   OUTSIDE energy: monitors wrap inference only; Docker scoring runs after, stopped.**
3. ✅ **BUILT (pending server run) — sweep + curves.** `scripts/run_energy_sweep.py`
   (per-γ: monitors→`run_inference --model-path <merged>`→`score_generations`→
   `join_energy`, single dedicated GPU, reload-per-γ). `scripts/build_curves.py`
   (accuracy-vs-cot + energy-vs-cot + format_fail-vs-γ, healthy-langs, csv/json/png).
   Sweep is **generation-only** (the SFT'd task). **TO RUN:** 1-γ trio smoke
   (`--gammas 1.0 --trio-only --limit 5`) → full 12-γ (`run_energy_sweep.py`) →
   `build_curves.py`.

Knob is proven (merged-knob PASS); no need to re-run before the sweep. The sweep
loads the merged model via `run_inference.py --model-path <merged>` (no adapter).

## 7. Open items / watch-list
- **Energy:** monitors + other-benchmark examples incoming from the user (step 6.2).
- **Behavioral gate** re-confirmed with a note (sampling noise); revisit only if a
  2nd model diverges the same direction.
- **McEval over-passing:** empty/garbage code can false-pass via prompt reconstruction
  in some langs — the §4 fence gate blocks it for generation; watch other tasks.
- **C model output** depressed (McEval Family-B re-stitch chokes on model-style C).
- **Other 3 models** still need Phases 1→4.

## 8. Command reference
```bash
# rebuild chain (after any contract/parse change; tokenskip_env + Docker)
python3 scripts/reparse_trajectories.py --model M --split train --count-tokens
python3 scripts/score_generations.py    --model M --task generation --split train --digest sha256:4735...
python3 scripts/build_corpus.py         --model M
python3 scripts/compress_corpus.py      --model M --task all --split train
python3 scripts/build_sft_dataset.py    --model M --count-tokens
python3 scripts/check_sft_dataset.py    --model M --cutoff-len 2048      # llamafactory_env

# SFT (llamafactory_env) -> edit the 3 machine paths in the yaml first
llamafactory-cli train configs/llamafactory/qwen2.5-coder-3b_lora.yaml

# knob validation (tokenskip_env, GPU)
python3 scripts/validate_knob.py --model M --adapter weights/M/lora_sft_run01 --limit 3
```
(`M = qwen2.5-coder-3b-instruct`)
