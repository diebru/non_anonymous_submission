# Phase 3 — LLaMA-Factory integration: design + as-built

Convert the Phase-2 multi-γ compressed corpus into LLaMA-Factory SFT data with the
**frozen γ-control prompt**, register it via `dataset_info.json`, prove
decontamination, and write the LoRA config — SFT-readiness for Qwen2.5-Coder-3B.
**Status: ✅ COMPLETE for Qwen2.5-Coder-3B (gate PASS, 2026-06-01).** Server build +
`check_sft_dataset` gate cleared after the §9 contract re-freeze: **423 correct
trajectories → 5,076 all-γ SFT examples** (balanced 423/γ), zero drops,
decontamination PASS, γ-marker survives the Qwen template, templated p100 = 1,211
tokens (< cutoff_len 2,048). Ready for Phase-4 LoRA SFT.

Source of truth: [`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md) §8 *Phase 3* + Decisions
#3/#6/#8. Inputs: [`phase2_plan.md`](phase2_plan.md) (the compressed corpus),
`src/tsmc/contract/` (frozen prompt/parse), `manifest/split_manifest.csv` (frozen
split), the vendored `TokenSkip/get_llamafactory_input.py` +
`TokenSkip/configs/examples/train_lora/...qwen_3B.yaml`.

## 1. Decisions resolved (were §4 open: P3-1/2/3)
- **P3-1 — Explanation SFT status → GENERATION-ONLY SFT.** Explanation's stage-1
  prompt carries no γ-marker; its compression is post-hoc/external (Decision #3), so
  the model never needs to *learn* to honor γ for it. **Phase-4 explanation runs on
  the un-SFT'd base instruct model** (so the SFT effect can't contaminate the
  information-bottleneck claim) — confirmed by the user. Completion excluded (no
  lever). Model-side explanation stays a future *separate* ablation.
- **P3-2 — γ density → ALL 12 γ per trajectory** (~2,928 gen examples). The 244-traj
  corpus is too thin for TokenSkip's one-random-γ (~20/γ); all-γ gives ~244/γ,
  balanced, and includes the γ=1.0 anchor. Memorization is acceptable — curves are
  measured on the **held-out test split**. `--gamma-sampling random-k` kept as an
  ablation knob.
- **P3-3 — format → ShareGPT `messages`** (`template: qwen`). 1:1 with inference's
  `chat_messages` (single user turn → assistant turn).

## 2. The re-join (the main plumbing)
Phase-2 records carry `cot_text`/`code_snippet`/`gamma`/… but NOT the McEval
`instruction`/`entry_point`. The builder recovers them by calling the **same**
`tsmc.inference.prompts.select_units("generation","train")` Phase-1/4 inference use,
indexed by `mceval_task_id` (carried in `_provenance` through Phase 2), then the
**same `reasoning_user_text(unit, gamma)`** — so the user turn (and the γ marker) is
byte-identical to inference *by construction*, not by re-derivation. This is the
critical-path freeze (roadmap §8): a train/inference γ-marker drift would mean the
SFT'd model never honors γ.

## 3. The training example (generation, model_side)
- **user** = `reasoning_user_text(unit, gamma)` → `instruction` → γ marker
  (`<|eot_id|>{γ}<|eot_id|>`, omitted at γ=1.0) → output-format directive.
- **assistant** = `build_assistant_target` = `{compressed_cot}\n{SENTINEL}\n` +
  one ` ```{fence_lang} ` block holding `code_snippet` verbatim.
- **Build-time round-trip assertion:** `parse_generation(assistant)` must recover
  `code_snippet` byte-identical on a clean branch (`sentinel`/`multi_fence`); else
  the example is **dropped with a logged reason** (`code_roundtrip_mismatch`,
  `bad_branch:*`, `marker_inconsistent`) — never train on a target our own parser
  rejects. γ-marker presence/omission is also guarded per example.

## 4. Decontamination (required gate)
`tsmc.sft.decontam.decontaminate`: load the manifest, assert (1) **no** SFT base
`problem_id` ∈ `test_problems` and (2) **every** id ∈ `train_problems` (canonical
lower-cased; idempotent on already-canonical ids). Vacuously true by construction
(train-only corpus) but checked explicitly and fails the build loudly.

## 5. Outputs (gitignored `sft_dir` = `<data_root>/sft`)
```
sft/<model>/generation_train.jsonl   ShareGPT messages (one example/line)
sft/<model>/dataset_info.json        LLaMA-Factory registration (sharegpt, messages tags)
sft/<model>/build_summary.json       coverage / drops / decontam / length stats
```
`dataset_info.json` registers `tsmc_<model>_generation` (formatting `sharegpt`,
`columns.messages=messages`, role/content tags). LLaMA-Factory reads it via
`dataset_dir` = that folder.
> ⚠ Gitignore note: the `sft/` ignore rule is **anchored** (`/sft/`) so it catches
> the data dir at repo root without also ignoring the `src/tsmc/sft/` source package.

## 6. cutoff_len safety (gate)
`build_sft_dataset.py` prints example length p50/p95/**p100**; with `--count-tokens`
it loads the Qwen tokenizer (server) and reports **true token** lengths +
`cutoff_len_recommendation` (next 512 multiple above p100). `cutoff_len` MUST exceed
the γ=1.0 worst case — a truncated full-CoT target loses the sentinel+code and
poisons the knob's top anchor (roadmap's named risk).

## 7. Code (CPU-testable core + thin LLaMA-Factory wiring)
- `src/tsmc/sft/format.py` — `build_example` (round-trip-asserted), `build_assistant_target`,
  `select_variants` (P3-2), `gamma_marker_consistent`. Pure, tokenizer-free.
- `src/tsmc/sft/decontam.py` — manifest test-leakage gate.
- `scripts/build_sft_dataset.py` — driver: re-join, emit dataset + `dataset_info.json`,
  run decontam, print per-γ/per-cell coverage + length stats. CPU; `--count-tokens`
  for the true cutoff_len input.
- `configs/llamafactory/qwen2.5-coder-3b_lora.yaml` — LoRA template adapted from
  TokenSkip's 3B (rank 8/α 16, `lora_target all`, `template qwen`, lr 5e-5, 3 epochs,
  bs1×ga8, bf16, val_size 0.1); `cutoff_len 4096` to confirm from `build_summary`.
- `tests/test_sft_format.py` — round-trip, γ-marker presence/omission,
  **byte-identity-with-inference guard**, γ-sampling, decontam (16 tests).
- Path wiring: `ProjectPaths.sft_dir` (+ `artifact_dirs`), `paths.example.yaml`.

## 8. Server recipe + remaining gates
```bash
# tokenskip_env (CPU dataset prep): build + register + decontaminate
git pull
python3 scripts/build_sft_dataset.py --model qwen2.5-coder-3b-instruct --count-tokens
#   -> reads build_summary.json: set cutoff_len in the LoRA yaml from cutoff_len_recommendation
# llamafactory_env: REMAINING Phase-3 gates
#   (a) dry-run dataset load passes
#   (b) LLaMA-Factory `qwen` template == tokenizer.apply_chat_template on a sample
#       (the classic silent train/inference drift; assert the user turn matches)
```
The actual LoRA SFT run is **Phase 4**. Outputs land under gitignored `sft/`.

## 9. Contract re-freeze (found by the Phase-3 gate, 2026-06-01)
`check_sft_dataset.py` caught that **1,812/2,928 examples (151/244 problems) had an
empty `code_snippet`**: the 3B frequently writes its code in a fenced block *inside
the reasoning* and emits a bare/empty trailing sentinel, so the post-sentinel region
was empty. The old `three_way_outcome` trusted McEval's verdict without requiring a
fence, so these were scored `pass` (McEval reconstructed from the prompt/reference) —
a textbook extraction-vs-reasoning confound. Fix (CPU, no GPU re-inference, since
`raw_full_output` is saved):
- **Parser re-frozen:** new `presentinel_salvage` branch recovers the **last
  pre-sentinel fenced block** as `code_snippet` and the text before it as the clean
  CoT; `three_way_outcome` now returns `format_fail` for any fenced (non-`direct_fill`)
  branch with `fence_found=False`. Prompt side unchanged → γ-marker freeze intact.
- **`build_example` guard:** drops `empty_code` targets (belt-and-suspenders).
- **Rebuild chain (server):** `reparse_trajectories.py` (re-parse saved raw outputs)
  → `score_generations` (McEval re-executes the recovered code) → `phase1_gates` →
  `build_corpus` → `compress_corpus` → `build_sft_dataset` → `check_sft_dataset`.
- Also fixed a `check_sft_dataset` token-count bug (`apply_chat_template(tokenize=True)`
  returned a dict on this transformers version → measured "2"; now counts the rendered
  strings).
