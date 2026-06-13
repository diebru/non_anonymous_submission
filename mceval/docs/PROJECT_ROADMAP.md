# PROJECT ROADMAP — McEval + TokenSkip

**Single source of truth for the project design.**
Status: **design phase — no implementation yet.**
Scope of this document: research goals, environment, workflow, per-task output contracts, Phase-0 decisions, stratification/manifest design, the long-format schema, and the Phase 0 → Phase 4 roadmap.

> Convention used throughout: **γ = fraction of CoT tokens *retained*** (γ = 1.0 → full CoT; lower γ → more aggressive compression). Confirming this matches TokenSkip's exact ratio convention (retained vs removed) is a Phase-0 empirical check.

---

## 1. Project overview and research goals

We apply **TokenSkip** (controllable Chain-of-Thought compression) to **multilingual code tasks**, scored by **McEval**'s execution-based harness, to characterize the trade-off between reasoning length, accuracy, and inference energy.

TokenSkip premise: a model can be LoRA-SFT-trained to emit variable-length CoT controlled by a target compression ratio γ, making reasoning length a tunable knob (self-distillation: the model generates its own CoT, which is filtered to correct, compressed, and used to fine-tune itself).

**Research goals**
1. **Concavity (primary):** accuracy vs. measured CoT-token-count is **concave** across compression ratios.
2. **Energy:** inference energy **decreases** as CoT shortens.
3. **Sweet spot:** **slight** compression preserves accuracy while saving energy.

**Three tasks** (all execution-based pass@1; verified in the McEval repo — no BLEU/ROUGE/embedding anywhere):
- **Generation (primary):** problem → reason → code. Genuine reasoning-compression. Classic concave curve expected.
- **Explanation (post-hoc compression):** two-pass — stage 1 the model *describes* code; stage 2 it regenerates code from its **own description** (which is the sole channel into stage 2); the reconstructed code is executed. The **stage-1 description is the compressible "CoT."** Mechanism = information bottleneck; plausibly steeper/earlier collapse than generation.
- **Completion (gated negative control):** FIM — fill `[MASK]` regions, execute. CoT is short/absent (especially `single`); used as the contrast case showing where compression has **no lever**.

**Key cross-cutting facts (verified in the McEval repo)**
- The three tasks are built from the **same base problems** (`Lang/N`). Distinct base problems = **2,066** = **2,007** present in all three tasks (carry a `level`) + **59** present in explanation+completion only (no `level`). Completion expands each base into masked variants (`Lang/N-k-{single|multi|span}`).
  - ⚠ **Verified in Phase 0 (Task 0.2):** the 2,066 count holds only after **case-normalizing the language** in `task_id`. The 59 shared problems are SQL, stored as `sql/N` in explanation but `SQL/N` in completion (generation has no SQL). On raw, case-sensitive ids the count is **2,125** and SQL would leak across the split. Always key base problems on `tsmc.mceval_data.canonical_base_id`. **Task 0.5:** McEval's `eval_all.py` excludes `sql` from execution → these 59 are **execution-unscored** (kept in the manifest for leakage tracking, but dropped from accuracy/behavioral stats; the primary generation task has no SQL). See `docs/phase0_findings.md`.
- Row counts: generation **2,007**, explanation **2,066**, completion **10,128** (~14.2K rows total).
- McEval **always re-runs its own per-language `extract()`** on whatever is placed in `raw_generation[0]`, so our output must land on McEval's happy path (a single language-tagged fenced block containing the exact `entry_point`).

---

## 2. Hardware and environment specs

**Server (all execution happens here):**
- CPU: Intel Xeon Gold 6326
- RAM: 256 GB
- GPU: 2× NVIDIA RTX A6000 (49 GB VRAM each; 98 GB total)

**Three environments (file-based hand-offs on server disk; pinned independently):**

| Environment | Runs | Why separate |
|---|---|---|
| `tokenskip_env` (conda) | vLLM inference (train-trajectory gen + test γ-sweep); TokenSkip; LLMLingua-2 compression; parsing | GPU generation throughput + compression tooling |
| `llamafactory_env` (conda) | LoRA SFT of the Qwen models | LLaMA-Factory training stack (peft/trl/accelerate), isolated from inference deps |
| McEval Docker (`multilingualnlp/mceval`, **pull by sha256 digest, never a floating tag**) | Execution-based evaluation across 40 language runtimes | Carries all language toolchains + McEval's hardcoded `/workspace/MMCodeEval/eval/tmp`; sandboxes untrusted generated code |

**Model matrix (Qwen-only; shared ~151k tokenizer → comparable token x-axis):**

| Role | Models |
|---|---|
| Controlled pair (code vs non-code, same size) | **Qwen2.5-3B-Instruct** ↔ **Qwen2.5-Coder-3B-Instruct** |
| Size axis (anchored on Coder-3B) | **Qwen2.5-Coder-3B → 7B → 14B-Instruct** |

All four feasible on 2× A6000 for vLLM inference and LoRA SFT. **70B excluded** (won't fit fp16; quantization would confound accuracy and energy).

**Feasibility summary (fp16 weights):** 3B ~6 GB, 7B ~15 GB, 8B ~16 GB, 14B ~28 GB — all fit a single A6000 for inference with KV-cache headroom; LoRA SFT fits on 1–2 GPUs.

---

## 3. Workflow constraint (local dev → git push → server git pull → execution)

**Execution is remote-only. Nothing runs locally except lightweight CPU-only tests.**

```
local dev (write/edit) → git add/commit → git push (origin/main)
                                            → server: git pull → execute
```

Rules (see `docs/WORKFLOW.md` for the full detail):
- **Never** run inference, Docker (McEval), or LLaMA-Factory locally.
- GPU scripts (inference, SFT) are **written locally, executed on the server**.
- CPU-only lightweight scripts (manifest generator, schema validators, parsers) **may** be tested locally if data is available; the **canonical run is always on the server**.
- **No hardcoded local paths** — every script reads paths from a config file or environment variables.
- Direct-to-`main` workflow (server pulls `main`); do not use feature branches that the server pull wouldn't see.

**Git artifact policy**
- **In git:** code, configs, and the split **manifest** (tiny text, defines the experiment).
- **Not in git (server/DVC/results branch):** generations, compressed corpora, model/adapter weights, eval dumps, any bulk `.jsonl`.

---

## 4. Per-task output-contract spec

Governing principle: **own the CoT/code separation** with a self-defined sentinel; write **only clean code** into `raw_generation[0]` as a single canonical language-tagged fenced block containing `entry_point`; let McEval's extractor act as a confirming second net. Record a **three-way outcome** (`format_fail` / `exec_fail` / `pass`) so extraction failures are never misread as reasoning failures (this is the central confound for the concavity result).

**γ placement:** at baseline (γ=1.0, pre-SFT) the prompt has no compression marker; at TokenSkip inference a **γ-control marker is structural scaffolding** placed before the reasoning region — never itself compressed. **Frozen format (Task 0.3, grounded in TokenSkip):** `<|eot_id|>{γ}<|eot_id|>` (for Qwen this is a *literal* text delimiter, not a special token), appended to the user content as the **last** region before the assistant turn, omitted entirely at γ=1.0. Single source: `tsmc.contract.gamma_marker`. Exact assembly order is frozen jointly with Phase-4 inference; place the marker last (TokenSkip's "ratio-last" convention) so the SFT'd model reads it immediately before generating.

> **Contract & schema are FROZEN as code (Task 0.3):** §4 → `tsmc.contract` (sentinel parsing, three-way outcome, prompt scaffolding); §7 → `tsmc.schema` (record + `validate_record`). Changing either is a contract change → bump the run-metadata `prompt_template_hash` and re-freeze.

### 4.1 Generation
- **Prompt regions (in order):** McEval `instruction` (structural) → γ-control marker (TokenSkip inference only) → output-format directive → model emits: **CoT reasoning → sentinel → fenced code block**.
- **Compressible:** the CoT reasoning region only. **Structural (never compressed):** γ marker, sentinel, fenced code + `entry_point`.
- **Parsing:** locate the **LAST** sentinel occurrence (defuses scratch code in CoT); everything after = code region, before = CoT. Take the **first** fenced block in the code region (multiple → `parser_branch=multi_fence`). Normalize to one canonical fenced block with `entry_point` into `raw_generation[0]`. No sentinel → `format_fail` (instrumented salvage = last fence in full output, `parser_branch=fallback`, never counts as clean pass). Truncation (`finish_reason=length`) → `truncated=true` → `format_fail`.

### 4.2 Explanation (two-pass, post-hoc compression)
- **Stage 1:** model is given canonical code (via `instruction`) → produces a natural-language **description**. **This description is the compressible CoT.**
- **Compression:** **post-hoc LLMLingua-2** pruning of the stage-1 description to target γ (PRIMARY path; model-side compression deferred as a future ablation). The compressed description becomes the stage-2 input.
- **Stage 2:** McEval template `"Write a {lang} function {signature} to solve the following problem:\n{compressed_description}"`. **Stage 2 MUST be CoT-free** — direct code, no reasoning, no sentinel. (Compressing stage-2 reasoning would create a second uncontrolled compression locus.)
- **Parsing:** stage-1 description = free text → `cot_text`; compress → tag `cot_origin`; feed to stage 2; extract stage-2 code with the same fence-first normalization (no CoT/sentinel to split). `raw_generation[0]` = stage-2 code; `raw_full_output` retains both passes.

### 4.3 Completion (FIM, gated negative control)
- **Subtypes:** `single` (1 masked line; CoT ≈ 0), `multi` (~3 lines; marginal), `span` (contiguous block; marginal).
- **Induced-CoT contract (when attempted):** mirror Generation — reason briefly, then sentinel, then one fenced block giving the complete code with the mask filled.
- **Empirical gate (tentative, validate in Phase 1):** for a (subtype × model), **skip TokenSkip** if median induced `cot_token_count < 30` **OR** `cot_token_count / code_snippet_tokens < 1.0`. Record `gate_decision`. Expect `single` → skip ("no lever").
- **Parsing:** with sentinel → parse as Generation. **No sentinel is EXPECTED for completion (not a failure):** `parser_branch=direct_fill`, `cot_text=""`, `cot_token_count=0`. `raw_generation[0]` = completed code, normalized to one fenced block.

### 4.4 Validation checks before Docker eval (populate `extraction_status`)
- **Generation:** sentinel found? fence found? `entry_point` in code? truncated?
- **Explanation:** stage-1 extracted? stage-1 compressed OK? stage-2 fence found? `entry_point` present? truncated? (record which stage failed)
- **Completion:** induced CoT present (measure length)? else `direct_fill`; fence found? `entry_point` present (where applicable)? truncated?
- **`extraction_status` fields:** `fence_found` (bool), `entry_point_found` (bool), `truncated` (bool), `parser_branch` ∈ {`sentinel`, `fence`, `direct_fill`, `multi_fence`, `fallback`, `none`}.

---

## 5. Phase-0 decision sheet

| # | Decision | Final choice | Notes |
|---|---|---|---|
| 1 | **Sentinel string** | `@@@FINAL_CODE_7F3A9@@@` (own line; nonce pinned at init) | Rejected `<<<FINAL_CODE>>>`: `>>>` appears in every Python docstring (REPL), `<<`/`>>` are shift/heredoc operators. `@@@` + uppercase + nonce ⇒ negligible collision; all-ASCII ⇒ tokenizer-stable; matched as decoded-text string. Structural, never compressed. |
| 2 | **γ grid (12 values)** | `{1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}` | Dense near 1.0 for Goal 3; tail for collapse. Plot vs **measured** `cot_token_count`, not γ. Knee unknown → add adaptive points after first sweep. |
| 3 | **Explanation compression path** | **Post-hoc LLMLingua-2 only** (model-side = future ablation) | Deterministic information-bottleneck; do not double explanation cost now. Tag arm with `compression_method` (generation test-time = `model_side`; explanation = `post_hoc`). Never merge curves across methods. |
| 4 | **Difficulty proxy** | **Propagate generation `level` by base `problem_id`**; fallback = tertiles of (solution lines, #test cases) calibrated to `level` for the 59 unlabeled | Difficulty is a base-problem property shared across tasks. Assign once per base problem, inherit to all task rows. Flag `difficulty_source`. |
| 5 | **Completion gate** | skip if median `cot_token_count < 30` **OR** `cot/code ratio < 1.0` (tentative) | Per (subtype × model). Validate against measured distribution in Phase 1. |
| 6 | **Model matrix** | Qwen2.5-3B-Instruct, Qwen2.5-Coder-3B/7B/14B-Instruct | Shared tokenizer → comparable x-axis. 70B excluded. γ grid applies to all; gate per model. |
| 7 | **Reproducibility** | **1 run per (model × γ × task)** | No variance bands initially; second run_id later as ablation. Greedy temp=0; note vLLM greedy is not bitwise-deterministic. |
| 8 | **Run metadata to pin** | HF model commit hash, vLLM version, TokenSkip commit, LLMLingua-2 checkpoint hash, McEval Docker **sha256 digest**, prompt-template hash (incl. sentinel), seed=42 | Hashing the template detects contract drift in the data, not just in git. |

**Grounding statistics (verified):** generation `level` = easy 1,221 / middle 401 / hard 385. Solution lines p25/50/75 = 4/10/17. ~asserts p50/75 = 3/6. Completion masked lines: `single` 1, `multi` 3, `span` 1-marker contiguous block.

---

## 6. Stratification plan and manifest design

**Atomic unit:** base problem `Lang/N` (2,066). **Stratum key:** `language × difficulty` (`task_type` is a **verification** dimension, not a split dimension, because base problems are shared across tasks).

**Verified split-relevant data**
- Distinct base problems: **2,066** = 2,007 in all three tasks + 59 in explanation+completion only (0 generation-only; explanation and completion share the same 2,066). **Verified Task 0.2** — but only after case-normalizing the language (`sql`↔`SQL`); raw count is 2,125. The 59 expl/compl-only are all SQL (no `level`).
- Languages uniform: ~50 each (min 50, max 53).
- **No thin cells:** 120 `language × difficulty` cells range **8 → 33** (median 10); none < 5.

**Split ratio: 80/20** → **~1,653 train / ~413 test** base problems.
- Rationale: training-data sufficiency after correct-trajectory filtering is the binding constraint; per-language test counts (~10) are too small for per-language behavioral conclusions either way.

**Base-problem integrity:** the manifest is keyed on the **case-normalized** base `problem_id` (`tsmc.mceval_data.canonical_base_id`, reconciling `sql/`↔`SQL/`); every row sharing the normalized `Lang/N` prefix (generation, explanation, completion `Lang/N-k-sub`) inherits the same split label. No task-variant can straddle the split → blocks cross-task leakage. **Keying on the raw `task_id` would leak the 59 SQL problems** (Task 0.2 finding).

**Allocation:** proportional within each `language × difficulty` cell, **largest-remainder rounding** to hit global 80/20. **Built in Task 0.4** (`tsmc.manifest`, `scripts/build_manifest.py`): 1,653 train / 413 test; membership 2,007 `gen+expl+compl` + 59 `expl+compl`; row-level gen 1606/401, expl 1653/413, compl 8098/2030; distributional gate passes. The 59 derived-difficulty problems are **all SQL** (one language, not ~1.5/language as earlier assumed) → they form the SQL language stratum, difficulty assigned by rank tertiles of docstring length (≈19/20/20), `difficulty_source=derived_proxy`. Manifest keys are **canonical (lower-cased)** ids (`sql/…`), so `problem_id`/`language` are lower-case.

**Balance gates**
- **Distributional (per cell):** train/test proportions within ±1 problem (near-exact by construction); plus a **row-level** check per task (completion variants multiply rows).
- **Behavioral (per task, AGGREGATE only):** baseline accuracy (γ=1.0, no SFT) within **±3%** train vs test, pooled across languages. **Not per-language** (test n≈10 → SE ≈ 15%). Even pooled, test n≈413 → difference SE ≈ 2.7%, so ±3% ≈ 1 SE — read a marginal failure as sampling noise.
- **One shared split serves all three tasks**, so a per-task failure cannot be fixed in isolation. Remedy order: (1) within sampling error → accept with note; (2) re-draw with new seed, re-check all per-task gates jointly; (3) coarsen difficulty strata (e.g., merge middle+hard) if repeated seeds fail.

**Manifest design (committed to git — the git exception)**
- Format: one line per base problem, deterministically sorted, columns:
  `problem_id, split, language, difficulty, difficulty_source, membership`
  (e.g., `AWK/1, train, AWK, easy, level_propagated, gen+expl+compl`). 2,066 lines. Use `.csv`/`.tsv` (not `.jsonl`, so it is not caught by the bulk-data ignore rule).
- **Seed: 42** (recorded for provenance). The **frozen manifest is authoritative**, not the seed.
- The manifest is *provisionally* frozen in Phase 0 (distributional gate) and *confirmed-frozen* after the Phase-1 behavioral gate.

**Effective split sizes (base problems split; rows inherit)**

| Level | Total | Train (~80%) | Test (~20%) |
|---|---|---|---|
| Base problems | 2,066 | ~1,653 | ~413 |
| — all-3-task core | 2,007 | ~1,606 | ~401 |
| — expl+compl-only | 59 | ~47 | ~12 |
| Generation rows | 2,007 | ~1,606 | ~401 |
| Explanation rows | 2,066 | ~1,653 | ~413 |
| Completion rows | 10,128 | ~8,100 | ~2,028 |

---

## 7. Long-format schema

One row per **(problem_id × task_type × completion_subtype × model_id × gamma × run_id)**.

| Field | Type | Semantics / per-task meaning | Nullable |
|---|---|---|---|
| `problem_id` | string | base problem `Lang/N` (split key; variants share it) | no |
| `task_type` | enum | `generation` / `explanation` / `completion` | no |
| `completion_subtype` | enum/null | `single` / `multi` / `span`; null for generation & explanation | yes |
| `model_id` | enum | one of the four Qwen models | no |
| `gamma` | float | target compression ratio (1.0 = baseline) | no |
| `run_id` | string | run/seed identifier (1 run for now) | no |
| `raw_full_output` | text | verbatim model output (both passes for explanation) | no |
| `cot_text` | text | generation: reasoning · explanation: stage-1 description · completion: induced reasoning (may be "") | yes |
| `code_snippet` | text | normalized solution written to `raw_generation[0]` | no |
| `cot_token_count` | int | measured tokens of `cot_text` (x-axis; 0 when no CoT) | no |
| `compression_ratio` | float | mirrors `gamma` (target) — kept explicit, distinct from measured count | no |
| `pass` | bool | from McEval execution | no |
| `extraction_status` | struct | `{fence_found, entry_point_found, truncated, parser_branch}` | no |
| `cot_origin` | enum | `original` (γ=1.0) / `compressed` | no |
| `compression_method` | enum | `model_side` / `post_hoc` (explanation = post_hoc) | no |
| `gate_decision` | enum/null | `applied` / `skipped_no_lever` (completion only) + measured median used | yes |
| `split` | enum | `train_problems` / `test_problems` | no |
| `lang` | string | language (stratification) | no |
| `difficulty` | enum | easy / middle / hard | no |
| `difficulty_source` | enum | `level_propagated` / `derived_proxy` | no |
| `energy_*` | reserved | populated later via PDU/NVIDIA-toolkit join on this key; inference records per-problem **timestamps** from Phase 1 onward | yes |

**Distinguishing task_type:** `task_type` is authoritative; `completion_subtype` non-null iff completion; `cot_token_count = 0` is meaningful (no/negligible CoT), not missing.

---

## 8. Phase 0 → Phase 4 roadmap

Cross-cutting principles: **per-model repetition** (TokenSkip self-distillation → Phases 1–4 run ×4 models, shared harness); **validation trio first** (Python + Rust [Family A] + C [Family B] before 40 languages); **manifest freeze loop** (distributional in Phase 0, behavioral-confirmed in Phase 1); **timestamps recorded from Phase 1** for later energy join.

### Phase 0 — Foundations
- **Goal:** repo + envs + McEval Docker executor working; contracts/schema frozen; empirical checks done; distributionally-validated manifest produced.
- **Deliverables:** repo skeleton (config-driven, `.gitignore`, env docs); pinned-version metadata template; **McEval Docker verified by digest** (stock eval on canonical solutions ≈100% on trio); frozen contract spec (sentinel, parsing, three-way outcome); frozen long-format schema; empirical checks (γ convention; base-problem overlap; difficulty-proxy calibration; sentinel-collision scan); **split manifest** (2,066 lines, seed 42, distributional gate passed, committed); contract↔extractor smoke test on Python/Rust/C.
- **Dependencies:** none (server access, conda envs, Docker pull rights).
- **Risks:** Docker hardcoded-path/volume issues; sentinel collision; γ convention misread; difficulty miscalibration; parser↔extractor mismatch (Family B).
- **Completion criteria:** Docker reproduces expected canonical pass rate (trio); manifest committed + distributional gate passed; schema/contract frozen; smoke test passes for all trio languages; pinned metadata recorded.
- **Effort:** Medium (~5–7 small scripts).

### Phase 1 — Train data generation
- **Goal:** baseline (γ=1.0, no SFT) CoT+code on **train** per model; eval; filter to **correct** trajectories → SFT raw material. Run γ=1.0 **test** pass for the behavioral gate.
- **Deliverables:** inference harness (vLLM, contract prompts, timestamps); two-pass explanation orchestration; completion induced-CoT mode; parsed long-format records (train @ γ=1.0, 4 models × 3 tasks) joined with McEval → three-way outcome; **behavioral ±3% gate** result; **completion induced-CoT length distribution** → `gate_decision` per (model × subtype); per-model correct-CoT corpus with per-cell counts.
- **Dependencies:** Phase 0.
- **Risks:** behavioral gate failure → re-draw manifest; high `format_fail` → contract tuning; 3B weak/short CoT → thin corpus; explanation plumbing; token budget for γ=1.0 worst case.
- **Completion criteria:** `format_fail` ≈ 0 on trio; behavioral ±3% gate passes per task (or manifest re-drawn) → **manifest confirmed-frozen**; per-model corpus produced; completion gate decisions recorded.
- **Effort:** Large (~6–8 scripts × 4 models; heavy compute).

### Phase 2 — Compression pipeline
- **Goal:** LLMLingua-2 at the 12 γ on correct CoTs → multi-γ compressed corpus (generation CoT; explanation stage-1 descriptions post-hoc; completion only where gated in).
- **Deliverables:** compression module (compress only the compressible region; scaffolding held out); 12 compressed variants per trajectory with **measured** `cot_token_count` + `cot_origin`; folder structure `model/task/γ/language`; completion `no-lever` marking; validation (monotonic γ→tokens; scaffolding intact; explanation critical-token spot-check).
- **Dependencies:** Phase 1; Phase 0 (γ grid, convention).
- **Risks:** LLMLingua-2 prunes reconstruction-critical tokens (esp. explanation); achieved ≠ target γ; scaffolding pruning; unstable on very short CoTs.
- **Completion criteria:** 12-γ corpus produced (generation all models, explanation descriptions, gated completion); monotonicity + scaffolding-intact verified; folder structure documented and populated.
- **Effort:** Medium (~3–5 scripts).

### Phase 3 — LLaMA-Factory integration
- **Goal:** convert multi-γ corpus to LLaMA-Factory format with the γ-control prompt format; register via `dataset_info.json`; reach SFT-readiness with decontamination.
- **Deliverables:** format-conversion module (`[instruction] + [γ marker] + [compressed CoT@γ] + [sentinel] + [code]`, Qwen chat template; γ-marker placement **frozen jointly with Phase-4 inference**); `dataset_info.json` per model; LoRA config templates (3B/7B/14B); **decontamination check** (zero test base problems in SFT data, cross-checked vs manifest); LLaMA-Factory dry-run load passes.
- **Dependencies:** Phase 2; Phase 0 (schema, manifest); llamafactory_env.
- **Risks:** format/chat-template mismatch; **γ-marker inconsistency** train vs inference → model won't honor γ; decontamination miss; max-length truncating γ=1.0 samples.
- **Completion criteria:** `dataset_info.json` validates + LLaMA-Factory loads (dry run); decontamination confirmed; configs ready for all 4 models; γ-marker byte-identical to inference contract.
- **Effort:** Medium (~3–4 scripts/config sets).

### Phase 4 — SFT + test inference + curves
- **Goal:** LoRA-SFT each model; run test-time inference at the 12 γ on **test**; eval; build accuracy-vs-`cot_token_count` curves per task and model (energy joins later).
- **Deliverables:** 4 LoRA-SFT'd models; **knob-validation** (γ modulates `cot_token_count` per model) before trusting curves; test inference (1 run per model × γ × task, parsed, timestamps); McEval eval → long-format records; curves per (task × model); **code-vs-non-code** contrast (3B pair); **size-axis** contrast (Coder 3B/7B/14B); **`format_fail`-vs-γ** confound diagnostic per task; completion `no-lever` reporting; schema energy-joinable.
- **Dependencies:** Phase 3; Phase 1 (confirmed-frozen manifest); Phase 0 (γ grid, contract).
- **Risks:** 3B doesn't honor γ post-SFT (knob fails) → curves uninterpretable; concavity cliff-like at 3B; explanation early-collapse; `format_fail` rising with γ (artifact); 14B SFT/inference cost; per-language curves noisy (report aggregate).
- **Completion criteria:** knob validated per model; full test sweep complete (1 × 12 γ × 3 tasks × 4 models) parsed + scored; per-(task × model) curves reproducible from pinned config + frozen manifest; confound diagnostic attached; concavity assessed (generation), bottleneck characterized (explanation), no-lever documented (completion); energy columns reserved and joinable.
- **Effort:** Large (SFT ×4 + knob validation + 12×3×4 test sweep + eval + aggregation/plotting; heavy compute).

**Critical-path gates (each can loop you backward):** Phase 0 Docker verification → Phase 1 behavioral ±3% (re-draw manifest) → Phase 3 γ-marker consistency → Phase 4 knob validation.

| Phase | One-line goal | Key gate | Effort |
|---|---|---|---|
| 0 Foundations | repo, Docker, contracts, manifest | Docker canonical pass; smoke test; distributional gate | Medium |
| 1 Train-data gen | baseline CoT+code on train, filter correct | behavioral ±3% → manifest frozen | Large |
| 2 Compression | LLMLingua-2 at 12 γ | monotonic γ→tokens; scaffolding intact | Medium |
| 3 LLaMA-Factory | format + register + decontam | zero test leakage; γ-marker matches inference | Medium |
| 4 SFT + test sweep | train, sweep γ, curves | knob validated; curves reproducible + diagnostic | Large |

---

## Open Phase-0 empirical checks (resolve before freezing)
Data-side checks resolved in **Task 0.2** (`scripts/phase0_empirical_checks.py`, guarded by `tests/`, full results in `docs/phase0_findings.md`):
1. **γ convention** — ✅ **RESOLVED from TokenSkip source (Task 0.3)**: `TokenSkip/LLMLingua.py` calls `compress_prompt(..., rate=compression_ratio)` and averages `compressed/original` tokens → the ratio is LLMLingua-2's **fraction of tokens RETAINED**, matching our γ (1.0 = full CoT). Marker format also grounded: `<|eot_id|>{γ}<|eot_id|>` appended to user content, **omitted at γ=1.0** (TokenSkip/get_llamafactory_input.py train, evaluation.py inference). Runtime spot-check still advisable in Phase 4 knob validation.
2. **Base-problem overlap** — ✅ **RESOLVED**: 2,066 normalized / 2,125 raw; 2,007 core + 59 SQL (expl/compl-only). Must key on the case-normalized id (`sql`↔`SQL`).
3. **Completion gate calibration** — ⏳ **OPEN**: measure induced-CoT distribution per subtype before locking X=30 / Y=1.0 (needs Phase-1 generations).
4. **Sentinel collision scan** — ✅ **RESOLVED**: `@@@FINAL_CODE_7F3A9@@@` absent from all 126,195 McEval string fields.
5. **Difficulty-proxy calibration** — ✅ **RESOLVED (data-side)**: LOC tertiles 4/9/15 separate levels cleanly, but SQL solutions are ~1 line (LOC p25/50/75 = 1/1/1) → the SQL proxy needs a non-LOC signal or a fixed assignment (decide in Task 0.4).

## Vendored components
- `McEval/` — benchmark + execution harness (paper: `McEval/2406.07436v1.pdf`).
- `TokenSkip/` — controllable CoT compression method.
- `LlamaFactory/` — SFT framework.
These are vendored as plain files (nested `.git` removed).
