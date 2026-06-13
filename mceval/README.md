# tokenskip_mceval3

Research project applying **TokenSkip** (controllable Chain-of-Thought compression) to **multilingual code tasks**, evaluated with the **McEval** execution-based harness, to characterize the trade-off between reasoning length, accuracy, and inference energy.

> **Status: Phase 1 (train-data generation) complete for Qwen2.5-Coder-3B; Phase 2 (compression) is next.** The full inference→score→gate→corpus pipeline runs end-to-end; the manifest is confirm-frozen; the per-model correct-CoT corpus is built. The other three models still need their Phase-1 pass. See [`docs/PHASE1_COMPLETE.md`](docs/PHASE1_COMPLETE.md) for results, [`docs/phase1_findings.md`](docs/phase1_findings.md) for the 40-language scoring-health map, [`docs/PHASE0_COMPLETE.md`](docs/PHASE0_COMPLETE.md) for foundations, and [`docs/PROJECT_ROADMAP.md`](docs/PROJECT_ROADMAP.md) for the authoritative design.

## Research goals
1. **Concavity (primary):** accuracy vs. measured CoT-token-count is concave across compression ratios.
2. **Energy:** inference energy decreases as CoT shortens.
3. **Sweet spot:** slight compression preserves accuracy while saving energy.

Three McEval tasks, all execution-based pass@1: **generation** (primary), **explanation** (post-hoc description compression, two-pass), and **completion** (gated negative control).

## Repository layout
```
.
├── README.md                  # this file
├── docs/
│   ├── PROJECT_ROADMAP.md     # single source of truth (design, decisions, schema, roadmap)
│   └── WORKFLOW.md            # local → push → pull → exec workflow; which scripts run where
├── McEval/                    # vendored: benchmark + execution harness (paper PDF inside)
├── TokenSkip/                 # vendored: controllable CoT compression
└── LlamaFactory/              # vendored: SFT framework
```
The three components are vendored as plain files (nested `.git` removed).

## Workflow constraint (read before running anything)
**All execution happens on a remote server, never locally.**

```
local dev (write/edit) → git add/commit → git push origin main
                                            → server: git pull → execute
```
- Never run inference, McEval Docker, or LLaMA-Factory locally.
- GPU scripts (inference, SFT) are written locally, executed on the server.
- CPU-only lightweight scripts (manifest generator, schema validators, parsers) may be tested locally if data is available; the canonical run is always on the server.
- No hardcoded local paths — use config files or environment variables.

Full detail: [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Environments

| Environment | Purpose |
|---|---|
| `tokenskip_env` (conda) | vLLM inference, TokenSkip, LLMLingua-2 compression, parsing |
| `llamafactory_env` (conda) | LoRA SFT of the Qwen models |
| McEval Docker `multilingualnlp/mceval` | execution-based evaluation (40 language runtimes) — **pull by sha256 digest, never a floating tag** |

## Server hardware
- CPU: Intel Xeon Gold 6326
- RAM: 256 GB
- GPU: 2× NVIDIA RTX A6000 (49 GB VRAM each)

## Model matrix (Qwen-only)
- Controlled pair: **Qwen2.5-3B-Instruct** ↔ **Qwen2.5-Coder-3B-Instruct** (code vs non-code, same size/tokenizer).
- Size axis: **Qwen2.5-Coder-3B → 7B → 14B-Instruct**.
- 70B excluded (won't fit fp16; quantization would confound results).

## Setup (server)
1. Create the two conda environments (`tokenskip_env`, `llamafactory_env`) — dependency specs to be added with the first implementation phase.
2. Pull the McEval Docker image **by sha256 digest** (digest recorded in run metadata, not a floating tag).
3. Configure paths via config/env vars (no hardcoded local paths).
4. `git pull` on the server before every execution.

## Git artifact policy
- **In git:** code, configs, and the split manifest (tiny text, defines the experiment).
- **Not in git:** generations, compressed corpora, model/adapter weights, eval dumps, bulk `.jsonl` (see [`.gitignore`](.gitignore)).

## Documentation
- [`docs/PHASE1_COMPLETE.md`](docs/PHASE1_COMPLETE.md) — **Phase 1 completion report** (Coder-3B): pipeline, results, behavioral gate + manifest confirm-freeze, completion gate, corpus, what's next.
- [`docs/phase1_findings.md`](docs/phase1_findings.md) — Phase-1 empirical findings: 40-language scoring-health map, the `bash -ic` toolchain fix, realized γ=1.0 baseline.
- [`docs/PHASE0_COMPLETE.md`](docs/PHASE0_COMPLETE.md) — **Phase 0 completion report**: everything built, frozen decisions, validation evidence, server setup, what's next.
- [`docs/phase0_findings.md`](docs/phase0_findings.md) — empirical findings (SQL casing, difficulty proxy, γ-convention, McEval execution quirks).
- [`docs/PROJECT_ROADMAP.md`](docs/PROJECT_ROADMAP.md) — overview, environment, contracts, Phase-0 decisions, stratification/manifest, schema, Phase 0→4 roadmap.
- [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — the execution workflow and per-script run location.
