# WORKFLOW — local development, remote execution

This document defines how work moves through the project. It is binding for all
contributors to this repository.

> **Golden rule: all execution happens on the remote server, never locally.**
> Local machines are for writing/editing code and committing it. The server pulls
> and runs it.

## 1. The loop

```
┌─────────────────────────┐        git push        ┌──────────────────────────┐
│  LOCAL (dev machine)    │ ─────────────────────►  │  remote git host          │
│  write / edit code      │                         │  (origin/main)            │
└─────────────────────────┘                         └────────────┬─────────────┘
            ▲                                                     │ git pull
            │ results summaries, manifest, figures (small)        ▼
            │                                        ┌──────────────────────────┐
            └────────────────────────────────────── │  SERVER (execute)        │
                                                     │  GPU inference / SFT      │
                                                     │  McEval Docker eval       │
                                                     └──────────────────────────┘
```

Every change follows:
1. **Write/edit locally.**
2. **`git add` + `git commit` + `git push`** to `origin main`.
3. **On the server: `git pull`**, then execute in the appropriate environment.

We use a **direct-to-`main`** flow because the server pulls `main`. Do **not** use
feature branches the server pull would not see.

## 2. What runs where

| Component | Environment | Runs locally? | Notes |
|---|---|---|---|
| Inference (vLLM) | `tokenskip_env` (server GPU) | ❌ never | needs A6000 GPUs |
| TokenSkip / LLMLingua-2 compression | `tokenskip_env` (server) | ❌ never (GPU/large model) | compressor checkpoint pinned |
| SFT (LoRA) | `llamafactory_env` (server GPU) | ❌ never | LLaMA-Factory |
| McEval evaluation | Docker `multilingualnlp/mceval` (server) | ❌ never | executes untrusted code across 40 runtimes |
| Manifest generator | CPU-only | ✅ testable locally | canonical run still on server |
| Schema validators | CPU-only | ✅ testable locally | |
| Parsers (CoT/code split) | CPU-only | ✅ testable locally if sample data present | canonical run on server |
| Aggregation / plotting | CPU-only | ✅ testable locally on result summaries | |

**CPU-only lightweight scripts** (manifest generator, schema validators, parsers,
aggregation) **may** be tested locally **if** the required data is available, but the
**canonical run is always on the server**. GPU scripts (inference, SFT) and Docker
(McEval) are **written locally, executed only on the server**.

## 3. Three environments

| Environment | Owns |
|---|---|
| `tokenskip_env` (conda) | vLLM inference; TokenSkip; LLMLingua-2 compression; parsing |
| `llamafactory_env` (conda) | LoRA SFT of the Qwen models |
| McEval Docker (`multilingualnlp/mceval`) | execution-based evaluation; **pull by sha256 digest, never a floating tag** |

Hand-offs between environments are **file-based on server disk** (e.g.,
`tokenskip_env` writes clean-code `.jsonl` → Docker scores it → filtered-correct
trajectories feed back to `llamafactory_env`). No environment imports another's
Python.

### McEval Docker note
McEval's `eval_all.py` hardcodes `os.chdir('/workspace/MMCodeEval/eval/tmp')` and
copies auxiliary language data there. **Run evaluation inside the pinned container**,
mounting our generations and a results directory as volumes. Do **not** fork McEval
to relocate the path; if a path must change, isolate it in a thin, committed shim and
record the image digest.

## 4. No hardcoded local paths

Every script reads paths from a **config file or environment variables** — never a
hardcoded local path. The same code must run unchanged locally (for CPU tests) and on
the server. Pin per-run metadata (model commit hash, vLLM version, TokenSkip commit,
LLMLingua-2 checkpoint hash, McEval Docker sha256 digest, prompt-template hash, seed)
so results are reproducible across the local→push→pull→exec cycle.

## 5. Git artifact policy

**Goes through git (small, defines the experiment):**
- code and configs
- the split **manifest** (2,066-line `.csv`/`.tsv`)
- small result summaries / figures

**Stays on the server (never git — see `.gitignore`):**
- `generations/` — raw model outputs and per-language `.jsonl`
- `compressed/` — LLMLingua-2 multi-γ corpora
- `weights/` — model and LoRA adapter weights
- `eval_dumps/` — McEval scored outputs
- any bulk `.jsonl`

Bulk artifacts are derived data keyed to the pinned config; they belong in a
data-versioning layer (DVC / results branch / server storage), not source control.
Committing them would bloat history, cause merge conflicts on regeneration, and slow
the push/pull loop.

## 6. Typical change, end to end
1. Edit a script locally; if CPU-only and data is available, smoke-test locally.
2. `git add <files>` → `git commit` → `git push origin main`.
3. SSH to the server; `git pull`.
4. `conda activate` the correct environment (or `docker run` the pinned McEval image).
5. Execute; write bulk outputs to the gitignored directories on server disk.
6. Pull small result summaries / figures back; commit those if they belong in git.
