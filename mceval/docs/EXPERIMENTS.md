# EXPERIMENTS — decoding-config matrix (penalty × max_tokens × model)

Tracker for the test-time sweeps that vary **decoding** settings to attack the low-γ
**runaway tail** (token regurgitation → ramble to max_tokens → `format_fail` → inflated
energy + wasted time). These settings — a repetition/frequency **penalty** and
**max_tokens** — change ONLY the sweep (test-time generation); they do **not** touch
Phase 1–4 training. So every cell **reuses the already-trained merged model** and just
re-runs the 12-γ sweep into **its own run-id**. Read with
[`PHASE4_RESULTS.md`](PHASE4_RESULTS.md) (why the tail inflates energy) and
[`PIPELINE_RUNBOOK.md`](PIPELINE_RUNBOOK.md) (how a sweep runs).

---

## ⚠️ DO NOT TOUCH — existing canonical data (must be preserved)

These are finished results; nothing below may overwrite them. New work goes to **new
run-ids only**.

- **14B (non-code), trained + swept:**
  - `weights/qwen2.5-14b-instruct/{lora_sft_run01, merged_sft_run01}` — the trained model (read-only; every 14B variant sweep loads this).
  - `generations/qwen2.5-14b-instruct/run01/**` — Phase-1 baseline (train+test, γ=1.0).
  - `generations/qwen2.5-14b-instruct/sft01/**` — the **done** baseline sweep (no-penalty, max_tokens 2048).
  - `corpus/`, `compressed/`, `sft/` for `qwen2.5-14b-instruct`.
- **Coder-3B (code control), full Phase 0–4:** `generations/qwen2.5-coder-3b-instruct/{run01,sft01}/**`, its `weights/`, `corpus/`, `compressed/`, `sft/`.

**Hard rule:** do **NOT** re-run `run_pipeline.py` on `qwen2.5-14b-instruct` — its
`p4_sweep` defaults to run-id `sft01` and would overwrite the finished baseline. 14B
variants are run with `run_energy_sweep.py` **directly**, each with a new `--run-id`.

---

## Run-id convention (this is what keeps cells from colliding)

A sweep only ever **writes** to `generations/<model>/<run-id>/...`; it **reads** the
merged model read-only. So distinct run-ids = distinct output dirs = existing data safe.

| decoding config | run-id |
|---|---|
| no penalty · max_tokens 2048 (**baseline**) | `sft01` |
| frequency_penalty 0.3 · max_tokens 2048 | `sft01_fp03` |
| no penalty · max_tokens 1024 | `sft01_mt1024` |
| frequency_penalty 0.3 · max_tokens 1024 | `sft01_fp03_mt1024` |

(`fp03` = frequency_penalty 0.3, the chosen "with-penalty" setting — count-scaled, code-safe,
available in vLLM 0.6.4. `mt1024` = max_tokens 1024.)

---

## The matrix (✅ done · ⬜ to run)

| model | needs one-time training? | no-pen/2048 `sft01` | fp03/2048 `sft01_fp03` | no-pen/1024 `sft01_mt1024` | fp03/1024 `sft01_fp03_mt1024` |
|---|---|---|---|---|---|
| **14B** (non-code) | no (done) | ✅ done | ⬜ | ⬜ | ⬜ |
| **7B** (non-code)  | **yes** (Phase 1→4) | ⬜ (from pipeline) | ⬜ | ⬜ | ⬜ |
| **3B** (non-code)  | **yes** (Phase 1→4) | ⬜ (from pipeline) | ⬜ | ⬜ | ⬜ |

- 7B/3B first need their **one-time** `run_pipeline.py` (Phase 1→4 → `run01` corpus/SFT +
  `merged_sft_run01`); that run also produces the **no-pen/2048 baseline** sweep at `sft01`.
- Each model = 1 baseline + 3 variant sweeps. 14B baseline already done ⇒ **11 new sweeps total** + **2 trainings**.
- **Coder-3B (code control):** optional re-sweep under the same 4 configs later, for the
  code-vs-non-code comparison under matched decoding. Not in the default matrix.

---

## Why each axis

- **frequency_penalty 0.3** — attacks the *cause*: subtracts a score penalty scaled by how
  many times a token already appeared, so the repetition loop gets throttled while normal
  code reuse is barely touched.
- **max_tokens 1024** — caps the *symptom*: a surviving runaway is cut at 1024 instead of
  2048, halving its wasted tokens/energy. Legitimate answers are <~800 tokens (box-plot
  maxes), so it should not truncate real answers — verify per model.
- **2×2** isolates each effect and their combination, so we can see whether the clean
  result comes from the penalty, the cap, or both — and whether aggregate energy finally
  **descends** with compression once the tail is gone.

---

## Suggested order (data-safe; time is not the constraint)

1. **14B variants** (model already trained): `sft01_fp03`, `sft01_mt1024`, `sft01_fp03_mt1024`. Answers the core question on the model we have.
2. **7B**: `run_pipeline.py --model qwen2.5-7b-instruct` (→ `sft01` baseline) → 3 variant sweeps.
3. **3B**: same.

Per cell, after the sweep: `build_curves.py` + `plot_answer_breakdown.py` + (optionally)
`plot_cot_boxplot.py`, all on that cell's run-id.

---

## How to run

Knobs live on `run_energy_sweep.py` (and `run_inference.py`): `--frequency-penalty 0.3`,
`--max-tokens 1024`. The digest resolves from `run_metadata.yaml`. The **data-safety
guard** makes a sweep abort if its `--run-id` already has records (unless `--skip-existing`
to resume / `--force` to overwrite) — so the variant run-ids below can never clobber `sft01`.

```bash
cd <repo-root>
M=qwen2.5-14b-instruct                       # already trained
MERGED=$PWD/weights/$M/merged_sft_run01

# --- 3 variant sweeps (each writes ONLY to its new run-id; sft01 untouched) ---
python3 scripts/run_energy_sweep.py --model $M --model-path $MERGED --run-id sft01_fp03         --frequency-penalty 0.3
python3 scripts/run_energy_sweep.py --model $M --model-path $MERGED --run-id sft01_mt1024       --max-tokens 1024
python3 scripts/run_energy_sweep.py --model $M --model-path $MERGED --run-id sft01_fp03_mt1024  --frequency-penalty 0.3 --max-tokens 1024

# --- analysis per cell ---
for RID in sft01_fp03 sft01_mt1024 sft01_fp03_mt1024; do
  python3 scripts/build_curves.py          --model $M --task generation --split test --run-id $RID
  python3 scripts/plot_answer_breakdown.py --model $M --task generation --split test --run-id $RID
done
```

**7B / 3B** — one-time pipeline first (gives the `sft01` baseline + merged model), then the
same 3 variant sweeps:
```bash
M=qwen2.5-7b-instruct        # then qwen2.5-3b-instruct
python3 scripts/run_pipeline.py --model $M               # Phase 1->4 (stops at gates; --force for unattended)
MERGED=$PWD/weights/$M/merged_sft_run01
python3 scripts/run_energy_sweep.py --model $M --model-path $MERGED --run-id sft01_fp03        --frequency-penalty 0.3
python3 scripts/run_energy_sweep.py --model $M --model-path $MERGED --run-id sft01_mt1024      --max-tokens 1024
python3 scripts/run_energy_sweep.py --model $M --model-path $MERGED --run-id sft01_fp03_mt1024 --frequency-penalty 0.3 --max-tokens 1024
```

Run detached for overnight, e.g. `setsid bash -c '…' >> matrix_$M.log 2>&1 < /dev/null &`.
Each sweep is single-GPU; pass `--gpu-index N` to pick a free card, `--no-pdu` if the PDU
is unreachable.

---

## Round 2 — max_tokens=512 axis + the 7B fix

**Result of round 1:** frequency_penalty 0.3 is a decisive win on 14B — at γ=0.1 it
eliminated format_fail (8.6%→0), cut energy/time ~59%, and *raised* accuracy
(0.507→0.549). So aggregate energy now **descends** with compression once the runaway
tail is removed. `mt1024` alone = partial (energy only); `both` ≈ `fp03`.
**7B was invalid** — the base Qwen2.5-7B-Instruct emits the sentinel immediately and
skips reasoning, so its corpus had ~0 CoT (eval_loss 0.004, median CoT 1). A probe
confirmed a reason-first **system prompt** restores CoT (256–616 tok).

**Two additions:**
1. **max_tokens axis extended to {2048, 1024, 512}.** New run-ids: `…_mt512`,
   `…_fp03_mt512`. ⚠ 512 may truncate legit answers at high γ (CoT+code ≈ 350–450 tok),
   so expect format_fail/accuracy to dip at γ≈1.0 — exploratory.
2. **7B fix:** re-run its whole pipeline with a pinned reason-first **system prompt**,
   under a **new run-id `run02`/`sft02`** (preserves `run01`/`sft01` = the no-CoT finding).
   `--system` is now threaded through run_pipeline → build_sft_dataset / validate_knob /
   run_energy_sweep so SFT↔inference stay byte-identical. The 7B then gets the full
   `{none,fp03} × {2048,1024,512}` matrix, all carrying `--system`.

The chosen system prompt (validated by the probe):
> *You are a careful programmer. Always reason step by step about the approach BEFORE
> writing any code. Do NOT write the final-code marker or the code block until after
> you have written your reasoning.*

### Round-2 commands
```bash
cd <repo-root>
SYS="You are a careful programmer. Always reason step by step about the approach BEFORE writing any code. Do NOT write the final-code marker or the code block until after you have written your reasoning."

# --- 14B & 3B: just the 512 cells (already trained; new run-ids) ---
for M in qwen2.5-14b-instruct qwen2.5-3b-instruct; do
  MG=$PWD/weights/$M/merged_sft_run01
  python3 scripts/run_energy_sweep.py --model $M --model-path $MG --run-id sft01_mt512       --max-tokens 512
  python3 scripts/run_energy_sweep.py --model $M --model-path $MG --run-id sft01_fp03_mt512  --frequency-penalty 0.3 --max-tokens 512
done

# --- 7B fix: full pipeline with the system prompt (run02/sft02), then its 5 variant cells ---
M=qwen2.5-7b-instruct
# --regen-yaml is REQUIRED: regenerates the LoRA yaml with output_dir=lora_sft_run02
# (else it keeps run01's yaml and the merge can't find the adapter).
python3 scripts/run_pipeline.py --model $M --run-id run02 --sweep-run-id sft02 --system "$SYS" --regen-yaml --force
MG=$PWD/weights/$M/merged_sft_run02
for cfg in "sft02_fp03 --frequency-penalty 0.3" "sft02_mt1024 --max-tokens 1024" \
           "sft02_fp03_mt1024 --frequency-penalty 0.3 --max-tokens 1024" \
           "sft02_mt512 --max-tokens 512" "sft02_fp03_mt512 --frequency-penalty 0.3 --max-tokens 512"; do
  set -- $cfg; RID=$1; shift
  python3 scripts/run_energy_sweep.py --model $M --model-path $MG --run-id $RID --system "$SYS" "$@"
done
```
(Then `build_curves` + `plot_curves` + `plot_answer_breakdown` per new run-id, as in round 1.)

---

## Status log

- 2026-06-04 — round 1 built + run. Results: **14B + 3B + Coder-3B valid; 7B invalid
  (no CoT).** `fp03` is the decisive energy win on 14B (see Round 2 header).
- 2026-06-04 — round 2 built: `--system` threading (run_pipeline → build_sft_dataset /
  validate_knob / run_energy_sweep; `build_example` gains a system message) + max_tokens
  512 axis. Verified (`--dry-run` threads `--system` to the 4 right stages; 17 SFT + 16
  inference tests green). Next: launch round-2 (14B/3B 512 cells + 7B `run02` fix).
