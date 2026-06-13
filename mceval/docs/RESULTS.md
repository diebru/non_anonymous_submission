# RESULTS — CoT-compression energy study

Consolidated findings as of 2026-06-08. Read with [`STATUS.md`](STATUS.md) (entry
point), [`PHASE4_RESULTS.md`](PHASE4_RESULTS.md) (the original Coder-3B sweep),
[`EXPERIMENTS.md`](EXPERIMENTS.md) (the decoding-matrix tracker + commands), and
[`PIPELINE_RUNBOOK.md`](PIPELINE_RUNBOOK.md) (how to run a model end-to-end).

The study applies **TokenSkip** controllable CoT compression (γ = fraction of CoT
retained) to **McEval** multilingual code generation, scoring accuracy by execution and
measuring **GPU+PDU energy** around inference only. Three goals: (1) accuracy-vs-CoT is
concave; (2) energy drops as CoT shortens; (3) a sweet spot preserves accuracy while
saving energy.

---

## 1. Headline result

**A standard decoding-time repetition penalty turns CoT compression into a real energy
win.** The obstacle in the original Coder-3B sweep was a low-γ **runaway tail**: under
aggressive compression a fraction of generations degenerate into token repetition and
ramble to `max_tokens=2048` → `format_fail` + huge token/energy cost, which *inverted*
the aggregate energy curve. **No repetition penalty had ever been applied** (plain greedy).

Adding **`frequency_penalty = 0.3`** (count-scaled, code-safe) eliminates the tail. On
**14B at γ=0.1** (the worst point), vs the no-penalty baseline:

| metric | base (no-pen, 2048) | **fp03** (freq_penalty 0.3) | Δ |
|---|---|---|---|
| format_fail | 0.086 | **0.000** | eliminated |
| GPU energy (J) | 60,119 | **24,715** | **−59%** |
| inference time (s) | 204 | **84** | **−59%** |
| healthy accuracy | 0.507 | **0.549** | **+0.04** |

With the tail gone, **aggregate energy now DESCENDS with compression** (γ=1.0 ~47k J →
γ=0.1 ~25k J) at preserved/better accuracy → **Goal 2 achieved**. The `max_tokens` cap
alone (`mt1024`) is a *partial* fix (energy −42%, but format_fail/accuracy unchanged — it
caps the symptom, not the cause); `both` ≈ `fp03` (the cap is redundant once the penalty
removes the runaways).

---

## 2. Models & validity

Validity is **per-model**: the γ-knob must actually compress CoT (median CoT falls with
γ). A model that emits no CoT can't be studied.

| model | base PASS CoT | SFT'd knob (med_cot 1.0→0.1) | verdict |
|---|---|---|---|
| **Qwen2.5-14B-Instruct** | ~194 | 215 → 23 | ✅ valid |
| **Qwen2.5-3B-Instruct** | 240 | 307 → 50 | ✅ valid (reasons *more* than 14B) |
| **Qwen2.5-Coder-3B-Instruct** | ~184 | 169 → 24 | ✅ valid (code control) |
| **Qwen2.5-7B-Instruct** (default prompt) | **0** | **1 → 1** | ❌ no CoT — see §4 |
| **Qwen2.5-7B-Instruct** (reason-first system prompt, `run02`) | ~320 | (run02 knob) | ✅ fixed |

CoT length is **not monotonic in model size** — 3B and 14B reason, the 7B-Instruct (by
default) does not. Coder-3B reasons *more* than the general 7B.

---

## 3. The decoding matrix

Each cell is the same trained+merged model, re-swept with different **decoding** settings
(no re-training). Axes: penalty **{none, frequency_penalty 0.3}** × max_tokens **{2048,
1024, 512}**. Run-id convention (per model; baseline = `sft01`, fixed-7B baseline = `sft02`):

| config | suffix |
|---|---|
| no-pen · 2048 | `sft01` / `sft02` |
| fp0.3 · 2048 | `…_fp03` |
| no-pen · 1024 | `…_mt1024` |
| fp0.3 · 1024 | `…_fp03_mt1024` |
| no-pen · 512 | `…_mt512` |
| fp0.3 · 512 | `…_fp03_mt512` |

14B and 3B have the full 6-cell matrix under `sft01*`; the fixed 7B under `sft02*`; the
old no-CoT 7B is preserved under `sft01*`/`run01`. The energy-sweep `run_energy_sweep.py`
has a **data-safety guard** (refuses to write a populated run-id unless `--skip-existing`/
`--force`), so cells never clobber each other.

---

## 4. The 7B story (why it failed, how it was fixed)

The base **Qwen2.5-7B-Instruct emits the sentinel immediately and skips reasoning** on the
default prompt (raw output starts `@@@FINAL_CODE_7F3A9@@@\n\n```…`). Its passing
trajectories had ~0 CoT; `filter_correct` kept exactly those; the SFT learned "no CoT"
(eval_loss 0.004) → a model with median CoT 1 at every γ. **Not a bug** — genuine model
behavior + a corpus selection effect.

**Fix:** a reason-first **system prompt** restores CoT (probe: 256–616 tok). To keep the
frozen SFT↔inference contract, `--system` is threaded consistently through
`run_pipeline.py` → `build_sft_dataset.py` / `validate_knob.py` / `run_energy_sweep.py`
(`run_inference.py` already had it); `build_example` accepts an optional system message;
`check_sft_dataset` accepts the `[system, user, assistant]` form. The 7B was re-run under
`run02`/`sft02` (preserving the no-CoT result in `run01`/`sft01`), and its corpus now has
real CoT (PASS median 320). The system prompt:
> *You are a careful programmer. Always reason step by step about the approach BEFORE
> writing any code. Do NOT write the final-code marker or the code block until after you
> have written your reasoning.*

**Caveat to disclose:** the 7B uses a different (system) prompt than the other models —
a per-model asymmetry. Report it as "the 7B-Instruct needs an explicit reasoning
instruction; with it, it behaves like the others."

---

## 5. Data layout & analysis toolkit

Per-cell data: `generations/<model>/<run-id>/generation/test/gamma<g>/{records,energy}/`.
Per-cell analysis (CPU, read-only) writes plots into the cell's split dir:
- `build_curves.py` → `curves.{csv,json,png}` (per-γ accuracy/energy/format_fail/token cols).
- `plot_curves.py` → `acc_vs_cot.png`, `acc_vs_pdu_energy.png` (needs `--corpus-run-id`).
- `plot_answer_breakdown.py` → `cot_code_bars_avgtokens.png`, `energy_vs_avgtokens_full_answer.png`
  (GPU+PDU), `cot_length_vs_gamma.png`, `full_answer_length_vs_gamma.png` (well-formed) +
  `full_answer_length_vs_gamma_all.png` (all outcomes), `inference_time_vs_gamma.png`,
  `accuracy_vs_avgtokens_only_cot.png`, `answer_breakdown.csv`, `cot_only.jsonl`,
  `code_only.jsonl`.
- `plot_cot_boxplot.py` → CoT-length box plot per γ, `--source records|compressed`.
- `plot_outcome_breakdown.py` → CoT/code by outcome at one γ (e.g. the raw train output).

Averages are **well-formed (pass+exec_fail)** unless a name says `_all`; the per-outcome
bars are over all records.

---

## 6. Caveats / threats to validity

- **Energy noise at mild γ** is single-GPU **preemption** (KV-pressure scheduling
  variance), a *measurement* artifact, not a γ effect. A clean re-run would use higher
  `gpu_memory_utilization`. The runaway tail at low γ is the *real* effect.
- **`max_tokens=512`** can truncate legitimate answers at high γ (CoT+code ≈ 350–450 tok)
  → expect format_fail/accuracy to dip at the *top* of the γ range; exploratory.
- **Cross-family x-axis** (`cot_token_count`) is tokenizer-specific → not comparable
  across model families; keep per-family or normalize.
- **1 run, greedy** (not bitwise-deterministic): ~5–10% run-to-run energy noise.
- The original `wf_J` (well-formed energy) was a token-weighted *estimate*; the penalty
  result makes the *measured* aggregate descend, which supersedes that estimate.

---

## 7. Next steps

1. Read the full decoding matrix (14B + 3B + fixed-7B × 6 cells): confirm `fp03`
   replicates the 14B energy win on 3B and the fixed 7B; characterize the 512 cap.
2. **Cross-family robustness — Llama-3.1-8B-Instruct** (see §below / the handoff prompt):
   ⚠ the γ-marker `<|eot_id|>{γ}<|eot_id|>` uses a string that is a **real special token
   in Llama-3** — it must be changed to a Llama-safe literal delimiter (a contract change),
   plus a `llama3` LlamaFactory template, and a per-family x-axis. Llama may also need a
   reasoning system prompt like the 7B.
3. Optional: a clean-energy re-run at higher `gpu_memory_utilization`; a 2nd run for
   variance bands; the explanation task.
