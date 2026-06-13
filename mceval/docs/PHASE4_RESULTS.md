# PHASE 4 — Energy-Instrumented γ-Sweep: Methods, Results, Next Steps

**Model:** Qwen2.5-Coder-3B-Instruct · **Task:** generation · **Split:** test · **Run:** `sft01`
**Date:** 2026-06-01/02 · **Status:** Phase 4 COMPLETE for Coder-3B/generation.

This is the post-reset entry point for Phase 4. Read with [`STATUS.md`](STATUS.md),
[`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md) §7–§8, and the auto-memory
`tokenskip-mceval-project.md`.

---

## 0. TL;DR

We LoRA-SFT'd Coder-3B for γ-controllable CoT compression, **merged** the adapter,
and ran the full **12-γ energy-instrumented test sweep** (GPU + PDU power around
inference only; McEval scoring strictly outside the energy window). Headline:

- **The knob works.** Median CoT 169→24, well-formed-mean CoT 195→22 (monotone).
- **Accuracy is preserved under mild compression** (~0.39–0.41 for γ=1.0→0.6) and
  declines under aggressive compression (0.32 at γ=0.2, 0.29 at γ=0.1). **Concave.**
- **Aggregate energy does NOT fall** — it is *minimized at γ=1.0* and rises ~+66% by
  γ=0.1. **Goal 2 (energy↓ as CoT↓) is refuted for the aggregate on this task.**
- **But the well-formed energy `wf_J` falls 12.7k→3.9k J (−70%)** — i.e. compression
  *does* save energy for the generations that stay coherent, recovering the
  classic TokenSkip result. The aggregate rises only because of a **destabilization
  tail**: under aggressive γ a growing fraction of generations ramble to
  `max_tokens` (`format_fail` 0.8%→29.5%), and those runaways dominate total tokens.
- **Why this task differs from math/QA:** code is a large, ~fixed share of the
  output (`mean_code` ≈ 161, roughly flat), so CoT compression has low leverage on
  total tokens; and a short math answer always terminates whereas a compressed code
  generation can run away. **Code-generation-specific negative result.**
- **Bonus:** SFT improved accuracy — γ=1.0 healthy-acc **0.398** vs the un-SFT'd base
  Phase-1 test baseline **0.262** (+0.14).

The actionable conclusion: **the obstacle to energy savings here is the runaway tail,
which is fixable** (teach the model to stop under heavy compression / add a
repetition penalty / tighter cap). Fix it and the aggregate collapses onto `wf_J`.

---

## 1. The data (full sweep)

Healthy-language scored rows, n=359/γ. `cot`/`code` are token counts; `cot` is the
reasoning region only (before the sentinel — code excluded). Energy is per-run.

| γ | acc | ffail | med_cot | wf_mean_cot | mean_cot | mean_code | gpu_J | **wf_J** | pdu_J | dur_s | trunc | tot_tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1.00|0.398|0.008|169|195.3|196.6|161.4|13031|**12695**|27651|46.1|2|158602|
|0.95|0.387|0.003|170|187.7|187.6|166.6|13100|**12925**|27588|46.0|1|153660|
|0.90|0.390|0.008|169|191.1|201.4|162.9|14174|**13619**|29617|49.7|3|156890|
|0.85|0.387|0.006|165|182.4|182.8|160.0|13683|**13312**|28868|48.2|2|151088|
|0.80|0.387|0.006|164|178.0|183.2|157.8|13719|**13341**|29019|48.4|2|148925|
|0.70|0.409|0.008|145|162.5|173.0|157.4|13507|**12933**|28418|47.6|3|144684|
|0.60|0.384|0.017|126|145.1|166.8|157.3|13842|**12464**|29264|48.4|7|144021|
|0.50|0.368|0.031| 96|111.3|160.1|161.5|13980|**11384**|29079|48.7|13|143395|
|0.40|0.379|0.086| 81| 94.2|246.9|151.2|15625|**9476**|32628|54.3|33|171727|
|0.30|0.365|0.106| 66| 81.9|290.0|141.4|16273|**8110**|33838|56.3|46|187816|
|0.20|0.320|0.198| 40| 43.9|440.3|128.1|19328|**5583**|40179|66.6|86|247666|
|0.10|0.290|0.295| 24| 22.3|619.7|106.5|21607|**3860**|45313|75.1|125|311681|

Full CSV + plots on the server (gitignored):
`generations/qwen2.5-coder-3b-instruct/sft01/generation/test/{curves.csv, curves.json,
curves.png, acc_vs_cot.png, acc_vs_pdu_energy.png}`.

### How to read the three CoT columns
- **`median_cot`** — the *typical* generation's reasoning length. Robust, monotone
  169→24. Use for the concavity x-axis.
- **`wf_mean_cot`** — mean over **well-formed** generations (pass+exec_fail; excludes
  the format-fail runaways). A meaningful *average* reasoning length, also monotone
  195→22. **Recommended "average".**
- **`mean_cot`** — raw average of `cot_token_count`. **Folds back** (196→160→620)
  because a runaway has no sentinel, so the parser counts its *whole* 2048-token
  output as "CoT" (a parse artifact, [parser.py](../src/tsmc/contract/parser.py) `fallback` branch). = TokenSkip's
  `avg_cot_length`, but polluted by the tail on this task.

---

## 2. The result, interpreted

**Three regimes** (γ → down = more compression):

| regime | γ | accuracy | format_fail | aggregate energy |
|---|---|---|---|---|
| **mild** | 1.0–0.7 | preserved ~0.39–0.41 | ~0 (≤0.8%) | flat ~13–14k J |
| **moderate** | 0.6–0.5 | slipping 0.38→0.37 | rising 1.7→3% | ~flat 13.8–14k J |
| **aggressive** | 0.4–0.1 | degrading 0.38→0.29 | **exploding 9→30%** | **rising 15.6→21.6k J** |

**Mechanism (now well-supported by the columns):**

1. **Code dominates the output.** `mean_code` ≈ 161 tokens and is **roughly flat**
   from γ=1.0 to 0.5 (then shrinks). At γ=1.0 the average generation is ~440 output
   tokens of which CoT (~195) is < half. So compressing CoT trims only a minority of
   the output → low leverage on aggregate energy. (Earlier guess that code *grows* to
   compensate is **not** supported — code is just a large fixed block.)
2. **Throughput penalty in the clean region.** `J/tok` rises 0.082→0.097 from γ=1.0→0.5
   — shorter, more-variable sequences batch less efficiently and the fixed prefill is
   amortized over fewer decode tokens. This cancels the ~10% token saving, so
   aggregate energy stays flat instead of descending there.
3. **Destabilization tail.** At γ≤0.4 a growing fraction of generations lose coherence
   and ramble to `max_tokens=2048` (`trunc` 2→125 of 359; `format_fail` 0.8%→29.5%).
   These few runaways carry huge token counts → total tokens **double** (143k→312k) →
   aggregate energy and duration rise (46s→75s; power ~const 283–290 W).

**The reconciliation with prior benchmarks (`wf_J`):** `wf_J` = run energy
token-weighted by the well-formed (non-runaway) share. It **descends 12.7k→3.9k J**
because (a) coherent generations genuinely shorten as CoT compresses, and (b) the
runaway energy is removed. So **TokenSkip's energy saving holds for coherent
generations**; the aggregate only rises because the runaway tail dominates. Your
previous descending curves (boolq/gsm8k/math/piqa) are the same mechanism *without* a
tail — a short final answer always terminates, so there were no runaways to invert it.

> ⚠ `wf_J` is a **token-weighted estimate**, not a direct measurement: run-level power
> can't separate concurrently-decoding requests, so it assumes ~uniform J/token. The
> *measured* quantity is the aggregate `gpu_J`/`pdu_J`. To make `wf_J` a hard number,
> re-run low-γ inference with runaways suppressed (repetition penalty / better SFT)
> and measure that directly.

---

## 3. Goal verdicts (1 model, 1 task, 1 run)

- **Goal 1 — concavity:** **supported.** Accuracy plateaus (~0.39) then declines vs
  CoT length.
- **Goal 2 — energy ↓ as CoT ↓:** **refuted for the aggregate** (energy minimized at
  γ=1.0, rises after); **supported for the well-formed subset** (`wf_J` descends).
- **Goal 3 — sweet spot (preserve accuracy AND save energy):** **marginal.** Mild
  compression (γ≈0.6–0.7) preserves accuracy (~0.38–0.41) at roughly-equal-or-slightly
  lower well-formed energy; the **aggregate** energy never drops, so there is no
  clean win-win until the runaway tail is fixed.

---

## 4. What was built (pipeline)

All committed; bulk outputs gitignored. SERVER = `the server`,
`<repo-root>`.

- **Adapter merge** — [`scripts/merge_lora.py`](../scripts/merge_lora.py) (peft
  `merge_and_unload`, fp32 merge → bf16 save, pins base commit `488639f1…`). Run in
  **`tokenskip_env`**. ⚠ **`llamafactory-cli export` is BROKEN on the server's
  transformers 5.2.0** (degenerate ~0-CoT merge); do NOT use it. The two conda envs
  have divergent transformers (llamafactory_env 5.2.0 vs tokenskip_env vLLM-0.6.4 /
  ~4.46), so a 5.2.0-saved tokenizer crashes vLLM — always merge in `tokenskip_env`.
  Merged checkpoint: `weights/qwen2.5-coder-3b-instruct/merged_sft_run01` (bf16).
- **Knob gate** — [`scripts/validate_knob.py`](../scripts/validate_knob.py)
  (`--model-path <merged>` with no `--adapter` validates the merge): merged-knob
  PASS, median CoT 160.5→22, matches base+adapter.
- **Energy core** — [`src/tsmc/energy/`](../src/tsmc/energy/): `core.py`
  (`integrate_power` trapezoid over the generate window; `summarize_run`),
  `monitors.py` (`EnergyMonitors` ctx). GPU primary / PDU secondary, RUN-level.
- **Pollers** — [`scripts/monitor_gpu.py`](../scripts/monitor_gpu.py) (per-line
  nvidia-smi parse + `--gpu-index`), [`scripts/monitor_pdu.py`](../scripts/monitor_pdu.py)
  (configurable SNMP; defaults 192.0.2.1/public/ePDUPhaseStatusActivePower.1).
- **Join** — [`scripts/join_energy.py`](../scripts/join_energy.py): integrate the
  curves over `run_meta.json`'s `generate_window` → `energy_summary.json` + stamp each
  record's reserved schema `energy` field.
- **Orchestrator** — [`scripts/run_energy_sweep.py`](../scripts/run_energy_sweep.py):
  per-γ → monitors wrap **inference only** → stop → `score_generations` (Docker,
  outside energy) → `join_energy`. Single dedicated GPU, reload-per-γ, fail-fast on
  missing digest.
- **Analysis** — [`scripts/build_curves.py`](../scripts/build_curves.py) (per-γ
  accuracy/energy/format_fail + token decomposition; `--x-axis median|mean|wf_mean`),
  [`scripts/plot_curves.py`](../scripts/plot_curves.py) (acc-vs-CoT with train-avg-CoT
  star + acc-vs-PDU-energy; `--gammas` filter),
  [`scripts/watch_sweep.py`](../scripts/watch_sweep.py) (live dashboard).

**Accuracy control stays OUTSIDE energy** (two guards): monitors are SIGINT'd before
the Docker scoring; and `join_energy` integrates only the decode `generate_window`
(46–75 s — the McEval tests, which take minutes, are never in the window).

---

## 5. How to reproduce (server)

```bash
cd <repo-root> && git pull
M=qwen2.5-coder-3b-instruct
DIGEST=sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5

# (one-time) merge the adapter in tokenskip_env, then validate the knob
python3 scripts/merge_lora.py --base Qwen/Qwen2.5-Coder-3B-Instruct \
  --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 \
  --adapter "$PWD/weights/$M/lora_sft_run01" --output "$PWD/weights/$M/merged_sft_run01"
python3 scripts/validate_knob.py --model $M --model-path "$PWD/weights/$M/merged_sft_run01" --limit 3

# full 12-γ energy sweep (single dedicated GPU), then curves + plots
nohup python3 scripts/run_energy_sweep.py --run-id sft01 \
  --model-path "$PWD/weights/$M/merged_sft_run01" --digest $DIGEST > sweep_sft01.log 2>&1 &
python3 scripts/watch_sweep.py --run-id sft01 --log sweep_sft01.log   # live (2nd terminal)
python3 scripts/build_curves.py --task generation --split test --run-id sft01
python3 scripts/plot_curves.py  --task generation --split test --run-id sft01
```

---

## 6. How to move forward (next steps, priority order)

1. **Fix the runaway tail (highest-value).** It is the sole obstacle to aggregate
   energy savings here. Options, cheapest first: (a) inference-time **repetition
   penalty** / `no_repeat_ngram` on the sweep; (b) **tighter `max_tokens`** (caps the
   damage, converts rambles to early truncation); (c) **more/curriculum SFT at low γ**
   so the model learns to stop under heavy compression. Re-run the low-γ points and
   check whether `gpu_J`/`pdu_J` collapse onto `wf_J` and descend → would turn Goal 2
   positive in the aggregate.
2. **Explanation task** (most likely to show a *real* aggregate energy descent). There
   the compressible CoT is the stage-1 description and stage-2 is code-free, so the
   compressed region is a *larger* share of the work — closer to your math/QA setup.
   Post-hoc LLMLingua-2 compression (Decision #3); generation harness already supports
   two-pass.
3. **The other 3 models** (Phases 1→4 on the shared harness): controlled pair
   **Qwen2.5-3B-Instruct** (non-code — does code-specialization change the CoT/code
   ratio and thus the leverage?), size ladder **Coder-7B / 14B** (prefer
   `--tensor-parallel-size 2`). Reuse `merge_lora.py` + the sweep.
4. **Analysis refinements:** pass-only population as a secondary curve (flag the
   selection-bias caveat — the passing set changes with γ); per-language/difficulty
   breakdowns; promote the `wf_J` vs `gpu_J` decomposition to the headline figure.
5. **Hardening:** the `wf_J` direct-measurement re-run (item 1 doubles as this);
   optional second `run_id` for variance bands (Decision #7 currently 1 run).

---

## 7. Watch-list / threats to validity

- **1 run, greedy (vLLM greedy not bitwise-deterministic).** ~5–10% run-to-run noise
  on energy; the clean-region "flat" is within that. Add a 2nd run for bands.
- **`wf_J` is an apportioned estimate**, not a measurement (see §2 box).
- **PDU is node-level** (single-tenant during the run, so usable as a cross-check, but
  GPU power is the attributable signal).
- **C model output** still depressed (McEval Family-B re-stitch); it's in the healthy
  set, so it drags accuracy a bit. Same caveats as Phases 1–3.
- **`mean_cot` non-monotonicity** is a *feature* (the runaway diagnostic), not a bug —
  use `median`/`wf_mean` for monotone x-axes.
```
