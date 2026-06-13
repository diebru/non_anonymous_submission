# PIPELINE RUNBOOK — run the full TokenSkip × McEval experiment on one model

The end-to-end recipe to take **one model** from base weights → LoRA-SFT'd
γ-controllable model → **energy-instrumented test sweep** → analysis. **Phases 1–4
repeat per model**; Phase 0 (foundations) is done once. Design rationale:
[`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md); reference result (Coder-3B):
[`PHASE4_RESULTS.md`](PHASE4_RESULTS.md); workflow: [`WORKFLOW.md`](WORKFLOW.md).

> **This round's goal (2026-06-02):** the Coder-3B result was a *negative* for energy
> (code dominates the output, so CoT compression had little leverage, and aggressive γ
> triggered repetition-loop degeneration that *raised* energy). Hypothesis: a **non-code
> `Qwen2.5-Instruct`** reasons more (CoT-dominated output), so CoT compression should
> have more leverage and the aggregate energy curve may actually **descend**. We repeat
> the pipeline on **Qwen2.5-3B / 7B / 14B-Instruct** and compare to the Coder-3B baseline.

---

## 1. Environments (all execution on the server)

| env | runs | notes |
|---|---|---|
| `tokenskip_env` (conda) | vLLM inference, LLMLingua-2 compression, **peft merge**, parsing, all analysis | transformers ≈ 4.46 (vLLM 0.6.4) |
| `llamafactory_env` (conda) | **LoRA SFT only** (`llamafactory-cli train`) | transformers 5.2.0 (bleeding edge) |
| McEval Docker | execution-based scoring | pin digest `sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5`; run via `bash -ic` |

Loop: **edit locally → `git push` → server `git pull` → run.** Bulk outputs
(`generations/ compressed/ sft/ weights/ eval_dumps/`) are gitignored.

---

## 2. Add a new model (one-time per model)

For this round, `qwen2.5-3b-instruct` is **already** registered; you must **add
`qwen2.5-7b-instruct` and `qwen2.5-14b-instruct`** (the non-code 7B/14B):

1. **`src/tsmc/constants.py`** → add the id(s) to `MODEL_IDS`.
2. **`configs/run_metadata.yaml`** → under `models:` add `hf_repo` (+ pinned `commit`):
   `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`.
3. **`configs/llamafactory/<model>_lora.yaml`** → copy `qwen2.5-coder-3b_lora.yaml`,
   change `model_name_or_path`, `dataset` (`tsmc_<model>_generation`), `dataset_dir`,
   `output_dir`. Keep `template: qwen`, rank 8 / α 16, lr 5e-5, 3 ep. Set `cutoff_len`
   from the Phase-3 `build_summary` (the gate prints the recommendation).
4. **7B / 14B**: add `--tensor-parallel-size 2` to inference / knob / sweep; SFT on
   2 GPUs (DDP) is automatic. 14B fp16 ≈ 28 GB (fits one A6000; TP=2 for headroom).

> All paths/ids are parameters — no code is model-specific. The only per-model files
> are the two YAMLs in step 3–4.

---

## 3. Phases 1–4 (set `M=<model_id>`, `DIGEST=sha256:4735…`)

### Phase 1 — train-data generation (`tokenskip_env` + Docker)
```bash
python3 scripts/run_inference.py     --task generation --split both --model $M   # +--tensor-parallel-size 2 for 7B/14B
python3 scripts/score_generations.py --task generation --split both --model $M --digest $DIGEST
python3 scripts/phase1_gates.py      --model $M        # behavioral ±3% gate (train vs test)
python3 scripts/build_corpus.py      --model $M        # correct-CoT corpus (train)
```
Gate: behavioral ±3% per task → confirm-freeze manifest. `format_fail` ≈ 0 on the trio.

### Phase 2 — compression (`tokenskip_env`, GPU for LLMLingua-2)
```bash
python3 scripts/compress_corpus.py    --model $M --task generation --split train
python3 scripts/validate_compression.py --model $M --split train     # monotonic γ→tokens
```

### Phase 3 — SFT dataset (`tokenskip_env`, then `llamafactory_env` for the gate)
```bash
python3 scripts/build_sft_dataset.py --model $M --count-tokens       # emits sft/$M/{generation_train.jsonl,...}
python3 scripts/check_sft_dataset.py --model $M --cutoff-len 2048    # llamafactory_env; set cutoff_len from p100
```

### Phase 4 — SFT + merge + knob + sweep + curves
```bash
# 4a. LoRA SFT  (llamafactory_env) — edit the 3 machine paths in the yaml first
llamafactory-cli train configs/llamafactory/${M}_lora.yaml

# 4b. MERGE via peft  (tokenskip_env) — NOT llamafactory-cli export (it's broken, see §5)
python3 scripts/merge_lora.py --base <hf_repo> --revision <sha> \
    --adapter "$PWD/weights/$M/lora_sft_run01" --output "$PWD/weights/$M/merged_sft_run01"
#   if vLLM can't load the merged tokenizer, restore the base tokenizer into the merged dir:
#   cp -Lf <hf_snapshot>/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt} weights/$M/merged_sft_run01/
#   rm -f weights/$M/merged_sft_run01/chat_template.jinja

# 4c. knob gate (tokenskip_env) — median CoT must fall monotonically with γ
python3 scripts/validate_knob.py --model $M --model-path "$PWD/weights/$M/merged_sft_run01" --limit 3

# 4d. energy-instrumented 12-γ test sweep (SINGLE dedicated GPU)
nohup python3 scripts/run_energy_sweep.py --model $M --run-id sft01 \
    --model-path "$PWD/weights/$M/merged_sft_run01" --digest $DIGEST > sweep_${M}.log 2>&1 &
python3 scripts/watch_sweep.py --model $M --run-id sft01 --log sweep_${M}.log   # live (2nd terminal)

# 4e. curves + plots
python3 scripts/build_curves.py --model $M --task generation --split test --run-id sft01
python3 scripts/plot_curves.py  --model $M --task generation --split test --run-id sft01
```

---

## 4. Analysis / plotting toolkit (all `tokenskip_env`, CPU)

Read the scored `records/` and produce tables/figures. All take `--model --run-id
--task --split [--gammas …]`.

| script | what |
|---|---|
| `build_curves.py` | accuracy / energy / format_fail + token decomposition; `--x-axis median\|mean\|wf_mean` |
| `plot_curves.py` | accuracy-vs-CoT (with train-avg-CoT star) + accuracy-vs-PDU-energy; `--gammas` |
| `plot_cot_vs_gamma.py` | avg CoT + avg code vs γ (well-formed) |
| `plot_cot_distribution.py` | full CoT-length distribution (median/mean/percentile bands) vs γ |
| `plot_cot_code_by_outcome.py` | 7-series CoT/code length by outcome vs γ |
| `quality_breakdown.py` | per-γ good-CoT / good-code / bad-both / union counts |
| `count_degeneration.py` | per-γ repetition-loop (degeneration) count (truncated + distinct-4gram) |
| `cot_code_split.py` | per-γ %CoT vs %code (well-formed) |
| `length_by_category.py` | avg answer length by outcome (pass/exec_fail/format_fail/all) |
| `inspect_runaways.py` | dump raw degenerate generations (the repetition loops) |
| `watch_sweep.py` | live sweep dashboard (GPU/PDU power + per-γ progress) |

---

## 5. Gotchas (learned the hard way — do not rediscover these)

- **MERGE is broken via LLaMA-Factory.** `llamafactory-cli export` on the server's
  transformers **5.2.0** produces a degenerate merge (SFT'd model collapses to ~0 CoT).
  **Use `scripts/merge_lora.py` (peft `merge_and_unload`, run in `tokenskip_env`).**
- **The two envs have divergent transformers** (llamafactory 5.2.0 vs tokenskip ≈ 4.46);
  a 5.2.0-saved tokenizer crashes vLLM (`'list' object has no attribute 'keys'`). Merge
  in `tokenskip_env`, or restore the base tokenizer into the merged dir (§3, 4b).
- **TokenSkip itself does NOT merge** — `eval.sh` uses base + `LoRARequest` (vLLM). Our
  peft merge was validated to reproduce the base+adapter knob, but if a merge ever looks
  off, compare against **base+adapter via vLLM LoRARequest** (open confound: does the
  merge change the degeneration tail? — untested).
- **Accuracy control stays OUTSIDE energy.** Monitors wrap inference only; the McEval
  Docker scoring runs after with monitors stopped; `join_energy` integrates only the
  `generate_window`. Don't reorder this.
- **Energy sweep = single dedicated GPU** (clean attribution; the GPU poller reads one
  `--gpu-index`). Set `CUDA_VISIBLE_DEVICES` = that index.
- McEval container must run via **`bash -ic`** (toolchains in `~/.bashrc`); run exactly
  **one** `score_generations` per task; pin the digest.
- **Never paste the `…` ellipsis glyph into a path** (it once created a literal `…/` dir).

---

## 6. Reference result to compare against — Coder-3B / generation (run `sft01`)

Full detail in [`PHASE4_RESULTS.md`](PHASE4_RESULTS.md). One-line: median CoT 169→24
(knob works); accuracy preserved ~0.39 to γ≈0.6 then declines; **aggregate energy
MINIMIZED at γ=1.0 and rises +66% by γ=0.1** (Goal 2 refuted) because (a) code is ~half
the output and γ-independent, and (b) a **repetition-loop degeneration tail** grows from
0.5% (γ=1.0) to **31% (γ=0.1)** and *adds* tokens. The well-formed energy `wf_J` does
descend −70%. **For the new (non-code) models, watch:** is the CoT a *larger* share of
the output (more leverage)? does degeneration appear (count_degeneration)? does the
aggregate energy finally descend?

---

## 7. Automation — `scripts/run_pipeline.py` (BUILT)

Master orchestrator: given `--model`, it runs the 15 stages of §3 end-to-end, shelling
out per stage with `conda run --no-capture-output -n <env> python3 -u …` (tokenskip_env
vs llamafactory_env) and Docker inside scoring. Every stage maps 1:1 to a §3 command, so
this doc stays the source of truth.

- **Gated:** STOPs at the two human checkpoints — `p1_gate` (behavioral ±3%) and
  `p4_knob` (median CoT must fall with γ) — printing the numbers, then halting. `--force`
  runs through gates unattended; resume with `--from-stage <key>` (also `--only`,
  `--stop-after`).
- **Per-model LoRA yaml** is auto-generated from the §2 template at `p4_yaml`
  (`configs/llamafactory/<model>_lora.yaml`; rank8/α16/lr5e-5/3ep/bs1×ga8/bf16/cutoff
  2048; paths filled from `run_metadata` + resolved `data_root`). Review it before SFT;
  `--regen-yaml` overwrites.
- **Size-aware:** 7B/14B auto-get `--tensor-parallel-size 2` on inference + knob; the
  energy sweep stays single-GPU by design. Reads `hf_repo`/`commit` (merge `--base`/
  `--revision`) and the McEval digest from `run_metadata`. Merge uses `merge_lora.py`
  (peft), never `llamafactory-cli export` (§5).

```bash
python3 scripts/run_pipeline.py --model qwen2.5-14b-instruct --dry-run   # preview all 15 stages
python3 scripts/run_pipeline.py --model qwen2.5-14b-instruct             # run; stops at p1_gate
python3 scripts/run_pipeline.py --model qwen2.5-14b-instruct --from-stage p1_corpus  # resume
```
Recommended queue order (non-code ladder): **14B → 7B → 3B**.
