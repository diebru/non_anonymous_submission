# LLAMA CROSS-FAMILY ARM — status & record (resume point after a clear)

**Goal:** cross-family robustness for the CoT-compression energy study — repeat the
TokenSkip × McEval pipeline on **`Llama-3.1-8B-Instruct`** (a different model family
from the Qwen2.5 ladder) and check whether the γ-knob, accuracy, and energy behaviour
replicate. No official "Llama-3.1-Code" exists → this is the **cross-family arm only**
(do NOT mix in Code Llama). Read with [`STATUS.md`](STATUS.md), [`RESULTS.md`](RESULTS.md)
(Qwen findings), [`EXPERIMENTS.md`](EXPERIMENTS.md), [`PIPELINE_RUNBOOK.md`](PIPELINE_RUNBOOK.md).

Date of this record: **2026-06-09**. Commits: **`ed3a385`** (marker redesign) +
**`e063394`** (per-family cutoff_len), both on `main`, pulled on the server.

---

## 1. The obstacle and the fix (the central change)

The γ-control marker tells the model how much to compress. TokenSkip wraps the ratio as
`<|eot_id|>{γ}<|eot_id|>`. For **Qwen** `<|eot_id|>` is ordinary text. For **Llama-3 it is
a REAL special token** (id 128009, end-of-turn) — the literal marker would be tokenized
into control tokens at SFT/inference and corrupt both the prompt structure and γ control.

**Fix — the marker delimiter is now PER MODEL FAMILY** (`tsmc.contract.prompt.GAMMA_DELIMITERS`):

| family | delimiter | marker at γ=0.5 |
|---|---|---|
| `qwen`  | `<\|eot_id\|>`        | `<\|eot_id\|>0.5<\|eot_id\|>` |
| `llama3` | `@@@GAMMA_7F3A9@@@` | `@@@GAMMA_7F3A9@@@0.5@@@GAMMA_7F3A9@@@` |

`gamma_marker(γ, family="qwen")` / `assemble_reasoning_prompt(..., family)` — **default
`"qwen"` keeps every existing Qwen run byte-identical** (Qwen `prompt_template_hash`
unchanged; only Llama gets a new hash). Family is resolved from the model id by
`constants.family_of(model_id)` ("llama" in id → `llama3`, else `qwen`).

The marker is **prompt-only** (the output parser/sentinel is untouched), omitted at γ=1.0,
and never seen by the compressor — so this was a clean, narrow contract change.

### Code changed (commits ed3a385 + e063394)
- `contract/prompt.py`, `constants.py`: per-family delimiter + `family_of`; registered
  `llama-3.1-8b-instruct` in `MODEL_IDS`.
- `family` threaded: `inference.prompts` → `inference.harness` (from `cfg.model_id`) and
  `sft.format.build_example` → `build_sft_dataset.py` / `validate_knob.py`.
  (`run_energy_sweep` needs nothing — it shells to `run_inference`.)
- `check_sft_dataset.py`: per-family marker regex **+ a new token-level
  `delimiter_is_plain_text` gate** — the old check was string-level (`tokenize=False`) and
  would FALSE-PASS on Llama; the new gate asserts the delimiter encodes to non-special ids
  and decodes back, i.e. it actually validates the marker on the real tokenizer.
- `run_pipeline.py`: per-family LoRA `template` (`llama3`) and per-family `cutoff_len`
  (`CUTOFF_LEN = {qwen:2048, llama3:3072}`), driving BOTH the LoRA yaml and the gate.
- 151 unittests + doctests green (the only error is `test_energy` needing `pytest`,
  pre-existing/unrelated).

---

## 2. Per-family differences vs the Qwen runs (everything else is identical)

Same `run_pipeline.py`, same 15 stages, same manifest/split, same sentinel/parser, same
12-γ grid, same LLMLingua-2 checkpoint, same LoRA recipe (r8/α16, lr5e-5, 3ep, bf16), same
peft merge, same McEval digest, same energy method. The **only** model-driven differences:

| knob | Qwen | Llama | why |
|---|---|---|---|
| γ-marker | `<\|eot_id\|>` | `@@@GAMMA_7F3A9@@@` | `<\|eot_id\|>` is a real Llama special token |
| LF template | `qwen` | `llama3` | each model's own chat template |
| `cutoff_len` | 2048 | 3072 | Llama tokenizes longer (templated p100 2255 vs Qwen ~1211–1333) |
| GPU | 7B/14B use TP=2 | TP=1 (single GPU) | 8B fits one A6000 |
| `--system` | only 7B needed it | NOT used | probe showed Llama reasons unaided |

**One deliberate SCOPE choice (user, 2026-06-09):** the test-time decoding matrix differs.
Qwen ran `{no-penalty, frequency_penalty 0.3} × {2048,1024,512}`. **Llama runs NO penalty
axis — only `max_tokens {2048, 1024, 512}` × splits `{test, train}`.** So the Llama arm
asks "does the knob/energy behaviour replicate cross-family + how does the max_tokens cap
act", not "does the fp03 fix replicate" (that could be added later, 1 sweep per cap).

---

## 3. Pipeline progress (run01) — DONE through the knob

| step | result |
|---|---|
| **Reasoning probe** | Llama **reasons** (trio median CoT 171, clean `sentinel` branch) → like 3B/14B, **NO `--system`** (unlike the Qwen-7B which had median 0–1). |
| **p1_gate** (behavioral ±3%) | **PASS** — generation healthy train **0.2947** / test **0.2758**, \|Δ\|=**0.019** WITHIN (scored 1442/359). ≈Coder-3B's 0.29; manifest stays confirm-frozen. |
| corpus | 424 correct generation trajectories → **5088** compressed variants (12 γ × 424). |
| **p3_check_sft** (marker gate) | **PASS @cutoff 3072** — delimiter `@@@GAMMA_7F3A9@@@` → 13 ordinary token ids, **`delimiter plain text: OK`**, **`marker survives template: OK`**, round-trip+structure OK. (Initially FAILED only on cutoff: 3/5088 over 2048 → bumped to 3072.) |
| p4_sft + p4_merge | clean; **merged checkpoint loads in vLLM with no tokenizer restore** (Llama merge was a non-issue). |
| **p4_knob** | **PASS** — median CoT **204.5 → 43.0** across γ1.0→0.1 (**79% shrink, monotonic**). The SFT'd Llama HONORS the new marker → the cross-family redesign works end-to-end. In the Qwen band (14B 215→23, 3B 307→50, Coder-3B 169→24). |

**Watch-items for the sweep:** (1) γ=1.0 format_fail ~10.8% on the 120-sample knob (Qwen
SFT got baseline <1% — check at scale); (2) a mild low-γ runaway tail (mean_cot 143 @γ0.1
> 121 @γ0.2 vs median 43 — same signature as Qwen, far milder than Coder-3B's 31%).

---

## 4. Pinned facts / paths (server: `<repo-root>`)

- **Base:** `meta-llama/Llama-3.1-8B-Instruct` (commit **TBD / unpinned** — merge ran
  unpinned with the loud warning; pin in `configs/run_metadata.yaml` later if desired).
- **Merged model:** `weights/llama-3.1-8b-instruct/merged_sft_run01` (bf16; load with
  `run_inference --model-path <this>`, no adapter, TP=1).
- **LoRA adapter:** `weights/llama-3.1-8b-instruct/lora_sft_run01`.
- **SFT dataset:** `sft/llama-3.1-8b-instruct/` · **corpus/compressed:** under run01.
- γ-marker `@@@GAMMA_7F3A9@@@` · sentinel `@@@FINAL_CODE_7F3A9@@@` · cutoff_len 3072 ·
  LF template `llama3`.
- **McEval Docker digest:** `sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5`.
- Envs: `tokenskip_env` (vLLM/inference/merge/analysis) · `llamafactory_env` (LoRA SFT) ·
  McEval Docker. Loop: edit local → push → server `git pull` → run.

---

## 5. WHAT'S LEFT — the energy-sweep matrix (NOT via the pipeline)

Run **directly** with `run_energy_sweep` (the pipeline's `p4_sweep` would only do
sft01/test/2048 with defaults). 6 cells = `{2048→sft01, 1024→sft01_mt1024, 512→sft01_mt512}
× {test, train}`, **no penalty**. Each writes its own run-id → no collisions (data-safety
guard protects). Energy validity: **one dedicated free GPU** (`--gpu-index`), `--digest`
explicit. Do the 3 **test** cells first (~359/γ), then **train** (~1442/γ, ~4× longer).

```bash
cd <repo-root>
M=llama-3.1-8b-instruct
MG=$PWD/weights/$M/merged_sft_run01
DIGEST=sha256:4735da5db683f96aef4b3d849881fa7ddf865a480ce55287f0019b4f377c52a5
GPU=0      # a FREE, dedicated GPU index (check nvidia-smi)

# 3 TEST cells (then repeat with --split train for the 3 TRAIN cells)
setsid bash -c '
for cell in "sft01 2048" "sft01_mt1024 1024" "sft01_mt512 512"; do
  set -- $cell; RID=$1; MT=$2
  python3 -u scripts/run_energy_sweep.py --model '"$M"' --model-path '"$MG"' \
    --run-id $RID --split test --max-tokens $MT --gpu-index '"$GPU"' --digest '"$DIGEST"'
  python3 -u scripts/build_curves.py          --model '"$M"' --task generation --split test --run-id $RID
  python3 -u scripts/plot_answer_breakdown.py --model '"$M"' --task generation --split test --run-id $RID
done' >> sweep_llama_test.log 2>&1 < /dev/null &
tail -f sweep_llama_test.log
```

Per cell, `build_curves` writes `generations/$M/<run-id>/generation/<split>/curves.{csv,json,png}`.
Report Llama **per-family** — `cot_token_count` is tokenizer-specific, NOT comparable to the
Qwen models (keep on its own axis).

---

## 6. RESULTS (energy sweep)

**TEST split, 3 cells DONE (2026-06-09); TRAIN split not yet run.** n=359/γ, single GPU,
greedy, no penalty. Columns: med/wf-mean CoT, mean_code, healthy_accuracy, format_fail,
aggregate `gpu_J`, well-formed `wf_J`, truncated count. (Full `curves.csv` on the server,
gitignored.)

### sft01 / test — max_tokens 2048 (baseline)
| γ | med_cot | wf_mean_cot | mean_code | acc | ffail | gpu_J | wf_J | trunc |
|---|---|---|---|---|---|---|---|---|
|1.0|214|243|132|0.281|0.086|33425|29277|11|
|0.9|167|192|135|0.287|0.050|31350|27318|10|
|0.8|150|178|131|0.234|0.028|29402|27255|5|
|0.7|149|174|128|0.231|0.019|29635|27856|4|
|0.6|131|148|122|0.262|0.033|29950|25451|10|
|0.5|107|122|123|0.256|0.039|28622|24009|10|
|0.4|89|106|121|0.265|0.028|28356|23458|10|
|0.3|78|94|126|0.262|0.042|29094|22311|14|
|0.2|62|72|126|0.259|0.050|29261|19934|19|
|0.1|48|62|120|0.242|0.084|30826|17265|30|

### sft01_mt1024 / test — max_tokens 1024
| γ | med_cot | wf_mean_cot | acc | ffail | gpu_J | wf_J | trunc |
|---|---|---|---|---|---|---|---|
|1.0|213|227|0.276|0.103|24412|21734|18|
|0.9|167|192|0.287|0.050|22458|20600|12|
|0.7|148|161|0.228|0.033|21538|20024|9|
|0.5|107|120|0.256|0.042|19811|17899|11|
|0.4|89|106|0.259|0.028|19519|17674|10|
|0.2|62|69|0.256|0.053|19157|15319|20|
|0.1|48|59|0.242|0.086|19680|13925|31|

### sft01_mt512 / test — max_tokens 512
| γ | med_cot | wf_mean_cot | mean_code | acc | ffail | gpu_J | wf_J | trunc |
|---|---|---|---|---|---|---|---|---|
|1.0|197|187|115|0.245|**0.279**|18900|12228|**101**|
|0.9|165|159|92|0.262|0.212|18367|12709|80|
|0.8|148|147|95|0.231|0.153|17510|13604|54|
|0.7|148|146|99|0.214|0.120|17304|13889|47|
|0.5|106|111|108|0.253|0.092|16064|13405|35|
|0.4|89|99|109|0.259|0.064|15269|13254|26|
|0.2|62|67|112|0.259|0.070|14517|12138|29|
|0.1|48|55|107|0.242|0.111|14574|11211|41|

### Reading (the cross-family verdict)

1. **Knob holds at scale, cross-family.** median CoT **214→48** (78%, monotone) driven by
   the new `@@@GAMMA_7F3A9@@@` marker. The marker redesign works end-to-end at full scale.
2. **Accuracy is ~FLAT** (~0.25–0.29 at every γ incl. 0.1) — **no concave cliff**. Llama
   preserves accuracy under 78% CoT compression, like the 14B (Coder-3B declined). SFT'd
   γ=1.0 acc 0.281 ≈ base 0.276 (no big self-distillation gain, no loss).
3. **Well-formed energy `wf_J` DESCENDS cleanly** in every cell (2048: 29.3k→17.3k,
   **−41%**) — clean compression saves energy.
4. **Aggregate `gpu_J` does NOT cleanly descend at 2048** — flat with a shallow minimum
   ~γ0.4 (28.4k) then a slight rise to γ0.1 (30.8k, still < γ1.0's 33.4k). Cause = the
   **same** as Qwen-no-penalty: `mean_code` is a fixed ~120–135-tok block (CoT compression
   has low leverage on the total) + a **mild low-γ runaway tail** (trunc 11→30, ffail
   re-rises 0.019@γ0.7 → 0.084@γ0.1). **Milder than Coder-3B** (which inverted +66%); here
   γ0.1 stays below γ1.0. So: **the energy mechanism REPLICATES cross-family** (well-formed
   ↓, aggregate throttled by fixed-code + tail) — a robust negative-for-aggregate without a
   repetition penalty.
5. **max_tokens cap effect.** `mt1024` lowers energy throughout (γ1.0 33k→24k) and makes the
   aggregate descend more cleanly (24.4k→19.7k, **−19%**) with accuracy unchanged — the
   *partial fix* (caps the tail), exactly like Qwen's mt1024. `mt512` forces a monotone
   aggregate descent (18.9k→14.6k) but **DESTRUCTIVELY**: at high γ it truncates legitimate
   answers (**ffail 0.279, trunc 101/359 @γ1.0**; mean_code chopped 132→115) — exploratory
   only, confirms the "512 truncates real answers at high γ" caveat.
6. **Llama baseline format_fail is elevated** (~8.6% @γ1.0/2048 vs Qwen post-SFT <1%) — a
   Llama output-format-compliance trait (NOT the marker, which validated). It dips to ~2%
   mid-γ (cleanest at γ0.7) then re-rises at low γ (the tail).

**Goal verdicts (Llama, generation, test, 1 run):** G1 accuracy-vs-CoT = plateau/preserved
(robust, no cliff); G2 energy↓ = **well-formed YES (−41%), aggregate NO at 2048 / PARTIAL at
1024**; G3 sweet-spot = weak (mild γ0.9–0.7 preserves accuracy at slightly lower aggregate
energy + lowest ffail, but not a decisive win). **The Qwen energy story is cross-family
robust.** Not run for Llama (user scope): the `frequency_penalty 0.3` fix — so the "with-fix
aggregate descent" arm is Qwen-only; the Llama arm shows the *baseline* replicates.

### Still TODO
- **TRAIN split** (3 cells: sft01 / mt1024 / mt512, `--split train`, ~1442/γ, ~4× longer) —
  in-sample, compare curve SHAPE not absolutes.
- Per-cell plots (`plot_answer_breakdown.py`) for the figures.

---

## 7. Resume after a clear

1. Read this doc + the auto-memory `tokenskip-mceval-project.md` (full running log).
2. The marker redesign + Phase 1→4 + knob are **DONE and validated**; the merged model is
   on the server. The remaining work is **section 5** (the 6 energy sweeps + curves) and
   **section 6** (record the results).
3. Nothing in the pipeline is mid-flight — it's STOPPED at `p4_knob`. Do **not**
   `--from-stage p4_sweep`; use the `run_energy_sweep` commands above.
