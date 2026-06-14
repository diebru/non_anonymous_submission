# auto/ — one-shot server automation

Wraps the existing repo scripts into a runnable pipeline. **All execution is on the
server.** Honors the energy rules: LoRA SFT + non-measured data-gen/compression use
**2 GPUs**; the energy-measured inference sweep is forced to **1 GPU**
(`CUDA_VISIBLE_DEVICES=0`).

## 1. Configure (the only thing you must fill in)
```bash
cd auto
cp config.env.example config.env
$EDITOR config.env      # paths, HF token, PDU IP, the 4 model commit SHAs
```

## 2. Run
```bash
bash 00_setup.sh        # envs, deps, LLaMA-Factory, base-model downloads (pinned), Docker pull, PDU patch
bash 10_reasoning.sh    # gsm8k/math/boolq/piqa x 4 models, end to end
bash 20_mceval.sh       # multilingual-code arm (stops at human gates unless MCEVAL_FORCE=1)
bash 30_plots.sh        # regenerate figures
# or: bash run_all.sh
```
`export DRY_RUN=1` to preview commands without running the heavy steps.

## What each step maps to
| Step | Repo pieces used | GPUs |
|---|---|---|
| baseline train-gen | `common/evaluation.py --data-type train` | 0,1 |
| compress + build SFT | `common/LLMLingua_iterato.py`, `common/get_llamafactory_input_all_{qwen,llama}.py` (via `_compress_build.py`) | 0,1 |
| register dataset | `LlamaFactory/data/dataset_info.json` (via `_register_dataset.py`) | — |
| LoRA SFT | auto-generated `common/configs/auto_*.yaml` + `llamafactory-cli train` | 0,1 |
| merge | `common/merge.py` (peft) | — |
| measured sweep | `common/evaluation.py` (merged model) + `common/monitor_gpu.py` + `common/monitor_pdu.py` | **0 only** |
| mceval | `mceval/scripts/run_pipeline.py` (configs via `_mceval_metadata.py`) | SFT 0,1 / sweep 0 |

## Notes / honest gaps
- **Untested end-to-end** (needs the GPU server + gated HF access); built directly from
  the repo's scripts and their hardcoded path conventions.
- `filter_formatted_outputs` in `LLMLingua_iterato.py` drops samples with `cot_length>1500`
  for all benchmarks (upstream rough edge) — left as-is.
- Figure aggregation into `plots/data/*.json` from raw runs is the one piece without a
  committed aggregator; ask for `auto/_aggregate.py` to wire it up.
- Decoding is greedy (temp 0, seed 42); vLLM greedy is **not** bitwise-deterministic.
