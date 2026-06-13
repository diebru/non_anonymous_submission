# common/scripts/ — full-sweep drivers

These are the original driver scripts used to run the **full experiment sweep**
(every compression ratio × max-new-tokens × repeated runs) with GPU/PDU energy
logging. They are the heavier counterpart to the per-benchmark `run.sh`.

| Script | Role |
|---|---|
| `Itera.sh` | full ratio × token × repeat inference sweep with energy monitors (one benchmark per invocation; set `BENCHMARK` at the top) |
| `inference_big_loop_on_qwen.sh`, `inference_big_loop_on_llama.sh` | batched inference sweep across model sizes and ratios |
| `eval.sh`, `eval_inference.sh` | single / inference-only evaluation entry points |
| `eval_training.sh`, `eval_training_math_gsm8k_llama.sh` | evaluate trained (merged) models |
| `merge_qwen.sh`, `merge_llama.sh` | batch-merge LoRA adapters into base models (call `../merge.py`) |

## Working-directory assumption
These drivers were written for a **flat layout** where `evaluation.py`, `configs/`,
and `datasets/` all sit in the current directory. Helper-script paths have been
updated to point at the shared engine (`../evaluation.py`, `../merge.py`,
`../monitor_*.py`), but the drivers still expect `configs/<benchmark>_*.json` and
the dataset to be resolvable from the working directory.

To use them in this split layout, run from the relevant benchmark folder (which
holds `configs/` and `datasets/`), e.g.:

```bash
cd ../../gsm8k && bash ../common/scripts/Itera.sh
```

For a simple single run, prefer the benchmark folder's own `run.sh`.

LoRA adapter paths (`../LlamaFactory/lora_saves/...`) refer to LLaMA-Factory
outputs produced during SFT; point them at your own training output directory.
