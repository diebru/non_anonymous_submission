# common/ — shared reasoning engine and tooling

Shared, benchmark-agnostic code for the reasoning/QA benchmarks (`gsm8k`, `math`,
`boolq`, `piqa`). Each benchmark folder holds only its dataset, configs, and a thin
`run.sh`; everything reusable lives here.

## Contents
- `evaluation.py` — the inference + scoring engine (driven by `--benchmark`). Reads
  `configs/<benchmark>_<data_type>.json` and the dataset path it points to, both
  **relative to the working directory** (so it is launched from a benchmark folder).
- `eval/`, `data_processing/` — answer extraction, scoring, and prompt/response
  processing imported by the engine.
- `LLMLingua.py`, `LLMLingua_iterato.py` — LLMLingua-2 CoT compression (builds the
  compressed training data at each compression ratio).
- `get_llamafactory_input_*.py` — convert compressed data into LLaMA-Factory SFT files.
- `merge.py` — merge a trained LoRA adapter into its base model.
- `monitor_gpu.py`, `monitor_pdu.py` — power/energy logging during inference.
- `configs/examples/train_lora/` — example LLaMA-Factory LoRA SFT configs.
- `requirements.txt` — pinned Python dependencies for this arm.
- `scripts/` — the original full-sweep drivers (see `scripts/README.md`).
- `Readme.md`, `assets/` — upstream TokenSkip documentation (method reference).

## Typical per-benchmark pipeline
1. (optional) `<benchmark>/prepare.py` — download/format the raw data.
2. `LLMLingua*.py` — build compressed CoT data across compression ratios.
3. `get_llamafactory_input_*.py` — produce LLaMA-Factory SFT files.
4. LoRA fine-tune with LLaMA-Factory (configs in `configs/examples/train_lora/`).
5. `merge.py` — merge the adapter into the base model.
6. `<benchmark>/run.sh` (single run) or `scripts/` drivers (full ratio/token sweep
   with energy logging) — inference + scoring.
