# Full reproduction guide

End-to-end steps to reproduce every result (reasoning/QA arm, multilingual-code arm,
and the figures) from a clean server. Read this top to bottom once before starting.

> **Two golden rules (energy correctness):**
> 1. **Training (LoRA SFT) uses 2 GPUs** — `CUDA_VISIBLE_DEVICES=0,1` (data-parallel).
> 2. **Energy-measured inference uses exactly 1 GPU** — `CUDA_VISIBLE_DEVICES=0`.
>    Pinning inference to a single GPU keeps the on-GPU energy (`nvidia-smi`) and the
>    rack-level PDU energy attributable to one device, so the two measurements are
>    consistent. Never run the measured inference with tensor/data parallelism.

---

## 0. Hardware and expected setup

Reference machine: Intel Xeon Gold 6326, 256 GB RAM, **2 × NVIDIA RTX A6000 (49 GB)**.
- 3B / 7B / 8B models: single GPU fits for inference.
- 14B model: fits in bf16 on one A6000 (~28 GB) for **single-GPU** measured inference;
  training still uses both GPUs.

## 1. Software environments

Create two conda environments (names are conventions used below):

```bash
# (a) inference / compression / scoring / plotting
conda create -n tokenskip_env python=3.10 -y
conda activate tokenskip_env
pip install -r common/requirements.txt          # torch 2.5.1, vllm 0.6.4.post1, transformers 4.47.0, peft 0.14.0, ...

# (b) LoRA supervised fine-tuning (LLaMA-Factory)
conda create -n llamafactory_env python=3.10 -y
conda activate llamafactory_env
pip install "llamafactory[torch,metrics]"        # or clone https://github.com/hiyouga/LLaMA-Factory and `pip install -e .`
```

The multilingual-code arm (`mceval/`) additionally needs the **McEval Docker image**,
pulled **by sha256 digest** (never a floating tag) so the 40 language runtimes are
fixed. See `mceval/docs/PIPELINE_RUNBOOK.md`.

## 2. Energy measurement setup (read before any measured run)

Energy per measured inference run = **on-GPU energy** (`common/monitor_gpu.py`, via
`nvidia-smi` on the single inference GPU) **+ rack PDU active power** integrated over
the run (`common/monitor_pdu.py`, via SNMP).

Configure the PDU host before measuring (it is a placeholder in the committed code):

```bash
export PDU_HOST=192.0.2.1            # <-- set to YOUR rack PDU's IP
export PDU_SNMP_COMMUNITY=public
# OID: PowerNet-MIB::ePDUPhaseStatusActivePower.1
```

Pass `--host "$PDU_HOST"` to `common/monitor_pdu.py` (the scripts default to the
placeholder `192.0.2.1`). The energy-sweep drivers in `common/scripts/` start both
monitors around the inference window and stop them before scoring.

---

## 3. Reasoning / QA arm — `gsm8k/`, `math/`, `boolq/`, `piqa/`

Run the following **per benchmark** (`gsm8k`, `math`, `boolq`, `piqa`) and **per
model** (`Qwen2.5-3B/7B/14B-Instruct`, `Llama-3.1-8B-Instruct`). Shared code is in
`common/`; each benchmark folder holds its data + configs.

### 3.1 Prepare data
- GSM8K and MATH-500 data are already provided under `<bench>/datasets/`.
- BoolQ / PIQA:
  ```bash
  cd boolq && python prepare.py     # then place the produced jsonl under datasets/boolq/
  cd piqa  && python prepare.py
  ```

### 3.2 Compress the CoT (build TokenSkip training data) — `tokenskip_env`
```bash
conda activate tokenskip_env
# Builds compressed CoT data across the compression-ratio grid (LLMLingua-2).
python common/LLMLingua.py            # single benchmark
python common/LLMLingua_iterato.py    # iterate over models × benchmarks × ratios
```

### 3.3 Build the LLaMA-Factory SFT files — `tokenskip_env`
```bash
python common/get_llamafactory_input_all_qwen.py     # Qwen models
python common/get_llamafactory_input_all_llama.py    # Llama model
# (per-task variants also exist: get_llamafactory_input_{boolq,piqa,math_gsm8k}.py)
```

### 3.4 LoRA fine-tune — `llamafactory_env` — **2 GPUs**
```bash
conda activate llamafactory_env
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \
    common/configs/examples/train_lora/myllama3_lora_sft_compressed_gsm8k_llmlingua2_qwen_7B.yaml
```
Use the matching `*_3B.yaml` / `*_14B.yaml` per model; adapt the dataset entry per
benchmark. LLaMA-Factory does multi-GPU DDP automatically across the two visible GPUs.

### 3.5 Merge the LoRA adapter into the base model — `tokenskip_env`
```bash
conda activate tokenskip_env
python common/merge.py --base <HF_BASE_MODEL> --adapter <LORA_OUTPUT_DIR> --output <MERGED_DIR>
# or batch: bash common/scripts/merge_qwen.sh   /   bash common/scripts/merge_llama.sh
```

### 3.6 Inference + energy + scoring — `tokenskip_env` — **1 GPU**

Single configuration (quick check), from a benchmark folder:
```bash
cd gsm8k
# edit run.sh: MODEL_PATH (base or merged), MODEL_TYPE, COMPRESSION_RATIO
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

Full measured sweep (all ratios × token budgets × repeats, with energy monitors) —
run from the benchmark folder so `configs/` and `datasets/` resolve, single GPU:
```bash
cd gsm8k
# set BENCHMARK, MODEL_*, ADAPTER/MERGED paths and PDU host inside the driver
CUDA_VISIBLE_DEVICES=0 bash ../common/scripts/Itera.sh
```
The driver starts `monitor_gpu.py` (GPU 0) and `monitor_pdu.py` (`$PDU_HOST`) around
the inference window, runs `evaluation.py` (greedy, seed 42), then scores. Results and
per-run energy land in the run's output directory.

> Keep `CUDA_VISIBLE_DEVICES` to a **single** index for every measured inference run.

---

## 4. Multilingual-code arm — `mceval/`

Self-contained package, orchestrated end-to-end per model:

```bash
cd mceval
cp configs/paths.example.yaml configs/paths.yaml                 # set data_root etc.
cp configs/run_metadata.example.yaml configs/run_metadata.yaml   # fill the pinned values
# fill run_metadata.yaml: model commit hashes, vllm version, McEval Docker sha256 digest
python scripts/run_pipeline.py --model qwen2.5-7b-instruct       # Phases 1->4, stops at review gates
```
Phases: train-data generation → LLMLingua-2 compression → LoRA SFT (multi-GPU) →
energy-instrumented evaluation sweep (**single-GPU** inference by design). The
authoritative recipe is `mceval/docs/PIPELINE_RUNBOOK.md`; design in
`mceval/docs/PROJECT_ROADMAP.md`.

---

## 5. Figures — `plots/` — `tokenskip_env` (CPU only)

After the runs, aggregate the per-run metrics into the two summaries
(`plots/data/inference_summary.json`, `plots/data/prompt_length_summary.json`), then:
```bash
cd plots
pip install -r requirements.txt
python plot_curves_normalized.py                 # accuracy/CoT/energy curves -> output/
python fitting_linear_cot_energy_normalized.py   # linear CoT-vs-energy fits -> output/
```
The committed summaries already reproduce the paper figures without re-running.

---

## 6. Reproduction notes
- Decoding is greedy (`temperature 0.0`, `seed 42`). vLLM greedy is **not**
  bitwise-deterministic, so numbers match within small noise, not digit-for-digit.
- Pin model commit hashes (Hugging Face) and the McEval Docker digest for exact runs.
- Bulk artifacts (weights, merged models, generations, eval dumps) are **not** in the
  repo — they are regenerated by the steps above.
- Energy validity depends on the 1-GPU inference rule (§0); a measured run that uses
  more than one GPU makes the GPU/PDU energy non-comparable.
