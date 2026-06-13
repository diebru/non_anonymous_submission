# plots/ — figure generation

Publication-quality figures for the accuracy / CoT-length / energy trade-off, built
from the **aggregated result summaries** in `data/` (so the figures regenerate
without re-running any inference).

## Contents
- `data/inference_summary.json` — per (model × task × compression-ratio) results:
  accuracy, average CoT/answer length, inference time, GPU and PDU energy.
- `data/prompt_length_summary.json` — per (model × benchmark) prompt-length stats.
- `plot_curves_normalized.py` — accuracy-vs-CoT, accuracy-vs-energy, and normalized
  curve plots, grouped by task and by model.
- `fitting_linear_cot_energy_normalized.py` — linear fits of normalized CoT vs
  energy (writes the fitting parameters and fitted plots).

## Run
```bash
pip install -r requirements.txt
python plot_curves_normalized.py
python fitting_linear_cot_energy_normalized.py
```
Each script reads `data/` and writes figures under `output/` (gitignored). Paths are
resolved relative to the script location, so run from anywhere.

The summaries are produced from the raw run outputs of the benchmark arms
(`../gsm8k`, `../math`, `../boolq`, `../piqa`, and `../mceval`).

## Energy measurement (how the `*_Energy_*` columns are produced)
Each measured inference run is pinned to a **single GPU** so the two energy sources
are consistent and comparable:
- **GPU energy** — `nvidia-smi` power on the one inference GPU (`common/monitor_gpu.py`).
- **PDU energy** — rack PDU active power over SNMP, integrated over the run
  (`common/monitor_pdu.py`). PDU host: **`192.0.2.1`** (placeholder; set to your rack
  PDU's IP via `--host`/`PDU_HOST`), SNMP community `public`, OID
  `PowerNet-MIB::ePDUPhaseStatusActivePower.1`.

LoRA training uses 2 GPUs; the energy-measured inference uses 1 GPU (see
[`../REPRODUCE.md`](../REPRODUCE.md) §0).
