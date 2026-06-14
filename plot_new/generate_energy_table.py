"""
generate_energy_table.py

Generates a LaTeX table of PDU Energy (J/sample) for each model x benchmark
at compression ratios gamma in {0.1, 0.5, 1.0}.

Layout:
  rows    -- benchmarks (BoolQ, GSM8K, MATH500, MCEval, PIQA)
  columns -- model x ratio  (3 ratios x 4 models = 12 data columns)

Output: output/tables/energy_pdu_table.tex
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_dir() -> str:
    # 1) explicit override (point this at your reproduced summary for the cross-check)
    env = os.environ.get("SUMMARY_DIR")
    if env:
        return env
    # 2) a local plot_new/data/ if present
    local = os.path.join(BASE_DIR, "data")
    if os.path.exists(os.path.join(local, "inference_summary.json")):
        return local
    # 3) fall back to the committed reference summaries in plots/data/
    return os.path.join(BASE_DIR, "..", "plots", "data")


DATA_DIR     = _resolve_data_dir()
SUMMARY_PATH = os.path.join(DATA_DIR, "inference_summary.json")
OUT_PATH     = os.path.join(BASE_DIR, "output", "tables", "energy_pdu_table.tex")

MODELS = ["Llama3.1_8b", "Qwen2.5_3b", "Qwen2.5_7b", "Qwen2.5_14b"]
DATASETS = ["boolq", "gsm8k", "math", "mceval", "piqa"]
RATIOS = [1.0]

MODEL_LABEL = {
    "Llama3.1_8b": r"Llama3.1 8B",
    "Qwen2.5_3b":  r"Qwen2.5 3B",
    "Qwen2.5_7b":  r"Qwen2.5 7B",
    "Qwen2.5_14b": r"Qwen2.5 14B",
}
DATASET_LABEL = {
    "boolq":  "BoolQ",
    "gsm8k":  "GSM8K",
    "math":   "MATH500",
    "mceval": "MCEval",
    "piqa":   "PIQA",
}


def load_data() -> dict:
    with open(SUMMARY_PATH) as f:
        records = json.load(f)

    allowed = set(RATIOS)
    data = {m: {d: {} for d in DATASETS} for m in MODELS}

    for rec in records:
        model  = rec.get("Model")
        task   = rec.get("Task")
        ratio  = rec.get("Ratio")
        n_samp = rec.get("N_Samples") or 0

        if model not in MODELS or task not in DATASETS:
            continue
        if ratio not in allowed:
            continue
        if n_samp <= 0 or rec.get("PDU_Energy_J") is None:
            continue

        data[model][task][ratio] = rec["PDU_Energy_J"] / n_samp

    return data


def fmt(val) -> str:
    """Format a float for the table, or --- if missing."""
    if val is None:
        return r"\texttt{---}"
    return f"{val:.4f}"


def build_table(data: dict) -> str:
    # ascending sort benchmarks by average PDU energy across models 
    def avg_energy(ds):
        vals = [data[m][ds].get(1.0) for m in MODELS if data[m][ds].get(1.0) is not None]
        return sum(vals) / len(vals) if vals else float("inf")

    sorted_datasets = sorted(DATASETS, key=avg_energy)

    col_spec = "l " + " ".join(["c"] * len(sorted_datasets))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{PDU energy consumption (J/sample) per model and benchmark"
                 r" at $\gamma = 1$ (no compression). Benchmarks sorted by average energy.}")
    lines.append(r"  \label{tab:pdu-energy}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # header: Model | ds1 | ds2 | ...
    ds_headers = " & ".join(f"\\textbf{{{DATASET_LABEL[ds]}}}" for ds in sorted_datasets)
    lines.append(f"    \\textbf{{Model}} & {ds_headers} \\\\")
    lines.append(r"    \midrule")

    # data rows: one per model, sorted by their total energy (ascending)
    def total_energy(model):
        vals = [data[model][ds].get(1.0) for ds in DATASETS if data[model][ds].get(1.0) is not None]
        return sum(vals) if vals else float("inf")

    sorted_models = sorted(MODELS, key=total_energy)

    for model in sorted_models:
        #print('data[model]',model,data[model])
        row_vals = [fmt(data[model][ds].get(1.0)) for ds in sorted_datasets]
        lines.append(f"    {MODEL_LABEL[model]} & {' & '.join(row_vals)} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def main():
    data = load_data()
    tex  = build_table(data)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(tex)

    print(f"Saved: {os.path.relpath(OUT_PATH, BASE_DIR)}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
