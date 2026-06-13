"""
fitting_linear_cot_energy_normalized.py

Linear fit: Average CoT Length vs PDU Energy per sample (J / sample).

For each (model, task) pair fits:
    pdu_energy_j / n_samples = slope * avg_cot_length + intercept

Generates:
  - One plot per (model, task) pair: scatter + fit line + R² / p / SE
  - One plot per benchmark (all models on the same axes)
  - One plot per model (all benchmarks on the same axes)
  - One combined plot with all (model, task) pairs

Data source: data/inference_summary.json
Output:
  output/fitting/linear_cot_energy_norm/          <- fit parameters (JSON)
  output/curve_plots/LinearFit/cot_energy_norm/   <- plots
"""

import os
import json
import itertools
import numpy as np
import scienceplots  # noqa: F401
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Publication-quality style  (matches plot_curves_normalized.py)
# ---------------------------------------------------------------------------

plt.style.use(["science", "ieee"])
mpl.rcParams.update({
    "figure.figsize":        (9, 6),
    "font.size":             18,
    "axes.titlesize":        20,
    "axes.labelsize":        18,
    "xtick.labelsize":       16,
    "ytick.labelsize":       16,
    "legend.fontsize":       15,
    "legend.title_fontsize": 16,
    "legend.framealpha":     0.9,
    "lines.linewidth":       2.5,
    "lines.markersize":      10,
    "font.family":           "sans-serif",
    #"font.sans-serif":       ["Arial"],
    #"text.usetex":           False,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
    "xtick.major.size":      5,
    "xtick.minor.size":      2.5,
    "ytick.major.size":      5,
    "ytick.minor.size":      2.5,
    "xtick.top":             False,
    "ytick.right":           False,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
})

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.join(BASE_DIR, "output", "LinearFitting_EnergyNorm_CoT")
FIT_DIR      = os.path.join(_ROOT, "fitting_params")
PLOT_DIR     = os.path.join(_ROOT, "fitting_plots")
SUMMARY_PATH = os.path.join(BASE_DIR, "data", "inference_summary.json")

ALL_MODELS  = ["Llama3.1_8b", "Qwen2.5_3b", "Qwen2.5_7b", "Qwen2.5_14b"]
ALL_TASKS   = ["boolq", "gsm8k", "math", "mceval", "piqa"]
KEEP_RATIOS = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

# Visual styles  -  consistent with plot_curves_normalized.py
_MODEL_COLOR = {
    "Llama3.1_8b": "#1f77b4",
    "Qwen2.5_3b":  "#2ca02c",
    "Qwen2.5_7b":  "#d62728",
    "Qwen2.5_14b": "#ff7f0e",
}
_MODEL_MARKER = {
    "Llama3.1_8b": "o",
    "Qwen2.5_3b":  "s",
    "Qwen2.5_7b":  "P",
    "Qwen2.5_14b": "X",
}
_MODEL_LABEL = {
    "Llama3.1_8b": "Llama3.1 8B",
    "Qwen2.5_3b":  "Qwen2.5 3B",
    "Qwen2.5_7b":  "Qwen2.5 7B",
    "Qwen2.5_14b": "Qwen2.5 14B",
}
_TASK_COLOR = {
    "boolq":  "#1f77b4",
    "gsm8k":  "#ff7f0e",
    "math":   "#d62728",
    "mceval": "#9467bd",
    "piqa":   "#2ca02c",
}
_TASK_MARKER = {
    "boolq":  "o",
    "gsm8k":  "X",
    "math":   "P",
    "mceval": "D",
    "piqa":   "s",
}
_TASK_LABEL = {
    "boolq":  "BoolQ",
    "gsm8k":  "GSM8K",
    "math":   "MATH500",
    "mceval": "MCEval",
    "piqa":   "PIQA",
}

XLABEL = "Average CoT Length (tokens)"
#YLABEL = "PDU Energy per Sample (J / sample)"
YLABEL = "PDU Energy (J / sample)"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_summary() -> dict:
    """Load inference_summary.json and index it as data[model][task] = [rows]."""
    with open(SUMMARY_PATH) as f:
        rows = json.load(f)
    data: dict = {}
    for row in rows:
        m = row["Model"]
        t = row["Task"]
        data.setdefault(m, {}).setdefault(t, []).append(row)
    return data


def _collect(rows: list) -> tuple[list, list, list]:
    """
    Extract (cot_length, energy_per_sample, ratio) for rows that pass KEEP_RATIOS.
    Uses Avg_Total_Length (CoT + code) when available, otherwise Avg_COT_Length.
    """
    xs, ys, rs = [], [], []
    for row in rows:
        cr = row.get("Ratio")
        if cr not in KEEP_RATIOS:
            continue
        tok = row.get("Avg_Total_Length") or row.get("Avg_COT_Length")
        e   = row.get("PDU_Energy_J")
        n   = row.get("N_Samples")
        if tok and e and n and n > 0:
            xs.append(float(tok))
            ys.append(float(e) / float(n))
            rs.append(float(cr))
    return xs, ys, rs

# ---------------------------------------------------------------------------
# Linear regression
# ---------------------------------------------------------------------------

def _fit(xs: list, ys: list) -> dict | None:
    """Return regression parameters, or None if there are fewer than 3 points."""
    if len(xs) < 3:
        return None
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return {
        "slope":     float(slope),
        "intercept": float(intercept),
        "r2":        float(r_value ** 2),
        "p_value":   float(p_value),
        "std_err":   float(std_err),
        "n_points":  len(x),
        "cot_range": [float(x.min()), float(x.max())],
        "e_range":   [float(y.min()), float(y.max())],
    }

# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _savefig(base: str, dpi: int = 300) -> None:
    for ext, kwargs in [(".png", {"dpi": dpi}), ("_slides.pdf", {"dpi": dpi})]:
        plt.savefig(f"{base}{ext}", bbox_inches="tight", **kwargs)
    try:
        plt.savefig(f"{base}.eps", format="eps", bbox_inches="tight", transparent=False)
    except Exception as exc:
        print(f"    EPS save skipped: {exc}")

# ---------------------------------------------------------------------------
# Individual (model, task) plot
# ---------------------------------------------------------------------------

def _plot_single(model: str, task: str,
                 xs: list, ys: list, rs: list,
                 params: dict, plots_dir: str) -> None:
    """Scatter + linear fit for a single (model, task) pair."""
    x     = np.array(xs)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = params["slope"] * x_fit + params["intercept"]

    fig, ax = plt.subplots()

    sc = ax.scatter(xs, ys, c=rs, cmap="plasma", s=100, zorder=3,
                    vmin=min(rs), vmax=max(rs),
                    edgecolors="white", linewidths=1.0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Compression ratio", fontsize=16)
    cbar.ax.tick_params(labelsize=15)

    for xi, yi, ri in zip(xs, ys, rs):
        ax.annotate(f"{ri:.1f}", (xi, yi),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=13, alpha=0.85)

    ax.plot(x_fit, y_fit, color="steelblue", linewidth=2.2, linestyle="--",
            label=f"$y = {params['slope']:.3f}x + {params['intercept']:.2f}$")

    # Stats box in the lower-right corner to stay clear of the fit line and legend
    stats_text = (
        f"$R^2 = {params['r2']:.4f}$\n"
        f"$p = {params['p_value']:.2e}$\n"
        f"$\\mathrm{{SE}} = {params['std_err']:.4f}$"
    )
    ax.text(0.97, 0.05, stats_text,
            transform=ax.transAxes, fontsize=14,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(f"{_MODEL_LABEL[model]} - {_TASK_LABEL[task]}")
    ax.legend(loc="upper left")
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    base = os.path.join(plots_dir, f"{model}_{task}_fit")
    _savefig(base)
    plt.close()
    print(f"  Plot => {os.path.relpath(base, BASE_DIR)}  (.png / _slides.pdf / .eps)")

# ---------------------------------------------------------------------------
# Per-benchmark plot: one line per model, same benchmark
# ---------------------------------------------------------------------------

def _plot_fits_by_task(task: str, all_series: list[dict]) -> None:
    """All models' scatter + fit on a single axes for one benchmark."""
    series = [s for s in all_series if s["task"] == task]
    if not series:
        return

    fig, ax = plt.subplots()
    legend_handles = []

    for s in series:
        model  = s["model"]
        xs     = np.array(s["xs"])
        ys     = np.array(s["ys"])
        color  = _MODEL_COLOR[model]
        marker = _MODEL_MARKER[model]

        ax.scatter(xs, ys, color=color, marker=marker, s=100, zorder=3,
                   edgecolors="white", linewidths=1.0)
        x_fit = np.linspace(xs.min(), xs.max(), 200)
        y_fit = s["params"]["slope"] * x_fit + s["params"]["intercept"]
        ax.plot(x_fit, y_fit, color=color, linewidth=2.2, linestyle="--")

        # Legend handle: line + marker combined
        legend_handles.append(
            mlines.Line2D([], [], color=color, marker=marker, linestyle="--",
                          linewidth=2.2, markersize=11, markerfacecolor="white",
                          markeredgewidth=1.5, label=_MODEL_LABEL[model])
        )

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(_TASK_LABEL[task])
    ax.legend(handles=legend_handles, title="Model", loc="best")
    plt.tight_layout()

    plots_dir = os.path.join(PLOT_DIR, "by_task")
    os.makedirs(plots_dir, exist_ok=True)
    base = os.path.join(plots_dir, f"{task}_all_models")
    _savefig(base)
    plt.close()
    print(f"  Plot => {os.path.relpath(base, BASE_DIR)}  (.png / _slides.pdf / .eps)")

# ---------------------------------------------------------------------------
# Per-model plot: one line per benchmark, same model
# ---------------------------------------------------------------------------

def _plot_fits_by_model(model: str, all_series: list[dict]) -> None:
    """All benchmarks' scatter + fit on a single axes for one model."""
    series = [s for s in all_series if s["model"] == model]
    if not series:
        return

    fig, ax = plt.subplots()
    legend_handles = []

    for s in series:
        task   = s["task"]
        xs     = np.array(s["xs"])
        ys     = np.array(s["ys"])
        color  = _TASK_COLOR[task]
        marker = _TASK_MARKER[task]

        ax.scatter(xs, ys, color=color, marker=marker, s=100, zorder=3,
                   edgecolors="white", linewidths=1.0)
        x_fit = np.linspace(xs.min(), xs.max(), 200)
        y_fit = s["params"]["slope"] * x_fit + s["params"]["intercept"]
        ax.plot(x_fit, y_fit, color=color, linewidth=2.2, linestyle="--")

        # Legend handle: line + marker combined
        legend_handles.append(
            mlines.Line2D([], [], color=color, marker=marker, linestyle="--",
                          linewidth=2.2, markersize=11, markerfacecolor="white",
                          markeredgewidth=1.5, label=_TASK_LABEL[task])
        )

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(_MODEL_LABEL[model])
    ax.legend(handles=legend_handles, title="Benchmark", loc="best")
    plt.tight_layout()

    plots_dir = os.path.join(PLOT_DIR, "by_model")
    os.makedirs(plots_dir, exist_ok=True)
    base = os.path.join(plots_dir, f"{model}_all_tasks")
    _savefig(base)
    plt.close()
    print(f"  Plot => {os.path.relpath(base, BASE_DIR)}  (.png / _slides.pdf / .eps)")

# ---------------------------------------------------------------------------
# Combined plot: all (model, task) pairs on one axes
# ---------------------------------------------------------------------------

def _plot_combined(all_series: list[dict]) -> None:
    """All (model, task) pairs on one axes; both legends outside to the right."""
    fig, ax = plt.subplots(figsize=(14, 9))

    for s in all_series:
        xs     = np.array(s["xs"])
        ys     = np.array(s["ys"])
        color  = _MODEL_COLOR[s["model"]]
        marker = _TASK_MARKER[s["task"]]

        ax.scatter(xs, ys, color=color, marker=marker, s=130, zorder=3,
                   edgecolors="white", linewidths=1.0)
        x_fit = np.linspace(xs.min(), xs.max(), 200)
        y_fit = s["params"]["slope"] * x_fit + s["params"]["intercept"]
        ax.plot(x_fit, y_fit, color=color, linewidth=2.5, linestyle="--")

    ax.set_xlabel(XLABEL, fontsize=24)
    ax.set_ylabel(YLABEL, fontsize=24)
    ax.tick_params(axis="both", labelsize=22)

    # Model legend: colored line, placed outside top-right
    model_handles = [
        mlines.Line2D([], [], color=_MODEL_COLOR[m], linewidth=3.0,
                      label=_MODEL_LABEL[m])
        for m in ALL_MODELS
        if any(s["model"] == m for s in all_series)
    ]
    # Benchmark legend: gray dashed line + marker, placed below model legend
    task_handles = [
        mlines.Line2D([], [], color="gray", marker=_TASK_MARKER[t],
                      linestyle="--", linewidth=2.5, markersize=14,
                      markerfacecolor="white", markeredgewidth=1.8,
                      label=_TASK_LABEL[t])
        for t in ALL_TASKS
        if any(s["task"] == t for s in all_series)
    ]

    leg1 = ax.legend(handles=model_handles, title="Model",
                     loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     fontsize=20, title_fontsize=22,
                     framealpha=0.9, borderpad=0.5, labelspacing=0.5,
                     handlelength=1.8)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=task_handles, title="Benchmark",
                     loc="upper left", bbox_to_anchor=(1.01, 0.50),
                     fontsize=20, title_fontsize=22,
                     framealpha=0.9, borderpad=0.5, labelspacing=0.5,
                     handlelength=1.8)

    plt.tight_layout()

    os.makedirs(PLOT_DIR, exist_ok=True)
    base = os.path.join(PLOT_DIR, "linearFitting_EnergyNorm_CoT_all_combined")
    # bbox_extra for legends outside the axes
    for ext, kwargs in [(".png", {"dpi": 300}), ("_slides.pdf", {"dpi": 300})]:
        fig.savefig(f"{base}{ext}", bbox_inches="tight",
                    bbox_extra_artists=(leg1, leg2), **kwargs)
    try:
        fig.savefig(f"{base}.eps", format="eps", bbox_inches="tight",
                    bbox_extra_artists=(leg1, leg2), transparent=False)
    except Exception as exc:
        print(f"    EPS save skipped: {exc}")
    plt.close()
    print(f"  Combined => {os.path.relpath(base, BASE_DIR)}  (.png / _slides.pdf / .eps)")

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_fitting(models: list = None, tasks: list = None) -> None:
    models     = models or ALL_MODELS
    tasks      = tasks  or ALL_TASKS
    all_data   = _load_summary()
    all_series = []

    # Individual (model, task) plots
    print("Fitting individual (model, task) pairs...")
    for model, task in itertools.product(models, tasks):
        rows       = all_data.get(model, {}).get(task, [])
        xs, ys, rs = _collect(rows)
        params     = _fit(xs, ys)

        if params is None:
            print(f"  Not enough points for {model}/{task}, skipping.")
            continue

        params["model"] = model
        params["task"]  = task

        fit_dir = os.path.join(FIT_DIR, model)
        os.makedirs(fit_dir, exist_ok=True)
        with open(os.path.join(fit_dir, f"{task}_params.json"), "w") as f:
            json.dump(params, f, indent=2)

        print(f"  {_MODEL_LABEL[model]} / {_TASK_LABEL[task]}"
              f"  ({len(xs)} points)  R² = {params['r2']:.4f}")
        _plot_single(model, task, xs, ys, rs, params,
                     os.path.join(PLOT_DIR, "individual", model))

        all_series.append({
            "model": model, "task": task,
            "xs": xs, "ys": ys,
            "params": params,
        })

    # Per-benchmark plots (all models, one benchmark per figure)
    print("\nPer-benchmark plots (all models)...")
    for task in tasks:
        _plot_fits_by_task(task, all_series)

    # Per-model plots (all benchmarks, one model per figure)
    print("\nPer-model plots (all benchmarks)...")
    for model in models:
        _plot_fits_by_model(model, all_series)

    # Combined plot
    print("\nGenerating combined plot...")
    _plot_combined(all_series)

    # Export summary JSON
    summary = {m: {t: {} for t in tasks} for m in models}
    for s in all_series:
        summary[s["model"]][s["task"]] = s["params"]
    os.makedirs(FIT_DIR, exist_ok=True)
    with open(os.path.join(FIT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Linear fit: CoT length vs normalized PDU energy."
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=ALL_TASKS, default=None,
        metavar="TASK",
        help=f"Benchmarks to include (default: all). Choices: {ALL_TASKS}",
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=None,
        metavar="MODEL",
        help=f"Models to include (default: all). Choices: {ALL_MODELS}",
    )
    args = parser.parse_args()
    run_fitting(models=args.models, tasks=args.tasks)
