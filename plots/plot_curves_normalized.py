"""
plot_curves_normalized.py --- Publication-quality curve plots from inference_summary.json.

Reads pre-aggregated data from data/inference_summary.json.
PDU energy is normalized by the number of benchmark samples (J / sample).
Datasets: boolq, gsm8k, math, mceval, piqa  (lcb excluded).
Ratios:   0.1, 0.2, ..., 0.9, 1.0.

Usage:
    python plot_curves_normalized.py                   # all plots
    python plot_curves_normalized.py --plot energy     # Accuracy vs Normalized Energy
    python plot_curves_normalized.py --plot cot        # Accuracy vs CoT Length
    python plot_curves_normalized.py --plot cot_energy # CoT Length vs Normalized Energy
"""

import os
import json
import argparse

import scienceplots  # noqa: F401
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Publication-quality style
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
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(BASE_DIR, "data", "inference_summary.json")

MODELS = ["Llama3.1_8b", "Qwen2.5_3b", "Qwen2.5_7b", "Qwen2.5_14b"]
DATASETS = ["boolq", "gsm8k", "math", "mceval", "piqa"]
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

MODEL_STYLE = {
    "Llama3.1_8b": {"color": "#1f77b4", "marker": "o", "label": "Llama3.1 8B"},
    "Qwen2.5_3b":  {"color": "#2ca02c", "marker": "s", "label": "Qwen2.5 3B"},
    "Qwen2.5_7b":  {"color": "#d62728", "marker": "P", "label": "Qwen2.5 7B"},
    "Qwen2.5_14b": {"color": "#ff7f0e", "marker": "X", "label": "Qwen2.5 14B"},
}
DATASET_STYLE = {
    "boolq":  {"color": "#1f77b4", "marker": "o", "label": "BoolQ"},
    "gsm8k":  {"color": "#ff7f0e", "marker": "X", "label": "GSM8K"},
    "math":   {"color": "#d62728", "marker": "P", "label": "MATH"},
    "mceval": {"color": "#9467bd", "marker": "D", "label": "MCEval"},
    "piqa":   {"color": "#2ca02c", "marker": "s", "label": "PIQA"},
}
dataset_names = {
    "boolq": "BoolQ",
    "gsm8k": "GSM8K",
    "math": "MATH500",
    "mceval": "McEval",
    "piqa": "PIQA",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary() -> dict:
    """
    Load inference_summary.json and restructure it as:
        data[model][task][ratio] = { "accuracy": ..., "norm_pdu_energy_j": ..., ... }

    Filters to DATASETS, MODELS, and RATIOS defined above.
    PDU energy is normalized: norm_pdu_energy_j = PDU_Energy_J / N_Samples.
    """
    with open(SUMMARY_PATH) as f:
        records = json.load(f)

    allowed_ratios = set(RATIOS)

    data: dict = {m: {d: {} for d in DATASETS} for m in MODELS}

    for rec in records:
        model  = rec.get("Model")
        task   = rec.get("Task")
        ratio  = rec.get("Ratio")
        n_samp = rec.get("N_Samples") or 0

        if model not in MODELS or task not in DATASETS:
            continue
        if ratio not in allowed_ratios:
            continue
        if n_samp <= 0 or rec.get("PDU_Energy_J") is None:
            continue

        data[model][task][ratio] = {
            "ratio":              ratio,
            "accuracy":           rec.get("Accuracy"),
            "avg_cot_length":     rec.get("Avg_COT_Length"),
            "norm_pdu_energy_j":  rec["PDU_Energy_J"] / n_samp,  # J per sample
            "pdu_energy_j":       rec["PDU_Energy_J"],
            "n_samples":          n_samp,
        }

    # energy_pct_baseline = (J/sample @ ratio) / (J/sample @ ratio 1.0)
    for model in MODELS:
        for task in DATASETS:
            baseline = data[model][task].get(1.0)
            if baseline is None:
                continue
            base_j = baseline["norm_pdu_energy_j"]
            for ratio in data[model][task]:
                data[model][task][ratio]["energy_pct_baseline"] = (
                    data[model][task][ratio]["norm_pdu_energy_j"] / base_j
                )

    return data


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _savefig(base: str, dpi: int = 300) -> None:
    for ext, kwargs in [
        (".png",        {"dpi": dpi}),
        ("_slides.pdf", {"dpi": dpi}),
    ]:
        plt.savefig(f"{base}{ext}", bbox_inches="tight", **kwargs)
    try:
        plt.savefig(f"{base}.eps", format="eps", bbox_inches="tight", transparent=False)
    except Exception as exc:
        print(f"  EPS save skipped: {exc}")


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

def _annotate_ratios(ax, xs, ys, ratios, color):
    """Label each data point with its compression ratio value."""
    for x, y, r in zip(xs, ys, ratios):
        ax.annotate(
            f"{r:.1f}", (x, y),
            textcoords="offset points", xytext=(5, 5),
            fontsize=12, color=color, alpha=0.85,
        )


# ---------------------------------------------------------------------------
# Sweet-spot (knee) detection — Kneedle algorithm
# ---------------------------------------------------------------------------

def _find_knee(xs: list, ys: list) -> int | None:
    """
    Return the index of the knee point in a monotone concave curve using the
    Kneedle method: normalise both axes to [0,1], then find the point on the
    curve with maximum perpendicular distance from the chord connecting the
    first and last points.

    Returns None if fewer than 3 points are available.
    """
    import numpy as np
    xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
    if len(xs) < 3:
        return None

    # Normalise to [0, 1]
    x_range = xs[-1] - xs[0]
    y_range = ys[-1] - ys[0]
    if x_range == 0 or y_range == 0:
        return None
    xn = (xs - xs[0]) / x_range
    yn = (ys - ys[0]) / y_range

    # Chord from first to last point: direction vector
    dx, dy = xn[-1] - xn[0], yn[-1] - yn[0]
    chord_len = np.hypot(dx, dy)
    if chord_len == 0:
        return None

    # Perpendicular distance of each point from the chord
    dists = np.abs(dy * xn - dx * yn + xn[-1] * yn[0] - yn[-1] * xn[0]) / chord_len
    return int(np.argmax(dists))


# ---------------------------------------------------------------------------
# Generic plot builders
# ---------------------------------------------------------------------------

def _plot_by_task(
    data: dict, x_key: str, y_key: str,
    xlabel: str, ylabel: str, title_prefix: str,
    plots_dir: str, fname_prefix: str, start_idx: int,
) -> None:
    """One figure per benchmark, one line per model."""
    os.makedirs(plots_dir, exist_ok=True)

    for img_idx, dataset in enumerate(DATASETS, start=start_idx):
        fig, ax = plt.subplots()
        has_data = False

        for model in MODELS:
            style = MODEL_STYLE[model]
            entries = data[model][dataset]

            # Sort by ratio to get a consistent line order
            sorted_ratios = sorted(entries.keys())
            xs = [entries[r][x_key] for r in sorted_ratios if entries[r].get(x_key) is not None]
            ys = [entries[r][y_key] for r in sorted_ratios if entries[r].get(x_key) is not None]
            rs = [r for r in sorted_ratios if entries[r].get(x_key) is not None]

            if not xs:
                continue

            ax.plot(
                xs, ys,
                marker=style["marker"], color=style["color"], label=style["label"],
                markerfacecolor="white", markeredgewidth=2.0,
            )
            #_annotate_ratios(ax, xs, ys, rs, style["color"])
            has_data = True

        if not has_data:
            plt.close()
            continue

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_title(f"{title_prefix} --- {DATASET_STYLE[dataset]['label']}")
        #ax.set_title(f"{DATASET_STYLE[dataset]['label']}")
        ax.legend(title="Model", loc="best")
        plt.tight_layout()

        base = os.path.join(plots_dir, f"img_{img_idx:02d}_{fname_prefix}_{dataset}")
        _savefig(base)
        plt.close()
        print(f"  Saved {os.path.basename(base)}  (.png / _slides.pdf / .eps)")


def _plot_by_model(
    data: dict, x_key: str, y_key: str,
    xlabel: str, ylabel: str, title_prefix: str,
    plots_dir: str, fname_prefix: str, start_idx: int,
) -> None:
    """One figure per model, one line per benchmark."""
    os.makedirs(plots_dir, exist_ok=True)

    for img_idx, model in enumerate(MODELS, start=start_idx):
        fig, ax = plt.subplots()
        has_data = False

        for dataset in DATASETS:
            style = DATASET_STYLE[dataset]
            entries = data[model][dataset]

            sorted_ratios = sorted(entries.keys())
            xs = [entries[r][x_key] for r in sorted_ratios if entries[r].get(x_key) is not None]
            ys = [entries[r][y_key] for r in sorted_ratios if entries[r].get(x_key) is not None]
            rs = [r for r in sorted_ratios if entries[r].get(x_key) is not None]

            if not xs:
                continue

            ax.plot(
                xs, ys,
                marker=style["marker"], color=style["color"], label=style["label"],
                markerfacecolor="white", markeredgewidth=2.0,
            )
            #_annotate_ratios(ax, xs, ys, rs, style["color"])
            has_data = True

        if not has_data:
            plt.close()
            continue

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_title(f"{title_prefix} --- {MODEL_STYLE[model]['label']}")
        #ax.set_title(f"{MODEL_STYLE[model]['label']}")
        ax.legend(title="Benchmark", loc="best")
        plt.tight_layout()

        base = os.path.join(plots_dir, f"img_{img_idx:02d}_{fname_prefix}_{model}")
        _savefig(base)
        plt.close()
        print(f"  Saved {os.path.basename(base)}  (.png / _slides.pdf / .eps)")


# ---------------------------------------------------------------------------
# Plot suites
# ---------------------------------------------------------------------------

def plot_energy_accuracy(data: dict) -> None:
    """Normalized PDU Energy (J/sample) vs Accuracy."""
    print("  [Normalized Energy vs Accuracy]")
    out_base = os.path.join(BASE_DIR, "output", "curve_plots_normalized", "Energy")
    #xlabel="Normalized PDU Energy (J / sample)"

    _plot_by_task(data,
        x_key="norm_pdu_energy_j", y_key="accuracy",
        xlabel="PDU Energy (J / sample)", ylabel="Accuracy",
        title_prefix="Accuracy vs. Normalized PDU Energy",
        plots_dir=os.path.join(out_base, "Task"),
        fname_prefix="Energy_Task", start_idx=1,
    )
    _plot_by_model(data,
        x_key="norm_pdu_energy_j", y_key="accuracy",
        xlabel="PDU Energy (J / sample)", ylabel="Accuracy",
        title_prefix="Accuracy vs. Normalized PDU Energy",
        plots_dir=os.path.join(out_base, "Model"),
        fname_prefix="Energy_Model", start_idx=6,
    )


def plot_cot_accuracy(data: dict) -> None:
    """Average CoT Length (tokens) vs Accuracy."""
    print("  [CoT Length vs Accuracy]")
    out_base = os.path.join(BASE_DIR, "output", "curve_plots_normalized", "CoT")

    _plot_by_task(data,
        x_key="avg_cot_length", y_key="accuracy",
        xlabel="Average CoT Length (tokens)", ylabel="Accuracy",
        title_prefix="Accuracy vs. Average CoT Length",
        plots_dir=os.path.join(out_base, "Task"),
        fname_prefix="CoT_Task", start_idx=10,
    )
    _plot_by_model(data,
        x_key="avg_cot_length", y_key="accuracy",
        xlabel="Average CoT Length (tokens)", ylabel="Accuracy",
        title_prefix="Accuracy vs. Average CoT Length",
        plots_dir=os.path.join(out_base, "Model"),
        fname_prefix="CoT_Model", start_idx=15,
    )


def plot_cot_energy(data: dict) -> None:
    """Average CoT Length (tokens) vs Normalized PDU Energy (J/sample)."""
    print("  [CoT Length vs Normalized Energy]")
    out_base = os.path.join(BASE_DIR, "output", "curve_plots_normalized", "CoT_Energy")
    #ylabel="Normalized PDU Energy (J / sample)"

    _plot_by_task(data,
        x_key="avg_cot_length", y_key="norm_pdu_energy_j",
        xlabel="Average CoT Length (tokens)", ylabel="PDU Energy (J / sample)",
        title_prefix="Normalized PDU Energy vs. CoT Length",
        plots_dir=os.path.join(out_base, "Task"),
        fname_prefix="CoT_Energy_Task", start_idx=14,
    )

    _plot_by_model(data,
        x_key="avg_cot_length", y_key="norm_pdu_energy_j",
        xlabel="Average CoT Length (tokens)", ylabel="PDU Energy (J / sample)",
        title_prefix="Normalized PDU Energy vs. CoT Length",
        plots_dir=os.path.join(out_base, "Model"),
        fname_prefix="CoT_Energy_Model", start_idx=19,
    )


def plot_energy_pct_baseline(data: dict) -> None:
    """Accuracy vs PDU Energy normalized to γ=1 (E(γ)/E(γ=1), adimensionale)."""
    print("  [Energy % baseline vs Accuracy]")
    out_base = os.path.join(BASE_DIR, "output", "curve_plots_normalized", "Energy_pct")
    xlabel = r"Normalized PDU Energy: $E(\gamma)\,/\,E(\gamma{=}1)$"

    def _by_task(data, plots_dir):
        import os as _os
        _os.makedirs(plots_dir, exist_ok=True)
        for img_idx, dataset in enumerate(DATASETS, start=1):
            fig, ax = plt.subplots()
            has_data = False
            for model in MODELS:
                style   = MODEL_STYLE[model]
                entries = data[model][dataset]
                sorted_ratios = sorted(entries.keys())
                xs = [entries[r]["energy_pct_baseline"] for r in sorted_ratios
                      if entries[r].get("energy_pct_baseline") is not None]
                ys = [entries[r]["accuracy"] for r in sorted_ratios
                      if entries[r].get("energy_pct_baseline") is not None
                      and entries[r].get("accuracy") is not None]
                if not xs:
                    continue
                ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                        label=style["label"], markerfacecolor="white", markeredgewidth=2.0)
                has_data = True
            if not has_data:
                plt.close()
                continue
            ax.axvline(1, color="gray", linewidth=1.2, linestyle="--", alpha=0.6,
                       label=r"$\gamma = 1$")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Accuracy")
            leg_loc = "center right" if dataset == "gsm8k" else "best"
            ax.legend(title="Model", loc=leg_loc)
            plt.tight_layout()
            base = _os.path.join(plots_dir, f"img_{img_idx:02d}_Energy_pct_Task_{dataset}")
            _savefig(base)
            plt.close()
            print(f"  Saved {_os.path.basename(base)}  (.png / _slides.pdf / .eps)")

    def _by_model(data, plots_dir):
        import os as _os
        _os.makedirs(plots_dir, exist_ok=True)
        for img_idx, model in enumerate(MODELS, start=6):
            fig, ax = plt.subplots()
            has_data = False
            for dataset in DATASETS:
                style   = DATASET_STYLE[dataset]
                entries = data[model][dataset]
                sorted_ratios = sorted(entries.keys())
                xs = [entries[r]["energy_pct_baseline"] for r in sorted_ratios
                      if entries[r].get("energy_pct_baseline") is not None]
                ys = [entries[r]["accuracy"] for r in sorted_ratios
                      if entries[r].get("energy_pct_baseline") is not None
                      and entries[r].get("accuracy") is not None]
                if not xs:
                    continue
                ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                        label=style["label"], markerfacecolor="white", markeredgewidth=2.0)
                has_data = True
            if not has_data:
                plt.close()
                continue
            ax.axvline(1, color="gray", linewidth=1.2, linestyle="--", alpha=0.6,
                       label=r"$\gamma = 1$")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Accuracy")
            ax.legend(title="Benchmark", loc="best")
            plt.tight_layout()
            base = _os.path.join(plots_dir, f"img_{img_idx:02d}_Energy_pct_Model_{model}")
            _savefig(base)
            plt.close()
            print(f"  Saved {_os.path.basename(base)}  (.png / _slides.pdf / .eps)")

    _by_task(data, os.path.join(out_base, "Task"))
    _by_model(data, os.path.join(out_base, "Model"))


def generate_all(data: dict = None) -> None:
    data = data or load_summary()
    plot_energy_accuracy(data)
    plot_cot_accuracy(data)
    plot_cot_energy(data)
    plot_energy_pct_baseline(data)
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate normalized curve plots.")
    parser.add_argument(
        "--plot",
        choices=["energy", "cot", "cot_energy", "energy_pct", "all"],
        default="all",
        help="Which plot suite to generate (default: all)",
    )
    args = parser.parse_args()

    data = load_summary()

    if args.plot == "energy":
        plot_energy_accuracy(data)
    elif args.plot == "cot":
        plot_cot_accuracy(data)
    elif args.plot == "cot_energy":
        plot_cot_energy(data)
    elif args.plot == "energy_pct":
        plot_energy_pct_baseline(data)
    else:
        generate_all(data)
