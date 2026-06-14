"""
generate_savings_table.py

For each (model, benchmark, compression ratio), relative to ratio=1.0:
  - Energy saved       (%): (E(1) - E(gamma)) / E(1)) * 100
  - Accuracy lost      (pp): (Acc(1) - Acc(gamma)) * 100  [negative = improvement]
  - CoT reduction      (%): (CoT(1) - CoT(gamma)) / CoT(1) * 100
  - Prompt/CoT ratio   (%): mean_prompt_tokens / avg_cot_tokens * 100

Output: output/tables/savings_summary.txt
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_dir() -> str:
    env = os.environ.get("SUMMARY_DIR")
    if env:
        return env
    local = os.path.join(BASE_DIR, "data")
    if os.path.exists(os.path.join(local, "inference_summary.json")):
        return local
    return os.path.join(BASE_DIR, "..", "plots", "data")


DATA_DIR     = _resolve_data_dir()
SUMMARY_PATH = os.path.join(DATA_DIR, "inference_summary.json")
PROMPT_PATH  = os.path.join(DATA_DIR, "prompt_length_summary.json")
OUT_PATH     = os.path.join(BASE_DIR, "output", "tables", "savings_summary.txt")

MODELS   = ["Llama3.1_8b", "Qwen2.5_3b", "Qwen2.5_7b", "Qwen2.5_14b"]
DATASETS = ["boolq", "gsm8k", "math", "mceval", "piqa"]
RATIOS   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

MODEL_LABEL = {
    "Llama3.1_8b": "Llama3.1 8B",
    "Qwen2.5_3b":  "Qwen2.5 3B",
    "Qwen2.5_7b":  "Qwen2.5 7B",
    "Qwen2.5_14b": "Qwen2.5 14B",
}
DATASET_LABEL = {
    "boolq":  "BoolQ",
    "gsm8k":  "GSM8K",
    "math":   "MATH500",
    "mceval": "MCEval",
    "piqa":   "PIQA",
}


def load_inference() -> dict:
    """data[model][task][ratio] = {energy, accuracy, cot}"""
    with open(SUMMARY_PATH) as f:
        records = json.load(f)

    allowed = set(RATIOS) | {1.0}
    data = {m: {d: {} for d in DATASETS} for m in MODELS}

    for rec in records:
        model = rec.get("Model")
        task  = rec.get("Task")
        ratio = rec.get("Ratio")
        n     = rec.get("N_Samples") or 0

        if model not in MODELS or task not in DATASETS:
            continue
        if ratio not in allowed or n <= 0 or rec.get("PDU_Energy_J") is None:
            continue

        data[model][task][ratio] = {
            "energy":   rec["PDU_Energy_J"] / n,
            "accuracy": rec.get("Accuracy"),
            "cot":      rec.get("Avg_COT_Length"),
        }
    return data


def load_prompt_lengths() -> dict:
    """prompt[model][task] = mean_tokens"""
    with open(PROMPT_PATH) as f:
        records = json.load(f)

    prompt = {}
    for rec in records:
        model = rec.get("Model")
        task  = rec.get("Benchmark", "").lower()
        if model and task:
            prompt.setdefault(model, {})[task] = rec.get("Mean_Tokens")
    return prompt


def fmt_pct(val) -> str:
    if val is None:
        return "n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"

def fmt_pp(val) -> str:
    if val is None:
        return "n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f} pp"


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    inference = load_inference()
    prompt    = load_prompt_lengths()

    lines = []

    for model in MODELS:
        for task in DATASETS:
            baseline = inference[model][task].get(1.0)
            if baseline is None:
                continue

            e1   = baseline["energy"]
            acc1 = baseline["accuracy"]
            cot1 = baseline["cot"]
            mean_prompt = prompt.get(model, {}).get(task)

            header = f"{MODEL_LABEL[model]}, {DATASET_LABEL[task]}"
            lines.append("=" * len(header))
            lines.append(header)
            lines.append("=" * len(header))
            lines.append(
                f"  Baseline (gamma1.0): energy={e1:.4f} J/sample, "
                f"accuracy={acc1:.4f}, CoT={cot1:.1f} tokens"
                + (f", prompt={mean_prompt:.1f} tokens" if mean_prompt else "")
            )
            lines.append("")

            ratio_rows = []
            for ratio in RATIOS:
                entry = inference[model][task].get(ratio)
                if entry is None:
                    continue

                e_saved   = ((e1 - entry["energy"]) )/ e1 * 100          if e1 else None
                acc_lost  = (acc1 - entry["accuracy"]) * 100           if acc1 is not None and entry["accuracy"] is not None else None
                acc_lost_rel = ((acc1 - entry["accuracy"]) / acc1) * 100 if acc1 and entry["accuracy"] is not None else None
                cot_red   = ((cot1 - entry["cot"]) / cot1 )* 100        if cot1 else None
                # prompt/CoT = mean_prompt_tokens / avg_cot_tokens_at_gamma * 100
                p_cot    = mean_prompt / entry["cot"] * 100            if mean_prompt and entry["cot"] else None

                ratio_rows.append({
                    "ratio": ratio, "e_saved": e_saved,
                    "acc_lost": acc_lost, "acc_lost_rel": acc_lost_rel,
                    "cot_red": cot_red, "p_cot": p_cot,
                })

                p_cot_str = (
                    f" | prompt/CoT [prompt_tokens/cot_tokens_at_gamma]: {fmt_pct(p_cot):>8}"
                    if p_cot is not None else ""
                )
                lines.append(
                    f"  gamma{ratio:.1f} | "
                    f"energy saved [(E(1)-E(gamma))/E(1)]: {fmt_pct(e_saved):>8} | "
                    f"accuracy [Acc(gamma)={entry['accuracy']:.4f}, Acc(1)={acc1:.4f}] "
                    f"lost [Acc(1)-Acc(gamma)]: {fmt_pp(acc_lost):>10} ({fmt_pct(acc_lost_rel):>8} of baseline) | "
                    f"CoT reduction [(CoT(1)-CoT(gamma))/CoT(1)]: {fmt_pct(cot_red):>8}"
                    + p_cot_str
                )

            # Best configurations
            valid = [r for r in ratio_rows if r["acc_lost"] is not None and r["e_saved"] is not None]
            if valid:
                best_acc    = min(valid, key=lambda r: r["acc_lost"])
                best_energy = max(valid, key=lambda r: r["e_saved"])

                # Best trade-off: normalise both metrics to [0,1] then maximise
                # (energy_saved_norm - acc_lost_norm) -- higher is better
                e_vals   = [r["e_saved"]  for r in valid]
                a_vals   = [r["acc_lost"] for r in valid]
                e_min, e_max = min(e_vals), max(e_vals)
                a_min, a_max = min(a_vals), max(a_vals)
                def tradeoff_score(r):
                    e_norm = (r["e_saved"]  - e_min) / (e_max - e_min) if e_max != e_min else 0
                    a_norm = (r["acc_lost"] - a_min) / (a_max - a_min) if a_max != a_min else 0
                    return e_norm - a_norm
                best_tradeoff = max(valid, key=tradeoff_score)

                lines.append("")
                lines.append(f"  >> Best accuracy:   gamma{best_acc['ratio']:.1f}"
                             f" | accuracy lost: {fmt_pp(best_acc['acc_lost'])}"
                             f" | energy saved: {fmt_pct(best_acc['e_saved'])}")
                lines.append(f"  >> Best energy:     gamma{best_energy['ratio']:.1f}"
                             f" | energy saved: {fmt_pct(best_energy['e_saved'])}"
                             f" | accuracy lost: {fmt_pp(best_energy['acc_lost'])}")
                lines.append(f"  >> Best trade-off:  gamma{best_tradeoff['ratio']:.1f}"
                             f" | energy saved: {fmt_pct(best_tradeoff['e_saved'])}"
                             f" | accuracy lost: {fmt_pp(best_tradeoff['acc_lost'])}"
                             f" [score = (E_saved(gamma)-E_saved_min)/(E_saved_max-E_saved_min)"
                             f" - (Acc_lost(gamma)-Acc_lost_min)/(Acc_lost_max-Acc_lost_min),"
                             f" where min/max are across gamma∈{{0.1,...,0.9}} for this block]")

            lines.append("")

    # -----------------------------------------------------------------------
    # Prompt length variation section
    # -----------------------------------------------------------------------
    lines.append("=" * 60)
    lines.append("PROMPT LENGTH VARIATION")
    lines.append("=" * 60)
    lines.append("")

    def pairwise(items: dict, label_map: dict) -> list[str]:
        """For each item, print % difference vs every other item."""
        out = []
        keys = list(items.keys())
        for k in keys:
            v = items[k]
            others = ", ".join(
                f"{label_map[o]}: {fmt_pct((v - items[o]) / items[o] * 100)}"
                for o in keys if o != k
            )
            out.append(f"    {label_map[k]:15s}: {v:.1f} tokens  |  vs  {others}")
        return out

    # Per dataset: pairwise comparison across models
    lines.append("-- Prompt length: pairwise comparison across models, per benchmark --")
    lines.append("")
    for task in DATASETS:
        vals = {m: prompt.get(m, {}).get(task) for m in MODELS}
        vals = {m: v for m, v in vals.items() if v is not None}
        if not vals:
            continue
        lines.append(f"  {DATASET_LABEL[task]}")
        lines += pairwise(vals, MODEL_LABEL)
        lines.append("")

    # Per model: pairwise comparison across benchmarks
    lines.append("-- Prompt length: pairwise comparison across benchmarks, per model --")
    lines.append("")
    for model in MODELS:
        vals = {d: prompt.get(model, {}).get(d) for d in DATASETS}
        vals = {d: v for d, v in vals.items() if v is not None}
        if not vals:
            continue
        lines.append(f"  {MODEL_LABEL[model]}")
        lines += pairwise(vals, DATASET_LABEL)
        lines.append("")

    output = "\n".join(lines)
    with open(OUT_PATH, "w") as f:
        f.write(output)

    print(output)
    print(f"\nSaved to {os.path.relpath(OUT_PATH, BASE_DIR)}")


if __name__ == "__main__":
    main()
