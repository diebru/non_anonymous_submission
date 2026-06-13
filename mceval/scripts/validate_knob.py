#!/usr/bin/env python3
"""Phase-4 knob validation: does the SFT'd model HONOR gamma? SERVER (GPU).

The critical Phase-4 gate (roadmap s8): before trusting any accuracy-vs-CoT curve,
confirm the LoRA-SFT'd model actually shortens its reasoning as the gamma marker
drops. We run the base model + our SFT adapter on a small held-out sample at several
gamma, parse each output with the frozen contract, and measure ``cot_token_count``
(the curve x-axis) per gamma. If the median does not fall monotonically with gamma,
the knob is broken and the curves would be meaningless.

This is a length/compliance check only -- it does NOT execute code (no Docker); the
full 12-gamma test sweep + McEval scoring is the next step once the knob is proven.

Usage (server, tokenskip_env, GPU):
    # base + adapter (the original Phase-4 knob gate)
    python3 scripts/validate_knob.py --model qwen2.5-coder-3b-instruct \
        --adapter /.../weights/qwen2.5-coder-3b-instruct/lora_sft_run01 --limit 3

    # MERGED checkpoint, no adapter (the merge-equivalence gate -- must match the above)
    python3 scripts/validate_knob.py --model qwen2.5-coder-3b-instruct \
        --model-path /.../weights/qwen2.5-coder-3b-instruct/merged_sft_run01 --limit 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from tsmc.config import get_paths  # noqa: E402
from tsmc.constants import MODEL_IDS, family_of  # noqa: E402
from tsmc.contract import parse_generation, three_way_outcome  # noqa: E402
from tsmc.inference import prompts as P  # noqa: E402
from tsmc.inference.runner import RunnerConfig, VLLMRunner  # noqa: E402

# Coarse gamma set for the knob check (endpoints + a few midpoints). The full
# GAMMA_GRID sweep is the later test run; here we just need the trend.
DEFAULT_GAMMAS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1)


def resolve_model_repo(model_id: str, override: str | None, paths) -> str | None:
    if override:
        return override
    for name in ("run_metadata.yaml", "run_metadata.example.yaml"):
        meta = paths.configs_dir / name
        if meta.is_file():
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            repo = ((data.get("models") or {}).get(model_id) or {}).get("hf_repo")
            if repo:
                return repo
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder-3b-instruct", choices=MODEL_IDS)
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir (Phase-4 SFT output). OMIT to validate a MERGED "
                         "checkpoint passed via --model-path (the merge-equivalence gate).")
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--limit", type=int, default=3, help="problems per language (small sample)")
    ap.add_argument("--trio-only", action="store_true", help="Python/C/Rust only (smallest sample)")
    ap.add_argument("--gammas", type=float, nargs="+", default=list(DEFAULT_GAMMAS))
    ap.add_argument("--model-path", default=None, help="base model source (else run_metadata hf_repo)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--system", default=None,
                    help="pinned system prompt (must match the model's SFT/inference contract)")
    ap.add_argument("--min-shrink", type=float, default=0.30,
                    help="require median CoT at the lowest gamma <= (1-this) x the gamma=1.0 median")
    args = ap.parse_args()
    paths = get_paths()

    repo = resolve_model_repo(args.model, args.model_path, paths)
    if not repo:
        print("ERROR: no base-model repo (set models.<id>.hf_repo or pass --model-path).", file=sys.stderr)
        return 2
    adapter = pathlib.Path(args.adapter) if args.adapter else None
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        print(f"ERROR: no adapter_config.json under {adapter}", file=sys.stderr)
        return 2
    if adapter is None and not args.model_path:
        # Without an adapter we validate the model at --model-path directly; that path
        # MUST be the merged checkpoint, else we'd silently test the bare base (no SFT).
        print("ERROR: validating a merged checkpoint needs --model-path <merged dir> "
              "(or pass --adapter to validate base+adapter).", file=sys.stderr)
        return 2

    units = P.select_units("generation", args.split, paths, trio_only=args.trio_only, limit=args.limit)
    gammas = sorted(set(args.gammas), reverse=True)
    src_tag = f"adapter={adapter.name}" if adapter else f"merged={pathlib.Path(repo).name}"
    print("=" * 70)
    print(f"Knob validation | model={args.model} {src_tag}")
    print(f"sample={len(units)} ({args.split}, limit={args.limit}, trio={args.trio_only}) gammas={gammas}")
    print("=" * 70)

    runner = VLLMRunner(RunnerConfig(
        model_path=repo, lora_path=str(adapter) if adapter else None, max_lora_rank=16,
        tensor_parallel_size=args.tensor_parallel_size, max_tokens=args.max_tokens,
    )).load()

    family = family_of(args.model)
    per_gamma: dict[float, dict] = {}
    for g in gammas:
        prompts = [runner.render(P.chat_messages(P.reasoning_user_text(u, g, family), args.system)) for u in units]
        outs = runner.generate(prompts)
        cots, fmt_fail = [], 0
        for u, o in zip(units, outs):
            pr = parse_generation(o.text, entry_point=u.entry_point, finish_reason=o.finish_reason)
            if three_way_outcome(pr.status, passed=True) == "format_fail":
                fmt_fail += 1
            cots.append(runner.count_tokens(pr.cot_text))
        per_gamma[g] = {
            "n": len(units),
            "median_cot": statistics.median(cots) if cots else 0,
            "mean_cot": round(statistics.mean(cots), 1) if cots else 0,
            "format_fail": fmt_fail,
            "format_fail_rate": round(fmt_fail / len(units), 3) if units else None,
        }
        s = per_gamma[g]
        print(f"  gamma={g:<5g} median_cot={s['median_cot']:<6} mean={s['mean_cot']:<7} "
              f"format_fail={s['format_fail']}/{s['n']} ({s['format_fail_rate']})")

    # --- gate: median CoT non-increasing as gamma falls + clear shrink end-to-end ---
    medians = [per_gamma[g]["median_cot"] for g in gammas]  # gamma-descending
    up_steps = [(gammas[i], gammas[i + 1]) for i in range(len(medians) - 1) if medians[i + 1] > medians[i]]
    hi, lo = medians[0], medians[-1]
    shrink = (hi - lo) / hi if hi else 0.0
    monotonic = not up_steps
    enough_shrink = shrink >= args.min_shrink
    ok = monotonic and enough_shrink

    out = {
        "model_id": args.model,
        "adapter": str(adapter) if adapter else None,
        "merged_model_path": None if adapter else repo,
        "split": args.split,
        "n_sample": len(units), "gammas": gammas, "per_gamma": {f"{g:g}": per_gamma[g] for g in gammas},
        "median_series": medians, "up_steps": up_steps,
        "shrink_top_to_bottom": round(shrink, 3), "min_shrink": args.min_shrink,
        "monotonic": monotonic, "pass": ok,
    }
    # Distinct artifact for the merged gate so it never clobbers the base+adapter result.
    fname = "knob_validation.json" if adapter else "knob_validation_merged.json"
    out_path = paths.generations_dir / args.model / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nmedian CoT gamma=1.0 -> {gammas[-1]:g}: {hi} -> {lo}  (shrink {shrink:.0%})")
    print(f"monotonic non-increasing: {'OK' if monotonic else 'VIOLATED ' + str(up_steps)}")
    print("\n" + ("RESULT: PASS  (the model honors gamma -> proceed to the full test sweep)"
                  if ok else "RESULT: FAIL  (knob not working -- curves would be meaningless)"))
    if not ok and not enough_shrink:
        print(f"  shrink {shrink:.0%} < required {args.min_shrink:.0%}", file=sys.stderr)
    print(f"\nwrote {out_path}")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
