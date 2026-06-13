#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model via peft -> standalone HF checkpoint.

SERVER-ONLY (loads the full model). **Run in `tokenskip_env`** (the inference env):
it carries the same transformers/peft stack vLLM uses, so the exported tokenizer is
byte-compatible with the engine (no post-hoc tokenizer restore needed) and we avoid
the `llamafactory-cli export` PEFT-merge bug seen on the server's transformers 5.2.0
(that export produced a degenerate model: the SFT'd Coder-3B collapsed to ~0 CoT at
every gamma; base+adapter via vLLM LoRARequest, by contrast, validates 159->23).

This is the standard, proven merge (peft `merge_and_unload`), adapted from the
project's reference `example_merge.py`. Hardened vs that reference:
  * --revision pins the base HF commit (merged = base + dLoRA; a base != the SFT
    base silently corrupts the merge).
  * merge math in fp32, weights saved in bf16 (--merge-dtype / --save-dtype): the
    delta is added at full precision, then cast to the bf16 the adapter ran at.
  * asserts the adapter dir is real before loading the (large) base model.

Usage (server, tokenskip_env, CPU is fine -- leaves the A6000s free):
    python3 scripts/merge_lora.py \
        --base Qwen/Qwen2.5-Coder-3B-Instruct \
        --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 \
        --adapter "$PWD/weights/qwen2.5-coder-3b-instruct/lora_sft_run01" \
        --output  "$PWD/weights/qwen2.5-coder-3b-instruct/merged_sft_run01"

Then validate the knob on the merged model (must reproduce ~159 -> 23 monotonic):
    python3 scripts/validate_knob.py --model qwen2.5-coder-3b-instruct \
        --model-path "$PWD/weights/qwen2.5-coder-3b-instruct/merged_sft_run01" --limit 3
"""
from __future__ import annotations

import argparse
import pathlib

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", required=True, help="base model repo id or local dir (== the SFT base)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (Phase-4 SFT output)")
    ap.add_argument("--output", required=True, help="destination dir for the merged checkpoint")
    ap.add_argument("--revision", default=None, help="pin the base HF commit (e.g. 488639f1...)")
    ap.add_argument("--merge-dtype", choices=list(_DTYPES), default="float32",
                    help="dtype the LoRA delta is added in (fp32 = no precision loss)")
    ap.add_argument("--save-dtype", choices=list(_DTYPES), default="bfloat16",
                    help="dtype the merged weights are saved in (bf16 = the inference dtype)")
    ap.add_argument("--device-map", default="cpu",
                    help="'cpu' (safe, leaves GPUs free) or 'auto' (uses GPU, faster)")
    args = ap.parse_args()

    adapter = pathlib.Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"ERROR: no adapter_config.json under {adapter}")
    out = pathlib.Path(args.output)
    if (out / "config.json").is_file():
        raise SystemExit(
            f"ERROR: {out} already has a model (config.json). Remove it first so stale "
            f"shards/index can't shadow the new merge:  rm -rf {out}"
        )

    merge_dtype = _DTYPES[args.merge_dtype]
    save_dtype = _DTYPES[args.save_dtype]
    print("=" * 70)
    print(f"[merge] base    = {args.base}" + (f" @ {args.revision}" if args.revision else ""))
    print(f"[merge] adapter = {adapter}")
    print(f"[merge] merge_dtype={args.merge_dtype}  save_dtype={args.save_dtype}  "
          f"device_map={args.device_map}")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained(args.base, revision=args.revision, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.revision, torch_dtype=merge_dtype,
        device_map=args.device_map, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    print("[merge] merge_and_unload() ...")
    merged = model.merge_and_unload()
    if save_dtype != merge_dtype:
        merged = merged.to(save_dtype)

    out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged checkpoint -> {out}")
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print("[merge] done. Next: validate the knob (must reproduce ~159 -> 23 monotonic).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
