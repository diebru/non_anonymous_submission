#!/usr/bin/env python3
"""Upload TokenSkip LoRA adapters to the Hub: one repo per Qwen model, 5 subfolders.

Layout produced:
  <ns>/<prefix>-qwen2.5-<size>/
      boolq/   gsm8k/   math/   piqa/      <- from LlamaFactory/lora_saves/<Model>/lora/<bench>_test
      mceval/                              <- from tokenskip_mceval3/weights/<model>/lora_sft_run01

Only adapter + tokenizer files are pushed (checkpoints, optimizer state, logs excluded).
Reproduction loads base + adapter live (vLLM LoRARequest), so base weights are NOT redistributed.
"""
import argparse, os, sys
from huggingface_hub import HfApi, upload_folder

ALLOW = [
    "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "special_tokens_map.json", "vocab.json", "merges.txt",
    "added_tokens.json", "chat_template.jinja", "README.md",
]
IGNORE = ["checkpoint-*/*", "*/optimizer.pt", "*/scheduler.pt", "global_step*/*"]

# size -> (reasoning model folder, mceval weights folder)
SIZE_DIRS = {
    "3b":  ("Qwen2.5-3B-Instruct",  "qwen2.5-3b-instruct"),
    "7b":  ("Qwen2.5-7B-Instruct",  "qwen2.5-7b-instruct"),
    "14b": ("Qwen2.5-14B-Instruct", "qwen2.5-14b-instruct"),
}


def adapter_src(size, bench, lora_saves, mceval_weights):
    rfolder, mfolder = SIZE_DIRS[size]
    if bench == "mceval":
        return os.path.join(mceval_weights, mfolder, "lora_sft_run01")
    return os.path.join(lora_saves, rfolder, "lora", f"{bench}_test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--prefix", default="tokenskip")
    ap.add_argument("--private", type=int, default=1)
    ap.add_argument("--sizes", default="3b 7b 14b")
    ap.add_argument("--benches", default="boolq gsm8k math piqa mceval")
    ap.add_argument("--lora-saves", required=True)
    ap.add_argument("--mceval-weights", required=True)
    ap.add_argument("--dry-run", type=int, default=0)
    a = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    for size in a.sizes.split():
        if size not in SIZE_DIRS:
            print(f"!! unknown size {size}, skip"); continue
        repo_id = f"{a.namespace}/{a.prefix}-qwen2.5-{size}"
        print(f"\n=== repo {repo_id} (private={bool(a.private)}) ===")
        if not a.dry_run:
            api.create_repo(repo_id, repo_type="model", private=bool(a.private), exist_ok=True)
        for bench in a.benches.split():
            src = adapter_src(size, bench, a.lora_saves, a.mceval_weights)
            cfg = os.path.join(src, "adapter_config.json")
            if not os.path.isfile(cfg):
                print(f"  -- skip {bench}: no adapter at {src}")
                continue
            print(f"  -> upload {bench}  ({src})")
            if a.dry_run:
                continue
            upload_folder(
                repo_id=repo_id, repo_type="model", token=token,
                folder_path=src, path_in_repo=bench,
                allow_patterns=ALLOW, ignore_patterns=IGNORE,
                commit_message=f"add {bench} TokenSkip LoRA adapter",
            )
    print("\nUpload pass complete.")


if __name__ == "__main__":
    main()
