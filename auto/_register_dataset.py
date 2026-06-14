#!/usr/bin/env python3
"""Register an SFT json in LLaMA-Factory's data/dataset_info.json (alpaca format).

Usage: python _register_dataset.py <dataset_info.json> <dataset_key> <file_name>
The <file_name> is resolved by LLaMA-Factory relative to its data/ dir, so pass the
basename of a json that lives in LlamaFactory/data/.
"""
import json, os, sys

info_path, key, file_name = sys.argv[1], sys.argv[2], sys.argv[3]
info = {}
if os.path.exists(info_path):
    with open(info_path) as f:
        info = json.load(f)
info[key] = {
    "file_name": file_name,
    "columns": {"prompt": "instruction", "query": "input", "response": "output"},
}
with open(info_path, "w") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print(f"registered dataset '{key}' -> {file_name} in {info_path}")
