#!/usr/bin/env python3
"""Run the repo's CoT-compression + SFT-file builder for ONE (model, benchmark).

Calls the existing functions in common/ (no logic duplicated):
  - common/LLMLingua_iterato.py::data_processing_gsm8k  (filter correct -> formatted -> LLMLingua-2 compress)
  - common/get_llamafactory_input_all_{qwen,llama}.py::process_dataset  (build the SFT json)

Usage:
  python _compress_build.py --bench gsm8k --model-folder Qwen2.5-7b-Instruct \
      --size 7b --mtype qwen --model-name qwen2.5 --builder qwen \
      --bench-dir /abs/repo/gsm8k --llmlingua /path/or/hf-id
"""
import argparse, importlib.util, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = os.path.join(REPO, "common")


def load(modfile):
    path = os.path.join(COMMON, modfile)
    spec = importlib.util.spec_from_file_location(modfile[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--model-folder", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--mtype", required=True, choices=["qwen", "llama3"])
    ap.add_argument("--model-name", required=True)          # qwen2.5 | llama3.1
    ap.add_argument("--builder", required=True, choices=["qwen", "llama"])
    ap.add_argument("--bench-dir", required=True)           # <repo>/<bench>
    ap.add_argument("--llmlingua", required=True)
    a = ap.parse_args()

    base = os.path.join(a.bench_dir, "outputs", a.model_folder, a.bench)
    input_dir = os.path.join(base, a.size) + os.sep        # holds Original/.. and Compression/..
    preds = os.path.join(input_dir, "Original", "train", "samples", "predictions.jsonl")
    if not os.path.exists(preds):
        sys.exit(f"ERROR: baseline train predictions not found: {preds}\n"
                 f"Run the baseline train-gen step (evaluation.py --data-type train) first.")

    # 1) compress (writes predictions_formatted.jsonl + Compression/train_outputs_compressed_ratio_*.jsonl)
    comp = load("LLMLingua_iterato.py")
    print(f">> compressing CoT for {a.model_folder}/{a.bench} ({a.size})", flush=True)
    comp.data_processing_gsm8k(input_dir=input_dir, model_type=a.mtype, llmlingua_path=a.llmlingua)

    # 2) build the LLaMA-Factory SFT json -> base/mydataset_<bench>_<model_name>_<size>.json
    builder = load(f"get_llamafactory_input_all_{a.builder}.py")
    print(f">> building SFT file for {a.model_folder}/{a.bench}", flush=True)
    builder.seed_everything(42)
    builder.process_dataset(dataset_name=a.bench, base_input_dir=input_dir,
                            output_dir=base + os.sep, model_name=a.model_name, model_size=a.size)

    out = os.path.join(base, f"mydataset_{a.bench}_{a.model_name}_{a.size}.json")
    print(f"SFT_JSON={out}")


if __name__ == "__main__":
    main()
