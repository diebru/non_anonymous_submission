#!/bin/bash
# Single evaluation run for the piqa benchmark. Run from this folder: bash run.sh
# The shared engine lives in ../common; this folder holds the dataset + configs.
set -e

BENCHMARK="piqa"
MODEL_TYPE="qwen"          # "qwen" or "llama3"
MODEL_SIZE="7b"
DATA_TYPE="test"           # "train" or "test"
COMPRESSION_RATIO=1.0      # 1.0 = baseline; <1.0 = TokenSkip-compressed model

# Base model (or a merged TokenSkip model) to evaluate.
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_PATH="${MODEL_PATH}"

OUTPUT_DIR="outputs/${BENCHMARK}/ratio_${COMPRESSION_RATIO}/"

CUDA_VISIBLE_DEVICES=0 python ../common/evaluation.py \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --model-size "${MODEL_SIZE}" \
    --model-type "${MODEL_TYPE}" \
    --data-type "${DATA_TYPE}" \
    --max_new_tokens 256 \
    --eval_batch_size 16 \
    --temperature 0.0 \
    --seed 42 \
    --benchmark "${BENCHMARK}" \
    --use_vllm \
    --compression_ratio "${COMPRESSION_RATIO}"
