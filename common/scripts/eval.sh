#!/bin/bash
BENCHMARK="math" # "gsm8k", "math"
MODEL_SIZE="8b"
MODEL_TYPE="llama3" # "llama3", "qwen"
DATA_TYPE="test" # "train", "test"

# 1. Set the compression ratio here (0.5, 0.6, 0.7, 0.8, 0.9)
COMPRESSION_RATIO=1.0

MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKENIZER_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"

OUPTUT_DIR="outputs/LLaMA-3.1-8B-Instruct/${BENCHMARK}/ratio_${COMPRESSION_RATIO}/"

# Generation Settings
MAX_NUM_EXAMPLES=100000000000000
MAX_NEW_TOKENS=512 # 512 for gsm8k, 1024 for math
EVAL_BATCH_SIZE=16
TEMPERATURE=0.0
SEED=42

CUDA_VISIBLE_DEVICES=0 python ../evaluation.py \
--output-dir ${OUPTUT_DIR} \
    --model-path ${MODEL_PATH} \
    --tokenizer-path ${TOKENIZER_PATH} \
    --model-size ${MODEL_SIZE} \
    --model-type ${MODEL_TYPE} \
    --data-type ${DATA_TYPE} \
    --max_num_examples ${MAX_NUM_EXAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --temperature ${TEMPERATURE} \
    --seed ${SEED} \
    --benchmark ${BENCHMARK} \
    --use_vllm \
    --compression_ratio ${COMPRESSION_RATIO}